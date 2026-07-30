"""Walk-forward player-season projections, one refit per graded season.

To project fantasy season Y honestly, nothing that happened in Y or later may
touch the model. So for each Y we refit from scratch: train on target seasons up
to Y-2, tune hyperparameters on Y-1, then predict Y. That mirrors the production
recipe in ``veteran.train`` (median point model clamped to a predictive IQR) but
keeps every season's models in memory instead of overwriting the shipped
artifacts.

Players without a projection here are drafted rookies and anyone who missed the
prior season entirely; :mod:`rookie_ppr.league.backtest_rosters` fills those from
a draft-slot baseline fit on earlier seasons only.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from rookie_ppr.config import MODELS_DIR
from rookie_ppr.league.adp_features import ADP_FEATURE_COLS, attach_adp_features
from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_SEASONS,
    LEAGUE_WF_METRICS_FILE,
    LEAGUE_WF_PROJECTIONS_CSV,
)
from rookie_ppr.league.opportunity_features import (
    OPPORTUNITY_FEATURE_COLS,
    attach_opportunity_features,
)
from rookie_ppr.model_score import (
    DEFAULT_INJURY_IQR_RELIEF,
    DEFAULT_IQR_CLAMP_STRENGTH,
    PARAM_GRID,
    RB_PARAM_GRID,
    RB_PARAM_GRID_WITH_MARKET,
    _clamp_pred_to_iqr,
    _feature_matrix,
    _fit_point_model,
    _fit_predictive_quartile_pair,
    _pearson_r,
    _tune_position_model,
)
from rookie_ppr.league.backtest_rosters import (
    adp_curve_value,
    fit_adp_curves,
    _adp_with_actuals,
    _actual_season_ppr,
)
from rookie_ppr.veteran.config import (
    POSITION_FEATURE_COLS,
    SKILL_POSITIONS,
    TARGET,
    VET_FEATURES_CSV,
)

PROJECTION_COLS = [
    "target_season",
    "gsis_id",
    "player_name",
    "position",
    "team",
    "predicted_ppr",
    "pred_q25",
    "pred_q75",
    "prior_ppr",
    "prior_games",
    "actual_ppr",
]

# Below this a position-season fit is not worth trusting; those rows fall back.
MIN_TRAIN_ROWS = 40
MIN_TUNE_ROWS = 8
# Roughly the skill-position depth of a 10-team 16-round draft, plus cushion.
DEFAULT_TRAIN_TOP_N = 200

# Elite RB mean-reversion fix: when the median undershoots the walk-forward ADP
# curve for a top-ADP back, blend toward the market. Only lifts; never pulls a
# higher model number down. Weight is the share kept on the model.
ELITE_RB_ADP_MAX = 24
ELITE_RB_MODEL_WEIGHT = 0.40

# Features forced upward / downward for RB so early-ADP and high-volume leaves
# cannot be learned as "more usage → more regression."
_RB_MONO_UP = frozenset(
    {
        "ppr",
        "ppr_per_game",
        "touches",
        "carries",
        "targets",
        "receptions",
        "rushing_yards",
        "receiving_yards",
        "rushing_tds",
        "receiving_tds",
        "carry_share",
        "touch_share",
        "target_share",
        "pos_carry_share",
        "pos_touch_share",
        "pos_target_share",
        "ppr_trail3_mean",
        "lag1_ppr",
        "off_snap_pct",
    }
)
_RB_MONO_DOWN = frozenset(
    {"adp_rank", "adp_log_rank", "adp_pos_rank", "depth_rank"}
)


def _param_grid(position: str, *, use_adp: bool = False) -> list[dict[str, Any]]:
    if position == "RB":
        return RB_PARAM_GRID_WITH_MARKET if use_adp else RB_PARAM_GRID
    return PARAM_GRID


def _rb_monotonic_cst(cols: list[str]) -> list[int] | None:
    """Monotonic constraints aligned to ``cols`` for the RB point model."""
    cst = []
    for col in cols:
        if col in _RB_MONO_UP:
            cst.append(1)
        elif col in _RB_MONO_DOWN:
            cst.append(-1)
        else:
            cst.append(0)
    return cst if any(cst) else None


def _elite_rb_adp_lift(
    pred: float,
    *,
    adp_rank: float,
    curve: float,
    model_weight: float = ELITE_RB_MODEL_WEIGHT,
    adp_max: float = ELITE_RB_ADP_MAX,
) -> float:
    """Blend up toward the ADP curve when an elite RB is mean-reverted below it."""
    if not np.isfinite(pred):
        return pred
    if not np.isfinite(adp_rank) or adp_rank <= 0 or adp_rank > adp_max:
        return float(pred)
    if not np.isfinite(curve):
        return float(pred)
    if pred >= curve:
        return float(pred)
    w = float(np.clip(model_weight, 0.0, 1.0))
    return float(w * pred + (1.0 - w) * curve)


def _usable_feature_cols(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    """Drop all-null / single-valued columns; HGB binning needs 2+ distinct values.

    This matters more here than in production: an early walk-forward step has no
    Next Gen Stats history at all, so those columns are simply absent for it.
    """
    usable: list[str] = []
    for col in cols:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if int(values.notna().sum()) == 0 or int(values.nunique(dropna=True)) < 2:
            continue
        usable.append(col)
    return usable


def _filter_top_tier(pool: pd.DataFrame, top_n: int | None) -> pd.DataFrame:
    """Keep only players the market ranked inside the top ``top_n`` that season.

    Training on the whole panel teaches the model to separate zero from 80
    points. Drafts are decided inside the top 200, where that skill is mostly
    useless. Filtering here changes the loss surface to match the evaluation.
    ADP is pre-season information for the row's target_season, so no leakage.
    """
    if top_n is None:
        return pool
    if "adp_rank" not in pool.columns:
        raise ValueError(
            "train_top_n requires ADP ranks on the feature frame; "
            "attach_adp_features must run first"
        )
    rank = pd.to_numeric(pool["adp_rank"], errors="coerce")
    return pool[rank.notna() & (rank <= top_n)].copy()


def load_veteran_features() -> pd.DataFrame:
    path = CSV_OUTPUT_DIR / VET_FEATURES_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m rookie_ppr.veteran.compile"
        )
    frame = pd.read_csv(path, low_memory=False)
    frame["target_season"] = pd.to_numeric(frame.get("target_season"), errors="coerce")
    frame[TARGET] = pd.to_numeric(frame.get(TARGET), errors="coerce")
    return frame


def _project_one_season(
    features: pd.DataFrame,
    season: int,
    *,
    use_adp: bool = False,
    use_opportunity: bool = False,
    train_top_n: int | None = None,
    clamp: bool = True,
    clamp_strength: float = DEFAULT_IQR_CLAMP_STRENGTH,
    injury_relief: float = DEFAULT_INJURY_IQR_RELIEF,
    adp_curves: dict[str, np.ndarray] | None = None,
    elite_rb_lift: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit on < season and predict it. Returns (rows, per-position diagnostics)."""
    labeled = features[features[TARGET].notna() & features["target_season"].notna()]
    train_pool = _filter_top_tier(
        labeled[labeled["target_season"] <= season - 2], train_top_n
    )
    tune_pool = _filter_top_tier(
        labeled[labeled["target_season"] == season - 1], train_top_n
    )
    predict_pool = features[features["target_season"] == season]

    # Top-tier training shrinks the pool a lot, especially early seasons / TE.
    min_train = max(20, MIN_TRAIN_ROWS // 2) if train_top_n else MIN_TRAIN_ROWS
    min_tune = max(5, MIN_TUNE_ROWS // 2) if train_top_n else MIN_TUNE_ROWS

    frames: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}

    for position in SKILL_POSITIONS:
        pos_train = train_pool[train_pool["position"] == position]
        pos_tune = tune_pool[tune_pool["position"] == position]
        pos_predict = predict_pool[predict_pool["position"] == position]
        if pos_predict.empty:
            continue
        if len(pos_train) < min_train:
            diagnostics[position] = {"skipped": "insufficient_train", "n_train": len(pos_train)}
            continue

        # Thin tuning fold: carve the most recent slice off the training pool
        # instead, so hyperparameters are still picked out of sample.
        if len(pos_tune) < min_tune:
            ordered = pos_train.sort_values("target_season")
            cut = max(int(len(ordered) * 0.85), min_train // 2)
            pos_tune = ordered.iloc[cut:]
            pos_train = ordered.iloc[:cut]
            if len(pos_tune) < 5 or len(pos_train) < min_train // 2:
                diagnostics[position] = {"skipped": "no_tuning_fold"}
                continue

        candidate_cols = list(POSITION_FEATURE_COLS[position])
        if use_adp:
            candidate_cols += ADP_FEATURE_COLS
        if use_opportunity:
            candidate_cols += OPPORTUNITY_FEATURE_COLS
        cols = _usable_feature_cols(pos_train, candidate_cols)
        if len(cols) < 3:
            diagnostics[position] = {"skipped": "insufficient_features", "n_features": len(cols)}
            continue

        mono = _rb_monotonic_cst(cols) if position == "RB" else None
        _, params, tune_r = _tune_position_model(
            position,
            pos_train,
            pos_tune,
            cols,
            target=TARGET,
            param_grid=_param_grid(position, use_adp=use_adp),
            monotonic_cst=mono,
        )
        # Refit on train + tune once hyperparameters are chosen; both folds are
        # strictly older than the season being projected.
        fit_df = pd.concat([pos_train, pos_tune], ignore_index=True)
        point = _fit_point_model(
            fit_df, cols, params, target=TARGET, monotonic_cst=mono
        )
        q25_model, q75_model = _fit_predictive_quartile_pair(
            fit_df, cols, params, target=TARGET, monotonic_cst=mono
        )

        X = _feature_matrix(pos_predict, cols)
        pred = point.predict(X)
        q25 = q25_model.predict(X)
        q75 = q75_model.predict(X)
        prior_games = pd.to_numeric(pos_predict.get("games"), errors="coerce").to_numpy()
        # Soft IQR pull (not a hard clip): the predictive band is fit on realized
        # seasons, so full clamping bakes injury mean-reversion into every point
        # estimate. strength < 1 leaves more of the raw median intact. Short prior
        # seasons also get pulled toward q75 before that soft clamp.
        clamped = (
            [
                _clamp_pred_to_iqr(
                    float(p),
                    float(a),
                    float(b),
                    strength=clamp_strength,
                    prior_games=float(g) if np.isfinite(g) else None,
                    injury_relief=injury_relief,
                )
                for p, a, b, g in zip(pred, q25, q75, prior_games)
            ]
            if clamp
            else [float(p) for p in pred]
        )

        # Elite RB ADP lift: median HGB mean-reverts 300+ PPR backs; when the
        # walk-forward ADP curve (fit on earlier seasons only) is higher, blend up.
        if (
            elite_rb_lift
            and use_adp
            and position == "RB"
            and adp_curves
            and "adp_rank" in pos_predict.columns
        ):
            adp_ranks = pd.to_numeric(pos_predict["adp_rank"], errors="coerce").to_numpy()
            lifted: list[float] = []
            n_lifted = 0
            for value, rank in zip(clamped, adp_ranks):
                curve = adp_curve_value(adp_curves, "RB", float(rank)) if np.isfinite(rank) else float("nan")
                new_val = _elite_rb_adp_lift(float(value), adp_rank=float(rank), curve=curve)
                if abs(new_val - float(value)) > 1e-6:
                    n_lifted += 1
                lifted.append(new_val)
            clamped = lifted
        else:
            n_lifted = 0

        frames.append(
            pd.DataFrame(
                {
                    "target_season": season,
                    "gsis_id": pos_predict["gsis_id"].to_numpy(),
                    "player_name": pos_predict["player_name"].to_numpy(),
                    "position": position,
                    "team": pos_predict.get("team", pd.Series(index=pos_predict.index)).to_numpy(),
                    "predicted_ppr": clamped,
                    "pred_q25": q25,
                    "pred_q75": q75,
                    "prior_ppr": pd.to_numeric(pos_predict.get("ppr"), errors="coerce").to_numpy(),
                    "prior_games": prior_games,
                    "actual_ppr": pd.to_numeric(pos_predict[TARGET], errors="coerce").to_numpy(),
                }
            )
        )

        diagnostics[position] = {
            "n_train": int(len(fit_df)),
            "n_features": len(cols),
            "n_predicted": int(len(pos_predict)),
            "tune_r": round(float(tune_r), 4) if pd.notna(tune_r) else None,
            "best_params": params,
            "train_top_n": train_top_n,
            "monotonic": bool(mono),
            "elite_rb_lifted": int(n_lifted),
        }

    if not frames:
        return pd.DataFrame(columns=PROJECTION_COLS), diagnostics
    return pd.concat(frames, ignore_index=True)[PROJECTION_COLS], diagnostics


def walk_forward_projections(
    seasons: list[int] | None = None,
    features: pd.DataFrame | None = None,
    *,
    use_adp: bool = False,
    use_opportunity: bool = False,
    train_top_n: int | None = None,
    clamp: bool = True,
    clamp_strength: float = DEFAULT_IQR_CLAMP_STRENGTH,
    injury_relief: float = DEFAULT_INJURY_IQR_RELIEF,
    elite_rb_lift: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    seasons = list(seasons or LEAGUE_SEASONS)
    features = load_veteran_features() if features is None else features
    # Top-tier filtering needs ADP ranks even when ADP is not a model feature.
    if use_adp or train_top_n is not None:
        features = attach_adp_features(features)
    if use_opportunity:
        features = attach_opportunity_features(features)

    adp_actuals = None
    if elite_rb_lift and use_adp:
        try:
            adp_actuals = _adp_with_actuals(_actual_season_ppr())
        except Exception:  # noqa: BLE001 - lift is optional; model still runs
            adp_actuals = None

    frames: list[pd.DataFrame] = []
    metrics: dict[str, Any] = {
        "seasons": {},
        "use_adp": use_adp,
        "use_opportunity": use_opportunity,
        "train_top_n": train_top_n,
        "clamp": clamp,
        "clamp_strength": clamp_strength if clamp else 0.0,
        "injury_relief": injury_relief if clamp else 0.0,
        "elite_rb_lift": bool(elite_rb_lift and use_adp),
        "elite_rb_adp_max": ELITE_RB_ADP_MAX,
        "elite_rb_model_weight": ELITE_RB_MODEL_WEIGHT,
    }
    for season in seasons:
        curves = (
            fit_adp_curves(adp_actuals, before_season=season)
            if adp_actuals is not None
            else None
        )
        rows, diagnostics = _project_one_season(
            features,
            season,
            use_adp=use_adp,
            use_opportunity=use_opportunity,
            train_top_n=train_top_n,
            clamp=clamp,
            clamp_strength=clamp_strength,
            injury_relief=injury_relief,
            adp_curves=curves,
            elite_rb_lift=elite_rb_lift,
        )
        graded = rows.dropna(subset=["actual_ppr", "predicted_ppr"])
        season_block: dict[str, Any] = {
            "n_projected": int(len(rows)),
            "n_graded": int(len(graded)),
            "train_through": season - 2,
            "tuned_on": season - 1,
            "by_position": diagnostics,
        }
        if len(graded) >= 5:
            season_block["pearson_r"] = round(
                _pearson_r(graded["actual_ppr"], graded["predicted_ppr"].to_numpy()), 4
            )
            season_block["baseline_prior_ppr_r"] = (
                round(
                    _pearson_r(
                        graded.dropna(subset=["prior_ppr"])["actual_ppr"],
                        graded.dropna(subset=["prior_ppr"])["prior_ppr"].to_numpy(),
                    ),
                    4,
                )
                if graded["prior_ppr"].notna().sum() >= 5
                else None
            )
        metrics["seasons"][str(season)] = season_block
        frames.append(rows)
        print(
            f"  {season}: projected={len(rows):<4} graded={len(graded):<4} "
            f"r={season_block.get('pearson_r')} "
            f"(prior-season baseline r={season_block.get('baseline_prior_ppr_r')})"
        )

    projections = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PROJECTION_COLS)
    )
    graded_all = projections.dropna(subset=["actual_ppr", "predicted_ppr"])
    if len(graded_all) >= 5:
        metrics["overall_pearson_r"] = round(
            _pearson_r(graded_all["actual_ppr"], graded_all["predicted_ppr"].to_numpy()), 4
        )
        metrics["overall_n_graded"] = int(len(graded_all))
    return projections, metrics


def save_projections(
    projections: pd.DataFrame, metrics: dict[str, Any], *, tag: str = ""
) -> str:
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name = LEAGUE_WF_PROJECTIONS_CSV
    metrics_name = LEAGUE_WF_METRICS_FILE
    if tag:
        name = name.replace(".csv", f"_{tag}.csv")
        metrics_name = metrics_name.replace(".json", f"_{tag}.json")
    path = CSV_OUTPUT_DIR / name
    projections.to_csv(path, index=False)
    (MODELS_DIR / metrics_name).write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build walk-forward player-season projections for league backtesting"
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=list(LEAGUE_SEASONS))
    parser.add_argument(
        "--with-adp",
        action="store_true",
        help="Add pre-season FantasyPros ADP rank as a model feature",
    )
    parser.add_argument(
        "--with-opportunity",
        action="store_true",
        help="Add team/position usage-share features (target_share, carry_share, ...)",
    )
    parser.add_argument(
        "--train-top-n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Train only on players ranked ADP <= N in their target season "
            f"(e.g. {DEFAULT_TRAIN_TOP_N}). Restricts the loss to the population "
            "drafts actually care about."
        ),
    )
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help="Skip pulling the point estimate toward the predictive IQR",
    )
    parser.add_argument(
        "--clamp-strength",
        type=float,
        default=DEFAULT_IQR_CLAMP_STRENGTH,
        metavar="S",
        help=(
            "How hard to pull predictions into the predictive IQR "
            f"(0=off, 1=hard clip; default {DEFAULT_IQR_CLAMP_STRENGTH})"
        ),
    )
    parser.add_argument(
        "--injury-relief",
        type=float,
        default=DEFAULT_INJURY_IQR_RELIEF,
        metavar="W",
        help=(
            "How far short prior seasons pull toward q75 before soft-clamping "
            f"(0=off; default {DEFAULT_INJURY_IQR_RELIEF})"
        ),
    )
    parser.add_argument(
        "--no-elite-rb-lift",
        action="store_true",
        help="Disable blending elite-ADP RBs up toward the walk-forward ADP curve",
    )
    parser.add_argument("--tag", default="", help="Suffix for the output filenames")
    args = parser.parse_args()

    print("walk-forward refit per season (train <= Y-2, tune on Y-1, predict Y):")
    if args.train_top_n:
        print(f"  training filter: ADP rank <= {args.train_top_n}")
    if args.with_opportunity:
        print("  opportunity features: on")
    if not args.no_clamp:
        print(f"  IQR clamp strength: {args.clamp_strength}")
        print(f"  injury IQR relief: {args.injury_relief}")
    if args.with_adp and not args.no_elite_rb_lift:
        print(
            f"  elite RB ADP lift: ADP<={ELITE_RB_ADP_MAX}, "
            f"model_weight={ELITE_RB_MODEL_WEIGHT}"
        )
    projections, metrics = walk_forward_projections(
        args.seasons,
        use_adp=args.with_adp,
        use_opportunity=args.with_opportunity,
        train_top_n=args.train_top_n,
        clamp=not args.no_clamp,
        clamp_strength=args.clamp_strength,
        injury_relief=args.injury_relief,
        elite_rb_lift=not args.no_elite_rb_lift,
    )
    path = save_projections(projections, metrics, tag=args.tag)

    print(f"\nrows={len(projections)} -> {path}")
    print(
        f"overall graded r={metrics.get('overall_pearson_r')} "
        f"on n={metrics.get('overall_n_graded')}"
    )
    metrics_name = (
        LEAGUE_WF_METRICS_FILE.replace(".json", f"_{args.tag}.json")
        if args.tag
        else LEAGUE_WF_METRICS_FILE
    )
    print(f"metrics -> {MODELS_DIR / metrics_name}")


if __name__ == "__main__":
    main()
