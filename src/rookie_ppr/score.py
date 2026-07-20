from __future__ import annotations

import argparse
import sys

from rookie_ppr.score_runner import score_player


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score an incoming rookie from partial inputs")
    parser.add_argument("--position", required=True, choices=["QB", "RB", "WR", "TE"])
    parser.add_argument("--draft-year", type=int, dest="draft_year")
    parser.add_argument("--draft-round", type=int, dest="draft_round")
    parser.add_argument("--draft-overall", type=int, dest="draft_overall")
    parser.add_argument("--college", dest="college")
    parser.add_argument("--draft-team", dest="draft_team")
    parser.add_argument("--recruiting-rank", type=float, dest="recruiting_rank")
    parser.add_argument("--recruiting-stars", type=float, dest="recruiting_stars")
    parser.add_argument("--hs-class", type=int, dest="hs_class")
    parser.add_argument("--age-at-draft", type=float, dest="age_at_draft")
    parser.add_argument("--ff-adp", type=float, dest="ff_adp")
    parser.add_argument("--ff-adp-rank", type=float, dest="ff_adp_rank")
    parser.add_argument("--forty", type=float, dest="forty")
    parser.add_argument("--cone", type=float, dest="cone")
    parser.add_argument("--shuttle", type=float, dest="shuttle")
    parser.add_argument("--vertical", type=float, dest="vertical")
    parser.add_argument("--broad-jump", type=float, dest="broad_jump")
    parser.add_argument("--wt", type=float, dest="wt")
    parser.add_argument("--ht", dest="ht")
    parser.add_argument("--cfb-pass-yards", type=float, dest="cfb_pass_yards")
    parser.add_argument("--cfb-pass-td", type=float, dest="cfb_pass_td")
    parser.add_argument("--cfb-rush-yards", type=float, dest="cfb_rush_yards")
    parser.add_argument("--cfb-rush-td", type=float, dest="cfb_rush_td")
    parser.add_argument("--cfb-rec", type=float, dest="cfb_rec")
    parser.add_argument("--cfb-rec-yards", type=float, dest="cfb_rec_yards")
    parser.add_argument("--cfb-rec-td", type=float, dest="cfb_rec_td")
    parser.add_argument("--sos-opp-win-pct", type=float, dest="sos_opp_win_pct")
    parser.add_argument("--team-opportunity-ppr", type=float, dest="team_opportunity_ppr")
    parser.add_argument("--off-pass-rate-proxy", type=float, dest="off_pass_rate_proxy")
    parser.add_argument("--off-pass-yards", type=float, dest="off_pass_yards")
    parser.add_argument("--off-rush-yards", type=float, dest="off_rush_yards")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    row = {k: v for k, v in vars(args).items() if v is not None}
    result = score_player(row)
    print(f"Position: {result['position']}")
    print(f"Predicted rookie PPR: {result['predicted_rookie_ppr']:.1f}")
    print(f"Success score (0-100): {result['success_score_0_100']:.1f}")
    print(
        f"Composite groups populated ({len(result['composite_populated'])}): "
        f"{', '.join(result['composite_populated']) or 'none'}"
    )
    if result["composite_missing"]:
        print(
            f"Composite groups missing ({len(result['composite_missing'])}): "
            f"{', '.join(result['composite_missing'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
