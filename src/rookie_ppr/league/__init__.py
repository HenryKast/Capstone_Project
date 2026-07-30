"""ESPN fantasy league track: ingest, walk-forward projections, season simulation.

Production defaults (see ``python -m rookie_ppr.league.compile``):

* **Team track** — base walk-forward model blended with the ADP curve
  (weight fit walk-forward on earlier drafted picks).
* **Player track** — ADP + opportunity-share walk-forward model for rankings.
"""

__all__: list[str] = []
