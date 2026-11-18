"""Export the aggregated results the interactive page needs to explorer/data.js.

Only derived analysis is exported (per-player ratings and cumulative season paths); no StatsBomb event data.
Players with at least 450 minutes are included.
Usage: python explorer/export.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MIN_MINUTES = 450


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def main():
    P = pd.read_csv(RES / "player_ratings.csv")
    P = P[P.minutes >= MIN_MINUTES].sort_values("rating", ascending=False).reset_index(drop=True)
    paths = pd.read_parquet(RES / "rating_paths.parquet")
    paths = paths[paths.player_id.isin(P.player_id)]
    hyp = pd.read_csv(RES / "hyperparameters.csv")
    rel = pd.read_csv(RES / "reliability.csv")
    chosen = pd.read_csv(RES / "model_choice.csv").chosen.iloc[0]
    rel = rel[(rel.measure == chosen) & (rel.reliability == 0.7)]
    cells = pd.read_csv(RES / "cell_means.csv")
    mse = pd.read_csv(RES / "rating_validation_mse.csv")
    m = mse[(mse.measure == chosen) & (mse.analysis == "H1->H2")].set_index("predictor")
    cov = pd.read_csv(RES / "rating_validation_coverage.csv")
    cov = cov[cov.measure == chosen].set_index("level").coverage
    parts = pd.read_csv(RES / "rating_validation_coverage_by_rating.csv")
    parts = parts[(parts.measure == chosen) & (parts.level == 0.8)].set_index("part").coverage

    players = [[int(x.player_id), x.player, x.team, x.league, x.group, int(round(x.minutes)), r(x.raw_rate),
                r(x.rating), r(x.lo80), r(x.hi80), r(x.reliability, 3), r(x.p_top10, 3),
                None if np.isnan(x.rank_in_cell) else int(x.rank_in_cell)] for x in P.itertuples()]
    path = {}
    for pid, g in paths.groupby("player_id"):
        path[int(pid)] = [[int(round(x.minutes)), r(x.post), r(x.lo80), r(x.hi80), r(x.raw)] for x in g.itertuples()]
    data = dict(
        players=players,
        fields=["id", "name", "team", "league", "group", "minutes", "raw", "rating", "lo80", "hi80", "reliability",
                "p_top10", "rank"],
        paths=path,
        cell_means={c: r(v) for c, v in zip(cells.cell, cells.mu, strict=True)},
        full_season_hyper={g: dict(tau=r(t), sigma=r(s))
                           for g, t, s in zip(hyp.group, hyp.tau, hyp.sigma, strict=True)},
        validated_hyper={g: dict(tau=r(t), sigma=r(s)) for g, t, s in zip(rel.group, rel.tau, rel.sigma, strict=True)},
        validation=dict(n_players=int(m.n_players.iloc[0]), wmse_bayes=r(m.wmse["bayes_posterior_mean"], 6),
                        wmse_raw=r(m.wmse["raw_first_half"], 6), wmse_mean=r(m.wmse["league_group_mean"], 6),
                        cov50=r(cov[0.5], 3), cov80=r(cov[0.8], 3), cov95=r(cov[0.95], 3),
                        cov80_bottom_third=r(parts["bottom third"], 3), cov80_top_third=r(parts["top third"], 3),
                        cov80_top10=r(parts["top 10%"], 3)),
    )
    out = ROOT / "explorer" / "data.js"
    out.write_text("const DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print("wrote", out, f"{out.stat().st_size / 1024:.0f} KB,", len(players), "players")


if __name__ == "__main__":
    main()
