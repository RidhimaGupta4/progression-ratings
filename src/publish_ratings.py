"""Full-season ratings for presentation (the procedure itself was validated H1 -> H2 in ratings.py).

For each player with any minutes: posterior mean and 80% interval of open-play progression value per 90,
minutes, reliability at those minutes, and the probability of ranking in the top 10% of his league and
position group. Also the season path of each player's rating (closed-form normal update with the
full-season hyperparameters held fixed), used by the interactive page as an illustration of how a rating
firms up as minutes accumulate.
The model (pre-registered, or the player-specific-noise variant) is the one selected by the decision rule
in DESIGN.md, as recorded by ratings.py in results/model_choice.csv.
Outputs: results/player_ratings.csv, results/rating_paths.parquet, results/hyperparameters.csv,
         results/cell_means.csv, results/full_season_fit.csv
"""
import numpy as np
import pandas as pd

from common import ROOT
from ratings import GROUPS, SEED, fit, player_matches


def main():
    chosen = pd.read_csv(ROOT / "results" / "model_choice.csv").chosen.iloc[0]
    hetero = chosen == "progression_player_noise"
    pm = player_matches("progression")
    f = fit(pm, seed=SEED + 7, hetero=hetero)
    P = f["players"].copy()
    names = pd.read_parquet(ROOT / "data" / "players.parquet").set_index("player_id").display_name
    P["player"] = P.player_id.map(names).fillna(P.player)
    lam, tau, sigma, mu = (f["post"][k] for k in ("lam", "tau", "sigma", "mu"))
    gi = P.group.map({g: i for i, g in enumerate(GROUPS)}).values
    P["minutes"] = P.t * 90
    P["raw_rate"] = P.y / P.t
    P["rating"] = lam.mean(0)
    P["lo80"], P["hi80"] = np.quantile(lam, [0.1, 0.9], axis=0)
    tm, sm = tau.mean(0)[gi], f["post"]["sigma_p"].mean(0)        # this player's noise level
    P["noise_sd"] = sm
    P["reliability"] = tm ** 2 / (tm ** 2 + sm ** 2 / P.t)
    # probability of being in the top 10% of the league x group (players with >= 900 minutes form the pool)
    P["p_top10"] = np.nan
    for _cell, g in P[P.minutes >= 900].groupby("cell"):
        idx = g.index.values
        draws = lam[:, idx]
        cut = np.quantile(draws, 0.9, axis=1, keepdims=True)
        P.loc[idx, "p_top10"] = (draws >= cut).mean(0)
    P["rank_in_cell"] = P[P.minutes >= 900].groupby("cell").rating.rank(ascending=False)
    cols = ["player_id", "player", "team", "league", "group", "minutes", "raw_rate", "rating", "lo80", "hi80",
            "reliability", "noise_sd", "p_top10", "rank_in_cell"]
    P[cols].sort_values(["league", "group", "rating"], ascending=[True, True, False]).to_csv(
        ROOT / "results" / "player_ratings.csv", index=False)
    hyp = pd.DataFrame(dict(group=GROUPS, tau=tau.mean(0), sigma=sigma.mean(0)))
    if hetero:
        hyp["omega"] = f["post"]["omega"].mean(0)
    hyp.to_csv(ROOT / "results" / "hyperparameters.csv", index=False)
    cells = pd.DataFrame(dict(cell=f["cells"], mu=mu.mean(0)))
    cells.to_csv(ROOT / "results" / "cell_means.csv", index=False)

    # season paths: after each appearance, conjugate normal posterior with hyperparameters fixed
    M = pd.read_parquet(ROOT / "data" / "matches.parquet")[["match_id", "date"]]
    pm2 = pm.merge(M, on="match_id").sort_values(["player_id", "date"])
    tau_of = dict(zip(hyp.group, hyp.tau, strict=True))
    noise_of = dict(zip(P.player_id, P.noise_sd, strict=True))
    cm = dict(zip(cells.cell, cells.mu, strict=True))
    rows = []
    for pid, g in pm2.groupby("player_id"):
        info = P.loc[P.player_id == pid].iloc[0]
        t_, s_ = tau_of[info.group], noise_of[pid]
        m0 = cm[info.cell]
        Y = T = 0.0
        for r in g.itertuples():
            Y += r.y
            T += r.t
            prec = 1 / t_ ** 2 + T / s_ ** 2
            mean = (m0 / t_ ** 2 + Y / s_ ** 2) / prec
            rows.append(dict(player_id=pid, date=r.date, minutes=T * 90, raw=Y / T, post=mean,
                             lo80=mean - 1.2816 / np.sqrt(prec), hi80=mean + 1.2816 / np.sqrt(prec)))
    pd.DataFrame(rows).to_parquet(ROOT / "results" / "rating_paths.parquet")
    pd.DataFrame([dict(model=chosen, players=len(P), rhat_max=f["rhat"], ess_min=f["ess"],
                       divergences=f["divergences"])]).to_csv(ROOT / "results" / "full_season_fit.csv", index=False)
    print(f"full-season fit: {len(P)} players, R-hat max {f['rhat']:.3f}, ESS min {f['ess']:.0f}, "
          f"divergences {f['divergences']}")
    show = P[P.minutes >= 1500].sort_values("rating", ascending=False)
    print(show.head(15)[["player", "team", "league", "group", "minutes", "rating", "lo80", "hi80", "p_top10"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
