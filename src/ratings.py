"""Bayesian player ratings with honest uncertainty, validated forward in time (H1 -> H2).

Measure: open-play ball-progression value = sum of action values of a player's open-play passes, crosses,
carries and dribbles (see DESIGN.md). Goalkeepers excluded.

Model (fitted on H1 only for validation; on the full season for the published ratings):
  y_pm ~ Normal(lambda_p * t_pm, sigma_p^2 * t_pm)      y: value in match m, t: minutes/90
  lambda_p ~ Normal(mu[league, group], tau_g)            partial pooling within league x position group
  sigma_p = sigma_g                                      pre-registered model
  sigma_p = sigma_g * exp(omega_g * z_p), z_p ~ N(0, 1)  variant: player-specific noise (DESIGN.md, deviation 5)
Priors are weakly informative and scaled using H1 data only (the H2 data never enter any fitted object).

Validation, players with >= 450 minutes in both halves:
  1. point prediction of the H2 rate: league-group mean vs raw H1 rate vs posterior mean
     (H2-minutes-weighted MSE; bootstrap over teams)
  2. coverage of 50% / 80% / 95% predictive intervals for the H2 rate, overall and by thirds of the H1 rating
  3. reliability vs minutes from (tau, sigma) and the empirical H1-H2 correlation by minutes band
Positive control: the same pipeline applied to open-play passes per 90 (known to be a stable trait).
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

from common import ROOT

numpyro.set_host_device_count(1)
GROUPS = ["CB", "FB/WB", "DM/CM", "AM/W", "CF"]
PROG_TYPES = ["pass", "cross", "carry", "dribble"]
MIN_MINUTES = 450
SEED = 20160101


# ----------------------------------------------------------------------------- data
def player_matches(measure="progression"):
    """One row per player-match appearance: y (measure total), t (minutes/90), team, group, league, half."""
    V = pd.read_parquet(ROOT / "data" / "values.parquet")
    Mi = pd.read_parquet(ROOT / "data" / "minutes.parquet")
    M = pd.read_parquet(ROOT / "data" / "matches.parquet")[["match_id", "league", "half"]]
    prog = V[V.type.isin(PROG_TYPES) & (V.subtype == "Open Play")]
    if measure == "progression":
        y = prog.groupby(["match_id", "player_id"]).value.sum()
    elif measure == "passes":   # positive control: count of open-play passes (completed or not)
        y = prog[prog.type.isin(["pass", "cross"])].groupby(["match_id", "player_id"]).size().astype(float)
    else:
        raise ValueError(measure)
    pm = Mi.groupby(["match_id", "player_id"]).agg(seconds=("seconds", "sum"), team=("team", "first"),
                                                   player=("player", "first")).reset_index()
    grp = (Mi.groupby(["match_id", "player_id", "group"]).seconds.sum().reset_index())
    pm = pm.merge(M, on="match_id")
    pm = pm.merge(y.rename("y").reset_index(), on=["match_id", "player_id"], how="left").fillna({"y": 0.0})
    pm = pm[pm.seconds > 0].copy()
    pm["t"] = pm.seconds / 5400.0
    # position group per player-half: the group with most minutes in that half
    g = grp.merge(M, on="match_id").groupby(["player_id", "half", "group"]).seconds.sum().reset_index()
    g = g.sort_values("seconds").groupby(["player_id", "half"]).tail(1)[["player_id", "half", "group"]]
    pm = pm.merge(g, on=["player_id", "half"])
    return pm[pm.group.isin(GROUPS)].reset_index(drop=True)


# ----------------------------------------------------------------------------- model
def model(y, t, pidx, cell_of_player, group_of_player, n_cells, n_groups, scale, team_of_player=None, n_teams=0,
          hetero=False):
    mu = numpyro.sample("mu", dist.Normal(0.0, 2.0 * scale).expand([n_cells]))
    tau = numpyro.sample("tau", dist.HalfNormal(2.0 * scale).expand([n_groups]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(10.0 * scale).expand([n_groups]))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([len(cell_of_player)]))
    centre = mu[cell_of_player]
    if team_of_player is not None:   # sensitivity: a team effect shared by the team's players
        tau_team = numpyro.sample("tau_team", dist.HalfNormal(2.0 * scale))
        zt = numpyro.sample("zt", dist.Normal(0.0, 1.0).expand([n_teams]))
        centre = centre + numpyro.deterministic("team_effect", tau_team * zt)[team_of_player]
    lam = numpyro.deterministic("lam", centre + tau[group_of_player] * z)   # non-centred
    sig_p = sigma[group_of_player]
    if hetero:   # variant: each player's noise level partially pooled towards his group's (log scale)
        omega = numpyro.sample("omega", dist.HalfNormal(1.0).expand([n_groups]))
        zs = numpyro.sample("zs", dist.Normal(0.0, 1.0).expand([len(cell_of_player)]))
        sig_p = sig_p * jnp.exp(omega[group_of_player] * zs)
    sig_p = numpyro.deterministic("sigma_p", sig_p)
    numpyro.sample("y", dist.Normal(lam[pidx] * t, sig_p[pidx] * jnp.sqrt(t)), obs=y)


def fit(pm, seed=SEED, warmup=1000, samples=1000, chains=4, team_adjusted=False, hetero=False):
    """Fit on the rows given (H1 rows for validation). Returns posterior draws and index maps."""
    players = pm.groupby("player_id").agg(player=("player", "first"), t=("t", "sum"), y=("y", "sum")).reset_index()
    for col in ("league", "team", "group"):   # where the player spent most minutes (transfers, role changes)
        main = pm.groupby(["player_id", col]).t.sum().reset_index().sort_values("t").groupby("player_id").tail(1)
        players = players.merge(main[["player_id", col]], on="player_id")
    players["cell"] = players.league + " | " + players.group
    cells = sorted(players.cell.unique())
    pid = {p: i for i, p in enumerate(players.player_id)}
    rate = players.y / players.t
    scale = float(np.sqrt(np.average((rate - np.average(rate, weights=players.t)) ** 2, weights=players.t)))
    data = dict(y=jnp.asarray(pm.y.values), t=jnp.asarray(pm.t.values),
                pidx=jnp.asarray(pm.player_id.map(pid).values),
                cell_of_player=jnp.asarray(players.cell.map({c: i for i, c in enumerate(cells)}).values),
                group_of_player=jnp.asarray(players.group.map({g: i for i, g in enumerate(GROUPS)}).values),
                n_cells=len(cells), n_groups=len(GROUPS), scale=scale, hetero=hetero)
    if team_adjusted:
        teams = sorted(players.team.unique())
        data.update(team_of_player=jnp.asarray(players.team.map({tm: i for i, tm in enumerate(teams)}).values),
                    n_teams=len(teams))
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warmup, num_samples=samples, num_chains=chains,
              chain_method="sequential", progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), **data)
    post = mc.get_samples(group_by_chain=True)
    keys = ("mu", "tau", "sigma", "lam") + (("omega", "sigma_p") if hetero else ())
    diag = summary({k: np.asarray(post[k]) for k in keys}, group_by_chain=True)
    rhat = max(float(np.max(v["r_hat"])) for v in diag.values())
    ess = min(float(np.min(v["n_eff"])) for v in diag.values())
    flat = {k: np.asarray(v).reshape(-1, *np.asarray(v).shape[2:]) for k, v in post.items()}
    divergences = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    return dict(post=flat, players=players, cells=cells, scale=scale, rhat=rhat, ess=ess,
                divergences=divergences)


# ----------------------------------------------------------------------------- validation
def validate(pm, measure, tag, seed=SEED, team_adjusted=False, hetero=False):
    h1, h2 = pm[pm.half == "H1"], pm[pm.half == "H2"]
    f = fit(h1, seed=seed, team_adjusted=team_adjusted, hetero=hetero)
    P = f["players"]
    lam = f["post"]["lam"]                                     # draws x players
    mu = f["post"]["mu"]
    P["post_mean"] = lam.mean(0)
    P["cell_mean"] = mu.mean(0)[P.cell.map({c: i for i, c in enumerate(f["cells"])}).values]
    P["raw_h1"] = P.y / P.t
    t2 = h2.groupby("player_id").agg(y2=("y", "sum"), t2=("t", "sum"))
    V = P.merge(t2, left_on="player_id", right_index=True)
    V = V[(V.t * 90 >= MIN_MINUTES) & (V.t2 * 90 >= MIN_MINUTES)].copy()
    V["rate_h2"] = V.y2 / V.t2
    w = V.t2.values

    def wmse(pred):
        return np.average((V.rate_h2.values - pred) ** 2, weights=w)

    n_teams = pm.groupby("player_id").team.nunique()
    V["single_team"] = V.player_id.map(n_teams).values == 1
    preds = {"league_group_mean": V.cell_mean.values, "raw_first_half": V.raw_h1.values,
             "bayes_posterior_mean": V.post_mean.values}
    base = {k: wmse(v) for k, v in preds.items()}
    # bootstrap over teams (players within a team are dependent)
    teams = V.team.unique()
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in preds}
    for _ in range(4000):
        pick = rng.choice(teams, len(teams))
        idx = np.concatenate([np.where(V.team.values == tm)[0] for tm in pick])
        for k, v in preds.items():
            boots[k].append(np.average((V.rate_h2.values[idx] - v[idx]) ** 2, weights=w[idx]))
    rows = []
    for k in preds:
        d = np.array(boots[k]) - np.array(boots["bayes_posterior_mean"])
        rows.append(dict(measure=measure, analysis=tag, predictor=k, n_players=len(V), wmse=base[k],
                         wmse_minus_bayes=base[k] - base["bayes_posterior_mean"],
                         diff_ci_lo=np.percentile(d, 2.5), diff_ci_hi=np.percentile(d, 97.5)))
    # predictive interval coverage for the H2 rate
    pi = P.set_index("player_id").index.get_indexer(V.player_id)
    gi = V.group.map({g: i for i, g in enumerate(GROUPS)}).values
    rng = np.random.default_rng(seed + 1)
    sig_draws = f["post"]["sigma_p"][:, pi]                   # draws x validation players
    draws = lam[:, pi] + rng.standard_normal(lam[:, pi].shape) * sig_draws / np.sqrt(V.t2.values)
    cover, cover_single = {}, {}
    st = V.single_team.values
    # thirds of the first-half rating within position group, and the top 10% (H1 information only)
    V["h1_third"] = V.groupby("group").post_mean.transform(lambda s: pd.qcut(s, 3, labels=False)).astype(int)
    V["h1_top10"] = V.groupby("group").post_mean.transform(lambda s: s >= s.quantile(0.9)).astype(bool)
    by_part = []
    for level in (0.5, 0.8, 0.95):
        lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
        inside = (V.rate_h2.values >= lo) & (V.rate_h2.values <= hi)
        cover[level], cover_single[level] = float(np.mean(inside)), float(np.mean(inside[st]))
        for name, mask in (("bottom third", V.h1_third.values == 0), ("middle third", V.h1_third.values == 1),
                           ("top third", V.h1_third.values == 2), ("top 10%", V.h1_top10.values)):
            by_part.append(dict(measure=measure, analysis=tag, level=level, part=name, n_players=int(mask.sum()),
                                coverage=float(np.mean(inside[mask])),
                                median_width=float(np.median((hi - lo)[mask]))))
    # 95% CIs for 80% coverage: bootstrap over teams (players in a team are dependent)
    lo80, hi80 = np.quantile(draws, [0.1, 0.9], axis=0)
    inside80 = (V.rate_h2.values >= lo80) & (V.rate_h2.values <= hi80)
    masks = {"all": np.ones(len(V), bool), "bottom third": V.h1_third.values == 0,
             "middle third": V.h1_third.values == 1, "top third": V.h1_third.values == 2,
             "top 10%": V.h1_top10.values}
    team_rows = [np.where(V.team.values == tm)[0] for tm in teams]
    rng_c = np.random.default_rng(seed + 2)
    cb = {k: [] for k in masks}
    for _ in range(2000):
        idx = np.concatenate([team_rows[i] for i in rng_c.integers(0, len(teams), len(teams))])
        for k, m in masks.items():
            sel = idx[m[idx]]
            cb[k].append(inside80[sel].mean())
    cov_ci = pd.DataFrame([dict(measure=measure, analysis=tag, level=0.8, part=k, n_players=int(m.sum()),
                                coverage=float(inside80[m].mean()), ci_lo=np.percentile(cb[k], 2.5),
                                ci_hi=np.percentile(cb[k], 97.5)) for k, m in masks.items()])
    for k, v in preds.items():   # sensitivity: players who stayed at one club all season
        rows.append(dict(measure=measure, analysis=tag + " | single-team players", predictor=k, n_players=int(st.sum()),
                         wmse=np.average((V.rate_h2.values[st] - v[st]) ** 2, weights=w[st])))
    # reliability vs minutes (model-implied) and empirical H1-H2 correlation by band
    tau_m, sig_m = f["post"]["tau"].mean(0), f["post"]["sigma"].mean(0)   # sigma_g: the median player's noise
    rel = []
    for gi_, (gname, tg, sg) in enumerate(zip(GROUPS, tau_m, sig_m, strict=True)):
        ratio = f["post"]["sigma"][:, gi_] ** 2 / f["post"]["tau"][:, gi_] ** 2   # per posterior draw
        for target in (0.5, 0.7, 0.8):
            minutes = 90 * target / (1 - target) * sg ** 2 / tg ** 2            # plug-in posterior means
            draws_m = 90 * target / (1 - target) * ratio
            rel.append(dict(measure=measure, analysis=tag, group=gname, reliability=target,
                            minutes_needed=minutes, minutes_ci_lo=np.percentile(draws_m, 2.5),
                            minutes_ci_hi=np.percentile(draws_m, 97.5), tau=tg, sigma=sg))
    # empirical H1-H2 correlation of raw rates (centred on the H1 league-group mean) vs the correlation the
    # model implies if a player's true rate is constant: sqrt(R(t1) R(t2)), R(t) = tau^2 / (tau^2 + sigma^2 / t)
    # The observed correlation is one Pearson correlation pooled over players of all groups, so the like-for-like
    # model value is pooled too: sum(tau^2) / sqrt(sum(tau^2 + sigma^2/t1) * sum(tau^2 + sigma^2/t2)).
    gi_tau, gi_sig = tau_m[gi], f["post"]["sigma_p"].mean(0)[pi]
    V["imp_cov"] = gi_tau ** 2
    V["imp_v1"] = gi_tau ** 2 + gi_sig ** 2 / V.t.values
    V["imp_v2"] = gi_tau ** 2 + gi_sig ** 2 / V.t2.values
    centre = V.cell_mean.values
    if team_adjusted:   # like-for-like: the implied correlation covers player skill only, so remove team effects
        teams_sorted = sorted(P.team.unique())
        te = f["post"]["team_effect"].mean(0)
        centre = centre + te[V.team.map({tm: i for i, tm in enumerate(teams_sorted)}).values]
    V["c1"], V["c2"] = V.raw_h1 - centre, V.rate_h2 - centre
    V["band"] = pd.cut(np.minimum(V.t, V.t2) * 90, [450, 900, 1350, 4000], right=False,
                       labels=["450-899", "900-1349", "1350+"])
    corr = []
    for b, g in V.groupby("band", observed=True):
        r = np.corrcoef(g.c1, g.c2)[0, 1]
        bs = []
        g_teams = g.team.unique()
        for _ in range(2000):   # bootstrap over teams, as for the MSE
            k = np.concatenate([np.where(g.team.values == tm)[0] for tm in rng.choice(g_teams, len(g_teams))])
            bs.append(np.corrcoef(g.c1.values[k], g.c2.values[k])[0, 1])
        corr.append(dict(measure=measure, analysis=tag, minutes_band=str(b), n_players=len(g), empirical_corr=r,
                         ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                         model_implied_corr=float(g.imp_cov.sum() / np.sqrt(g.imp_v1.sum() * g.imp_v2.sum()))))
    return dict(fit=f, table=V, mse=pd.DataFrame(rows), coverage=cover, coverage_single=cover_single,
                reliability=pd.DataFrame(rel), corr=pd.DataFrame(corr), coverage_parts=pd.DataFrame(by_part),
                coverage_ci=cov_ci)


ANALYSES = (  # name, measure, team-adjusted, player-specific noise
    ("progression", "progression", False, False),            # pre-registered model
    ("progression_player_noise", "progression", False, True),
    ("passes", "passes", False, False),                      # positive control
    ("passes_player_noise", "passes", False, True),
    ("progression_team_adjusted", "progression", True, False),
)


def choose_model(cov, parts):
    """Decision rule fixed in DESIGN.md (deviation 5) before the variant's H1 -> H2 results were seen."""
    c80 = cov.set_index(["measure", "level"]).coverage
    top = parts[(parts.level == 0.8) & (parts.part == "top third")].set_index("measure").coverage
    a = abs(c80[("progression_player_noise", 0.8)] - 0.8) <= 0.03
    b = abs(top["progression_player_noise"] - 0.8) < abs(top["progression"] - 0.8)
    chosen = "progression_player_noise" if (a and b) else "progression"
    return pd.DataFrame([dict(chosen=chosen, rule_a_overall_80_within_3pts=bool(a),
                              rule_b_top_third_closer_to_80=bool(b),
                              overall80_original=c80[("progression", 0.8)],
                              overall80_player_noise=c80[("progression_player_noise", 0.8)],
                              top_third80_original=top["progression"],
                              top_third80_player_noise=top["progression_player_noise"])])


def main():
    out = {}
    for name, measure, team_adj, hetero in ANALYSES:
        pm = player_matches(measure)
        res = validate(pm, name, "H1->H2", team_adjusted=team_adj, hetero=hetero)
        f = res["fit"]
        print(f"\n=== {name}: players fitted {len(f['players'])}, appearances {len(pm[pm.half == 'H1'])}, "
              f"R-hat max {f['rhat']:.3f}, ESS min {f['ess']:.0f}, divergences {f['divergences']}", flush=True)
        print(res["mse"].round(6).to_string(index=False))
        print("predictive interval coverage:", {k: round(v, 3) for k, v in res["coverage"].items()})
        print(res["coverage_parts"][res["coverage_parts"].level == 0.8].round(3).to_string(index=False))
        print(res["reliability"][res["reliability"].reliability == 0.7].round(3).to_string(index=False))
        print(res["coverage_ci"].round(3).to_string(index=False))
        print(res["corr"].round(3).to_string(index=False), flush=True)
        out[name] = res
    R = ROOT / "results"
    pd.concat([r["mse"] for r in out.values()]).to_csv(R / "rating_validation_mse.csv", index=False)
    cov = pd.DataFrame([dict(measure=m, level=k, coverage=v, coverage_single_team=r["coverage_single"][k],
                             n_players=len(r["table"]), n_single_team=int(r["table"].single_team.sum()),
                             rhat_max=r["fit"]["rhat"], ess_min=r["fit"]["ess"], divergences=r["fit"]["divergences"])
                        for m, r in out.items() for k, v in r["coverage"].items()])
    cov.to_csv(R / "rating_validation_coverage.csv", index=False)
    parts = pd.concat([r["coverage_parts"] for r in out.values()])
    parts.to_csv(R / "rating_validation_coverage_by_rating.csv", index=False)
    pd.concat([r["coverage_ci"] for r in out.values()]).to_csv(R / "rating_validation_coverage80_ci.csv", index=False)
    pd.concat([r["reliability"] for r in out.values()]).to_csv(R / "reliability.csv", index=False)
    pd.concat([r["corr"] for r in out.values()]).to_csv(R / "rating_h1_h2_correlation.csv", index=False)
    choice = choose_model(cov, parts)
    choice.to_csv(R / "model_choice.csv", index=False)
    chosen = choice.chosen.iloc[0]
    print("\nmodel choice:", choice.T.to_string(header=False))
    keep = ["player_id", "player", "team", "league", "group", "t", "y", "t2", "y2", "raw_h1", "rate_h2", "cell_mean",
            "post_mean", "single_team", "h1_third", "h1_top10"]
    for m in ("progression", "progression_player_noise"):
        out[m]["table"][keep].to_csv(R / f"rating_validation_players_{m}.csv", index=False)
    a = out[chosen]["table"].set_index("player_id").post_mean
    b = out["progression_team_adjusted"]["table"].set_index("player_id").post_mean.loc[a.index]
    rho = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    pd.DataFrame([dict(spearman_chosen_vs_team_adjusted=rho, chosen=chosen, n_players=len(a))]).to_csv(
        R / "team_adjustment_rank_agreement.csv", index=False)
    print("rank agreement chosen model vs team-adjusted posterior means:", round(rho, 3))


if __name__ == "__main__":
    sys.exit(main())
