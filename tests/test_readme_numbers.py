"""Every headline number in README.md is recomputed here from results/ and must appear in the README as written."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
README = (ROOT / "README.md").read_text()
README_FLAT = " ".join(README.split())   # line wraps do not matter


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def contains(s):
    assert " ".join(s.split()) in README_FLAT, f"README does not contain {s!r}"


@pytest.fixture(scope="module")
def mse():
    m = pd.read_csv(RES / "rating_validation_mse.csv")
    return m[(m.measure == "progression") & (m.analysis == "H1->H2")].set_index("predictor")


def test_model_choice_is_preregistered_model():
    c = pd.read_csv(RES / "model_choice.csv").iloc[0]
    assert c.chosen == "progression" and not c.rule_a_overall_80_within_3pts and not c.rule_b_top_third_closer_to_80
    contains(f"overall {pct(c.overall80_player_noise)}, top third {pct(c.top_third80_player_noise)}")


def test_vaep_numbers():
    V = pd.read_csv(RES / "vaep_validation.csv")
    s = V[V.label == "scores"].pivot(index="held_out_league", columns="model", values="auc")
    c = V[V.label == "concedes"].pivot(index="held_out_league", columns="model", values="auc")
    contains(f"ROC AUC {s.vaep.min():.3f}–{s.vaep.max():.3f}")
    contains(f"(location-only model: {s.location_only.min():.3f}–{s.location_only.max():.3f})")
    contains(f"and {c.vaep.min():.3f}–{c.vaep.max():.3f} for")
    for lab, txt in (("scores", "for scoring"), ("concedes", "for conceding")):
        ll = V[V.label == lab].pivot(index="held_out_league", columns="model", values="log_loss")
        red = 100 * (1 - ll.vaep / ll.base_rate)
        contains(f"{red.min():.1f}–{red.max():.1f}% below the base rate {txt}" if lab == "scores"
                 else f"{red.min():.1f}–{red.max():.1f}% {txt}")
    gain_c, gain_s = c.vaep - c.location_only, s.vaep - s.location_only
    assert 0.05 <= round(gain_c.min(), 2) and round(gain_c.max(), 2) <= 0.07
    assert 0.02 <= round(gain_s.min(), 2) and round(gain_s.max(), 2) <= 0.03
    C = pd.read_csv(RES / "vaep_calibration.csv")
    top = C[(C.bin == 9) & (C.label == "scores")]
    assert (top.predicted - top.observed).abs().max() <= 0.002


def test_forward_validation_numbers(mse):
    b, r, m = (mse.wmse[k] for k in ("bayes_posterior_mean", "raw_first_half", "league_group_mean"))
    contains(f"{int(mse.n_players.iloc[0])} players")
    contains("**19% lower error than raw first-half rates**")
    assert round(100 * (1 - b / r)) == 19 and round(100 * (1 - b / m)) == 39
    contains(f"weighted MSE {b:.5f} vs {r:.5f}")
    rr = mse.loc["raw_first_half"]
    contains(f"difference 95% CI {rr.diff_ci_lo:.5f} to {rr.diff_ci_hi:.5f}")
    contains(f"league-position average ({m:.5f})")


def test_coverage_numbers():
    C = pd.read_csv(RES / "rating_validation_coverage.csv")
    c = C[C.measure == "progression"].set_index("level")
    contains(f"contained {100 * c.coverage[0.5]:.1f} / {100 * c.coverage[0.8]:.1f} / {100 * c.coverage[0.95]:.1f}%")
    contains(f"80% coverage {pct(c.coverage_single_team[0.8])}")
    contains(f"({int(c.n_single_team.iloc[0])} players")
    p = C[C.measure == "passes"].set_index("level").coverage
    contains(f"({pct(p[0.8])} at 80%,")
    P = pd.read_csv(RES / "rating_validation_coverage_by_rating.csv")
    pp = P[(P.measure == "progression") & (P.level == 0.8)].set_index("part").coverage
    contains(f"{pct(pp['bottom third'])} of second-half rates for the bottom third")
    contains(f"{pct(pp['middle third'])} for the middle third")
    contains(f"**{pct(pp['top third'])} for the top third and {pct(pp['top 10%'])} for the top 10%**")
    q = P[(P.measure == "passes") & (P.level == 0.8)].set_index("part").coverage
    contains(f"same top-third pattern ({pct(q['top third'])})")
    mse = pd.read_csv(RES / "rating_validation_mse.csv")
    v = mse[(mse.measure == "progression_player_noise") & (mse.analysis == "H1->H2")].set_index("predictor")
    contains(f"MSE {v.wmse['bayes_posterior_mean']:.5f}")
    assert v.diff_ci_lo["raw_first_half"] < 0 < v.diff_ci_hi["raw_first_half"]   # "no better than raw rates"


def test_bias_by_level():
    T = pd.read_csv(RES / "rating_validation_players_progression.csv")
    top3 = T[T.h1_third == 2]
    raw_drop = np.average(top3.rate_h2 - top3.raw_h1, weights=top3.t2)
    contains(f"by {abs(raw_drop):.3f} per 90")
    t10 = T[T.h1_top10]
    contains(f"(+{np.average(t10.rate_h2 - t10.post_mean, weights=t10.t2):.3f} per 90)")
    sd = T.assign(e=T.rate_h2 - T.post_mean).groupby("h1_third").e.std()
    contains(f"(standard deviation {sd[2]:.3f} in the top third")
    contains(f"against\n{sd[0]:.3f} in the bottom third")


def test_reliability_numbers():
    R = pd.read_csv(RES / "reliability.csv")
    r = R[(R.measure == "progression") & (R.reliability == 0.7)].set_index("group").minutes_needed
    contains(f"Defensive/central midfielders {r['DM/CM']:,.0f} · attacking midfielders/wingers {r['AM/W']:,.0f} · "
             f"centre-backs {r['CB']:,.0f} · centre-forwards {r['CF']:,.0f} · full-backs/wing-backs {r['FB/WB']:,.0f}")
    p = R[(R.measure == "passes") & (R.reliability == 0.7)].minutes_needed
    contains(f"Reliability 0.7 after {p.min():.0f}–{p.max():.0f} minutes")
    assert 2 <= p.min() / 90 and p.max() / 90 <= 4.5
    h = R[(R.measure == "progression") & (R.group == "FB/WB")].iloc[0]
    assert round(h.tau ** 2 / (h.tau ** 2 + h.sigma ** 2 / 38), 2) in (0.66, 0.67)
    C = pd.read_csv(RES / "rating_h1_h2_correlation.csv")
    c = C[C.measure == "progression"].set_index("minutes_band").loc[["450-899", "900-1349", "1350+"]]
    contains(", ".join(f"{a:.2f} vs {b:.2f}" for a, b in zip(c.model_implied_corr[:2], c.empirical_corr[:2],
                                                             strict=True))
             + f" and {c.model_implied_corr.iloc[2]:.2f} vs {c.empirical_corr.iloc[2]:.2f}")
    p = C[C.measure == "passes"].empirical_corr
    contains(f"observed half-to-half correlation {p.min():.2f}–{p.max():.2f}")


def test_team_and_convergence():
    t = pd.read_csv(RES / "team_adjustment_rank_agreement.csv").iloc[0]
    contains(f"Spearman {t.spearman_chosen_vs_team_adjusted:.3f}")
    R = pd.read_csv(RES / "reliability.csv")
    tc = R[(R.group == "CB") & (R.reliability == 0.7)].set_index("measure").tau
    contains(f"(tau {tc['progression']:.3f} to {tc['progression_team_adjusted']:.3f})")
    C = pd.read_csv(RES / "rating_validation_coverage.csv").drop_duplicates("measure").set_index("measure")
    F = pd.read_csv(RES / "full_season_fit.csv").iloc[0]
    others = C.drop(index="passes_player_noise")
    assert others.rhat_max.max() <= 1.0105 and F.rhat_max <= 1.0105
    assert (C.divergences == 0).all() and F.divergences == 0
    contains(f"has R-hat {C.rhat_max['passes_player_noise']:.3f}")


def test_top_players():
    P = pd.read_csv(RES / "player_ratings.csv")
    top = P[P.minutes >= 1500].sort_values("rating", ascending=False).head(6)
    m = top.iloc[0]
    contains(f"{m.player} {m.rating:.3f} per 90 (80% interval {m.lo80:.3f} to {m.hi80:.3f})")
    for r in top.iloc[1:].itertuples():
        contains(f"{r.player} {r.rating:.3f}")
    contains(f"({len(P):,} outfield players rated)")


def test_interval_uncertainty_numbers():
    C = pd.read_csv(RES / "rating_validation_coverage80_ci.csv")
    c = C[C.measure == "progression"].set_index("part")
    contains(f"(80%: 95% CI {100 * c.ci_lo['all']:.1f} to {100 * c.ci_hi['all']:.1f}%)")
    contains(f"(95% CIs {100 * c.ci_lo['top third']:.1f} to {100 * c.ci_hi['top third']:.1f}% and "
             f"{100 * c.ci_lo['top 10%']:.1f} to {100 * c.ci_hi['top 10%']:.1f}%)")
    assert c.ci_hi["top third"] < 0.8 and c.ci_hi["top 10%"] < 0.8 and c.ci_lo["bottom third"] > 0.8
    assert c.ci_lo["all"] < 0.8 < c.ci_hi["all"]
    contains(f"bottom third got {pct(c.coverage['bottom third'])} ({100 * c.ci_lo['bottom third']:.1f} to "
             f"{100 * c.ci_hi['bottom third']:.1f}%)")
    q = C[(C.measure == "passes") & (C.part == "all")].iloc[0]
    contains(f"95% CI {100 * q.ci_lo:.1f} to {100 * q.ci_hi:.1f}%), with the same top-third")
    R = pd.read_csv(RES / "reliability.csv")
    r = R[(R.measure == "progression") & (R.reliability == 0.7)].set_index("group")
    contains(f"from {r.minutes_ci_lo['DM/CM']:,.0f}–{r.minutes_ci_hi['DM/CM']:,.0f} (central midfielders) to "
             f"{r.minutes_ci_lo['FB/WB']:,.0f}–{r.minutes_ci_hi['FB/WB']:,.0f} (full-backs)")
    contains(f"full-backs is {r.minutes_ci_lo['FB/WB']:,.0f} to {r.minutes_ci_hi['FB/WB']:,.0f} minutes and for "
             f"centre-forwards {r.minutes_ci_lo['CF']:,.0f} to {r.minutes_ci_hi['CF']:,.0f}")
    K = pd.read_csv(RES / "rating_h1_h2_correlation.csv")
    k = K[K.measure == "progression"]
    assert ((k.model_implied_corr >= k.ci_lo) & (k.model_implied_corr <= k.ci_hi)).all()
    contains("every model value is inside the observed value's 95% CI")
