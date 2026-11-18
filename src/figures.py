"""All README figures. Every figure carries the StatsBomb logo and a data credit (StatsBomb user agreement 1.4).

Colours: position groups use five fixed categorical slots (validated for colour-vision deficiency; always
direct-labelled or in a legend); single-hue blue for magnitude; grey for references.
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from common import ROOT

FIG, RES = ROOT / "figures", ROOT / "results"
SURF, INK, INK2, MUTED, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#d6d5d0"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GROUP_COL = {"CB": BLUE, "FB/WB": ORANGE, "DM/CM": AQUA, "AM/W": YELLOW, "CF": MAGENTA}
GROUP_NAME = {"CB": "Centre-backs", "FB/WB": "Full/wing-backs", "DM/CM": "Defensive/central mid",
              "AM/W": "Attacking mid/wingers", "CF": "Centre-forwards"}
MODEL_COL = {"base_rate": MUTED, "location_only": ORANGE, "vaep": BLUE}
MODEL_NAME = {"base_rate": "Base rate", "location_only": "Location only", "vaep": "Action-value model"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
LOGO = plt.imread(ROOT / "assets" / "statsbomb_logo.png")

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": LINE, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
                     "axes.titlelocation": "left"})


def chosen():
    return pd.read_csv(RES / "model_choice.csv").chosen.iloc[0]


MEAS_NAME = {"progression": "Pre-registered model (one noise level per position)",
             "progression_player_noise": "Player-specific noise",
             "progression_team_adjusted": "Team-adjusted variant",
             "passes": "Positive control: passes per 90",
             "passes_player_noise": "Passes per 90, player-specific noise"}
MEAS_STYLE = {"progression": (BLUE, "o"), "progression_player_noise": (MUTED, "o"),
              "progression_team_adjusted": (AQUA, "s"), "passes": (ORANGE, "^"), "passes_player_noise": (YELLOW, "^")}


def credit(fig):
    """StatsBomb logo (top right) and data credit on every figure: figures are often shared on their own."""
    w, h = fig.get_size_inches()
    logo_w = 1.25 / w                                    # 1.25 inch wide
    logo_h = logo_w * w / h * LOGO.shape[0] / LOGO.shape[1]
    ax = fig.add_axes([0.995 - logo_w, 0.995 - logo_h, logo_w, logo_h])
    ax.imshow(LOGO)
    ax.axis("off")
    fig.text(0.995 - logo_w - 0.008, 0.995 - logo_h / 2, "Data: StatsBomb Open Data, 2015/16", ha="right",
             va="center", fontsize=7.5, color=MUTED)


def save(fig, name):
    credit(fig)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------- fig 1: action-value model
def fig_vaep():
    V = pd.read_csv(RES / "vaep_validation.csv")
    C = pd.read_csv(RES / "vaep_calibration.csv")
    leagues = sorted(V.held_out_league.unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), gridspec_kw=dict(hspace=0.55, wspace=0.28))
    for j, label in enumerate(("scores", "concedes")):
        ax = axes[0, j]
        sub = V[V.label == label]
        y = np.arange(len(leagues))
        for k, model in enumerate(("base_rate", "location_only", "vaep")):
            vals = sub[sub.model == model].set_index("held_out_league").auc.loc[leagues].values
            ax.scatter(vals, y + (k - 1) * 0.18, s=46, color=MODEL_COL[model], label=MODEL_NAME[model],
                       zorder=3, edgecolor=SURF, linewidth=1)
        ax.set_yticks(y, leagues)
        ax.set_xlim(0.45, 0.9)
        ax.set_xlabel("ROC AUC on the held-out league (both halves)")
        ax.grid(axis="x", color=LINE, lw=0.6)
        what = "the acting team scores" if label == "scores" else "the acting team concedes"
        ax.set_title(f"P({what} within 10 actions)")
        if j == 0:
            ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.2, 0.5), fontsize=8.5)
        ax = axes[1, j]
        sub = C[C.label == label]
        lim = [sub[["predicted", "observed"]].min().min() * 0.8, sub[["predicted", "observed"]].max().max() * 1.25]
        ax.plot(lim, lim, color=MUTED, lw=1, ls="--", label="Perfect calibration")
        for i, L in enumerate(leagues):
            s = sub[sub.held_out_league == L]
            ax.plot(s.predicted, s.observed, marker="o", ms=4.5, lw=1.4,
                    color=[BLUE, ORANGE, AQUA, YELLOW][i], label=L)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Predicted probability (decile of predictions)")
        ax.set_ylabel("Observed frequency")
        ax.set_title("Calibration on the held-out league")
        if j == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("The action-value model, scored only on leagues it never saw", x=0.06, ha="left",
                 fontweight="bold", fontsize=13, y=1.0)
    save(fig, "fig1_action_value_model.png")


# ----------------------------------------------------------------------------- fig 2: forward validation
def fig_forward():
    M = pd.read_csv(RES / "rating_validation_mse.csv")
    Cv = pd.read_csv(RES / "rating_validation_coverage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(wspace=0.45, width_ratios=[1.15, 1]))
    ax = axes[0]
    ch = chosen()
    m = M[(M.measure == ch) & (M.analysis == "H1->H2")].set_index("predictor")
    order = ["league_group_mean", "raw_first_half", "bayes_posterior_mean"]
    names = ["League x position mean", "Raw first-half rate", "Bayesian rating (posterior mean)"]
    cols = [MUTED, ORANGE, BLUE]
    vals = m.wmse.loc[order].values * 1e3
    ax.barh(range(3), vals, color=cols, height=0.55)
    for i, (v, k) in enumerate(zip(vals, order, strict=True)):
        txt = f"{v:.2f}"
        if k != "bayes_posterior_mean":
            r = m.loc[k]
            txt += f"   (+{r.wmse_minus_bayes * 1e3:.2f}; 95% CI {r.diff_ci_lo * 1e3:.2f} to {r.diff_ci_hi * 1e3:.2f})"
        ax.text(v + 0.04, i, txt, va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(3), names)
    ax.set_xlim(0, vals.max() * 1.9)
    ax.invert_yaxis()
    ax.set_xlabel("Error predicting each player's second-half rate (weighted MSE x 1000)")
    n = int(m.n_players.iloc[0])
    ax.set_title(f"Predicting the second half from the first ({n} players)", pad=22)
    ax.text(0, 1.015, "Lower is better. In brackets: extra error vs the Bayesian rating, "
            "95% CI (bootstrap over teams).",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    ax = axes[1]
    lv = np.array([0.5, 0.8, 0.95])
    ax.plot([0.4, 1.0], [0.4, 1.0], color=MUTED, ls="--", lw=1)
    for k, meas in enumerate(("progression", "progression_team_adjusted", "progression_player_noise", "passes")):
        c, mk = MEAS_STYLE[meas]
        s = Cv[Cv.measure == meas].set_index("level").coverage.loc[lv].values
        ax.plot(lv + (k - 1.5) * 0.006, s, marker=mk, color=c, lw=1.4, ms=6, label=MEAS_NAME[meas],
                zorder=5 - k)
    ax.set_xticks(lv, ["50%", "80%", "95%"])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlim(0.42, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Nominal predictive interval")
    ax.set_ylabel("Share of second-half rates inside")
    ax.set_title("Are the intervals honest overall?")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(color=LINE, lw=0.5)
    save(fig, "fig2_forward_validation.png")


# ----------------------------------------------------------------------------- fig 3: coverage by rating level
def fig_coverage_parts():
    C = pd.read_csv(RES / "rating_validation_coverage_by_rating.csv")
    C = C[C.level == 0.8]
    CI = pd.read_csv(RES / "rating_validation_coverage80_ci.csv")
    parts = ["bottom third", "middle third", "top third", "top 10%"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw=dict(wspace=0.3))
    for ax, col, title, lab in ((axes[0], "coverage", "80% intervals: share of second-half rates inside",
                                 "Coverage of the 80% predictive interval"),
                                (axes[1], "median_width", "80% intervals: how wide they are",
                                 "Median interval width (value per 90)")):
        x = np.arange(len(parts))
        for k, meas in enumerate(("progression", "progression_player_noise")):
            c, _ = MEAS_STYLE[meas]
            s = C[C.measure == meas].set_index("part").loc[parts]
            ax.bar(x + (k - 0.5) * 0.36, s[col], width=0.34, color=c, label=MEAS_NAME[meas])
            top = s[col].values
            if col == "coverage":
                e = CI[CI.measure == meas].set_index("part").loc[parts]
                ax.errorbar(x + (k - 0.5) * 0.36, e.coverage, yerr=[e.coverage - e.ci_lo, e.ci_hi - e.coverage],
                            fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
                top = e.ci_hi.values
            for xi, v, tp in zip(x + (k - 0.5) * 0.36, s[col], top, strict=True):
                ax.text(xi, tp + (0.01 if col == "coverage" else 0.0), f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8, color=INK2)
        n = C[C.measure == "progression"].set_index("part").n_players.loc[parts]
        ax.set_xticks(x, [f"{p}\n(n={k})" for p, k in zip(parts, n, strict=True)])
        ax.set_ylabel(lab)
        ax.set_title(title)
        if col == "coverage":
            ax.axhline(0.8, color=INK, lw=1, ls="--")
            ax.set_ylim(0, 1.18)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    fig.text(0.06, -0.04, "Players grouped by their first-half rating within position group "
             "(first-half information only). Error bars: 95% CI, bootstrap over teams.", fontsize=8, color=MUTED)
    save(fig, "fig3_coverage_by_rating.png")


# ----------------------------------------------------------------------------- fig 3: minutes needed
def fig_reliability():
    R = pd.read_csv(RES / "reliability.csv")
    Co = pd.read_csv(RES / "rating_h1_h2_correlation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw=dict(wspace=0.32))
    ax = axes[0]
    r = R[(R.measure == chosen()) & (R.reliability == 0.7)].set_index("group")
    mins = np.linspace(1, 3420, 400)
    for g, c in GROUP_COL.items():
        tau, sig = r.loc[g, ["tau", "sigma"]]
        rel = tau ** 2 / (tau ** 2 + sig ** 2 / (mins / 90))
        ax.plot(mins, rel, color=c, lw=2, label=f"{GROUP_NAME[g]}: {r.loc[g, 'minutes_needed']:,.0f} min "
                f"({r.loc[g, 'minutes_ci_lo']:,.0f}–{r.loc[g, 'minutes_ci_hi']:,.0f})")
    ax.axhline(0.7, color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(0, 3420)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 900, 1800, 2700, 3420], ["0", "900\n(10 full games)", "1800", "2700", "3420\n(full season)"])
    ax.set_xlabel("Minutes of data")
    ax.set_ylabel("Reliability, typical player")
    ax.set_title("How many minutes before the rating means something?")
    ax.legend(frameon=False, fontsize=7.6, loc="lower right", title="Minutes to reliability 0.7 (95% interval)",
              title_fontsize=7.6)
    ax = axes[1]
    bands = ["450-899", "900-1349", "1350+"]
    x = np.arange(3)
    for k, (meas, c, name) in enumerate(((chosen(), BLUE, "Ball-progression rating"),
                                         ("passes", ORANGE, "Positive control: passes per 90"))):
        s = Co[Co.measure == meas].set_index("minutes_band").loc[bands]
        off = (k - 0.5) * 0.22
        ax.errorbar(x + off, s.empirical_corr, yerr=[s.empirical_corr - s.ci_lo, s.ci_hi - s.empirical_corr],
                    fmt="o", color=c, ms=6, capsize=3, lw=1.3, label=f"{name}: observed")
        ax.scatter(x + off, s.model_implied_corr, marker="_", s=320, color=INK, lw=2, zorder=4,
                   label="Model-implied" if k == 0 else None)
    ax.set_xticks(x, [b + " min" for b in bands])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Minutes in the shorter half (each player must have 450+ in both)")
    ax.set_ylabel("Correlation, first half vs second half")
    ax.set_title("Does the model's reliability match reality?")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", color=LINE, lw=0.5)
    save(fig, "fig4_reliability.png")


# ----------------------------------------------------------------------------- fig 4: top ratings
def fig_top(n=20, min_minutes=1500):
    P = pd.read_csv(RES / "player_ratings.csv")
    top = P[P.minutes >= min_minutes].sort_values("rating", ascending=False).head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7.4))
    y = np.arange(len(top))
    for i, r in enumerate(top.itertuples()):
        c = GROUP_COL[r.group]
        ax.plot([r.lo80, r.hi80], [i, i], color=c, lw=3, solid_capstyle="round", alpha=0.85)
        ax.scatter(r.rating, i, s=44, color=c, edgecolor=SURF, lw=1.2, zorder=3)
        ax.scatter(r.raw_rate, i, s=18, facecolor="none", edgecolor=INK2, lw=0.9, zorder=4)
    ax.set_yticks(y, [f"{r.player}  ·  {r.team}" for r in top.itertuples()], fontsize=8.5)
    ax.set_xlabel("Open-play ball-progression value per 90 (goal-probability units), 80% interval")
    ax.grid(axis="x", color=LINE, lw=0.5)
    handles = [plt.Line2D([], [], color=c, lw=3, label=GROUP_NAME[g]) for g, c in GROUP_COL.items()
               if g in set(top.group)]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, label="Raw rate (no shrinkage)"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"Highest-rated ball progressors, Big-4 leagues 2015/16 (players with {min_minutes:,}+ minutes)")
    save(fig, "fig5_top_ratings.png")


# ----------------------------------------------------------------------------- fig 5: shrinkage
def fig_shrinkage():
    T = pd.read_csv(RES / f"rating_validation_players_{chosen()}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(wspace=0.5))
    ax = axes[0]
    P = pd.read_csv(RES / "player_ratings.csv")
    cm = pd.read_csv(RES / "cell_means.csv").set_index("cell").mu
    P["cell"] = P.league + " | " + P.group
    P = P[P.minutes >= 90]
    x = P.raw_rate - P.cell.map(cm)
    yv = P.rating - P.cell.map(cm)
    sc = ax.scatter(x, yv, c=np.minimum(P.minutes, 3420), cmap=SEQ, s=10, lw=0, alpha=0.9, vmin=0, vmax=3420)
    lim = [np.percentile(x, 0.5), np.percentile(x, 99.5)]
    ax.plot(lim, lim, color=MUTED, ls="--", lw=1)
    ax.axhline(0, color=LINE, lw=0.8)
    ax.set_xlim(lim)
    ax.set_xlabel("Raw rate minus league x position mean")
    ax.set_ylabel("Bayesian rating minus the same mean")
    ax.set_title("Few minutes, strong pull to the mean")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Minutes played")
    cb.outline.set_visible(False)
    ax = axes[1]
    xs = T.raw_h1 - T.cell_mean
    ax.scatter(xs, T.rate_h2 - T.cell_mean, s=10, color=ORANGE, alpha=0.55, lw=0, label="Raw first-half rate")
    ax.scatter(T.post_mean - T.cell_mean, T.rate_h2 - T.cell_mean, s=10, color=BLUE, alpha=0.55, lw=0,
               label="Bayesian rating from the first half")
    lim = [np.percentile(xs, 0.5), np.percentile(xs, 99.5)]
    ax.plot(lim, lim, color=MUTED, ls="--", lw=1)
    for est, c in ((T.raw_h1 - T.cell_mean, ORANGE), (T.post_mean - T.cell_mean, BLUE)):   # binned means
        b = pd.qcut(est, 10, labels=False)
        g = pd.DataFrame(dict(x=est, y=T.rate_h2 - T.cell_mean, w=T.t2, b=b)).groupby("b")
        mx = g.apply(lambda d: np.average(d.x, weights=d.w), include_groups=False)
        my = g.apply(lambda d: np.average(d.y, weights=d.w), include_groups=False)
        ax.plot(mx, my, color=c, lw=2.2, marker="o", ms=5, mec=SURF, zorder=5)
    ax.set_xlim(lim)
    ax.set_xlabel("First-half estimate (centred on league x position mean)")
    ax.set_ylabel("Actual second-half rate (centred)")
    ax.set_title("Shrunk estimates line up with the future", pad=16)
    ax.text(0, 1.01, "Lines: means of 10 equal-size bins. On the dashed line = no systematic over- or under-rating.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "fig6_shrinkage.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_vaep()
    fig_forward()
    fig_coverage_parts()
    fig_reliability()
    fig_top()
    fig_shrinkage()
