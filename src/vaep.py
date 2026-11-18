"""Action-value model (VAEP; Decroos et al., 2019), fitted with leave-one-league-out on first halves only.

For each league L, two gradient-boosted classifiers (P(score within 10 actions), P(concede within 10
actions)) are trained on the H1 actions of the OTHER three leagues and applied to every action of L.
No action value in L is ever produced by a model that saw L, or that saw anything played after any
league's midpoint (see DESIGN.md).

Outputs: data/values.parquet (per action: p_score, p_concede, value, offensive, defensive),
         results/vaep_validation.csv (per held-out league: log loss, Brier, AUC vs baselines),
         results/vaep_calibration.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from common import ROOT

TYPES = ["pass", "cross", "carry", "dribble", "shot", "interception", "tackle", "clearance", "recovery",
         "block", "keeper", "foul", "bad_touch", "own_goal"]
GOAL = np.array([120.0, 40.0])
HGB = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
           early_stopping=False, random_state=0)


# ----------------------------------------------------------------------------- labels
def labels(A, k=10):
    """scores/concedes: does the acting team score / concede within actions i..i+k-1 of the same period?"""
    scores = np.zeros(len(A), dtype=np.int8)
    concedes = np.zeros(len(A), dtype=np.int8)
    for _, g in A.groupby(["match_id", "period"], sort=False):
        idx = g.index.values
        team = g.team.values
        gf = g.goal_for.values
        for j in np.where(pd.notna(gf))[0]:
            lo = max(0, j - k + 1)
            for i in range(lo, j + 1):
                if gf[j] == team[i]:
                    scores[idx[i]] = 1
                else:
                    concedes[idx[i]] = 1
    return scores, concedes


# ----------------------------------------------------------------------------- features
def features(A):
    """Features of the current action a0 and the two previous actions a1, a2 (same period only).
    Previous actions by the other team are mirrored into a0's team perspective. Only a0 and earlier."""
    F = _F32()
    key = A.match_id.astype(str) + "_" + A.period.astype(str)
    same_seq = [np.ones(len(A), bool)]
    for lag in (1, 2):
        same_seq.append((key.shift(lag) == key).values)
    for lag in (0, 1, 2):
        S = A.shift(lag) if lag else A
        ok = same_seq[lag] if lag else np.ones(len(A), bool)
        sameteam = (S.team.values == A.team.values) if lag else np.ones(len(A), bool)
        x, y, ex, ey = (S[c].values.astype(float) for c in ("x", "y", "end_x", "end_y"))
        flip = ~sameteam
        x, ex = np.where(flip, 120 - x, x), np.where(flip, 120 - ex, ex)
        y, ey = np.where(flip, 80 - y, y), np.where(flip, 80 - ey, ey)
        p = f"a{lag}_"
        F[p + "present"] = ok.astype(float)
        for c, v in (("x", x), ("y", y), ("end_x", ex), ("end_y", ey)):
            F[p + c] = np.where(ok, v, np.nan)
        for which, (px, py) in (("start", (x, y)), ("end", (ex, ey))):
            dx, dy = GOAL[0] - px, np.abs(GOAL[1] - py)
            F[p + f"{which}_dist"] = np.where(ok, np.hypot(dx, dy), np.nan)
            F[p + f"{which}_angle"] = np.where(ok, np.arctan2(dy, dx), np.nan)
        F[p + "dx"] = np.where(ok, ex - x, np.nan)
        F[p + "dy"] = np.where(ok, ey - y, np.nan)
        F[p + "success"] = np.where(ok, S.success.values.astype(float), np.nan)
        F[p + "sameteam"] = np.where(ok, sameteam.astype(float), np.nan)
        tcol = S.type.values
        for t in TYPES:
            F[p + "type_" + t] = np.where(ok, (tcol == t).astype(float), np.nan)
        sub = S.subtype.values
        F[p + "setpiece"] = np.where(ok, np.isin(sub, ["Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"])
                                     .astype(float), np.nan)
        F[p + "penalty"] = np.where(ok, (sub == "Penalty").astype(float), np.nan)
        bp = S.body_part.values
        F[p + "head"] = np.where(ok, (bp == "Head").astype(float), np.nan)
    t0, t1, t2 = A.t.values, A.t.shift(1).values, A.t.shift(2).values
    F["dt_01"] = np.where(same_seq[1], t0 - t1, np.nan)
    F["dt_12"] = np.where(same_seq[2], t1 - t2, np.nan)
    F["period"] = A.period.values.astype(float)
    F["time_in_period"] = A.t.values / 60.0
    F["goal_diff_before"] = A["goal_diff_before"].values.astype(float)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in F.items()}, index=A.index)


class _F32(dict):
    """dict that stores every feature as float32 immediately (peak-memory control)."""

    def __setitem__(self, k, v):
        super().__setitem__(k, np.asarray(v, dtype=np.float32))


def goal_diff_before(A):
    """Goal difference from the acting team's perspective, counting only goals in EARLIER actions."""
    out = np.zeros(len(A))
    for _, g in A.groupby("match_id", sort=False):
        teams = pd.unique(g.team)
        cnt = {t: 0 for t in teams}
        gd = np.empty(len(g))
        for i, (tm, gf) in enumerate(zip(g.team.values, g.goal_for.values, strict=True)):
            other = [t for t in cnt if t != tm]
            gd[i] = cnt[tm] - (cnt[other[0]] if other else 0)
            if pd.notna(gf):
                cnt[gf] = cnt.get(gf, 0) + 1
        out[A.index.get_indexer(g.index)] = gd
    return out


# ----------------------------------------------------------------------------- values
def action_values(A, ps, pc):
    """VAEP value: [P_score(after) - P_score(before)] - [P_concede(after) - P_concede(before)].
    'before' = the previous action's probabilities from the acting team's perspective (swapped if the
    previous action was by the opponent); zero at the start of a period and right after a goal."""
    key = (A.match_id.astype(str) + "_" + A.period.astype(str)).values
    prev_same_seq = np.r_[False, key[1:] == key[:-1]]
    prev_team_same = np.r_[False, A.team.values[1:] == A.team.values[:-1]]
    prev_goal = np.r_[False, pd.notna(A.goal_for.values[:-1])]
    ps_prev = np.r_[0.0, ps[:-1]]
    pc_prev = np.r_[0.0, pc[:-1]]
    b_s = np.where(prev_team_same, ps_prev, pc_prev)
    b_c = np.where(prev_team_same, pc_prev, ps_prev)
    reset = ~prev_same_seq | prev_goal
    b_s, b_c = np.where(reset, 0.0, b_s), np.where(reset, 0.0, b_c)
    off, dfn = ps - b_s, -(pc - b_c)
    return off + dfn, off, dfn


def scores(y, p):
    return dict(log_loss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p), auc=roc_auc_score(y, p))


def main():
    A = pd.read_parquet(ROOT / "data" / "actions.parquet")
    M = pd.read_parquet(ROOT / "data" / "matches.parquet")[["match_id", "league", "half"]]
    A = A.merge(M, on="match_id", how="left").reset_index(drop=True)
    A["goal_diff_before"] = goal_diff_before(A)
    A["scores"], A["concedes"] = labels(A)
    X = features(A)
    import gc
    gc.collect()
    print("features", X.shape, f"{X.memory_usage(deep=True).sum() / 1e9:.2f} GB", flush=True)
    loc_cols = ["a0_x", "a0_y", "a0_end_x", "a0_end_y"]
    ps, pc = np.full(len(A), np.nan), np.full(len(A), np.nan)
    rows, calib = [], []
    for L in sorted(A.league.unique()):
        tr = ((A.league != L) & (A.half == "H1")).values
        te = (A.league == L).values
        assert not (tr & te).any() and set(A.loc[tr, "league"]) .isdisjoint({L})
        for target, store in (("scores", ps), ("concedes", pc)):
            y_tr, y_te = A.loc[tr, target].values, A.loc[te, target].values
            full = HistGradientBoostingClassifier(**HGB).fit(X.loc[tr], y_tr)
            loc = HistGradientBoostingClassifier(**HGB).fit(X.loc[tr, loc_cols], y_tr)
            p_full = full.predict_proba(X.loc[te])[:, 1]
            p_loc = loc.predict_proba(X.loc[te, loc_cols])[:, 1]
            store[te] = p_full
            base = np.full(te.sum(), y_tr.mean())
            for name, p in (("base_rate", base), ("location_only", p_loc), ("vaep", p_full)):
                rows.append(dict(held_out_league=L, label=target, model=name, n=int(te.sum()),
                                 positives=int(y_te.sum()), **scores(y_te, p)))
            bins = np.quantile(p_full, np.linspace(0, 1, 11))
            b = np.clip(np.searchsorted(bins, p_full, side="right") - 1, 0, 9)
            for k in range(10):
                s = b == k
                calib.append(dict(held_out_league=L, label=target, bin=k, predicted=p_full[s].mean(),
                                  observed=y_te[s].mean(), n=int(s.sum())))
        print(L, "done", flush=True)
    A["p_score"], A["p_concede"] = ps, pc
    A["value"], A["offensive"], A["defensive"] = action_values(A, ps, pc)
    keep = ["match_id", "index", "period", "t", "league", "half", "team", "player_id", "player", "position", "type",
            "subtype", "success", "x", "y", "end_x", "end_y", "play_pattern", "goal_for", "scores", "concedes",
            "goal_diff_before", "p_score", "p_concede", "value", "offensive", "defensive"]
    A[keep].to_parquet(ROOT / "data" / "values.parquet")
    V = pd.DataFrame(rows)
    V.to_csv(ROOT / "results" / "vaep_validation.csv", index=False)
    pd.DataFrame(calib).to_csv(ROOT / "results" / "vaep_calibration.csv", index=False)
    print(V.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
