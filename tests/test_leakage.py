"""Leakage and data-integrity guards. Run after the pipeline: `python -m pytest -q`.

Tests that need the derived data in data/ are skipped if it is absent (the data are not redistributed;
see README > Reproduce).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"
needs_data = pytest.mark.skipif(not (DATA / "values.parquet").exists(), reason="derived data not built")


@pytest.fixture(scope="module")
def matches():
    return pd.read_parquet(DATA / "matches.parquet")


@pytest.fixture(scope="module")
def actions():
    return pd.read_parquet(DATA / "actions.parquet")


# ----------------------------------------------------------------------------- time split
@needs_data
def test_halves_are_strictly_ordered_in_time(matches):
    for league, g in matches.groupby("league"):
        assert g[g.half == "H1"].date.max() < g[g.half == "H2"].date.min(), league


@needs_data
def test_split_is_at_first_round_20_fixture(matches):
    for league, g in matches.groupby("league"):
        cut = g[g["round"] == 20].date.min()
        assert (g.half == np.where(g.date < cut, "H1", "H2")).all(), league


@needs_data
def test_match_counts(matches):
    assert len(matches) == 1517
    assert matches.match_id.is_unique


# ----------------------------------------------------------------------------- goals and minutes
@needs_data
def test_goals_per_team_equal_scores(matches, actions):
    gf = actions[actions.goal_for.notna()].groupby(["match_id", "goal_for"]).size()
    for m in matches.itertuples():
        assert gf.get((m.match_id, m.home_team_events), 0) == m.home_score, m.match_id
        assert gf.get((m.match_id, m.away_team_events), 0) == m.away_score, m.match_id


@needs_data
def test_never_more_than_eleven_on_the_pitch():
    mi = pd.read_parquet(DATA / "minutes.parquet")
    worst = 0
    for (_, _), g in mi.groupby(["match_id", "team"]):
        ev = np.r_[g.start.values, g.stop.values]
        delta = np.r_[np.ones(len(g)), -np.ones(len(g))]
        order = np.lexsort((delta, ev))          # at equal times, process exits before entries
        worst = max(worst, int(np.cumsum(delta[order]).max()))
    assert worst == 11


@needs_data
def test_minutes_are_plausible():
    mi = pd.read_parquet(DATA / "minutes.parquet")
    per_match = mi.groupby(["match_id", "player_id"]).seconds.sum()
    assert (per_match >= 0).all() and per_match.max() < 110 * 60


# ----------------------------------------------------------------------------- VAEP labels and features
def _toy(n=30, goal_at=None, team_seq=None):
    team_seq = team_seq or ["A"] * n
    rows = []
    for i in range(n):
        rows.append(dict(match_id=1, index=i, period=1, t=float(i), team=team_seq[i], player_id=i,
                         type="pass", subtype="Open Play", success=True, body_part="Right Foot",
                         x=50.0 + i % 7, y=40.0, end_x=60.0, end_y=41.0, play_pattern="Regular Play",
                         goal_for=None, under_pressure=False, league="L", half="H1"))
    A = pd.DataFrame(rows)
    if goal_at is not None:
        A.loc[goal_at, "type"] = "shot"
        A.loc[goal_at, "success"] = True
        A.loc[goal_at, "goal_for"] = A.loc[goal_at, "team"]
    return A


def test_labels_window_is_ten_actions_including_current():
    from vaep import labels
    A = _toy(30, goal_at=20)
    s, c = labels(A)
    assert s.tolist() == [0] * 11 + [1] * 10 + [0] * 9     # actions 11..20 score, nothing after the goal
    assert c.sum() == 0


def test_labels_do_not_cross_periods():
    from vaep import labels
    A = _toy(30, goal_at=3)
    A.loc[:1, "period"] = 1
    A.loc[2:, "period"] = 2
    s, _ = labels(A)
    assert s[:2].sum() == 0 and s[2:4].sum() == 2


def test_features_do_not_depend_on_later_actions():
    from vaep import features
    A = _toy(30)
    A["goal_diff_before"] = 0
    X1 = features(A)
    B = A.copy()
    B.loc[15:, ["x", "end_x", "t"]] = [[5.0, 110.0, 999.0]] * 15
    B.loc[15:, "type"] = "shot"
    X2 = features(B)
    pd.testing.assert_frame_equal(X1.iloc[:15], X2.iloc[:15])


def test_goal_difference_excludes_the_current_goal():
    from vaep import goal_diff_before
    A = _toy(10, goal_at=4)
    gd = goal_diff_before(A)
    assert gd[4] == 0 and gd[5] == 1


# ----------------------------------------------------------------------------- VAEP training sets
def test_vaep_training_set_excludes_held_out_league_and_second_halves():
    src = (ROOT / "src" / "vaep.py").read_text()
    assert 'tr = ((A.league != L) & (A.half == "H1")).values' in src


# ----------------------------------------------------------------------------- ratings
@needs_data
def test_validation_fit_sees_first_half_only(monkeypatch):
    import ratings
    seen = {}

    def spy(pm, **kw):
        seen["halves"] = set(pm.half)
        raise RuntimeError("stop")

    monkeypatch.setattr(ratings, "fit", spy)
    pm = ratings.player_matches("progression")
    with pytest.raises(RuntimeError, match="stop"):
        ratings.validate(pm, "progression", "test")
    assert seen["halves"] == {"H1"}


@needs_data
def test_progression_measure_uses_open_play_ball_moving_actions_only():
    import ratings
    V = pd.read_parquet(DATA / "values.parquet", columns=["match_id", "player_id", "type", "subtype", "value"])
    pm = ratings.player_matches("progression")
    keep = V[V.type.isin(["pass", "cross", "carry", "dribble"]) & (V.subtype == "Open Play")]
    expected = keep.groupby(["match_id", "player_id"]).value.sum()
    got = pm.set_index(["match_id", "player_id"]).y
    common = got.index.intersection(expected.index)
    assert len(common) > 30000
    np.testing.assert_allclose(got.loc[common].values, expected.loc[common].values, rtol=1e-9, atol=1e-12)


@needs_data
def test_actions_fall_inside_players_time_on_pitch(actions):
    mi = pd.read_parquet(DATA / "minutes.parquet")
    A = actions[["match_id", "period", "t", "player_id"]]
    p1 = A[A.period == 1].groupby("match_id").t.max().rename("p1")
    A = A.merge(p1, on="match_id")
    A = A.assign(mt=A.t + np.where(A.period == 2, A.p1, 0.0))
    k = A.merge(mi[["match_id", "player_id", "start", "stop"]], on=["match_id", "player_id"], how="left")
    assert k.start.notna().all()
    outside = ((k.mt < k.start - 60) | (k.mt > k.stop + 60)).sum()
    assert outside == 7   # README: "Seven of the 3.23 million actions" (substitutions logged late)
