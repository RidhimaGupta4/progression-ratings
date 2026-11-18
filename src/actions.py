"""StatsBomb events -> one row per on-ball action, plus minutes played and match metadata.

Outputs (data/):
  matches.parquet  match_id, league, date, round, half (H1/H2 by date: see DESIGN.md)
  actions.parquet  one row per action, in event order, with type/result/locations/team/player/goal flags
  minutes.parquet  player x match: minutes on pitch, team, minutes per position group

Usage: python src/actions.py <statsbomb_data_dir>
"""
import sys
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = {2: "Premier League", 7: "Ligue 1", 11: "La Liga", 12: "Serie A"}
SEASON = 27
SET_PIECE_PASS = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
POS_GROUP = {
    "Goalkeeper": "GK",
    "Center Back": "CB", "Left Center Back": "CB", "Right Center Back": "CB",
    "Right Back": "FB/WB", "Left Back": "FB/WB", "Right Wing Back": "FB/WB", "Left Wing Back": "FB/WB",
    "Center Defensive Midfield": "DM/CM", "Left Defensive Midfield": "DM/CM", "Right Defensive Midfield": "DM/CM",
    "Center Midfield": "DM/CM", "Left Center Midfield": "DM/CM", "Right Center Midfield": "DM/CM",
    "Center Attacking Midfield": "AM/W", "Left Attacking Midfield": "AM/W", "Right Attacking Midfield": "AM/W",
    "Left Midfield": "AM/W", "Right Midfield": "AM/W", "Left Wing": "AM/W", "Right Wing": "AM/W",
    "Center Forward": "CF", "Left Center Forward": "CF", "Right Center Forward": "CF", "Secondary Striker": "CF",
}
GK_ACTIONS = {"Shot Saved", "Collected", "Punch", "Keeper Sweeper", "Smother", "Save", "Shot Saved To Post",
              "Shot Saved Off Target"}


def clock(mmss):
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def ts_seconds(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def build_matches(raw):
    rows = []
    for cid, league in LEAGUES.items():
        ms = orjson.loads((raw / "matches" / str(cid) / f"{SEASON}.json").read_bytes())
        mid_date = min(m["match_date"] for m in ms if m["match_week"] == 20)
        for m in ms:
            rows.append(dict(match_id=m["match_id"], league=league, date=m["match_date"], round=m["match_week"],
                             half="H1" if m["match_date"] < mid_date else "H2",
                             home_team=m["home_team"]["home_team_name"], away_team=m["away_team"]["away_team_name"],
                             home_team_id=m["home_team"]["home_team_id"], away_team_id=m["away_team"]["away_team_id"],
                             home_score=m["home_score"], away_score=m["away_score"]))
    return pd.DataFrame(rows)


def action_of(e):
    """Map one StatsBomb event to (type, subtype, result, end_location) or None if it is not an action."""
    t = e["type"]["name"]
    loc = e.get("location")
    if t == "Pass":
        p = e["pass"]
        sub = p.get("type", {}).get("name", "Open Play")
        if sub in ("Recovery", "Interception"):
            sub = "Open Play"
        kind = "cross" if p.get("cross") else "pass"
        return kind, sub, "outcome" not in p, p.get("end_location"), p.get("body_part", {}).get("name")
    if t == "Carry":
        return "carry", "Open Play", True, e["carry"].get("end_location"), None
    if t == "Dribble":
        return "dribble", "Open Play", e["dribble"]["outcome"]["name"] == "Complete", loc, None
    if t == "Shot":
        s = e["shot"]
        return ("shot", s.get("type", {}).get("name", "Open Play"), s["outcome"]["name"] == "Goal",
                (s.get("end_location") or loc)[:2], s.get("body_part", {}).get("name"))
    if t == "Interception":
        return "interception", "", not e["interception"]["outcome"]["name"].startswith("Lost"), loc, None
    if t == "Duel" and e.get("duel", {}).get("type", {}).get("name") == "Tackle":
        oc = e["duel"].get("outcome", {}).get("name", "")
        return "tackle", "", oc in ("Won", "Success", "Success In Play", "Success Out"), loc, None
    if t == "Clearance":
        return "clearance", "", True, loc, None
    if t == "Ball Recovery":
        return "recovery", "", not e.get("ball_recovery", {}).get("recovery_failure", False), loc, None
    if t == "Block":
        return "block", "", True, loc, None
    if t == "Goal Keeper" and e["goalkeeper"].get("type", {}).get("name") in GK_ACTIONS:
        return "keeper", e["goalkeeper"]["type"]["name"], True, loc, None
    if t == "Foul Committed":
        return "foul", "", False, loc, None
    if t in ("Miscontrol", "Dispossessed"):
        return "bad_touch", "", False, loc, None
    if t == "Own Goal Against":
        return "own_goal", "", False, loc, None
    return None


def build_actions(raw, match):
    ev = orjson.loads((raw / "events" / f"{match.match_id}.json").read_bytes())
    ev.sort(key=lambda e: e["index"])
    # Team names differ between match files and event files (e.g. "Olympique de Marseille" vs "Marseille"),
    # so the opponent is resolved by team id and named with the event-file name used everywhere downstream.
    ids = {match.home_team_id, match.away_team_id}
    ev_name = {e["team"]["id"]: e["team"]["name"] for e in ev if "team" in e}
    assert set(ev_name) == ids, (match.match_id, ev_name, ids)
    rows = []
    for e in ev:
        if e["period"] > 4:
            continue
        a = action_of(e)
        if a is None or e.get("location") is None:
            continue
        kind, sub, ok, end, body = a
        team = e["team"]["name"]
        other = ev_name[(ids - {e["team"]["id"]}).pop()]
        goal_for = team if (kind == "shot" and ok) else (other if kind == "own_goal" else None)
        x, y = e["location"][:2]
        ex, ey = (end[:2] if end is not None else (x, y))
        rows.append(dict(
            match_id=match.match_id, index=e["index"], period=e["period"],
            t=ts_seconds(e["timestamp"]), team=team, player_id=e.get("player", {}).get("id"),
            player=e.get("player", {}).get("name"), position=e.get("position", {}).get("name"),
            type=kind, subtype=sub, success=bool(ok), body_part=body or "",
            x=float(x), y=float(y), end_x=float(ex), end_y=float(ey),
            play_pattern=e["play_pattern"]["name"], goal_for=goal_for,
            under_pressure=bool(e.get("under_pressure", False))))
    return rows, ev_name[match.home_team_id], ev_name[match.away_team_id]


def build_minutes(raw, match):
    """Minutes on pitch per player, from events (not line-up intervals).

    The line-up files contain inconsistencies: e.g. a player who leaves for treatment ('Player Off') and is
    then substituted is still listed as coming back on ('Player On') until the final whistle, which puts
    12 players on the pitch. Events record what happened: starters from 'Starting XI', exits and entries
    from 'Substitution', dismissals from red cards. Temporary treatment absences are ignored (player
    remains on the pitch). Position group per player-match: most common position across his events.
    """
    ev = orjson.loads((raw / "events" / f"{match.match_id}.json").read_bytes())
    ev.sort(key=lambda e: e["index"])
    pend = {}
    for e in ev:
        if e["period"] in (1, 2):
            pend[e["period"]] = max(pend.get(e["period"], 0.0), ts_seconds(e["timestamp"]))

    def mt(e):  # match time in seconds: period 2 starts after the full length of period 1
        return ts_seconds(e["timestamp"]) + (pend[1] if e["period"] == 2 else 0.0)

    end_time = pend[1] + pend[2]
    on, off, team_of, name_of = {}, {}, {}, {}
    pos_counts = {}
    for e in ev:
        t = e["type"]["name"]
        if t == "Starting XI":
            for p in e["tactics"]["lineup"]:
                pid = p["player"]["id"]
                on[pid], team_of[pid], name_of[pid] = 0.0, e["team"]["name"], p["player"]["name"]
                pos_counts.setdefault(pid, {}).setdefault(p["position"]["name"], 0)
        elif t == "Substitution":
            out_id, in_p = e["player"]["id"], e["substitution"]["replacement"]
            off[out_id] = mt(e)
            on[in_p["id"]], team_of[in_p["id"]], name_of[in_p["id"]] = mt(e), e["team"]["name"], in_p["name"]
        else:
            card = (e.get("foul_committed", {}) or {}).get("card", {}).get("name") or \
                   (e.get("bad_behaviour", {}) or {}).get("card", {}).get("name")
            if card in ("Red Card", "Second Yellow") and "player" in e:
                off[e["player"]["id"]] = min(off.get(e["player"]["id"], np.inf), mt(e))
        if "player" in e and "position" in e:
            c = pos_counts.setdefault(e["player"]["id"], {})
            c[e["position"]["name"]] = c.get(e["position"]["name"], 0) + 1
    rows = []
    for pid, start in on.items():
        stop = min(off.get(pid, end_time), end_time)
        pc = pos_counts.get(pid, {})
        pos = max(pc, key=pc.get) if pc else "unknown"
        rows.append(dict(match_id=match.match_id, team=team_of[pid], player_id=pid, player=name_of[pid],
                         group=POS_GROUP.get(pos, "other"), seconds=max(0.0, stop - start),
                         start=start, stop=max(start, stop)))
    return rows


def main(raw):
    raw = Path(raw)
    (ROOT / "data").mkdir(exist_ok=True)
    matches = build_matches(raw)
    acts, mins, home_ev, away_ev = [], [], [], []
    for m in matches.itertuples():
        rows, h, a = build_actions(raw, m)
        acts += rows
        home_ev.append(h)
        away_ev.append(a)
        mins += build_minutes(raw, m)
    matches["home_team_events"], matches["away_team_events"] = home_ev, away_ev
    A = pd.DataFrame(acts)
    Mi = pd.DataFrame(mins)
    names = {}
    for m in matches.itertuples():
        for team in orjson.loads((raw / "lineups" / f"{m.match_id}.json").read_bytes()):
            for pl in team["lineup"]:
                names[pl["player_id"]] = pl.get("player_nickname") or pl["player_name"]
    pd.DataFrame(dict(player_id=list(names), display_name=list(names.values()))).to_parquet(
        ROOT / "data" / "players.parquet")
    matches.to_parquet(ROOT / "data" / "matches.parquet")
    A.to_parquet(ROOT / "data" / "actions.parquet")
    Mi.to_parquet(ROOT / "data" / "minutes.parquet")
    print(matches.groupby(["league", "half"]).size().unstack())
    print("actions", len(A), "| types", A.type.value_counts().to_dict())
    print("goals from actions", A.goal_for.notna().sum(), "| goals from match scores",
          int((matches.home_score + matches.away_score).sum()))
    # per-match, per-side check: goals credited to each team equal that team's score in the match file
    gf = A[A.goal_for.notna()].groupby(["match_id", "goal_for"]).size()
    bad = sum(int(gf.get((m.match_id, m.home_team_events), 0)) != m.home_score or
              int(gf.get((m.match_id, m.away_team_events), 0)) != m.away_score for m in matches.itertuples())
    assert bad == 0, f"{bad} matches where per-team goals disagree with the score"
    assert A.goal_for.dropna().isin(A.team.unique()).all()
    print("per-match per-team goal check: OK for all", len(matches), "matches")


if __name__ == "__main__":
    main(sys.argv[1])
