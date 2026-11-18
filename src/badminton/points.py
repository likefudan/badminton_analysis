"""Who won each rally, how it ended, and each player's strokes, points and errors.

Built on what earlier steps wrote: the hits with their hitter and stroke type
(`badminton strokes`), the skeletons, the shuttle track and the court
calibration. Nothing here looks at the video.

A rally is a run of hits with no pause longer than hits.RALLY_GAP_S. Who won
it is not seen directly. The clue used is where the shuttle lay afterwards:

  * the pick-up. After a rally somebody bends down to pick the shuttle up: a
    hand at floor level during the dead time. (A shuttle scooped up with the
    racket while standing is not seen; a crouch without a hand at the floor
    was tried and fired on lunges.) The
    hand is at the floor, so its pixel goes through the court calibration to a
    court position: the spot where the shuttle lay. Inside the lines, the half
    it lay in lost the rally (the shot was not returned, or never crossed the
    net). Clearly outside the lines, that half won (the other side hit it out).
  * when nobody is seen picking it up, the end of the shuttle track after the
    last hit, put on the floor the same way. A shuttle lost in the air lands
    too far from the camera, so this only counts when the track ends coming
    down, and it is marked as a weak call.

"The winner serves next" is not used: in games without a score whoever picks
the shuttle up serves, or knocks it over to the other side first. The side of
the next first hit is written down for reference only.

Between rallies the shuttle is often knocked back to the server and caught.
That looks like a rally of one or two hits; when nobody picks the shuttle up
after such a run it is taken for that and left out (`dead_ball`).

How the rally ended follows from the winner and the last hit found:
  * the last hitter's side won: 制胜球 (the others did not get it back);
  * the last hitter's side lost: 失误 by the last hitter (没过网 when the
    shuttle lay in their own half, 出界 when it lay outside the other half).
    It is 受迫 (forced) when the shot it answered was an attacking one
    (杀球 / 推扑) or came less than FORCED_S before; otherwise 主动失误.
A missed hit at the end of a rally leaves the winner right but gives the
winner or the error to the wrong player. docs/points.md has what this got
right on the recordings checked by eye.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .court import LENGTH, NET_Y, SINGLES_INSET, WIDTH, CourtCalibration
from .hits import KPT_THR, RALLY_GAP_S, _upright_height

SCHEMA = "badminton.points/1"

PICK_HAND_BH = 0.2  # a hand lower than this many body heights above the feet is at the floor
PICK_MIN_S = 0.15  # ... for at least this long
PICK_AFTER_S = 0.5  # looked for from this long after the last hit (a lunge at the last hit is not a pick-up)
PICK_WINDOW_S = 15.0  # ... and for at most this long
IN_M = 0.3  # inside the lines, or outside by less than this: in
OUT_M = 0.9  # outside the lines by more than this: out. In between: no call
FAR_OFF_M = 5.0  # further outside than this is not this court's shuttle
AFTER_S = 2.5  # the shuttle track is followed for this long after the last hit (longer than any flight)
START_S = 0.6  # the flight must start this soon after the last hit, or it is something else
MIN_FLIGHT_S = 0.2  # shorter than this says nothing about where the shuttle went
DOWN_S = 0.15  # the track must be going down the picture over this long at its end
TRACK_OUT_M = 1.5  # "out" from a track end needs this much (a point in the air lands "outside" easily)
NET_ZONE_M = 0.6  # a track end this close to the net line could be either half: no call
DEAD_BALL_HITS = 2  # a run of at most this many hits that nobody picks up after: the shuttle being passed back
FORCED_S = 0.55  # an error on a shot that arrived faster than this was forced
ATTACK_GROUPS = ("杀球", "推扑")
OTHER = {"near": "far", "far": "near"}
SIDE_ZH = {"near": "近端", "far": "远端"}
SIDES = ("near", "far")


def split_rallies(strokes: list[dict], fps: float, gap_s: float = RALLY_GAP_S) -> list[list[dict]]:
    """Runs of hits with no pause longer than gap_s, in time order."""
    out: list[list[dict]] = []
    for s in sorted(strokes, key=lambda s: s["frame"]):
        if out and s["frame"] - out[-1][-1]["frame"] <= gap_s * fps:
            out[-1].append(s)
        else:
            out.append([s])
    return out


# ---------------------------------------------------------------- the spot where the shuttle lay


def judge_spot(court_xy, singles: bool, in_m: float = IN_M, out_m: float = OUT_M) -> dict:
    """A floor point in court metres -> which half, and in / out (None: too close to the line to call)."""
    x, y = float(court_xy[0]), float(court_xy[1])
    inset = SINGLES_INSET if singles else 0.0
    outside = max(inset - x, x - (WIDTH - inset), -y, y - LENGTH, 0.0)
    return {"court_xy": [round(x, 2), round(y, 2)], "half": "near" if y < NET_Y else "far",
            "outside_m": round(outside, 2),
            "in": True if outside <= in_m else False if outside > out_m else None}


def hand_heights(data: dict, tracking) -> list[dict]:
    """Per tracked player, on the skeleton's rows: how high the lower hand is above the feet
    (in standing body heights, NaN when not seen) and that hand's pixel."""
    out = []
    n = len(data["frame"])
    for tr in tracking.tracks:
        hand = np.full((n, 2), np.nan)
        box_h = np.full(n, np.nan)
        feet = np.full(n, np.nan)
        for i in np.flatnonzero(tr.det_index >= 0):
            p = tr.det_index[i]
            b = data["bbox"][i, p]
            box_h[i], feet[i] = b[3] - b[1], b[3]
            ok = [k for k in (9, 10) if data["scores"][i, p, k] >= KPT_THR]
            if ok:
                k = max(ok, key=lambda k: data["keypoints"][i, p, k, 1])
                hand[i] = data["keypoints"][i, p, k]
        height = (feet - hand[:, 1]) / _upright_height(box_h)
        with np.errstate(invalid="ignore"):
            down = height < PICK_HAND_BH
        out.append({"id": tr.id, "side": SIDES[tr.side], "height": height, "spot": hand, "down": down})
    return out


def find_pickup(hands: list[dict], frames: np.ndarray, fps: float, f0: int, f1: int,
                calib: CourtCalibration, singles: bool) -> dict | None:
    """The first time, between video frames f0 and f1, that a player's hand stays at the floor.
    Returns who, when, and the court position of the hand at its lowest."""
    lo, hi = np.searchsorted(frames, [f0, f1])
    if hi - lo < 2:
        return None
    step = float(np.median(np.diff(frames[lo:hi]))) / fps
    need = max(2, int(round(PICK_MIN_S / step)))
    best = None
    for h in hands:
        low = h["down"][lo:hi]
        run = 0
        for j, v in enumerate(low):
            run = run + 1 if v else 0
            if run >= need:
                a = lo + j - run + 1
                b = a
                while b < hi and h["down"][b]:
                    b += 1
                k = a + int(np.nanargmin(h["height"][a:b]))
                if best is None or frames[a] < best["frame"]:
                    spot = judge_spot(calib.to_court(h["spot"][k][None])[0], singles)
                    best = {"frame": int(frames[a]), "time_s": round(float(frames[a]) / fps, 2),
                            "player_id": h["id"], "player_side": h["side"], **spot}
                break
    return best


def find_track_end(xy: np.ndarray, f0: int, f1: int, fps: float) -> dict | None:
    """Where the flight after the hit at frame f0 ends (frames before f1). xy: (N, 2) shuttle
    pixels per video frame, NaN where not seen. None when there is no usable flight."""
    f1 = min(f1, len(xy), f0 + int(round(AFTER_S * fps)))
    run = np.flatnonzero(np.isfinite(xy[f0:f1, 0])) + f0
    if not len(run) or run[0] - f0 > START_S * fps or run[-1] - run[0] < MIN_FLIGHT_S * fps:
        return None
    tail = run[run >= run[-1] - max(2, int(round(DOWN_S * fps)))]
    return {"frame": int(run[-1]), "xy_px": [round(float(v), 1) for v in xy[run[-1]]],
            "flight_s": round(float(run[-1] - f0) / fps, 2),
            "descending": bool(len(tail) >= 2 and xy[tail[-1], 1] > xy[tail[0], 1])}


def spot_vote(spot: dict | None) -> tuple[str | None, str | None]:
    """(winning side, 界内 / 出界) from where the shuttle lay; (None, None) when it cannot be called."""
    if not spot or spot["outside_m"] > FAR_OFF_M or spot["in"] is None:
        return None, None
    return (OTHER[spot["half"]], "界内") if spot["in"] else (spot["half"], "出界")


# ---------------------------------------------------------------- rallies


def error_kind(rally: list[dict], fps: float) -> str:
    """主动失误 (unforced) or 受迫失误 (forced) for an error on the rally's last hit, from the shot it answered."""
    if len(rally) < 2:
        return "unforced"  # a serve
    last, before = rally[-1], rally[-2]
    if before.get("side") == last.get("side"):
        return "unforced"  # the shot in between was missed: nothing known about it
    quick = (last["frame"] - before["frame"]) / fps < FORCED_S
    attack = before.get("group") in ATTACK_GROUPS and not before.get("needs_review")
    return "forced" if quick or attack else "unforced"


def judge_rallies(strokes: list[dict], shuttle_xy: np.ndarray, calib: CourtCalibration, fps: float,
                  singles: bool, data: dict | None = None, tracking=None) -> list[dict]:
    """One dict per run of hits: the clues, the winner and how it ended. Without the skeleton
    (`data`, `tracking`) only the shuttle track is used."""
    groups = split_rallies(strokes, fps)
    hands = hand_heights(data, tracking) if data is not None and tracking is not None else []
    frames = np.asarray(data["frame"], int) if data is not None else np.zeros(0, int)
    end_frame = int(frames[-1]) + 1 if len(frames) else len(shuttle_xy)
    out = []
    for i, g in enumerate(groups):
        last = g[-1]
        nxt = groups[i + 1] if i + 1 < len(groups) else None
        f1 = nxt[0]["frame"] if nxt else end_frame
        pick = None
        if hands:
            pick = find_pickup(hands, frames, fps, last["frame"] + int(round(PICK_AFTER_S * fps)),
                               min(f1, last["frame"] + int(round(PICK_WINDOW_S * fps))), calib, singles)
        track = find_track_end(shuttle_xy, last["frame"], f1, fps)
        if track:
            track.update(judge_spot(calib.to_court(np.array([track["xy_px"]]))[0], singles, out_m=TRACK_OUT_M))
        winner, where = spot_vote(pick)
        conf = "high" if winner else None
        if not winner and track and track["descending"] and abs(track["court_xy"][1] - NET_Y) >= NET_ZONE_M:
            winner, where = spot_vote(track)
            conf = "low" if winner else None
        side = last.get("side")
        if winner and where == "出界" and side == winner:
            conf = "low"  # the shuttle lies outside the last hitter's own half: a hit after it was missed
        row = {
            "n": 0, "start_s": g[0]["time_s"], "end_s": last["time_s"], "hits": len(g),
            "dead_ball": bool(len(g) <= DEAD_BALL_HITS and pick is None and bool(hands)),
            "last_hit": {"frame": last["frame"], "time_s": last["time_s"], "player_id": last.get("player_id"),
                         "side": side, "type": last.get("type"), "group": last.get("group"),
                         "needs_review": bool(last.get("needs_review"))},
            "clues": {"pickup": pick, "track_end": track, "shuttle_lay": where,
                      "next_first_hit_side": nxt[0].get("side") if nxt else None},
            "winner_side": winner, "confidence": conf, "ending": None, "error": None,
            "error_kind_if_lost": error_kind(g, fps),
        }
        if row["dead_ball"]:
            row.update(winner_side=None, confidence=None)
        row["program"] = {k: row[k] for k in ("winner_side", "confidence", "dead_ball")}
        out.append(row)
    settle(out)
    return out


def settle(rallies: list[dict]) -> None:
    """How each rally ended (from its winner and last hit), and the rally numbers (dead balls get 0)."""
    n = 0
    for row in rallies:
        row["ending"] = row["error"] = None
        winner, side = row["winner_side"], row["last_hit"]["side"]
        if not row["dead_ball"] and winner and side:
            if winner == side:
                row["ending"] = "winner"
            else:
                row["ending"] = "error"
                row["error"] = {"kind": row["error_kind_if_lost"], "how": _how(row)}
        n += not row["dead_ball"]
        row["n"] = 0 if row["dead_ball"] else n


def _how(row: dict) -> str | None:
    """没过网 / 出界, from where the shuttle lay; only when that spot agrees with who lost."""
    clues = row["clues"]
    spot = clues["pickup"] or clues["track_end"]
    by_spot, where = spot_vote(spot)
    if not spot or by_spot != row["winner_side"]:
        return None
    return "出界" if where == "出界" else "没过网" if spot["half"] == row["last_hit"]["side"] else None


# ---------------------------------------------------------------- calls a person made on the result page

LABEL_MATCH_S = 1.5  # a label belongs to the rally whose last hit is this close to the one it was made on
LABEL_WINNERS = ("near", "far", "none")  # none: not a rally (the shuttle passed back, a warm-up)


def load_labels(path: str | Path) -> list[dict]:
    """<video>.points.labels.csv (frame,winner,note): frame = the last hit of the rally as the page showed it."""
    import csv

    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("frame") or "").strip()]
    return [{"frame": int(float(r["frame"])), "winner": r["winner"].strip(), "note": (r.get("note") or "").strip()}
            for r in rows if r.get("winner", "").strip() in LABEL_WINNERS]


def apply_labels(rallies: list[dict], labels: list[dict], fps: float) -> int:
    """Put a person's calls over the program's: each label goes to the rally whose last hit is nearest
    (within LABEL_MATCH_S). "none" takes the run out like a dead ball; near / far make it a rally with
    that winner, even one the program had taken for a dead ball. Returns how many labels found a rally."""
    used = 0
    for lab in labels:
        best = min(rallies, key=lambda r: abs(r["last_hit"]["frame"] - lab["frame"]), default=None)
        if best is None or abs(best["last_hit"]["frame"] - lab["frame"]) > LABEL_MATCH_S * fps:
            continue
        used += 1
        if lab["winner"] == "none":
            best.update(dead_ball=True, winner_side=None, confidence="manual")
        else:
            best.update(dead_ball=False, winner_side=lab["winner"], confidence="manual")
    settle(rallies)
    return used


COUNTED = ("high", "manual")  # calls that go into the scores: a pick-up seen, or a person's


def player_table(strokes: list[dict], rallies: list[dict], names: dict | None = None) -> list[dict]:
    """Per player: strokes by group (the sure ones in `strokes`, the unsure ones by their likeliest
    group in `strokes_guess`), rally-ending winners and errors (only of
    rallies called with a pick-up seen). Hits of runs marked dead_ball are left out."""
    names = names or {}
    skip = [(r["start_s"], r["end_s"]) for r in rallies if r["dead_ball"]]
    table: dict = {}
    for s in strokes:
        pid = s.get("player_id")
        if pid is None or any(a <= s["time_s"] <= b for a, b in skip):
            continue
        p = table.setdefault(pid, {"id": pid, "name": names.get(str(pid), "") or s.get("player_name", ""),
                                   "side": s.get("side"), "hits": 0, "hits_unsure": 0, "strokes": {}, "strokes_guess": {},
                                   "winners": 0, "errors": 0, "unforced_errors": 0, "forced_errors": 0,
                                   "winners_by_type": {}, "errors_by_type": {}})
        p["hits"] += 1
        if s.get("needs_review"):
            p["hits_unsure"] += 1
        into = p["strokes_guess"] if s.get("needs_review") else p["strokes"]
        into[s["group"]] = into.get(s["group"], 0) + 1
    for r in rallies:
        p = table.get(r["last_hit"]["player_id"])
        if p is None or not r["ending"] or r["confidence"] not in COUNTED:
            continue
        group = r["last_hit"]["group"] + ("?" if r["last_hit"]["needs_review"] else "")
        if r["ending"] == "winner":
            p["winners"] += 1
            p["winners_by_type"][group] = p["winners_by_type"].get(group, 0) + 1
        else:
            p["errors"] += 1
            p[r["error"]["kind"] + "_errors"] += 1
            p["errors_by_type"][group] = p["errors_by_type"].get(group, 0) + 1
    return [table[k] for k in sorted(table)]


def side_table(rallies: list[dict]) -> dict:
    """Points per side, counting only the rallies called with a pick-up seen."""
    out = {s: {"points": 0, "winners": 0, "opponent_errors": 0, "opponent_unforced_errors": 0} for s in SIDES}
    for r in rallies:
        w = r["winner_side"]
        if not w or r["confidence"] not in COUNTED:
            continue
        out[w]["points"] += 1
        if r["ending"] == "winner":
            out[w]["winners"] += 1
        elif r["ending"] == "error":
            out[w]["opponent_errors"] += 1
            out[w]["opponent_unforced_errors"] += r["error"]["kind"] == "unforced"
    return out


def points_json(strokes: list[dict], rallies: list[dict], fps: float, video: str = "", inputs: dict | None = None,
                names: dict | None = None, singles: bool = False) -> dict:
    real = [r for r in rallies if not r["dead_ball"]]
    return {
        "schema": SCHEMA, "video": video, "fps": fps, "inputs": inputs or {}, "format": "单打" if singles else "双打",
        "rules": {"rally_gap_s": RALLY_GAP_S, "forced_s": FORCED_S, "attack_groups": list(ATTACK_GROUPS),
                  "in_m": IN_M, "out_m": OUT_M, "pick_hand_bh": PICK_HAND_BH},
        "how_to_read": [
            "谁赢这一分是推出来的，不是看见的：看死球后是谁弯腰把球捡起来、在哪捡（站着用拍子把球挑起来的认不出）。球躺在界内，它所在的半场丢分；明显在界外，它所在的半场得分。",
            "confidence：manual = 人在结果页上点的；high = 看到了捡球；low = 没看到捡球，只按最后一拍之后球轨迹的终点猜（不可靠，不计入得分和失误）；都没有就不判。program 是程序自己的判断。",
            "最后一拍漏掉时，得分的一边通常还对，但制胜球或失误会记到错的人头上。",
            "主动失误 = 丢分的最后一拍，而且它接的那一拍不是杀球/推扑，来球间隔也不短于 forced_s 秒；否则算受迫失误。",
            "dead_ball 的不是回合：一两拍之后没人捡球，是把球打回给发球的人。这些拍不计入统计。",
            "hits_unsure 是类型没把握的击球：不计入 strokes，按最可能的大类计入 strokes_guess（是猜的）。winners_by_type / errors_by_type 里带 ? 的大类也是猜的。",
            "每个回合的第一拍按规则算发球（击球人离网不超过 5.3 米时；长短看接发球的人站在哪），其余拍不会被认成发球。"
        ],
        "rallies_total": len(real), "rallies_judged": sum(1 for r in real if r["confidence"] in COUNTED),
        "rallies_labelled": sum(1 for r in rallies if r["confidence"] == "manual"),
        "rallies_guessed": sum(1 for r in real if r["confidence"] == "low"),
        "dead_ball_runs": len(rallies) - len(real),
        "sides": side_table(real), "players": player_table(strokes, rallies, names), "rallies": rallies,
    }


def _kinds(counts: dict | None) -> str:
    return "、".join(f"{k} {v}" for k, v in sorted((counts or {}).items(), key=lambda kv: -kv[1])) or "无"


def summary_lines(doc: dict) -> list[str]:
    """What the command prints: the score by side, then one line per player."""
    lab = doc.get("rallies_labelled", 0)
    lines = [f"{doc['rallies_total']} 个回合，判出胜负的 {doc['rallies_judged']} 个" + (f"（其中 {lab} 个是人标的）" if lab else "")
             + f"（另有 {doc['rallies_guessed']} 个只能按球轨迹猜，不计分；{doc['dead_ball_runs']} 段是把球打回给对方，不算回合）"]
    for side, s in doc["sides"].items():
        lines.append(f"  {SIDE_ZH[side]}：得 {s['points']} 分（制胜球 {s['winners']}，对方失误 {s['opponent_errors']}，"
                     f"其中主动失误 {s['opponent_unforced_errors']}）")
    for p in doc["players"]:
        kinds, guess = _kinds(p.get("strokes")), _kinds(p.get("strokes_guess"))
        lines.append(f"  {p['name'] or '球员 ' + str(p['id'])}（{SIDE_ZH.get(p['side'], '?')}）：{p['hits']} 拍"
                     f"，有把握的：{kinds}；没把握的 {p['hits_unsure']} 拍猜是：{guess}；制胜球 {p['winners']}，"
                     f"失误 {p['errors']}（主动 {p['unforced_errors']}，受迫 {p['forced_errors']}）")
    return lines


def digest(doc: dict) -> dict:
    """What the AI review is given: the totals and one short row per rally."""
    rows = []
    for r in doc["rallies"]:
        if r["dead_ball"]:
            continue
        rows.append({"n": r["n"], "start_s": r["start_s"], "end_s": r["end_s"], "hits": r["hits"],
                     "winner_side": r["winner_side"] if r["confidence"] in COUNTED else None,
                     "ending": r["ending"] if r["confidence"] in COUNTED else None,
                     "error": r["error"] if r["confidence"] in COUNTED else None,
                     "shuttle_lay": r["clues"]["shuttle_lay"] if r["confidence"] in COUNTED else None,
                     "by_person": r["confidence"] == "manual",
                     "last_hit": {k: r["last_hit"][k] for k in ("player_id", "side", "type", "needs_review")}})
    return {"format": doc["format"], "rallies_total": doc["rallies_total"], "rallies_judged": doc["rallies_judged"],
            "rallies_labelled": doc.get("rallies_labelled", 0),
            "sides": doc["sides"], "players": doc["players"], "rallies": rows}


def digest_text(d: dict) -> list[str]:
    """【回合和得失分】 for the review prompt."""
    lab = d.get("rallies_labelled", 0)
    lines = [f"【回合和得失分】按击球切出 {d['rallies_total']} 个回合，其中 {d['rallies_judged']} 个判出了谁赢"
             + (f"，{lab} 个是球友在结果页上自己标的谁赢" if lab else "")
             + "（程序的办法：死球后谁弯腰捡球、在哪捡；球在界内，它所在半场丢分；明显界外，它所在半场得分。没看到捡球的不判）。"
             "没判出的回合不能当成平分，也不能据此说谁赢了这场。击球类型是模型认的（每回合第一拍按规则算发球），「没把握」的单独列出、只是猜的大类，说的时候要和有把握的分开。"]
    for side, s in d["sides"].items():
        lines.append(f"- {SIDE_ZH[side]}：判出的回合里得 {s['points']} 分（自己打出制胜球 {s['winners']}，"
                     f"对方失误 {s['opponent_errors']}，其中主动失误 {s['opponent_unforced_errors']}）")
    for p in d["players"]:
        kinds, guess = _kinds(p.get("strokes")), _kinds(p.get("strokes_guess"))
        lines.append(f"- {p['name'] or str(p['id']) + ' 号'}（{SIDE_ZH.get(p['side'], '?')}）：{p['hits']} 拍，"
                     f"有把握的：{kinds}；另有 {p['hits_unsure']} 拍类型没把握，按最可能的大类猜是：{guess}。制胜球 {p['winners']}，"
                     f"失误 {p['errors']}（主动 {p['unforced_errors']}，受迫 {p['forced_errors']}）")
    for r in d["rallies"]:
        if not r["winner_side"]:
            continue
        last = r["last_hit"]
        how = "制胜球" if r["ending"] == "winner" else \
            {"unforced": "主动失误", "forced": "受迫失误"}[r["error"]["kind"]] + (f"（{r['error']['how']}）" if r["error"]["how"] else "")
        lines.append(f"- 回合 {r['n']}（{r['start_s']:.0f}–{r['end_s']:.0f} 秒，{r['hits']} 拍）：{SIDE_ZH[r['winner_side']]}得分"
                     + ("（人标的）" if r.get("by_person") else "") + "，"
                     f"最后一拍 {last['player_id']} 号 {last['type']}{'（类型没把握，是猜的）' if last['needs_review'] else ''}，{how}")
    return lines


def run(strokes_json_path, shuttle_csv, skeleton_npz, court: CourtCalibration, players_per_side: int,
        names: dict | None = None, labels_csv=None) -> dict:
    """Files in, the points document out. labels_csv: the winners a person set on the result page."""
    from .bst import read_shuttle_csv
    from .tracking import load_npz, track_players

    doc = json.loads(Path(strokes_json_path).read_text())
    strokes, fps = doc["strokes"], float(doc["fps"])
    data = load_npz(skeleton_npz)
    tracking = track_players(data, court, players_per_side)
    xy = read_shuttle_csv(shuttle_csv, int(data["frame"][-1]) + 1)
    rallies = judge_rallies(strokes, xy, court, fps, players_per_side == 1, data, tracking)
    inputs = {"strokes": str(strokes_json_path), "shuttle": str(shuttle_csv), "skeleton": str(skeleton_npz)}
    if labels_csv and Path(labels_csv).exists():
        apply_labels(rallies, load_labels(labels_csv), fps)
        inputs["labels"] = str(labels_csv)
    return points_json(strokes, rallies, fps, doc.get("video", ""), inputs, names, players_per_side == 1)
