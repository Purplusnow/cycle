"""정적 사이트 생성기.

DB → Jinja2 → ``dist/`` 정적 HTML. 서버가 없으므로 전부 빌드 타임에 확정한다.

이 사이트가 반드시 지켜야 하는 것.

* **예상이 언제 만들어졌는지 화면에 적는다.** 발주 전에 확정 저장한 기록과
  시간순 교차검증으로 사후 산출한 기록은 뜻이 전혀 다르다. 표에서 구분되지
  않으면 읽는 사람이 둘을 같은 것으로 받아들인다.
* **빗나간 예상을 지우지 않는다.** 결과 아카이브는 적중·불발을 그대로 싣는다.
* **표본 수를 함께 적는다.** 30경주짜리 60% 와 3,000경주짜리 48% 는 다른 말이다.
* **베이스라인을 같이 적는다.** 경륜은 득점 하나로 대부분 설명되는 종목이라,
  적중률 60% 는 그 자체로는 아무 말도 하지 않는다. '득점 최고를 그냥 찍었을
  때'와 나란히 놓아야 모델이 무엇을 보탰는지 보인다.

    python -m cycleai.site --db data/cycleai.sqlite --out dist
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .clock import now_kst, today_kst
from .kcycle.store import session
from .korean import josa, josa_of
from .marks import MARK_MEANING, MARK_THRESHOLDS, assign_marks
from .simulate import TACTIC_DESC, simulate as run_simulation
from .strategy import build_report as strategy_report
from .verify import BET_ORDER, build_report

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path("templates")
STATIC_DIR = Path("static")

# 발주 전에 확정 저장한 공개 기록 / 시간순 교차검증으로 산출한 검증 기록.
LIVE_VERSION = "v1"
OOS_VERSION = "v1-oos"
VERSION_LABEL = {
    LIVE_VERSION: ("공개", "발주 전에 확정 저장한 예상"),
    OOS_VERSION: ("검증", "시간순 교차검증으로 산출한 과거 재현 기록"),
}

WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]

# 그날 편성 전체. 예상이 없는 경주도 포함한다 — 자동 실행이 늦어 발주가 지난
# 경주에는 예상을 만들지 않는데(동결 규칙), 그렇다고 목록에서 빼면 1R 이 사라지고
# 8R 부터 시작하는 이상한 시간표가 된다. 편성은 그대로 두고 예상 자리만 비운다.
DAY_SQL = """
SELECT r.race_key, r.race_ymd, r.stnd_yr, r.meet_nm, r.week_tcnt, r.day_tcnt,
       r.race_no, r.post_time, r.race_grade, r.race_len, r.round_cnt, r.field_size,
       COALESCE(r.has_result, 0) AS has_result,
       (SELECT COUNT(*) FROM predictions p WHERE p.race_key = r.race_key
                                       AND p.model_version = ?)  AS n_pred
FROM races r
WHERE r.race_ymd = ?
  AND EXISTS (SELECT 1 FROM entries e WHERE e.race_key = r.race_key)
ORDER BY r.race_no ASC
"""

RACE_SQL = """
SELECT r.race_key, r.race_ymd, r.stnd_yr, r.meet_nm, r.week_tcnt, r.day_tcnt,
       r.race_no, r.post_time, r.race_grade, r.race_len, r.round_cnt, r.field_size,
       COALESCE(r.has_result, 0) AS has_result,
       (SELECT COUNT(*) FROM predictions p WHERE p.race_key = r.race_key
                                       AND p.model_version = ?)  AS n_pred,
       (SELECT COUNT(DISTINCT v.pool) FROM payoffs v
         WHERE v.race_key = r.race_key)                          AS n_pool
FROM races r
WHERE EXISTS (SELECT 1 FROM predictions p WHERE p.race_key = r.race_key
                                      AND p.model_version = ?)
-- 경주는 **항상 1R 부터** 늘어놓는다. 하루 안의 순서는 시행 순서이고,
-- 그것이 사람이 경주를 찾는 순서다. 날짜만 최신이 위로 온다.
ORDER BY r.race_ymd DESC, r.race_no ASC
"""

RUNNER_SQL = """
SELECT p.back_no, p.racer_nm, p.p_win, p.p_top2, p.p_top3, p.pred_rank,
       e.racer_grd, e.grade_cur, e.grade_bef, e.period_no, e.age, e.trng_plc,
       e.color_nm, e.gear_rate, e.rec_200m,
       e.pre_win_cnt, e.pas_win_cnt, e.brk_win_cnt, e.mrk_win_cnt,
       e.win_tot_cnt, e.run_day_cnt,
       e.win_rate, e.high_rate, e.high_3_rate, e.tot_avg_scr, e.area_avg_scr,
       e.bf1_meet, e.bf1_ymd, e.bf1_d1, e.bf1_d2, e.bf1_d3,
       e.bf2_meet, e.bf2_ymd, e.bf2_d1, e.bf2_d2, e.bf2_d3,
       e.bf3_meet, e.bf3_ymd, e.bf3_d1, e.bf3_d2, e.bf3_d3,
       res.ord
FROM predictions p
LEFT JOIN entries e   ON e.race_key = p.race_key AND e.back_no = p.back_no
LEFT JOIN results res ON res.race_key = p.race_key AND res.racer_nm = p.racer_nm
WHERE p.race_key = ? AND p.model_version = ?
ORDER BY p.back_no
"""

# 전법. 화면에서 쓰는 순서와 이름을 한 곳에 둔다.
TACTICS = [("선행", "pre_win_cnt"), ("젖히기", "pas_win_cnt"),
           ("추입", "brk_win_cnt"), ("마크", "mrk_win_cnt")]


def _dict(row: sqlite3.Row) -> Dict:
    return {k: row[k] for k in row.keys()}


def _iso(ymd: Optional[str]) -> Optional[str]:
    """``"20260813"`` → ``"2026-08-13"``. 사이트맵의 lastmod 형식."""
    s = str(ymd or "")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 and s[:8].isdigit() else None


def fmt_date(ymd: Optional[str]) -> str:
    """``"20260813"`` → ``"2026-08-13 (목)"``"""
    if not ymd or len(str(ymd)) < 8:
        return ""
    s = str(ymd)
    try:
        d = dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return s
    return f"{d.isoformat()} ({WEEKDAY_KO[d.weekday()]})"


def _post_dt(row: Dict) -> Optional["dt.datetime"]:
    """경주일자 + 발주시각 → datetime. 하나라도 없으면 None.

    '아직 안 달린 경주'를 빌드 시점에 가려내는 데 쓴다. 화면에서 지난 경주를
    흐리게 하는 것은 **보는 사람의 시계**로 하지만(base.html), 어느 날을 첫
    화면에 걸지는 빌드 때 정해야 한다.
    """
    ymd, hhmm = row.get("race_ymd"), row.get("post_time")
    if not ymd or not hhmm:
        return None
    try:
        return dt.datetime.strptime(f"{str(ymd)[:8]} {str(hhmm)[:5]}", "%Y%m%d %H:%M")
    except ValueError:
        return None


def fmt_md(ymd: Optional[str]) -> str:
    """``"0523"`` 또는 ``"20260523"`` → ``"5.23"``. 최근 성적 칸의 날짜."""
    s = "".join(ch for ch in str(ymd or "") if ch.isdigit())
    if len(s) == 8:
        s = s[4:]
    if len(s) != 4:
        return ""
    return f"{int(s[:2])}.{int(s[2:])}"


# ---------------------------------------------------------------------------
# 조회
# ---------------------------------------------------------------------------

def load_races(conn: sqlite3.Connection, version: str) -> List[Dict]:
    rows = [_dict(r) for r in conn.execute(RACE_SQL, (version, version))]
    for r in rows:
        r["date_label"] = fmt_date(r["race_ymd"])
        r["version"] = version
    return rows


def load_runners(conn: sqlite3.Connection, race_key: str, version: str) -> List[Dict]:
    runners = [_dict(r) for r in conn.execute(RUNNER_SQL, (race_key, version))]
    if not runners:
        return []
    # 기호는 예측 순위에서 나온다. 규칙은 marks 한 곳에만 있다.
    assign_marks(runners)

    # 같은 훈련지가 둘 이상이면 라인 후보다. 화면에서 색으로 묶어 주기 위해
    # 그룹 번호를 여기서 매긴다 — 템플릿에서 세면 페이지마다 달라진다.
    groups: Dict[str, List[Dict]] = {}
    for r in runners:
        if r.get("trng_plc"):
            groups.setdefault(r["trng_plc"], []).append(r)
    gi = 0
    for plc, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        gi += 1
        for m in members:
            m["line_group"] = gi
            m["line_size"] = len(members)

    for r in runners:
        r["is_winner"] = (r.get("ord") == 1)
        # 전법 비율. 입상 이력이 없으면 성향을 말할 수 없으므로 비워 둔다 —
        # 0% 로 채우면 '전 전법 0%인 선수'가 되어 마크형과 구분되지 않는다.
        total = sum((r.get(c) or 0) for _, c in TACTICS)
        r["tactic_total"] = total
        r["tactics"] = [{
            "name": name,
            "cnt": r.get(col) or 0,
            "ratio": ((r.get(col) or 0) / total) if total else None,
        } for name, col in TACTICS]
        r["main_tactic"] = (max(r["tactics"], key=lambda t: t["cnt"])["name"]
                            if total else None)
        # 최근 3회전 착순. 앞이 오래된 쪽(3회전 전)이 되도록 뒤집는다.
        r["recent"] = []
        for i in (3, 2, 1):
            r["recent"].append({
                "meet": r.get(f"bf{i}_meet"),
                "date": fmt_md(r.get(f"bf{i}_ymd")),
                "ranks": [r.get(f"bf{i}_d{d}") for d in (1, 2, 3)],
            })
    return sorted(runners, key=lambda x: x["back_no"])


def load_payoffs(conn: sqlite3.Connection, race_key: str) -> List[Dict]:
    rows = [_dict(r) for r in conn.execute(
        "SELECT pool, combo, payout FROM payoffs WHERE race_key = ?", (race_key,))]
    order = {"단승": 0, "연승1": 1, "연승2": 2, "쌍승": 3, "복승": 4, "삼복승": 5}
    # 이름을 확인하지 못한 칸(pool7·pool8)은 화면에 올리지 않는다. 승식 이름
    # 없이 배당만 적으면 읽는 사람이 알아서 짐작하게 되고, 그 짐작이 틀린다.
    return sorted([r for r in rows if r["pool"] in order],
                  key=lambda r: order.get(r["pool"], 99))


def betting_combos(runners: List[Dict]) -> List[Dict]:
    """모델 확률을 승식 형식으로 옮겨 적는다.

    구매 권유가 아니라 확률의 다른 표기이며, 화면에도 그렇게 적는다. 회수율이
    100% 를 넘지 않는다는 사실을 같은 화면에 함께 둔다.
    """
    picks = [r["back_no"] for r in sorted(runners, key=lambda x: x["pred_rank"] or 99)]
    if len(picks) < 3:
        return []
    return [
        {"label": "단승", "value": f"{picks[0]}", "note": "1순위"},
        {"label": "복승", "value": f"{picks[0]}-{picks[1]}", "note": "1·2순위 (순서 무관)"},
        {"label": "쌍승", "value": f"{picks[0]}→{picks[1]}", "note": "1·2순위 (순서까지)"},
        {"label": "삼복승", "value": f"{picks[0]}-{picks[1]}-{picks[2]}",
         "note": "1~3순위 (순서 무관)"},
        {"label": "복승 박스", "value": "-".join(str(x) for x in sorted(picks[:3])),
         "note": "상위 3명에서 2명 — 3통"},
    ]


def load_simulation(conn: sqlite3.Connection, race_key: str,
                    runners: List[Dict]) -> Dict:
    """전개 시뮬레이션. 저장된 것이 있으면 그것을 쓴다.

    공개(v1) 예상은 발주 전에 시뮬레이션까지 함께 확정 저장한다. 검증(v1-oos)
    기록은 설명용이므로 빌드 때 계산한다 — 1만 3천 경주를 미리 돌려 저장할
    이유가 없고, 저장한다고 그 숫자가 더 참이 되지도 않는다.
    """
    row = conn.execute("SELECT payload FROM simulations WHERE race_key=?",
                       (race_key,)).fetchone()
    if row:
        try:
            sim = json.loads(row["payload"])
            sim["frozen"] = True
            return sim
        except ValueError:
            pass
    # 시뮬레이션은 정규화된 확률과 전법 비율을 입력으로 받는다.
    src = []
    for r in runners:
        d = dict(r)
        d["p_win"] = r.get("p_win")
        for t in r.get("tactics", []):
            key = {"선행": "pre_ratio", "젖히기": "pas_ratio",
                   "추입": "brk_ratio", "마크": "mrk_ratio"}[t["name"]]
            d[key] = t["ratio"]
        src.append(d)
    sim = run_simulation(src, n_sims=800)
    if sim:
        sim["frozen"] = False
    return sim


# 체크 포인트로 올릴 항목.
#   (표시 이름, 컬럼, 큰 값이 좋은가, 단위, 표본 컬럼, 최소 표본)
FOCUS_SPECS = [
    ("종합 평균득점 최고", "tot_avg_scr", True, "점", None, 0),
    ("광명 평균득점 최고", "area_avg_scr", True, "점", None, 0),
    ("연대율 최고", "high_rate", True, "%", "run_day_cnt", 10),
    ("200m 기록 최고", "rec_200m", False, "초", None, 0),
    ("입상률 최고", "win_ratio_calc", True, "%", "run_day_cnt", 10),
]


def focus_points(runners: List[Dict]) -> List[Dict]:
    """'무엇을 보고 골랐나' 를 항목별로 뒤집어 보여준다.

    같은 표를 선수 중심이 아니라 항목 중심으로 한 번 더 보여주면 훑어보기가
    빨라진다. 값이 비었거나 전원이 같으면 그 항목은 아무것도 말하지 않으므로
    내보내지 않는다. 표본이 얕은 값도 마찬가지다 — 출전 2회 100% 는 정보가
    아니라 잡음이다.
    """
    for r in runners:
        run = r.get("run_day_cnt") or 0
        r["win_ratio_calc"] = (100.0 * (r.get("win_tot_cnt") or 0) / run) if run else None

    out = []
    for label, col, bigger, unit, cnt_col, min_n in FOCUS_SPECS:
        vals = []
        for r in runners:
            v = r.get(col)
            if v is None:
                continue
            if cnt_col and (r.get(cnt_col) or 0) < min_n:
                continue
            vals.append((v, r))
        if len(vals) < 2:
            continue
        nums = [v for v, _ in vals]
        if max(nums) == min(nums):
            continue
        best = (max if bigger else min)(vals, key=lambda x: x[0])
        n = best[1].get(cnt_col) if cnt_col else None
        out.append({
            "label": label, "back_no": best[1]["back_no"],
            "racer_nm": best[1]["racer_nm"],
            "value": f"{best[0]:.4g}{unit}",
            "sample": f"{int(n)}회" if n else "",
        })
    return out


def race_picks(runners: List[Dict]) -> List[Dict]:
    """일곱 명 전부를 **배번 순**으로. 결과와 무관하게 항상 만든다.

    상위 몇 명만 적으면 나머지를 어떻게 보는지가 안 보이고, 순위 순으로 늘어
    놓으면 카드마다 배번 자리가 달라져 여러 경주를 훑을 때 눈이 매번 다시
    자리를 찾아야 한다. **배번 순으로 고정하면 칸이 줄 맞춰져** 1번끼리,
    7번끼리 세로로 비교된다. 순위는 기호(★◎○▲△)가 알려준다.
    """
    if not runners:
        return []
    return [{"back_no": r["back_no"], "racer": r["racer_nm"], "mark": r.get("mark"),
             "p_win": r.get("p_win"), "ord": r.get("ord"),
             "tactic": r.get("main_tactic"), "line": r.get("line_group")}
            for r in sorted(runners, key=lambda x: x["back_no"])]


def race_outcome(runners: List[Dict]) -> Dict:
    """경주가 끝났으면 우리 예상이 어땠는지 한 줄로 요약한다."""
    if not any(r.get("ord") for r in runners):
        return {}
    by_rank = sorted(runners, key=lambda x: x["pred_rank"] or 99)
    top1 = by_rank[0]
    order = {r["pred_rank"]: r.get("ord") for r in by_rank}
    top3 = {order.get(i) for i in (1, 2, 3)} - {None}
    return {
        "top1_back_no": top1["back_no"],
        "top1_racer": top1["racer_nm"],
        "top1_ord": top1.get("ord"),
        "hit_win": top1.get("ord") == 1,
        "hit_place": bool(top1.get("ord")) and top1["ord"] <= 2,
        "hit_quinella": {order.get(1), order.get(2)} == {1, 2},
        "hit_trio": top3 == {1, 2, 3},
    }


# 자금 곡선에 쓸 계열. 여섯 규칙을 다 그리면 선이 엉켜 아무것도 안 읽힌다.
# 질문이 향하는 셋만 남긴다 — 정액(기준선), 마틴게일(실패하면 두 배),
# 손실회수형(이번에 맞히면 다 덮는 크기). 나머지는 아래 표에 그대로 있다.
CURVE_SERIES = ["정액", "마틴게일(2배)", "손실회수형"]
CURVE_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"]

CHART_W, CHART_H = 760, 300
PAD_L, PAD_R, PAD_T, PAD_B = 62, 96, 16, 34


def build_curve_chart(stage: Dict, start_bankroll: float) -> Dict:
    """자금 곡선을 SVG 좌표로 바꾼다.

    빌드 타임에 좌표를 확정한다 — 브라우저에서 계산할 이유가 없고, 자바스크립트가
    막힌 환경에서도 그림은 보여야 한다.
    """
    runs = {r["name"]: r for r in stage["runs"]}
    series = [runs[n] for n in CURVE_SERIES if n in runs and runs[n].get("curve")]
    if not series:
        return {}

    max_x = max(max(p[0] for p in s["curve"]) for s in series)
    max_y = max(start_bankroll, max(max(p[1] for p in s["curve"]) for s in series))
    max_x = max(max_x, 1)

    def sx(x): return PAD_L + (x / max_x) * (CHART_W - PAD_L - PAD_R)
    def sy(y): return CHART_H - PAD_B - (y / max_y) * (CHART_H - PAD_T - PAD_B)

    out = []
    for i, s in enumerate(series):
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in s["curve"])
        last = s["curve"][-1]
        out.append({
            "name": s["name"], "color": CURVE_COLORS[i % len(CURVE_COLORS)],
            "points": pts, "end_x": sx(last[0]), "end_y": sy(last[1]),
            "n_bets": s["n_bets"], "ruined": s["ruined"], "curve": s["curve"],
        })

    # 끝점 라벨을 겹치지 않게 **위로** 쌓는다. 파산한 계열은 모두 0원에서
    # 끝나 같은 자리에 모이는데, 아래로 밀면 가로축 눈금 글씨와 부딪힌다.
    floor = CHART_H - PAD_B - 4
    used: List[float] = []
    for ser in sorted(out, key=lambda r: (r["end_x"], r["end_y"])):
        y = min(ser["end_y"] + 4, floor)
        while any(abs(y - u) < 13 for u in used):
            y -= 13
        ser["label_y"] = max(y, PAD_T + 10)
        used.append(ser["label_y"])

    xt = [0, max_x // 4, max_x // 2, (max_x * 3) // 4, max_x]
    yt = [0, max_y * 0.25, max_y * 0.5, max_y * 0.75, max_y]
    return {
        "w": CHART_W, "h": CHART_H, "series": out,
        "zero_y": sy(0), "start_y": sy(start_bankroll),
        "xticks": [{"x": sx(v), "label": f"{int(v):,}"} for v in xt],
        "yticks": [{"y": sy(v), "label": f"{int(v / 10000):,}만"} for v in yt],
        "max_x": max_x, "max_y": max_y,
        "plot": {"l": PAD_L, "r": CHART_W - PAD_R, "t": PAD_T, "b": CHART_H - PAD_B},
    }


def load_metrics(path: Path = Path("models/metrics.json")) -> Dict:
    """학습·검증 규모를 화면에 그대로 노출한다.

    적중률만 크게 적고 표본 수와 베이스라인을 빼면 스스로를 속이게 된다.
    경륜에서는 특히 그렇다 — 득점 최고를 그냥 찍어도 60% 가 나오는 종목이라,
    '적중률 60%' 라는 문장만으로는 모델이 한 일이 없다는 사실을 감출 수 있다.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    wf = raw.get("walk_forward") or {}
    out = {
        "trained_races": raw.get("trained_races"),
        "trained_rows": raw.get("trained_rows"),
        "n_features": raw.get("n_features"),
        "date_max": raw.get("date_max"),
        "verified_races": wf.get("n_races"),
        "hit_win": wf.get("top1_win"),
        "hit_place": wf.get("top1_top2"),
        "roi_win": wf.get("roi_win"),
        "score_win": wf.get("score_win"),
        "score_roi": wf.get("score_roi"),
        "back1_win": wf.get("back1_win"),
        "auc": wf.get("auc"),
        "logloss": wf.get("logloss"),
    }
    if out.get("hit_win") is not None and out.get("score_win") is not None:
        out["edge_win"] = out["hit_win"] - out["score_win"]
    return out


# ---------------------------------------------------------------------------
# 빌드
# ---------------------------------------------------------------------------

def asset_versions(static_dir: Path = STATIC_DIR) -> Dict[str, str]:
    """정적 파일의 내용 해시.

    ``style.css?v=ab12cd34`` 처럼 붙여 브라우저가 옛 파일을 계속 쓰지 못하게
    한다. 이게 없으면 CSS 를 고쳐도 화면이 안 바뀌고, 고치는 쪽은 코드가
    틀렸다고 착각하며 엉뚱한 곳을 파게 된다.
    """
    import hashlib

    out: Dict[str, str] = {}
    if not static_dir.exists():
        return out
    for f in static_dir.iterdir():
        if f.is_file():
            out[f.name] = hashlib.sha1(f.read_bytes()).hexdigest()[:8]
    return out


def build(db: str, out: Path, cfg: Dict) -> None:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True, lstrip_blocks=True,
    )
    env.filters["pct"] = lambda v, d=1: "—" if v is None else f"{v * 100:.{d}f}%"
    env.filters["num"] = lambda v, d=1: "—" if v is None else f"{v:,.{d}f}"
    env.filters["date_ko"] = fmt_date
    env.filters["josa"] = josa
    env.filters["jo"] = josa_of

    out.mkdir(parents=True, exist_ok=True)
    if STATIC_DIR.exists():
        shutil.copytree(STATIC_DIR, out / "static", dirs_exist_ok=True)
    # GitHub Pages 는 기본으로 Jekyll 을 돌린다. 이 파일이 없으면 밑줄로
    # 시작하는 이름이 조용히 무시된다.
    (out / ".nojekyll").write_text("", encoding="utf-8")
    # 커스텀 도메인은 **배포 산출물 안에** CNAME 이 있어야 유지된다.
    #
    # 다만 github.io 주소는 커스텀 도메인이 아니다. 그걸 CNAME 에 적으면 Pages 가
    # 자기 자신을 커스텀 도메인으로 잡아 무한 리다이렉트에 빠진다 — 프로젝트
    # 페이지로 먼저 올려 두고 도메인은 나중에 붙이는 경우가 정확히 이 상황이다.
    host = urlparse(cfg.get("site", {}).get("url") or "").hostname
    if host and not host.endswith(".github.io"):
        (out / "CNAME").write_text(host + "\n", encoding="utf-8")
    # ads.txt 가 없으면 애드센스가 '승인되지 않은 판매자'로 보고 게재를 막는다.
    client = (cfg.get("adsense", {}) or {}).get("client") or ""
    if client.startswith("ca-"):
        (out / "ads.txt").write_text(
            f"google.com, {client[3:]}, DIRECT, f08c47fec0942fa0\n", encoding="utf-8")

    bcfg = cfg.get("build", {})
    # 배포 기준 경로. 환경변수가 설정을 이긴다 — 같은 소스로 로컬(루트)과
    # 배포(/저장소명)를 모두 구울 수 있어야 한다.
    base = (os.environ.get("CYCLEAI_BASE")
            or cfg.get("site", {}).get("base_path") or "").rstrip("/")
    ctx_base = {
        "site": cfg.get("site", {}),
        "built_at": now_kst().strftime("%Y-%m-%d %H:%M"),
        "mark_meaning": MARK_MEANING,
        "star_threshold": MARK_THRESHOLDS["star"],
        "version_label": VERSION_LABEL,
        "bet_order": BET_ORDER,
        "tactic_desc": TACTIC_DESC,
        "assets": asset_versions(),
        "base": base,
        "adsense": cfg.get("adsense", {}) or {},
        "analytics": cfg.get("analytics", {}) or {},
        "verification": cfg.get("verification", {}) or {},
    }

    with session(db) as conn:
        metrics = load_metrics()
        live = load_races(conn, LIVE_VERSION)
        oos = load_races(conn, OOS_VERSION)

        today = today_kst().strftime("%Y%m%d")
        # ── 첫 화면에 실을 '그날 전 경주' ────────────────────────
        #
        # **끝난 경주를 목록에서 빼지 않는다.** 결과가 들어왔다고 그 경주만
        # 빠지면 그날 시간표가 앞뒤로 갈려, 1R 은 사라지고 5R 부터 보이는
        # 이상한 목록이 된다. 무엇을 밀었는지도 함께 사라진다.
        live_sorted = sorted(live, key=lambda r: (r["race_ymd"] or "",
                                                  r["race_no"] or 0))
        # 아직 착순이 안 들어온 개최일들. 여기에 **하루 이상이 겹친다.**
        pending_days = sorted({r["race_ymd"] for r in live_sorted
                               if r["race_ymd"] and not r["has_result"]})

        # 그중 **아직 발주하지 않은 경주가 남은** 가장 이른 날을 첫 화면에 건다.
        #
        # 경정에서는 'has_result 가 없는 가장 이른 날' 로 골라도 됐다. 수·목
        # 편성이라 다음 개최일이 오기 전에 결과가 들어왔기 때문이다. **경륜은
        # 금·토·일 연속 편성이라 그 가정이 깨진다** — 금요일 결과는 토요일
        # 아침에야 올라오는데, 그때는 토요일 경주도 미결과라 min() 이 이미 다
        # 끝난 금요일을 고른다. 실제로 8/14 저녁에 첫 화면이 8/14 를 계속
        # 보여주고 8/15 예상은 어디에도 나오지 않았다.
        future = [d for d in pending_days
                  if any(_post_dt(r) and _post_dt(r) > now_kst()
                         for r in live_sorted if r["race_ymd"] == d)]
        if future:
            target_day = future[0]
        elif pending_days:
            # 남은 경주가 없으면 가장 **최근** 개최일. 여기서 min 을 쓰면
            # 결과가 늦게 들어오는 옛날 하루에 첫 화면이 붙들린다.
            target_day = pending_days[-1]
        else:
            days_all = [r["race_ymd"] for r in live_sorted if r["race_ymd"]]
            target_day = max(days_all) if days_all else None

        # 예상이 있는 경주만 뽑으면 편성에 구멍이 난다. 그날 전체를 다시 읽는다.
        def _day_races(ymd: str) -> List[Dict]:
            rows = []
            for row in conn.execute(DAY_SQL, (LIVE_VERSION, ymd)):
                r = _dict(row)
                r["date_label"] = fmt_date(r["race_ymd"])
                r["version"] = LIVE_VERSION
                rows.append(r)
            return rows

        upcoming = _day_races(target_day) if target_day else []
        # 첫 화면에 걸지 않은 미결과 개최일도 페이지와 칩은 만들어 둔다.
        # 그러지 않으면 오늘 경주가 끝난 순간, 결과가 들어오기 전까지 그날
        # 예상이 사이트에서 통째로 사라진다 — 발주 전에 확정 저장했다는 기록이
        # 정작 가장 볼 만한 시점에 없어지는 셈이다.
        other_pending = [r for d in pending_days if d != target_day
                         for r in _day_races(d)]

        # 결과 아카이브는 미결과 개최일을 빼고 쌓는다 — 같은 경주가 두 번
        # 나오지 않게.
        shown = set(pending_days)
        finished = [r for r in live if r["has_result"] and r["race_ymd"] not in shown]
        finished += [r for r in oos if r["has_result"] and r["race_ymd"] not in shown]
        finished = finished[: bcfg.get("results_limit", 300)]

        # ── 경주 상세 ────────────────────────────────────────────
        detail_targets = (upcoming + other_pending
                          + finished[: bcfg.get("past_races", 400)])
        seen = set()
        race_pages: List[Dict] = []
        for r in detail_targets:
            if r["race_key"] in seen:
                continue
            seen.add(r["race_key"])
            runners = load_runners(conn, r["race_key"], r["version"])
            if not runners:
                continue
            page = dict(r)
            page.update({
                "runners": runners,
                "combos": betting_combos(runners),
                "payoffs": load_payoffs(conn, r["race_key"]),
                "picks": race_picks(runners),
                "outcome": race_outcome(runners),
                "sim": load_simulation(conn, r["race_key"], runners),
                "focus": focus_points(runners),
            })
            race_pages.append(page)
            _write(out / "race" / r["race_key"] / "index.html",
                   env.get_template("race.html").render(
                       race=page, page_url=f"/race/{r['race_key']}/", **ctx_base))

        by_key = {p["race_key"]: p for p in race_pages}
        for lst in (upcoming, other_pending, finished):
            for r in lst:
                p = by_key.get(r["race_key"])
                r["picks"] = p.get("picks") if p else []
                r["outcome"] = p.get("outcome") if p else {}
                r["has_page"] = p is not None

        # ── 경주일 ───────────────────────────────────────────────
        days: Dict[str, List[Dict]] = {}
        for r in upcoming + other_pending + finished:
            if r["race_ymd"]:
                days.setdefault(r["race_ymd"], []).append(r)
        for ymd, rows in days.items():
            rows = sorted(rows, key=lambda x: x["race_no"] or 0)
            _write(out / "day" / ymd / "index.html",
                   env.get_template("day.html").render(
                       ymd=ymd, date_label=fmt_date(ymd), races=rows,
                       page_url=f"/day/{ymd}/", **ctx_base))

        # ── 검증 ─────────────────────────────────────────────────
        reports = {v: build_report(conn, v) for v in (LIVE_VERSION, OOS_VERSION)}

        # 고배당 카드에서 상세 페이지로 갈 수 있는 것만 링크한다. 오래된
        # 경주는 상세를 굽지 않으므로(build.past_races) 링크가 깨진다.
        have_page = {p["race_key"] for p in race_pages}
        for rep in reports.values():
            for group in (rep.get("highlights") or {}).values():
                for h in group:
                    h["has_page"] = h["race_key"] in have_page

        # ── 베팅 전략 ────────────────────────────────────────────
        strat = strategy_report(conn, OOS_VERSION)
        if not strat.get("empty"):
            for stage in strat["stages"]:
                stage["chart"] = build_curve_chart(stage, strat["start_bankroll"])

    _write(out / "strategy" / "index.html", env.get_template("strategy.html").render(
        s=strat, page_url="/strategy/", **ctx_base))

    _write(out / "index.html", env.get_template("index.html").render(
        upcoming=upcoming, finished=finished[:20], metrics=metrics,
        today=today, day_list=sorted(days, reverse=True)[:12],
        highlights=(reports.get(OOS_VERSION) or {}).get("highlights"),
        reports=reports, page_url="/", **ctx_base))

    _write(out / "results" / "index.html", env.get_template("results.html").render(
        races=finished, page_url="/results/", **ctx_base))

    _write(out / "accuracy" / "index.html", env.get_template("accuracy.html").render(
        reports=reports, metrics=metrics, min_sample=bcfg.get("min_sample", 30),
        page_url="/accuracy/", **ctx_base))

    # ── 검색엔진용 ───────────────────────────────────────────────
    site_url = cfg.get("site", {}).get("url") or ""
    if site_url:
        today_iso = today_kst().isoformat()
        urls = [{"path": "/", "freq": "daily", "priority": "1.0", "lastmod": today_iso},
                {"path": "/accuracy/", "freq": "weekly", "priority": "0.9",
                 "lastmod": today_iso},
                {"path": "/strategy/", "freq": "monthly", "priority": "0.7",
                 "lastmod": today_iso},
                {"path": "/results/", "freq": "daily", "priority": "0.8",
                 "lastmod": today_iso}]
        for ymd in sorted(days, reverse=True):
            urls.append({"path": f"/day/{ymd}/", "freq": "monthly", "priority": "0.6",
                         "lastmod": _iso(ymd) or today_iso})
        for p_ in race_pages:
            urls.append({"path": f"/race/{p_['race_key']}/", "freq": "monthly",
                         "priority": "0.7",
                         "lastmod": _iso(p_.get("race_ymd")) or today_iso})
        _write(out / "sitemap.xml", env.get_template("sitemap.xml").render(
            urls=urls, site=cfg.get("site", {}), base=base))
        (out / "robots.txt").write_text(
            "User-agent: *\nAllow: /\n\n"
            f"Sitemap: {site_url}{base}/sitemap.xml\n", encoding="utf-8")
        log.info("사이트맵 %d개 주소", len(urls))

    log.info("경주 상세 %d개 · 경주일 %d개 · 다가올 %d경주 · 결과 %d경주",
             len(race_pages), len(days), len(upcoming), len(finished))
    log.info("빌드 완료 → %s", out)


def _write(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="정적 사이트 생성")
    ap.add_argument("--db", default="data/cycleai.sqlite")
    ap.add_argument("--out", default="dist")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    except OSError:
        cfg = {}
    build(args.db, Path(args.out), cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
