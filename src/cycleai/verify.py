"""공개(기록)했던 예측의 실제 적중률 검증 — 승식 판정.

백테스트가 아니라 **저장된 예측**을 결과와 대조한다. 백테스트 숫자는 얼마든지
예쁘게 만들 수 있으므로, 믿을 지표는 이쪽이어야 한다.

**배당을 받을 수 있는 승식만 판정한다** (단승·연승·쌍승·복승·삼복승). 경주결과
API 에는 이름 없는 칸이 둘 더 있지만(pool7·pool8) 어느 승식인지 확인할 수 없어
집계에 넣지 않았다 — 이름을 잘못 붙인 회수율은 틀렸다는 것조차 알 수 없다.

**배당표가 없는 경주는 판정하지 않는다.** 배당은 적중 조합만 저장되므로 '표에
없으면 불발'인데, 자료 자체가 안 들어온 경주까지 그렇게 세면 맞힌 경주가
불발로 남는다. 자료가 올 때까지 집계에서 빼는 것이 맞다.

    python -m cycleai.verify --db data/cycleai.sqlite
    python -m cycleai.verify --version v1-oos      # 사후 워크포워드 기록
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .kcycle.store import session

log = logging.getLogger(__name__)

# 표본이 이보다 얕으면 숫자가 잡음이다. 구간을 나눌수록 더 그렇다.
MIN_GROUP_RACES = 30

# 배당을 받을 수 있는 승식만. 순서가 있는 승식은 조합을 정렬하지 않는다.
BET_ORDER = ["단승", "연승", "쌍승", "복승", "삼복승"]

VERIFY_SQL = """
SELECT
    p.race_key, p.back_no, p.racer_nm, p.pred_rank, p.p_win, p.p_top2, p.p_top3,
    p.model_version, p.created_at,
    r.race_ymd, r.stnd_yr, r.week_tcnt, r.day_tcnt, r.race_no,
    r.race_grade, r.field_size,
    res.ord,
    e.racer_grd, e.grade_cur, e.trng_plc
FROM predictions p
JOIN races r          ON r.race_key = p.race_key
LEFT JOIN results res ON res.race_key = p.race_key AND res.racer_nm = p.racer_nm
LEFT JOIN entries e   ON e.race_key = p.race_key AND e.back_no = p.back_no
WHERE p.model_version = ? AND COALESCE(r.has_result, 0) = 1
"""


def load_payoffs(conn: sqlite3.Connection) -> Dict[str, Dict[str, Dict]]:
    """{race_key: {pool: {"combo": ..., "payout": ...}}}"""
    out: Dict[str, Dict[str, Dict]] = {}
    for r in conn.execute("SELECT race_key, pool, combo, payout FROM payoffs"):
        out.setdefault(r["race_key"], {})[r["pool"]] = {
            "combo": r["combo"], "payout": r["payout"]}
    return out


def _settle(picks: List[int], table: Dict[str, Dict],
            actual: Optional[List[int]] = None) -> Dict[str, Dict]:
    """추천 배번(순위순)과 그 경주의 배당표로 승식별 손익을 낸다.

    배당표에는 **적중 조합만** 들어 있다. 우리 조합이 그 조합과 같으면 적중이고
    배당이 곧 회수액이다(1배 = 원금). 매수(cost)는 몇 통을 샀는지이며,
    회수율의 분모가 된다 — 박스로 넓게 사면 원금도 함께 늘어나야 공정하다.

    **배당률 API 는 조합을 주지 않는다** (배당값만 온다). 그래서 조합을 모르는
    승식은 실제 착순(``actual``)으로 판정한다 — 배당은 배당표에서, 적중 여부는
    착순에서 온다. 둘 다 사실이므로 섞어 써도 된다.
    """
    if not picks:
        return {}
    out: Dict[str, Dict] = {}
    act = [int(x) for x in (actual or [])]

    def _won_set(pool: str, size: int, ordered: bool) -> Optional[List[str]]:
        """그 경주의 적중 조합. 배당표에 없으면 착순에서 만든다."""
        info = table.get(pool) or {}
        if info.get("combo"):
            return [info["combo"]]
        if len(act) >= size:
            c = act[:size]
            return ["-".join(str(x) for x in (c if ordered else sorted(c)))]
        return None

    def hit(pool: str, mine: List[str], *, ordered: bool, size: int) -> None:
        info = table.get(pool)
        if not info or info.get("payout") is None:
            return  # 배당 자료 없음 — 판정하지 않는다
        won = _won_set(pool, size, ordered)
        payout = 0.0
        if won:
            keys = {w if ordered else "-".join(sorted(w.split("-"))) for w in won}
            for c in mine:
                cc = c if ordered else "-".join(sorted(c.split("-")))
                if cc in keys:
                    payout = float(info["payout"])
                    break
        out[pool] = {"cost": len(mine), "payout": payout, "hit": payout > 0}

    j = lambda *xs: "-".join(str(x) for x in xs)  # noqa: E731

    hit("단승", [j(picks[0])], ordered=True, size=1)

    # 연승은 1착·2착 두 자리에 각각 다른 배당이 붙는다(연승1·연승2). 우리 픽이
    # 둘 중 어디에 들어왔느냐로 회수액이 갈리므로, **그 자리의 배당이 있어야만**
    # 판정한다. 한쪽이 비었는데 다른 쪽 배당으로 대신 세면 회수율이 조용히
    # 틀어진다 — 1착 배당은 2착 배당보다 낮으므로 한 방향으로만 틀린다.
    place = {1: table.get("연승1"), 2: table.get("연승2")}
    have = {k: v for k, v in place.items() if v and v.get("payout") is not None}
    if have:
        won2 = act[:2]
        if picks[0] in won2:
            slot = won2.index(picks[0]) + 1        # 1착이면 1, 2착이면 2
            info = have.get(slot)
            if info:
                out["연승"] = {"cost": 1, "payout": float(info["payout"]), "hit": True}
        else:
            # 빗나간 경우는 어느 자리 배당인지 따질 필요가 없다.
            out["연승"] = {"cost": 1, "payout": 0.0, "hit": False}

    if len(picks) >= 2:
        hit("쌍승", [j(picks[0], picks[1])], ordered=True, size=2)   # 순서까지
        hit("복승", [j(picks[0], picks[1])], ordered=False, size=2)  # 순서 무관
    if len(picks) >= 3:
        hit("삼복승", [j(*picks[:3])], ordered=False, size=3)
    return out


def _settle_box(picks: List[int], table: Dict[str, Dict], n: int, pool: str,
                size: int, actual: Optional[List[int]] = None) -> Optional[Dict]:
    """박스 매수. 상위 n명에서 size 개를 고르는 모든 조합을 산다."""
    info = table.get(pool)
    if not info or info.get("payout") is None or len(picks) < n:
        return None
    mine = {"-".join(str(x) for x in sorted(c)) for c in combinations(picks[:n], size)}
    won = info.get("combo")
    if won:
        key = "-".join(sorted(won.split("-")))
    elif actual and len(actual) >= size:
        key = "-".join(str(x) for x in sorted(actual[:size]))
    else:
        return None
    payout = float(info["payout"]) if key in mine else 0.0
    return {"cost": len(mine), "payout": payout, "hit": payout > 0}


def load_verified(conn: sqlite3.Connection, version: str) -> pd.DataFrame:
    df = pd.read_sql_query(VERIFY_SQL, conn, params=[version])
    if df.empty:
        return df
    for c in ("ord", "pred_rank", "p_win", "back_no", "stnd_yr", "race_no"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["race_date"] = pd.to_datetime(df["race_ymd"], format="%Y%m%d", errors="coerce")
    # 1착이 정확히 한 명으로 확인되는 경주만 (실격·낙차 제외)
    ok = df.groupby("race_key")["ord"].transform(lambda s: (s == 1).sum() == 1)
    return df[ok].copy()


def race_level(df: pd.DataFrame, payoffs: Dict) -> pd.DataFrame:
    """경주 단위 적중 여부 테이블."""
    if df.empty:
        return df
    rows = []
    for key, g in df.groupby("race_key"):
        g = g.sort_values("pred_rank")
        if (g["pred_rank"] == 1).sum() != 1:
            continue
        table = payoffs.get(key)
        if not table:
            # 배당 자료가 없는 경주는 집계에서 뺀다. 여기서 0원으로 세면
            # 맞힌 경주가 불발로 남는다.
            continue
        picks = [int(v) for v in g["back_no"].tolist() if pd.notna(v)]
        actual = [int(r.back_no) for r in
                  g.dropna(subset=["ord"]).sort_values("ord").itertuples()
                  if pd.notna(r.back_no)]
        bets = _settle(picks, table, actual)
        for label, (pool, n, size) in {
            "복승박스3": ("복승", 3, 2),
            "삼복승박스4": ("삼복승", 4, 3),
        }.items():
            b = _settle_box(picks, table, n, pool, size, actual)
            if b:
                bets[label] = b

        t1 = g[g["pred_rank"] == 1].iloc[0]
        o = {int(r.pred_rank): (r.ord if pd.notna(r.ord) else None)
             for r in g.itertuples()}
        top_n = lambda n: {v for k, v in o.items() if k <= n and v}  # noqa: E731

        rows.append({
            "race_key": key, "bets": bets,
            "race_date": g["race_date"].iloc[0],
            "stnd_yr": g["stnd_yr"].iloc[0],
            "race_no": g["race_no"].iloc[0],
            "race_grade": g["race_grade"].iloc[0],
            "top1_back_no": int(t1["back_no"]) if pd.notna(t1["back_no"]) else None,
            "top1_racer": t1["racer_nm"],
            "top1_grd": t1["grade_cur"],
            "top1_ord": o.get(1),
            "hit_win": float(o.get(1) == 1),
            "hit_place": float(bool(o.get(1)) and o.get(1) <= 2),
            "hit_top3_has_winner": float(1.0 in top_n(3)),
            "payout_win": (float(bets.get("단승", {}).get("payout") or 0.0)),
        })
    return pd.DataFrame(rows)


def ci95(returns: pd.Series) -> Optional[float]:
    """회수율의 95% 신뢰구간 반폭.

    **회수율은 표본이 얕을 때 100% 를 우습게 넘나든다.** 공개 기록 155경주에서
    단승 회수율이 106.1% 로 나왔는데, 오차가 ±16.4%p 라 구간이 [89.7%, 122.5%]
    였다 — 100% 를 품는다. 같은 지표가 18,968경주 검증에서는 92.9% 였다.

    숫자만 적으면 읽는 쪽은 '수익이 난다'로 읽는다. ± 를 옆에 두면 그 문장이
    아직 성립하지 않는다는 것이 같은 줄에서 보인다. 절단 회수율이 '꼬리에
    매달렸는가'를 잡는다면, 이 값은 '표본이 말할 수 있는가'를 잡는다.
    """
    s = pd.to_numeric(returns, errors="coerce").dropna()
    if len(s) < 2:
        return None
    return float(1.96 * s.std(ddof=1) / (len(s) ** 0.5))


def summarize(rl: pd.DataFrame) -> Dict:
    if rl.empty:
        return {"n_races": 0}
    return {
        "n_races": int(len(rl)),
        "hit_win": float(rl["hit_win"].mean()),
        "hit_place": float(rl["hit_place"].mean()),
        "hit_top3_has_winner": float(rl["hit_top3_has_winner"].mean()),
        "roi_win": float(rl["payout_win"].sum() / len(rl)),
        "roi_win_ci": ci95(rl["payout_win"]),
        "avg_win_payout": (float(rl.loc[rl["hit_win"] == 1, "payout_win"].mean())
                           if (rl["hit_win"] == 1).any() else None),
        "first_date": str(rl["race_date"].min())[:10],
        "last_date": str(rl["race_date"].max())[:10],
    }


def bet_summary(rl: pd.DataFrame) -> List[Dict]:
    """승식별 누적 적중률과 회수율.

    회수율은 '그 방식으로 매 경주 균등하게 샀다면 얼마가 돌아왔나'다. 박스는
    매수가 늘어난 만큼 원금도 늘어나므로 분모에 그대로 반영된다 — 그래야 넓게
    사는 방식이 유리해 보이는 착시가 없다.

    **절단 회수율(roi_trim)을 함께 낸다.** 삼복승처럼 조합이 큰 승식은 회수율이
    소수의 고배당에 통째로 매달린다 — 실측에서 삼복승 회수율은 135% 였지만
    회수액 상위 1% 를 잘라내면 82% 로 떨어졌다. 앞의 숫자만 적으면 '이 승식은
    수익이 난다'로 읽히는데, 그것은 800배짜리 한 판을 맞힌 적이 있다는 뜻이지
    다음에도 그럴 것이라는 뜻이 아니다. 둘을 나란히 두면 그 차이가 곧 경고가 된다.
    """
    if rl.empty or "bets" not in rl:
        return []
    agg: Dict[str, Dict[str, float]] = {}
    rets: Dict[str, List[float]] = {}
    pers: Dict[str, List[float]] = {}
    for bets in rl["bets"]:
        for name, b in (bets or {}).items():
            a = agg.setdefault(name, {"n": 0, "hit": 0, "cost": 0.0, "payout": 0.0})
            a["n"] += 1
            a["hit"] += int(b["hit"])
            a["cost"] += b["cost"]
            a["payout"] += b["payout"]
            rets.setdefault(name, []).append(float(b["payout"]))
            # 박스는 통 수가 다르므로 **통당** 회수액으로 모아야 구간이 맞는다.
            pers.setdefault(name, []).append(float(b["payout"]) / max(b["cost"], 1))

    def _trim(name: str, cost: float) -> Optional[float]:
        """회수액 상위 1% 를 그 분위값으로 눌러 다시 계산한 회수율."""
        vals = rets.get(name) or []
        if len(vals) < 100 or not cost:
            return None
        arr = pd.Series(vals)
        cap = float(arr.quantile(0.99))
        return float(arr.clip(upper=cap).sum() / cost)

    out, tot = [], {"n": 0, "hit": 0, "cost": 0.0, "payout": 0.0}
    for name in BET_ORDER + ["복승박스3", "삼복승박스4"]:
        a = agg.get(name)
        if not a or not a["n"]:
            continue
        # 통합 회수율에는 기본 다섯 승식만 넣는다. 박스는 승식이 아니라
        # 매수 방식이라, 섞으면 '박스가 불리하다'는 잘못된 결론으로 읽힌다.
        if name in BET_ORDER:
            for k in tot:
                tot[k] += a[k]
        out.append({
            "name": name, "n_races": int(a["n"]),
            "tickets": round(a["cost"] / max(1, a["n"]), 1),
            "hit_rate": a["hit"] / a["n"],
            "roi": a["payout"] / a["cost"] if a["cost"] else None,
            "roi_trim": _trim(name, a["cost"]),
            "roi_ci": ci95(pd.Series(pers.get(name, []))),
        })
    if tot["cost"]:
        races = max((r["n_races"] for r in out if r["name"] in BET_ORDER), default=0)
        all_rets = [v for name in BET_ORDER for v in rets.get(name, [])]
        cap = float(pd.Series(all_rets).quantile(0.99)) if len(all_rets) >= 100 else None
        out.append({
            "name": "다섯 승식 통합", "n_races": races,
            "tickets": round(tot["cost"] / races, 1) if races else 0,
            "hit_rate": tot["hit"] / tot["n"], "roi": tot["payout"] / tot["cost"],
            "roi_trim": (float(pd.Series(all_rets).clip(upper=cap).sum() / tot["cost"])
                         if cap is not None else None),
            "roi_ci": ci95(pd.Series([v for n in BET_ORDER for v in pers.get(n, [])])),
            "is_total": True,
        })
    return out


def breakdown(rl: pd.DataFrame, col: str, label: str,
              min_races: int = MIN_GROUP_RACES) -> List[Dict]:
    """축 하나로 갈라 집계한다. 표본이 얕은 구간은 내보내지 않는다."""
    if rl.empty or col not in rl:
        return []
    out = []
    for name, g in rl.groupby(col):
        if not str(name) or str(name) == "nan" or len(g) < min_races:
            continue
        row = summarize(g)
        row[label] = str(name)
        out.append(row)
    return out


# 성과 카드에 올릴 최소 배당. 이보다 낮으면 '고배당'이라 부르기 어렵다.
HIGHLIGHT_MIN_ODDS = 10.0


def highlights(rl: pd.DataFrame, limit: int = 6) -> Dict[str, List[Dict]]:
    """고배당 적중 기록.

    적중률·회수율 표는 성실하지만 눈에 걸리지 않는다. 처음 온 사람이 이 사이트를
    한 번 더 볼 이유는 '삼복승 47배가 맞았다' 같은 구체적인 장면이다. 표에 이미
    들어 있는 사실을 앞으로 꺼내는 것이므로 없는 말을 지어내지 않는다.

    **한 경주에서 여러 승식이 맞아도 가장 큰 것 하나만 남긴다.** 같은 경주가
    카드로 세 번 나오면 성과가 여럿인 것처럼 보여 오히려 신뢰를 깎는다.
    """
    if rl.empty or "bets" not in rl:
        return {"top": [], "recent": []}
    best: Dict[str, Dict] = {}
    for r in rl.itertuples():
        n_hit = sum(1 for k in BET_ORDER if (r.bets or {}).get(k, {}).get("hit"))
        for name in BET_ORDER:
            b = (r.bets or {}).get(name)
            if not b or not b["hit"]:
                continue
            odds = b["payout"] / max(1, b["cost"])
            if odds < HIGHLIGHT_MIN_ODDS:
                continue
            cur = best.get(r.race_key)
            if cur and cur["odds"] >= odds:
                continue
            best[r.race_key] = {
                "race_key": r.race_key, "date": str(r.race_date)[:10],
                "race_no": int(r.race_no) if pd.notna(r.race_no) else None,
                "bet": name, "odds": round(odds, 1),
                "back_no": r.top1_back_no, "racer": r.top1_racer,
                "n_hit": n_hit,
            }
    rows = list(best.values())
    top = sorted(rows, key=lambda h: -h["odds"])[:limit]
    recent = sorted(rows, key=lambda h: (h["date"], h["odds"]), reverse=True)[:limit]
    # 최근 목록에 있는 것을 역대에 또 걸면 같은 카드가 두 번 나온다.
    seen = {h["race_key"] for h in recent}
    return {"top": [h for h in top if h["race_key"] not in seen], "recent": recent}


def build_report(conn: sqlite3.Connection, version: str) -> Dict:
    df = load_verified(conn, version)
    rl = race_level(df, load_payoffs(conn))
    if rl.empty:
        return {"version": version, "overall": {"n_races": 0}, "by_bet": [],
                "monthly": [], "by_back_no": [], "by_race_grade": [],
                "by_grade": [], "recent": [], "highlights": {"top": [], "recent": []}}

    rl = rl.sort_values("race_date")
    monthly = []
    for period, g in rl.groupby(rl["race_date"].dt.to_period("M")):
        s = summarize(g)
        s["month"] = str(period)
        monthly.append(s)

    recent = rl.sort_values(["race_date", "race_no"], ascending=[False, False]).head(60)
    recent_rows = [{
        "race_key": r.race_key, "date": str(r.race_date)[:10],
        "race_no": int(r.race_no) if pd.notna(r.race_no) else None,
        "pick_back_no": r.top1_back_no, "pick": r.top1_racer,
        "ord": int(r.top1_ord) if pd.notna(r.top1_ord) else None,
        "hit_bets": [k for k in BET_ORDER if (r.bets or {}).get(k, {}).get("hit")],
    } for r in recent.itertuples()]

    return {
        "version": version,
        "overall": summarize(rl),
        "by_bet": bet_summary(rl),
        "highlights": highlights(rl),
        "monthly": monthly[-24:],
        "by_back_no": breakdown(rl, "top1_back_no", "back_no"),
        "by_race_grade": breakdown(rl, "race_grade", "race_grade"),
        "by_grade": breakdown(rl, "top1_grd", "grade"),
        "recent": recent_rows,
    }


def report_text(rep: Dict) -> str:
    o = rep.get("overall", {})
    ver = rep.get("version", "")
    kind = "시간순 교차검증(과거 재현)" if ver.endswith("-oos") else "발주 전 확정 저장(공개)"
    if not o.get("n_races"):
        return (f"[{ver}] 검증 가능한 예측이 없습니다.\n"
                "  (예측을 만들고 경주 결과·배당이 들어와야 집계됩니다)")

    lines = [
        f"예측 검증  [{ver}] {kind}",
        f"  {o['first_date']} ~ {o['last_date']}   총 {o['n_races']:,}경주",
        "-" * 62,
        f"  1순위 1착         {o['hit_win']:6.1%}",
        f"  1순위 2착 이내    {o['hit_place']:6.1%}",
        f"  상위 3명 안에 1착 {o['hit_top3_has_winner']:6.1%}",
    ]
    if o.get("avg_win_payout"):
        lines.append(f"  적중 시 평균 단승배당 {o['avg_win_payout']:.1f}배")

    if rep.get("by_bet"):
        lines += ["", f"  {'승식':<14}{'경주':>7}{'매수':>6}{'적중률':>9}{'회수율':>9}"
                      f"{'절단':>9}", "  " + "-" * 54]
        for b in rep["by_bet"]:
            roi = f"{b['roi']:.1%}" if b.get("roi") is not None else "  n/a"
            trim = f"{b['roi_trim']:.1%}" if b.get("roi_trim") is not None else "  n/a"
            mark = "▸ " if b.get("is_total") else "  "
            lines.append(f"  {mark}{b['name']:<12}{b['n_races']:>7,}{b['tickets']:>6}"
                         f"{b['hit_rate']:>9.1%}{roi:>9}{trim:>9}")
        lines.append("  * 절단 = 회수액 상위 1% 를 눌러 다시 계산한 회수율. "
                     "둘의 차이가 크면 그 회수율은 소수의 고배당에 매달려 있다.")
    if rep.get("by_back_no"):
        lines += ["", "  1순위로 고른 배번별 성적", "  " + "-" * 45]
        for r in sorted(rep["by_back_no"], key=lambda x: x["back_no"]):
            lines.append(f"    {r['back_no']}번  {r['n_races']:>6,}경주  "
                         f"1착 {r['hit_win']:5.1%}  회수율 {r['roi_win']:6.1%}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="예측 적중률 검증")
    ap.add_argument("--db", default="data/cycleai.sqlite")
    ap.add_argument("--version", default="v1", help="model_version (사후 기록은 v1-oos)")
    ap.add_argument("--out", default="data/accuracy.json")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with session(args.db) as conn:
        rep = build_report(conn, args.version)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(report_text(rep))
    print(f"\n검증 리포트 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
