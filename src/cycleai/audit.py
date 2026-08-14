"""자료 결손 점검.

**없는 줄 모르는 자료가 가장 위험하다.** 수집 파이프라인은 API 가 0건을 줘도
정상 종료한다. 어디에도 경고가 없고, 그 조용함이 신호처럼 보인다.

경륜에서 이 함정이 특히 잘 나는 자리가 넷이다.

* **출주표는 광명뿐이다.** 착순은 창원·부산도 들어오므로, 그쪽 경주는 항상
  '출주표 결손'으로 보인다. 그것은 결손이 아니라 원래 없는 것이므로 **광명만
  센다** — 이걸 구분하지 않으면 매 실행마다 수백 건의 가짜 경고가 뜨고, 진짜
  결손은 그 안에 묻힌다.
* **회차가 한 자리일 때 조용히 빈다.** ``tms``·``week_tcnt`` 는 2자리 0채움이
  아니면 오류 없이 0건을 준다. 연초(1~9회차)에만 터지므로 여름에 만든 수집기가
  이듬해 1월에 조용히 비는 식이다.
* **경주결과순위의 페이지네이션이 불안정하다.** 연 단위로 훑으면 행이 중복되고
  누락된다. 회차 단위 수집이 맞는데, 그 규칙이 어디선가 깨지면 착순만 조금씩
  빈다.
* **일곱 명이 다 안 들어온 경주.** 결장·실격이면 정상이지만, 수집 누락이면
  경주 하나의 확률 정규화가 통째로 틀어진다.

그래서 굽기 전에 자료가 있어야 할 자리에 있는지 따로 센다. 부족하면
``--strict`` 로 **비정상 종료**해 눈에 띄게 한다 — 경고를 로그에만 남기면
아무도 보지 않는다.

    python -m cycleai.audit --db data/cycleai.sqlite
    python -m cycleai.audit --days 60 --strict
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import Dict, List

import pandas as pd

from .clock import today_kst
from .kcycle.store import session

# 배당을 받을 수 있는 승식 수 (단승·연승1·연승2·쌍승·복승·삼복승).
EXPECTED_POOLS = 6
# 경륜은 7명 편성이 기본이다.
FULL_FIELD = 7
# 출주표(=피처)가 있는 유일한 경륜장.
CARD_VENUE = "광명"


def check(conn, days: int) -> List[Dict]:
    """최근 며칠을 훑어 결손을 찾는다. 반환값은 사람이 읽을 문제 목록이다."""
    since = (today_kst() - dt.timedelta(days=days)).strftime("%Y%m%d")
    df = pd.read_sql_query(
        """
        SELECT g.race_key, g.race_ymd, g.stnd_yr, g.meet_nm, g.week_tcnt,
               g.day_tcnt, g.race_no, g.post_time,
               COALESCE(g.has_result, 0) AS has_result,
               (SELECT COUNT(*) FROM entries e WHERE e.race_key = g.race_key) AS n_entry,
               (SELECT COUNT(*) FROM results r WHERE r.race_key = g.race_key
                                             AND r.ord IS NOT NULL)           AS n_ord,
               (SELECT COUNT(DISTINCT v.pool) FROM payoffs v
                 WHERE v.race_key = g.race_key)                               AS n_pool,
               (SELECT COUNT(*) FROM predictions p
                 WHERE p.race_key = g.race_key)                               AS n_pred
        FROM races g
        WHERE g.race_ymd IS NOT NULL AND g.race_ymd >= ?
        """, conn, params=[since])
    if df.empty:
        return []

    # 광명만 본다. 창원·부산은 출주표도 배당도 공개 API 에 없으므로, 함께 세면
    # 있지도 않은 자료를 매번 결손으로 보고하게 된다.
    home = df[df["meet_nm"] == CARD_VENUE]
    done = home[home["has_result"] == 1]
    issues: List[Dict] = []

    def add(kind: str, rows: pd.DataFrame, note: str) -> None:
        if rows.empty:
            return
        issues.append({
            "kind": kind, "n": len(rows), "note": note,
            "races": [f"{r.race_ymd} {int(r.race_no)}R "
                      f"({int(r.stnd_yr)}년 {int(r.week_tcnt)}회 {int(r.day_tcnt)}일)"
                      for r in rows.head(8).itertuples()],
        })

    # ── 배당 ─────────────────────────────────────────────────────
    add("배당 결손", done[done["n_pool"] == 0],
        "시행된 경주에 배당이 하나도 없다 — 이 경주들은 승식 판정에서 제외된다")
    add("배당 일부", done[(done["n_pool"] > 0) & (done["n_pool"] < EXPECTED_POOLS)],
        f"승식 {EXPECTED_POOLS}종 중 일부만 들어왔다")

    # ── 착순 ─────────────────────────────────────────────────────
    # 당일 경주는 아직 안 끝났을 수 있으므로 어제 이전만 본다.
    yesterday = (today_kst() - dt.timedelta(days=1)).strftime("%Y%m%d")
    add("착순 결손", done[(done["n_ord"] == 0) & (done["race_ymd"] <= yesterday)],
        "전날 이전 경주인데 착순이 없다")
    add("착순 일부", done[(done["n_ord"] > 0) & (done["n_ord"] < done["n_entry"])],
        "출주 인원보다 착순이 적다 (결장·실격이면 정상, 아니면 수집 누락)")

    # ── 출주표 ───────────────────────────────────────────────────
    add("출주표 결손", home[home["n_entry"] == 0],
        "광명 경주인데 출주표가 없다 — 피처도 평가도 만들 수 없다")
    add("편성 부족", home[(home["n_entry"] > 0) & (home["n_entry"] < FULL_FIELD)],
        f"출주표가 {FULL_FIELD}명 미만이다 (결장이면 정상)")

    # ── 회차 0채움 사고의 흔적 ────────────────────────────────────
    #
    # 한 자리 회차만 통째로 비어 있으면 그것은 개최가 없었던 것이 아니라
    # 0채움을 빠뜨린 것이다. 두 자리 회차는 멀쩡하므로 총량만 보면 눈치채지
    # 못한다 — 그래서 따로 센다.
    single = home[home["week_tcnt"] <= 9]
    if len(single) and not len(single[single["n_entry"] > 0]):
        issues.append({
            "kind": "한 자리 회차 전멸", "n": len(single),
            "note": "1~9회차에 출주표가 하나도 없다. week_tcnt/tms 의 2자리 "
                    "0채움이 빠졌을 때 나타나는 모양이다",
            "races": [f"{int(r.stnd_yr)}년 {int(r.week_tcnt)}회" for r in
                      single.head(8).itertuples()],
        })

    # ── 예측 ─────────────────────────────────────────────────────
    # 예측을 남기기 시작한 이후의 경주만 본다. 그 전은 없는 것이 정상이다.
    first = conn.execute(
        "SELECT MIN(g.race_ymd) FROM races g JOIN predictions p "
        "ON p.race_key = g.race_key").fetchone()[0]
    if first:
        add("예측 누락", home[(home["n_entry"] > 0) & (home["n_pred"] == 0) &
                           (home["has_result"] == 1) & (home["race_ymd"] >= first)],
            "출주표가 있는데 예측이 없다 — 그날 생성이 빠졌다")

    return issues


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="자료 결손 점검")
    ap.add_argument("--db", default="data/cycleai.sqlite")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--strict", action="store_true",
                    help="결손이 있으면 1 로 끝낸다 (자동화에서 빨갛게 뜬다)")
    args = ap.parse_args(argv)

    with session(args.db) as conn:
        issues = check(conn, args.days)

    if not issues:
        print(f"자료 점검 최근 {args.days}일 — 결손 없음")
        return 0

    print(f"자료 점검 최근 {args.days}일 — 결손 {len(issues)}종")
    for it in issues:
        print(f"\n  ▸ {it['kind']} {it['n']}경주 — {it['note']}")
        for r in it["races"]:
            print(f"      {r}")
        if it["n"] > len(it["races"]):
            print(f"      … 외 {it['n'] - len(it['races'])}경주")
    print()
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
