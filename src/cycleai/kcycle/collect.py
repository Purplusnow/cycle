"""수집기 — 오픈API → SQLite.

**연 단위 일괄 수집이 기본이다.** 경정에서는 배당을 경주 하나씩 받아야 해서
회차 좌표를 훑고 다녔지만, 경륜은 출주표·착순·배당·선수정보가 전부 연도 하나로
통째로 온다. 20년치가 1,000회 호출 안에 들어온다.

    python -m cycleai.kcycle.collect backfill      # 2005~올해
    python -m cycleai.kcycle.collect daily         # 올해분 갱신
    python -m cycleai.kcycle.collect --years 2019 2020

경주일 흐름은 ``daily`` → ``predict upcoming`` → ``site`` →
(경주 후) ``daily`` → ``verify`` → ``audit`` → ``site`` 순이다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..clock import today_kst
from . import normalize as N
from .client import KcycleApiError, KcycleClient
from .endpoints import REGISTRY, to_api_params
from .store import (DEFAULT_DB, counts, dumps, link_result_back_no, log_fetch,
                    mark_flags, already_fetched, session, upsert)

log = logging.getLogger(__name__)

# 출주표·착순 모두 2003년부터 있다(2002 는 0건). 그 이전을 물어봐야 빈 응답만
# 돌아오므로 시작 연도를 박아 둔다 — 호출량은 유한하다.
FIRST_YEAR = 2003

# 상대전적은 연 44,000행이고 그 해 전체의 누적값이라 과거 연도를 모아 봐야
# 시점이 어긋난다. 최근 몇 해만 받는다.
OPPO_YEARS = 3

# 회차별 득점은 회차마다 한 번씩 불러야 한다(연 33회). 20년이면 660회라
# 백필에서 가장 비싼 항목이므로 기본은 최근 연도만 받는다.
TMS_YEARS = 3
MAX_WEEK = 40


def _years(args_years: Optional[List[int]], mode: str) -> List[int]:
    this = today_kst().year
    if args_years:
        return sorted(args_years)
    if mode == "backfill":
        return list(range(FIRST_YEAR, this + 1))
    # daily 는 올해와 작년. 연초에는 작년 회차가 아직 갱신되는 일이 있다.
    return [this - 1, this]


def collect_year(client: KcycleClient, conn, year: int, *, force: bool = False,
                 heavy: bool = True) -> Dict[str, int]:
    """한 해분을 통째로 받아 넣는다.

    **API 구간마다 커밋한다.** 끝에서 한 번만 커밋하면, 뒤쪽 구간이 네트워크로
    실패하는 순간 앞에서 이미 받아 놓은 것까지 함께 롤백된다 — 실측
    (2026-08-16)에서 상대전적 구간이 타임아웃 나면서 그 실행의 착순·배당이
    통째로 날아갔고, 사이트는 하루 전 자료로 다시 구워졌다.

    해외 러너에서 data.go.kr 접속은 자주 끊긴다. 한 번에 다 받는 것을 전제로
    두면 안 되고, **매 실행이 조금씩이라도 전진하게** 만들어야 한다.
    """
    y = str(year)
    out: Dict[str, int] = {}
    this_year = today_kst().year

    def done(endpoint: str, coord: str) -> bool:
        # 올해 자료는 계속 늘어나므로 절대 건너뛰지 않는다. 지난 해는 한 번
        # 받았으면 다시 묻지 않는다 — 그것이 백필을 며칠에 나눠 돌릴 수 있게 한다.
        if force or year >= this_year:
            return False
        return already_fetched(conn, endpoint, coord)

    # --- 출주표: 피처. 광명만 나온다 (창원·부산은 출주표 자체가 없다) ---
    if not done("race_card", y):
        recs = client.fetch(REGISTRY["race_card"].path, {"stnd_yr": y},
                            rows=1000, max_pages=60)
        races = {r["race_key"]: r for r in
                 filter(None, (N.race_row_from_entry(x) for x in recs))}
        entries = []
        for rec in recs:
            row = N.entry_row(rec)
            if row and row.get("back_no"):
                row["raw_json"] = dumps(rec)
                entries.append(row)
        upsert(conn, "races", list(races.values()), ["race_key"])
        upsert(conn, "entries", entries, ["race_key", "back_no"])
        log_fetch(conn, "race_card", y, len(recs))
        conn.commit()
        out["entries"] = len(entries)

    # --- 착순: 레이블. **창원·부산도 넣는다** (광명 선수의 원정 성적이다) ---
    #
    # **연 단위로 받으면 안 된다.** 이 API 는 정렬이 고정돼 있지 않아, 페이지를
    # 넘기면 같은 행이 두 번 오고 어떤 행은 한 번도 오지 않는다. 실측: 2026년을
    # numOfRows 1000/500/300 으로 각각 훑으면 받은 행은 14,455 로 같은데
    # 유니크는 11,349 / 10,957 / 10,630 으로 매번 달랐다 — 페이지 크기가 결과를
    # 바꾼다는 것은 페이지 사이에 순서가 없다는 뜻이다.
    #
    # 회차 하나는 한 페이지(1000행)에 다 들어가므로 페이지를 넘길 일이 없고,
    # 그래서 결과가 확정적이다. 회차별로 훑으면 39회 호출로 14,455건 전부가
    # 중복 없이 들어온다 — totalCount 와 정확히 일치한다.
    if not done("race_rank", y):
        total = 0
        for wk in range(1, MAX_WEEK + 1):
            recs = client.fetch(REGISTRY["race_rank"].path,
                                to_api_params("race_rank",
                                              {"stnd_yr": y, "week_tcnt": wk}),
                                rows=1000, max_pages=3)
            if not recs:
                continue
            upsert(conn, "races", N.race_rows_from_result(recs), ["race_key"])
            rows = N.result_rows(recs)
            for row, rec in zip(rows, recs):
                row["raw_json"] = dumps(rec)
            upsert(conn, "results", rows, ["race_key", "racer_nm"])
            total += len(rows)
        log_fetch(conn, "race_rank", y, total)
        conn.commit()
        out["results"] = total

    # --- 배당: 검증. 연 단위로 통째로 온다 ---
    if not done("payoff", y):
        recs = client.fetch(REGISTRY["payoff"].path, {"stnd_yr": y},
                            rows=1000, max_pages=20)
        rows = [r for rec in recs for r in N.payoff_rows(rec)]
        upsert(conn, "payoffs", rows, ["race_key", "pool"])
        log_fetch(conn, "payoff", y, len(recs))
        conn.commit()
        out["payoffs"] = len(rows)

    # --- 선수 연도별 집계: 사전값 ---
    if not done("racer_info", y):
        recs = client.fetch(REGISTRY["racer_info"].path, {"stnd_yr": y},
                            rows=1000, max_pages=10)
        rows = list(filter(None, (N.racer_year_row(x) for x in recs)))
        upsert(conn, "racer_year", rows, ["stnd_yr", "racer_nm"])
        log_fetch(conn, "racer_info", y, len(recs))
        conn.commit()
        out["racer_year"] = len(rows)

    # --- 낙차·사고: 선수 단위 이력 ---
    if not done("down_accident", y):
        recs = client.fetch(REGISTRY["down_accident"].path,
                            to_api_params("down_accident", {"stnd_yr": y}),
                            rows=1000, max_pages=10)
        rows = [r for rec in recs for r in N.accident_rows(rec)]
        upsert(conn, "accidents", rows,
               ["stnd_yr", "week_tcnt", "day_tcnt", "race_no", "racer_nm"])
        log_fetch(conn, "down_accident", y, len(recs))
        conn.commit()
        out["accidents"] = len(rows)

    if not heavy:
        return out

    # --- 상대전적: 최근 연도만 ---
    if year > this_year - OPPO_YEARS and not done("oppo_win", y):
        # 연 6만 3천 행이라 페이지가 63장이다. max_pages 를 60 으로 두었더니
        # 정확히 60,000 행에서 잘렸는데, **딱 떨어지는 숫자가 유일한 단서였다** —
        # 오류도 경고도 없이 뒤쪽 선수들만 조용히 빠진다.
        recs = client.fetch(REGISTRY["oppo_win"].path, {"stnd_yr": y},
                            rows=1000, max_pages=120)
        rows = list(filter(None, (N.oppo_row(x) for x in recs)))
        upsert(conn, "oppo", rows, ["stnd_yr", "racer_nm", "oppo_nm"])
        log_fetch(conn, "oppo_win", y, len(recs))
        conn.commit()
        out["oppo"] = len(rows)

    # --- 회차별 득점: 회차마다 한 번씩. 최근 연도만 ---
    if year > this_year - TMS_YEARS:
        total = 0
        for wk in range(1, MAX_WEEK + 1):
            coord = f"{y}-{wk:02d}"
            if done("tms_score", coord):
                continue
            try:
                recs = client.fetch(REGISTRY["tms_score"].path,
                                    {"stnd_yr": y, "week_tcnt": wk},
                                    rows=1000, max_pages=5)
            except KcycleApiError as e:
                if e.fatal:
                    raise
                log.warning("  득점 %s 실패: %s", coord, e)
                continue
            rows = list(filter(None, (N.tms_score_row(x) for x in recs)))
            upsert(conn, "tms_score", rows,
                   ["stnd_yr", "week_tcnt", "day_tcnt", "meet_nm", "racer_nm"])
            log_fetch(conn, "tms_score", coord, len(recs))
            conn.commit()
            total += len(rows)
            # 그 해가 아직 안 온 회차는 0건이다. 연속 0건이면 뒤도 비어 있다.
            if not recs and wk > 1:
                break
        out["tms_score"] = total

    return out


def collect_sanctions(client: KcycleClient, conn) -> int:
    """제재선수 현황. 연도 축이 없는 '현재 상태' 라 매번 새로 받는다."""
    recs = client.fetch(REGISTRY["sanction"].path, {}, rows=1000, max_pages=5)
    rows = list(filter(None, (N.sanction_row(x) for x in recs)))
    upsert(conn, "sanctions", rows, ["racer_nm", "kind", "reason"])
    log_fetch(conn, "sanction", "current", len(recs))
    return len(rows)


def collect_results_for(client: KcycleClient, conn, race_keys: List[str]) -> int:
    """경주결과 API 로 배당 조합과 1~3착을 채운다.

    **경주 하나에 호출 하나다** — 파라미터 다섯 개가 전부 필수라 다른 방법이
    없다. 그래서 배당률(연 단위)로 못 채운 경주에만 쓴다. 조합을 알아야 하는
    경주, 즉 우리가 예상을 낸 경주만 넘겨라.
    """
    n = 0
    for key in race_keys:
        c = N.coords_from_key(key)
        if not c:
            continue
        try:
            recs = client.fetch(REGISTRY["race_result"].path, {
                "stnd_yr": c["stnd_yr"], "meet_nm": c["meet_nm"],
                "week_tcnt": c["week_tcnt"], "day_tcnt": c["day_tcnt"],
                "race_no": c["race_no"]}, rows=10, max_pages=1)
        except KcycleApiError as e:
            if e.fatal:
                raise
            log.warning("  경주결과 %s 실패: %s", key, e)
            continue
        for rec in recs:
            upsert(conn, "races", [x for x in [N.race_row_from_result(rec)] if x],
                   ["race_key"])
            upsert(conn, "payoffs", N.result_payoff_rows(rec), ["race_key", "pool"])
            # 1~3착만이라도 채운다. 착순 API 가 늦게 오는 날에 쓸모가 있다.
            upsert(conn, "results", N.result_top3(rec), ["race_key", "racer_nm"])
            n += 1
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="경륜 오픈API 수집기")
    ap.add_argument("mode", nargs="?", default="daily",
                    choices=["backfill", "daily", "prune"])
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--force", action="store_true", help="수집 이력을 무시하고 다시 받는다")
    ap.add_argument("--light", action="store_true",
                    help="상대전적·회차별 득점을 건너뛴다 (호출량 절약)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 원본 응답 정리는 네트워크가 필요 없다. 서비스키를 확인하기 전에 처리한다.
    if args.mode == "prune":
        from .store import prune_raw_json

        with session(Path(args.db)) as conn:
            n = prune_raw_json(conn)
        log.info("오래된 raw_json %d행 정리", n)
        return 0

    try:
        client = KcycleClient.from_env()
    except ValueError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2

    years = _years(args.years, args.mode)
    log.info("수집 대상 연도: %s ~ %s (%d개)", years[0], years[-1], len(years))

    with session(Path(args.db)) as conn:
        failed = []
        for year in years:
            try:
                got = collect_year(client, conn, year, force=args.force,
                                   heavy=not args.light)
            except KcycleApiError as e:
                # 한 해가 막혔다고 나머지를 버리지 않는다. 호출 한도 초과라면
                # 여기서 멈추는 것이 맞고, 그때 이미 받은 해는 DB 에 남는다.
                log.error("%d년 수집 실패: %s", year, e)
                failed.append(str(year))
                if e.code in ("22", "29"):
                    log.error("호출 한도에 걸렸습니다. 내일 이어서 돌리면 "
                              "이미 받은 해는 건너뜁니다.")
                    break
                continue
            conn.commit()
            log.info("  %d: %s", year, got or "이미 받음")

        # **제재선수 하나가 전체 수집을 죽이면 안 된다.**
        #
        # 실측(2026-08-16 02:13): 이 API 가 ConnectTimeout 을 내면서 예외가
        # main 밖으로 나갔고, 워크플로는 설계대로 '기존 자료로 진행' 했다. 그
        # 결과 **그날 새로 올라온 배당이 통째로 빠진 채 사이트가 다시 구워졌다.**
        # 없어도 예측·검증이 되는 보조 자료 때문에 핵심 자료를 잃은 셈이다.
        #
        # 연도별 수집은 이미 개별로 감싸져 있었는데 여기만 맨몸이었다.
        try:
            collect_sanctions(client, conn)
        except KcycleApiError as e:
            log.warning("제재선수 수집 실패 — 건너뜁니다: %s", e)
            failed.append("제재선수")

        linked = link_result_back_no(conn)
        flags = mark_flags(conn)
        conn.commit()
        log.info("배번 연결 %d행, 플래그 %s", linked, flags)
        log.info("합계: %s", counts(conn))
        if failed:
            # 받은 것은 이미 커밋됐다. 실패를 남기되 **0 으로 끝낸다** —
            # 여기서 1 을 돌려주면 워크플로가 통째로 재시도하면서, 방금 받아
            # 놓은 것까지 '실패한 실행'으로 취급한다.
            log.warning("일부 수집 실패: %s (받은 자료는 그대로 남았습니다)",
                        ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
