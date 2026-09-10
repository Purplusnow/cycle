"""자동 실행 보고문.

**'성공했습니다'만 보내면 사흘이면 안 읽게 되고, 그러면 정작 실패한 날도
지나친다.** 매번 무엇이 달라졌는지를 담아야 읽을 이유가 생긴다 — 몇 경주를
게재했고, 어디까지 정산했고, 누적 성적이 얼마인지.

    python -m cycleai.report --db data/cycleai.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from .clock import now_kst
from .site import build_status, fmt_date
from .verify import build_report


def _pct(v, d=1):
    return "—" if v is None else f"{v * 100:.{d}f}%"


def compose(db: str) -> str:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        st = build_status(conn)
        rep = build_report(conn, "v1")
    finally:
        conn.close()

    o = rep.get("overall") or {}
    L = [f"경륜 분석 · {now_kst().strftime('%m/%d %H:%M')} KST", ""]

    if st["upcoming_total"]:
        days = " · ".join(fmt_date(d) for d in st["upcoming_days"])
        L += ["■ 게재", f"  발주 전 {st['upcoming_total']}경주 ({days})"]
        if st["upcoming_unpredicted"]:
            L.append(f"  ⚠ 그중 {st['upcoming_unpredicted']}경주는 아직 예상 없음")
    else:
        L += ["■ 게재", "  발주 전 경주 없음 (다음 편성 대기)"]

    if st["settle_pending"]:
        L += ["", "■ 밀린 정산",
              f"  {st['settle_pending']}경주 ({' · '.join(st['settle_pending_days'])})"]

    if o.get("n_races"):
        ci = o.get("roi_win_ci")
        band = (f" (95% 구간 {_pct(o['roi_win'] - ci, 0)}~{_pct(o['roi_win'] + ci, 0)})"
                if ci else "")
        L += ["", "■ 공개 기록 (발주 전 확정 저장)",
              f"  {o['first_date']} ~ {o['last_date']} · {o['n_races']:,}경주",
              f"  1순위 1착 {_pct(o['hit_win'])} · 2착 이내 {_pct(o['hit_place'])}",
              f"  단승 회수율 {_pct(o['roi_win'])}{band}"]
    else:
        L += ["", "■ 공개 기록", "  아직 정산된 경주가 없습니다"]
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="자동 실행 보고문")
    ap.add_argument("--db", default="data/cycleai.sqlite")
    args = ap.parse_args(argv)
    print(compose(args.db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
