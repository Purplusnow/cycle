"""기호 기준의 보정.

``marks.MARK_THRESHOLDS['star']`` 를 손으로 정하지 않기 위한 도구다. ★ 는
'우세가 뚜렷하다'는 뜻인데, 그 기준을 눈대중으로 잡으면 ★ 가 너무 흔해져
아무것도 가리지 않거나 너무 드물어 화면에 나오지 않는다.

시간순 교차검증 기록(v1-oos)에서 후보 기준마다 **출현 빈도와 그때의 적중률**을
같이 낸다. 둘을 함께 봐야 곡선의 무릎이 보인다 — 적중률만 보면 기준은 끝없이
올라가고, 출현만 보면 끝없이 내려간다.

    python -m cycleai.calibrate --db data/cycleai.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
from typing import List, Optional

import pandas as pd

# 경륜은 예측이 잘 되는 종목이라 1순위의 2착 이내 확률이 전반적으로 높다.
# 경정에서 쓰던 0.6~0.75 구간은 여기서 '출현 80%' 라 기호가 아무것도 가리지
# 못한다 — 후보를 위쪽으로 옮겨야 한다.
CANDIDATES = [0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96]

SQL = """
SELECT p.race_key, p.pred_rank, p.p_top2, res.ord
FROM predictions p
JOIN races r          ON r.race_key = p.race_key
LEFT JOIN results res ON res.race_key = p.race_key AND res.racer_nm = p.racer_nm
WHERE p.model_version = ? AND COALESCE(r.has_result, 0) = 1
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="예상 기호 기준 보정")
    ap.add_argument("--db", default="data/cycleai.sqlite")
    ap.add_argument("--version", default="v1-oos")
    args = ap.parse_args(argv)

    conn = sqlite3.connect(args.db)
    try:
        df = pd.read_sql_query(SQL, conn, params=[args.version])
    finally:
        conn.close()
    if df.empty:
        print("보정할 기록이 없습니다. 먼저 python -m cycleai.predict backfill")
        return 1

    df["ord"] = pd.to_numeric(df["ord"], errors="coerce")
    # 1착이 정확히 한 명으로 확인되는 경주만
    ok = df.groupby("race_key")["ord"].transform(lambda s: (s == 1).sum() == 1)
    df = df[ok]
    top = df[df["pred_rank"] == 1].dropna(subset=["p_top2"])
    if top.empty:
        print("1순위 기록이 없습니다.")
        return 1

    n = len(top)
    base_win = float((top["ord"] == 1).mean())
    base_plc = float((top["ord"] <= 2).mean())
    print(f"v1-oos 워크포워드 {n:,}경주 · 무작위는 1착 14.3% / 2착이내 28.6%")
    print(f"1순위 전체 평균은 1착 {base_win:.1%} / 2착이내 {base_plc:.1%}\n")
    print(f"  {'기준':>6}{'★출현':>8}{'★1착':>9}{'★2착이내':>11}")
    for t in CANDIDATES:
        sub = top[top["p_top2"] >= t]
        if not len(sub):
            continue
        print(f"  {t:>6.2f}{len(sub) / n:>8.0%}{(sub['ord'] == 1).mean():>9.1%}"
              f"{(sub['ord'] <= 2).mean():>11.1%}")
    print("\n출현이 10~20% 인 구간에서 적중률 증가가 꺾이는 지점을 고른다 —")
    print("경주일마다 두세 번 나오면 드물지도 흔하지도 않다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
