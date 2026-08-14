"""피처 생성.

경륜의 성질이 피처 설계를 크게 좌우한다. 경정과 갈리는 지점이 넷이다.

* **자리가 유리함을 뜻하지 않는다.** 경정의 1코스는 그 자체로 압도적이었지만,
  경륜의 배번은 출발 위치일 뿐이다. 그 자리를 **전법**(선행·젖히기·추입·마크)이
  차지한다. 출주표가 전법별 입상 횟수를 주므로, 횟수보다 **비율**을 만든다 —
  30번 입상 중 20번이 마크인 선수와 3번 중 2번이 마크인 선수는 같은 성향이지만
  횟수로 보면 전혀 다른 값이다.

* **혼자 달리지 않는다.** 경륜은 라인을 짜고 함께 들어온다. 그래서 '이 선수가
  얼마나 강한가'만큼 '이 경주에 자기 편이 있는가'가 결과를 가른다. 훈련지가
  같은 동료 수와, 상대전적의 **동반입상 횟수**를 라인 대리 지표로 쓴다.

* **최근 성적이 회전 단위로 온다.** 회차가 3일 편성이라 최근 3회전 × 3일차 =
  아홉 개의 착순이 출주표에 실려 있다. 우리가 과거 결과에서 다시 계산할 필요가
  없고, 계산하지 않으므로 **누수 위험도 없다**.

* **등급이 두 체계다.** 경주 급(선발·우수·특선)과 선수 등급(A1~B3)이 따로 있고,
  등급 조정 전후 값도 온다. 조정으로 올라온 선수와 내려온 선수는 같은 등급이어도
  다른 상태다.

한 경주에서 이기는 선수는 정확히 하나다. 따라서 절대값만큼이나 **같은 경주 안의
상대 위치**가 중요하다. 평균득점 85점은 다른 여섯이 80점대면 강점이지만 90점대면
약점이다. 그래서 주요 지표마다 경주 내 편차·표준점수·순위를 함께 만든다.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# 경주 내 상대 위치를 함께 만들 지표들.
RELATIVE = [
    "tot_avg_scr", "area_avg_scr", "win_rate", "high_rate", "high_3_rate",
    "rec_200m", "gear_rate", "bf_avg", "period_no", "age",
    "pre_ratio", "pas_ratio", "brk_ratio", "mrk_ratio",
    "win_per_day", "line_mates", "line_same_win", "prior_high_rate",
    "grade_rank",
]

CATEGORICAL = ["back_no", "racer_grd", "grade_cur", "race_grade", "grade_move",
               "trng_plc"]

# 선수 등급의 실제 서열. 특선급(SS·S1~S3) → 우수급(A1~A3) → 선발급(B1~B3).
# 작을수록 높은 등급이다. 사전순 비교로는 SS 가 S1 보다 뒤로 가서 뒤집힌다.
GRADE_ORDER = {g: i for i, g in enumerate(
    ["SS", "S1", "S2", "S3", "A1", "A2", "A3", "B1", "B2", "B3"])}

BASE_NUMERIC = [
    "age", "period_no", "gear_rate", "rec_200m",
    "pre_win_cnt", "pas_win_cnt", "brk_win_cnt", "mrk_win_cnt",
    "pre_ratio", "pas_ratio", "brk_ratio", "mrk_ratio",
    "win_tot_cnt", "run_day_cnt", "win_per_day",
    "win_rate", "high_rate", "high_3_rate", "tot_avg_scr", "area_avg_scr",
    "bf1_d1", "bf1_d2", "bf1_d3", "bf2_d1", "bf2_d2", "bf2_d3",
    "bf3_d1", "bf3_d2", "bf3_d3", "bf_avg", "bf_cnt", "bf_best", "bf_win_cnt",
    "line_mates", "line_same_win", "line_opp_win",
    "prior_run_cnt", "prior_win_rate", "prior_high_rate", "prior_high_3_rate",
    "prior_down_cnt", "prior_elim_cnt",
    "acc_prior_cnt",
    # 등급은 범주이자 서열이다. 범주형으로만 넣으면 'A3 와 B1 은 인접하다'는
    # 것을 모델이 표본으로 다시 배워야 한다. 서열을 숫자로도 준다.
    "grade_rank",
    "field_size", "race_len", "round_cnt",
]

TRAIN_SQL = """
SELECT
    e.*,
    r.race_ymd, r.stnd_yr, r.meet_nm, r.week_tcnt, r.day_tcnt, r.race_no,
    r.race_grade, r.race_len, r.round_cnt, r.field_size, r.has_result,
    r.post_time,
    res.ord,
    -- 단승 배당은 경주 단위 값이다. 회수율 계산에만 쓰고 피처로는 절대 넣지
    -- 않는다 — 경주가 끝나야 정해지는 값이다.
    win.payout AS win_payout
FROM entries e
JOIN races r          ON r.race_key = e.race_key
LEFT JOIN results res ON res.race_key = e.race_key AND res.racer_nm = e.racer_nm
LEFT JOIN payoffs win ON win.race_key = e.race_key AND win.pool = '단승'
"""


def _tactic_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """전법별 입상 비율.

    횟수는 경력 길이에 끌려간다 — 마크 20회는 베테랑이면 흔하고 신인이면 전부다.
    비율로 바꾸면 '이 선수가 어떻게 타는 사람인가'만 남는다. 횟수도 그대로
    남겨 둔다: 비율이 같아도 표본이 3회인지 300회인지는 다른 이야기다.
    """
    cols = {"pre": "pre_win_cnt", "pas": "pas_win_cnt",
            "brk": "brk_win_cnt", "mrk": "mrk_win_cnt"}
    total = sum(pd.to_numeric(df[c], errors="coerce").fillna(0)
                for c in cols.values() if c in df)
    for name, col in cols.items():
        v = pd.to_numeric(df.get(col), errors="coerce")
        # 입상이 한 번도 없으면 성향을 말할 수 없다. 0 으로 채우면 '전 종목
        # 0%인 선수'가 되어 마크형과 구분되지 않는다 — 결측으로 둔다.
        df[f"{name}_ratio"] = (v / total.replace(0, np.nan))
    return df


def _line_features(df: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """라인 대리 지표.

    경륜에서 승부는 혼자 나지 않는다. 같은 지역(훈련지) 선수끼리 라인을 짜고
    앞사람이 바람을 막아 주면 뒷사람이 살아 온다. 공개 자료에 '오늘 누가 누구와
    라인을 짤 것인가'는 없으므로 둘로 대신한다.

    * **훈련지 동료 수** — 같은 경주에 훈련지가 같은 선수가 몇인가.
    * **동반입상 이력** — 상대전적 API 의 same_win_tcnt 를 이 경주의 다른 여섯
      명에 대해 합한 값. 함께 입상해 온 사이라면 오늘도 함께 갈 가능성이 높다.

    **작년 자료만 쓴다.** 상대전적은 그 해 전체의 누적값이라, 같은 해 것을 쓰면
    오늘 경주의 결과가 이미 그 숫자 안에 들어가 있다.
    """
    df["line_mates"] = (df.groupby(["race_key", "trng_plc"])["racer_nm"]
                        .transform("count") - 1)
    df.loc[df["trng_plc"].isna(), "line_mates"] = np.nan

    try:
        oppo = pd.read_sql_query(
            "SELECT stnd_yr, racer_nm, oppo_nm, same_win_cnt, win_cnt, lose_cnt "
            "FROM oppo", conn)
    except (ValueError, sqlite3.Error):
        oppo = pd.DataFrame()
    if oppo.empty:
        df["line_same_win"] = np.nan
        df["line_opp_win"] = np.nan
        return df

    # 25만 행을 돌므로 조회를 딕셔너리로 둔다. MultiIndex 의 .loc 는 행마다
    # 인덱싱 비용이 붙어 같은 일에 수십 배가 걸린다.
    oppo["join_yr"] = pd.to_numeric(oppo["stnd_yr"], errors="coerce") + 1
    pair = {(int(r.join_yr), r.racer_nm, r.oppo_nm):
            (float(r.same_win_cnt or 0), float(r.win_cnt or 0))
            for r in oppo.itertuples() if pd.notna(r.join_yr)}

    # 결과를 행 인덱스에 직접 담는다. 리스트에 모아 뒤에 붙이면 groupby 순서와
    # 프레임 순서가 어긋나는 순간 **다른 선수의 라인 값이 들어간다** — 오류
    # 없이 조용히 틀리는 종류의 사고다.
    same = pd.Series(np.nan, index=df.index, dtype=float)
    opp = pd.Series(np.nan, index=df.index, dtype=float)
    for _, race in df.groupby("race_key", sort=False):
        names = list(race["racer_nm"])
        yr = race["stnd_yr"].iloc[0]
        if pd.isna(yr):
            continue
        yr = int(yr)
        for idx, me in zip(race.index, names):
            s = w = 0.0
            found = False
            for other in names:
                if other == me:
                    continue
                hit = pair.get((yr, me, other))
                if hit is None:
                    continue
                found = True
                s += hit[0]
                w += hit[1]
            if found:
                same[idx], opp[idx] = s, w
    df["line_same_win"] = same
    df["line_opp_win"] = opp
    return df


def _prior_year(df: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """작년 성적을 사전값으로 붙인다.

    선수정보 API 는 연도별 집계다. **작년 것만** 쓴다 — 올해 것에는 오늘 경주가
    이미 들어가 있어, 쓰면 그 순간 미래를 본다.
    """
    try:
        ry = pd.read_sql_query(
            "SELECT stnd_yr, racer_nm, run_cnt, win_rate, high_rate, high_3_rate,"
            " down_cnt, elim_cnt FROM racer_year", conn)
    except (ValueError, sqlite3.Error):
        ry = pd.DataFrame()
    cols = ["prior_run_cnt", "prior_win_rate", "prior_high_rate",
            "prior_high_3_rate", "prior_down_cnt", "prior_elim_cnt"]
    if ry.empty:
        for c in cols:
            df[c] = np.nan
        return df
    ry["join_yr"] = pd.to_numeric(ry["stnd_yr"], errors="coerce") + 1
    ry = ry.rename(columns={"run_cnt": "prior_run_cnt", "win_rate": "prior_win_rate",
                            "high_rate": "prior_high_rate",
                            "high_3_rate": "prior_high_3_rate",
                            "down_cnt": "prior_down_cnt", "elim_cnt": "prior_elim_cnt"})
    return df.merge(ry[["join_yr", "racer_nm"] + cols],
                    left_on=["stnd_yr", "racer_nm"], right_on=["join_yr", "racer_nm"],
                    how="left").drop(columns=["join_yr"], errors="ignore")


def _accident_history(df: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """낙차·사고 이력 (지난 해까지의 누적).

    낙차 API 에는 경륜장이 없어 경주에 정확히 붙일 수 없다. 그래서 **경주가
    아니라 선수에게** 붙이고, 시점도 연 단위로만 자른다 — 정밀하지 않은 것을
    정밀한 척 쓰는 것보다, 거친 것을 거친 대로 쓰는 편이 낫다.
    """
    try:
        acc = pd.read_sql_query(
            "SELECT stnd_yr, racer_nm, COUNT(*) n FROM accidents GROUP BY 1,2", conn)
    except (ValueError, sqlite3.Error):
        acc = pd.DataFrame()
    if acc.empty:
        df["acc_prior_cnt"] = np.nan
        return df
    acc["stnd_yr"] = pd.to_numeric(acc["stnd_yr"], errors="coerce")
    acc = acc.sort_values("stnd_yr")
    acc["cum"] = acc.groupby("racer_nm")["n"].cumsum()
    acc["join_yr"] = acc["stnd_yr"] + 1
    out = df.merge(acc[["join_yr", "racer_nm", "cum"]],
                   left_on=["stnd_yr", "racer_nm"], right_on=["join_yr", "racer_nm"],
                   how="left").drop(columns=["join_yr"], errors="ignore")
    out["acc_prior_cnt"] = out.pop("cum")
    return out


def add_relative(df: pd.DataFrame) -> pd.DataFrame:
    """경주 내 편차·표준점수·순위를 붙인다.

    셋을 다 만드는 이유는 서로 다른 것을 말하기 때문이다. 편차는 '얼마나
    앞서는가', 표준점수는 '이 경주의 흩어진 정도에 비해 얼마나', 순위는 '몇
    번째인가'다. 압도적인 한 명이 있는 경주와 고만고만한 경주를 순위만으로는
    구분할 수 없다.
    """
    for col in RELATIVE:
        if col not in df:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        mean = v.groupby(df["race_key"]).transform("mean")
        std = v.groupby(df["race_key"]).transform("std")
        df[f"rel_{col}"] = v - mean
        # 표준편차가 0(전원 동일)이면 편차도 0이다. 나눗셈으로 inf 를 만들지 않는다.
        df[f"z_{col}"] = (v - mean) / std.replace(0, np.nan)
        df[f"rank_{col}"] = v.groupby(df["race_key"]).rank(ascending=False,
                                                           method="average")
    return df


def build_frame(conn: sqlite3.Connection, *, with_labels: bool = True) -> pd.DataFrame:
    """학습·추론 공통 프레임."""
    df = pd.read_sql_query(TRAIN_SQL, conn)
    if df.empty:
        return df

    num = ["age", "period_no", "gear_rate", "rec_200m", "win_rate", "high_rate",
           "high_3_rate", "tot_avg_scr", "area_avg_scr", "win_tot_cnt",
           "run_day_cnt", "bf_avg", "bf_cnt", "field_size", "race_len",
           "round_cnt", "ord", "back_no", "stnd_yr", "week_tcnt", "day_tcnt",
           "race_no", "pre_win_cnt", "pas_win_cnt", "brk_win_cnt", "mrk_win_cnt"]
    for c in num + [f"bf{i}_d{d}" for i in (1, 2, 3) for d in (1, 2, 3)]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = _tactic_ratios(df)
    # 출전 한 번당 입상 — 입상 횟수를 경력으로 나눈 값. 경력이 긴 선수의
    # 누적 입상 횟수에 모델이 끌려가는 것을 막는다.
    df["win_per_day"] = (df["win_tot_cnt"] /
                         pd.to_numeric(df["run_day_cnt"], errors="coerce")
                         .replace(0, np.nan))

    bf_cols = [f"bf{i}_d{d}" for i in (1, 2, 3) for d in (1, 2, 3)]
    bf = df[bf_cols]
    df["bf_best"] = bf.min(axis=1)
    df["bf_win_cnt"] = (bf == 1).sum(axis=1)

    # 등급 조정 방향. 같은 B2 라도 올라온 B2 와 내려온 B2 는 다른 상태다.
    #
    # **문자열로 비교하면 안 된다.** 등급은 SS > S1 > S2 > S3 > A1 … > B3 인데
    # 사전순으로는 "S1" < "SS" 라서, 특선급 최고 등급으로 올라간 선수가 강등된
    # 것으로 뒤집힌다(전체의 2%인 SS 만 틀리므로 표에서 눈에 띄지도 않는다).
    rank = df["grade_cur"].map(GRADE_ORDER)
    rank_bef = df["grade_bef"].map(GRADE_ORDER)
    df["grade_rank"] = rank
    df["grade_move"] = np.where(
        rank.isna() | rank_bef.isna(), "유지",
        np.where(rank < rank_bef, "상승", np.where(rank > rank_bef, "하락", "유지")))

    df = _prior_year(df, conn)
    df = _accident_history(df, conn)
    df = _line_features(df, conn)
    df = add_relative(df)

    # 시간 정렬 키. 워크포워드 검증에서 순서가 틀리면 미래로 과거를 맞힌다.
    ymd = pd.to_datetime(df["race_ymd"], format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(df["stnd_yr"].astype("Int64").astype(str) + "0101",
                              format="%Y%m%d", errors="coerce")
    df["race_date"] = ymd.fillna(fallback)
    df["order_key"] = (df["stnd_yr"].astype("Int64").astype(str).str.zfill(4) +
                       df["week_tcnt"].astype("Int64").astype(str).str.zfill(2) +
                       df["day_tcnt"].astype("Int64").astype(str).str.zfill(1) +
                       df["race_no"].astype("Int64").astype(str).str.zfill(2))

    if with_labels:
        o = df["ord"]
        df["y_win"] = (o == 1).astype(float).where(o.notna())
        df["y_top2"] = (o <= 2).astype(float).where(o.notna())
        df["y_top3"] = (o <= 3).astype(float).where(o.notna())
    return df


def build_training_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """레이블이 온전한 경주만 남긴 학습 프레임.

    1착이 정확히 한 명으로 확인되는 경주만 쓴다. 실격·낙차로 착순이 깨진 경주를
    그대로 넣으면 '이긴 사람이 없는 경주'가 음성 표본만 잔뜩 만든다.
    """
    df = build_frame(conn, with_labels=True)
    if df.empty:
        return df
    ok = df.groupby("race_key")["y_win"].transform(lambda s: (s == 1).sum() == 1)
    df = df[ok & df["ord"].notna()].copy()
    log.info("학습 프레임 %d행 / %d경주 (%s ~ %s)", len(df), df["race_key"].nunique(),
             str(df["race_date"].min())[:10], str(df["race_date"].max())[:10])
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    """실제로 존재하는 피처 컬럼만 고정된 순서로 돌려준다.

    순서를 고정하는 이유는 모델을 피클로 저장했다가 다시 쓰기 때문이다. 학습과
    추론의 컬럼 순서가 어긋나면 오류 없이 **틀린 예측**이 나온다.
    """
    cols = [c for c in BASE_NUMERIC if c in df]
    cols += [f"{p}{c}" for c in RELATIVE for p in ("rel_", "z_", "rank_")
             if f"{p}{c}" in df]
    cols += [c for c in CATEGORICAL if c in df]
    return cols
