"""파라미터 정규화 규칙 — 어기면 오류가 아니라 '빈 결과'가 된다.

이 셋은 전부 프로브로 실측한 게이트웨이 동작이고, 셋 다 실패가 조용하다.
조용한 실패는 테스트로만 지킬 수 있다 — 깨져도 실행은 성공으로 끝나기 때문이다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleai.kcycle.endpoints import (  # noqa: E402
    ALT_PARAM_KEYS, norm_meet, norm_ymd, to_api_params,
)


def test_race_no_는_2자리_0채움():
    # 실측: race_no="01" → 1건, "1" → 0건 (오류 없이 빈 결과)
    assert to_api_params("race_result", {"race_no": 1})["race_no"] == "01"
    assert to_api_params("race_result", {"race_no": "7"})["race_no"] == "07"
    assert to_api_params("race_result", {"race_no": 15})["race_no"] == "15"


def test_회차도_2자리_0채움():
    # 실측: 출주표 week_tcnt="09" → 448건, "9" → 0건.
    # 한 자리 회차(=연초)에서만 터지므로 두 자리만 보고 지나치기 쉽다.
    assert to_api_params("race_card", {"week_tcnt": 9})["week_tcnt"] == "09"
    assert to_api_params("race_card", {"week_tcnt": 24})["week_tcnt"] == "24"


def test_이름을_바꾼_뒤에_0채움한다():
    # 순서가 뒤집히면 week_tcnt 만 채워지고 tms 는 그냥 지나가, 경주결과순위가
    # 1~9회차에서 조용히 빈다.
    out = to_api_params("race_rank", {"week_tcnt": 9})
    assert out == {"tms": "09"}


def test_출주표_날짜를_그대로_넘기면_안된다():
    # 출주표는 "2026.01.02", 조조연습현황은 8자리만 받는다.
    out = to_api_params("exercise", {"stnd_yr": "2026", "race_ymd": "2026.01.02"})
    assert out["race_ymd"] == "20260102"


def test_월일만_온_날짜는_연도를_붙인다():
    # 경주결과의 race_ymd 는 "0102" 로 월일만 온다.
    assert norm_ymd("0102", "2026") == "20260102"
    # 연도를 모르면 지어내지 않는다 — 지어내면 그 행은 다른 해의 경주가 된다.
    assert norm_ymd("0102") == "0102"


def test_경륜장명_꼬리공백():
    # 출주표는 "광명  ", 경주결과순위는 "광명". 조인 키로 그대로 쓰면 안 붙는다.
    assert norm_meet("광명  ") == norm_meet("광명") == "광명"


def test_파라미터명이_어긋나는_API():
    # 경주결과순위·낙차사고만 stnd_year/tms/day_ord/race_day 를 쓴다.
    assert ALT_PARAM_KEYS == {"race_rank", "down_accident"}
    out = to_api_params("race_rank", {"stnd_yr": "2026", "week_tcnt": 24,
                                      "day_tcnt": 2, "race_ymd": "2026.06.13"})
    assert out == {"stnd_year": "2026", "tms": "24", "day_ord": 2,
                   "race_day": "20260613"}
    # 다른 API 는 원래 이름 그대로여야 한다.
    assert "stnd_yr" in to_api_params("race_card", {"stnd_yr": "2026"})
