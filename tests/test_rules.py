"""값의 뜻을 지키는 규칙들.

여기 있는 것은 전부 **깨져도 프로그램이 멈추지 않는** 종류다. 등급이 뒤집혀도
예측은 나오고, 조사가 틀려도 페이지는 뜬다. 그래서 테스트로만 지킬 수 있다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleai.features import GRADE_ORDER  # noqa: E402
from cycleai.korean import josa  # noqa: E402
from cycleai.kcycle.normalize import (  # noqa: E402
    parse_clock, parse_past_rank, parse_pool_value, parse_seconds, race_key,
)


# ── 등급 ────────────────────────────────────────────────────────────

def test_등급_서열은_사전순이_아니다():
    # 특선급 최고는 SS 다. 사전순으로는 "S1" < "SS" 라 SS 승급이 강등으로
    # 뒤집힌다 — 전체의 2%인 SS 만 틀리므로 표에서 눈에 띄지도 않는다.
    assert GRADE_ORDER["SS"] < GRADE_ORDER["S1"] < GRADE_ORDER["S3"]
    assert GRADE_ORDER["S3"] < GRADE_ORDER["A1"]     # 특선급이 우수급보다 위
    assert GRADE_ORDER["A3"] < GRADE_ORDER["B1"]     # 우수급이 선발급보다 위
    # 사전순으로는 "S1" 이 "SS" 보다 앞이다. 옛 코드는 '문자열이 작으면 상위
    # 등급'으로 봤으므로, S1 → SS 승급이 강등으로 뒤집혔다.
    assert "S1" < "SS"
    assert GRADE_ORDER["SS"] < GRADE_ORDER["S1"]     # 실제로는 SS 가 위다


# ── 조사 ────────────────────────────────────────────────────────────

def test_조사():
    assert josa("김용진", "이/가") == "김용진이"
    assert josa("이주영", "이/가") == "이주영이"
    assert josa("김종재", "은/는") == "김종재는"
    assert josa("선행", "으로/로") == "선행으로"
    # 'ㄹ' 받침은 '으로' 가 아니라 '로' 다.
    assert josa("서울", "으로/로") == "서울로"
    assert josa("마크", "으로/로") == "마크로"


# ── 출주표 파싱 ─────────────────────────────────────────────────────

def test_같은_구분자가_다른_뜻이다():
    # dptre_tm 은 발주 시각, rec_200m_scr 은 초 단위 기록이다. 구분자가 같다는
    # 이유로 한 함수에 몰면 12시 55분이 12.55초가 된다.
    assert parse_clock('12"55') == "12:55"
    assert parse_seconds('11"78') == 11.78


def test_최근성적은_경주번호가_앞이다():
    # 실측으로 확정했다: 2026-05-23 강석호 "선발 5-1" = 부산 5경주 1착.
    # 뒤집어 읽으면 7착을 1착으로 배우게 된다.
    p = parse_past_rank("선발 5-1")
    assert (p["race_no"], p["rank"]) == (5, 1)
    assert (p["grade"], p["round"]) == ("선발", "예선")
    assert parse_past_rank("선준 5-2")["round"] == "준결승"
    assert parse_past_rank("선결 6-7")["round"] == "결승"


def test_결장은_성적이_아니다():
    # 0 이나 7 로 채우면 '아주 나쁜 성적'이 된다. 결장은 성적 없음이다.
    p = parse_past_rank("결  장")
    assert p["rank"] is None and p["absent"] is True


# ── 배당 ────────────────────────────────────────────────────────────

def test_배당_조합_파싱():
    # 경주결과는 조합과 배당을 한 문자열로, 배당률은 배당만 준다.
    assert parse_pool_value("(7-3-4)4.4") == ("7-3-4", 4.4)
    assert parse_pool_value("2.7") == (None, 2.7)
    assert parse_pool_value("") == (None, None)


# ── 경주 키 ─────────────────────────────────────────────────────────

def test_경주키에_경륜장이_들어간다():
    # 광명·창원·부산이 같은 날 같은 회차 같은 경주번호를 각자 쓴다. 경정처럼
    # 경륜장을 빼면 세 경주가 한 행에서 서로를 덮어쓴다.
    a = race_key(2026, "광명  ", 24, 1, 3)
    b = race_key(2026, "창원", 24, 1, 3)
    assert a == "2026-광명-24-1-03"
    assert a != b
