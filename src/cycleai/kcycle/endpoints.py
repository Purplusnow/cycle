"""경륜 오픈API 엔드포인트 레지스트리.

경로·파라미터는 data.go.kr 데이터셋 페이지에 임베드된 Swagger 명세에서 그대로
받아 ``config/openapi/<데이터셋번호>.json`` 에 보관했다. 추측한 값이 하나도
없으므로 ``NO_OPENAPI_SERVICE_ERROR`` 는 여기 적힌 경로를 쓰는 한 나지 않는다.

승인 여부와 **실제 응답 필드**는 ``probe`` 로 확인한다. 경륜 API 12개 중 4개
(경주결과순위·낙차사고·제재선수·선수정보)는 포털 명세에 응답 정의가 아예 비어
있어, 실물을 보지 않으면 파서를 필드명 추측 없이 쓸 수 없다.

    python -m cycleai.kcycle.probe

**명세가 틀린 곳이 있다 — 오퍼레이션 세그먼트는 대소문자를 가린다.** 자료실
서비스(SRVC_WEB_CRA_MBR_INFO)의 명세는 오퍼레이션 8개를 전부 대문자로 적어
놨는데, 실제 게이트웨이는 그중 5개를 **소문자로만** 받는다(코드 12). 같은
서비스 안에서 스타트사진·경주동영상은 대문자라야 한다. 규칙이 없으므로 여기
적힌 값은 전부 프로브로 실측해 확정한 것이다.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

RESOLVED_PATH = Path(os.environ.get("CYCLEAI_ENDPOINTS", "config/endpoints.resolved.json"))

# 경륜은 세 곳에서 시행한다. 경정에 없던 축이라, 같은 선수라도 어디서 타느냐로
# 성적이 갈린다 (출주표에 광명 전용 평균득점 필드가 따로 있는 것이 그 증거다).
VENUES = ("광명", "창원", "부산")

# **경륜장명에 꼬리 공백이 붙어 온다.** 실측: 출주표·상대전적은 "광명  ",
# 경주결과·경주결과순위는 "광명". 이걸 그대로 조인 키로 쓰면 오류 없이 한 건도
# 붙지 않는다. 읽어 들이는 모든 자리에서 norm_meet 을 통과시킨다.
def norm_meet(value: object) -> str:
    return str(value or "").strip()

# 배번은 1~7. **경정의 정번과 달리 그 자체가 유리함을 뜻하지 않는다.**
# 경륜에서 자리의 의미는 전법(선행·젖히기·추입·마크)과 라인이 만든다.
LANES = (1, 2, 3, 4, 5, 6, 7)


@dataclass
class Endpoint:
    """하나의 오픈API 오퍼레이션."""

    key: str                    # 내부 식별자
    title: str                  # 한글 명칭
    service: str                # GW 서비스 세그먼트 (SRVC_...)
    operation: str              # 오퍼레이션 세그먼트 (TODZ_...)
    dataset_pk: str             # data.go.kr 데이터셋 번호
    required: List[str] = field(default_factory=list)   # serviceKey·페이징 외 필수
    optional: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def path(self) -> str:
        """BASE 이후의 전체 경로. GW 는 서비스 세그먼트를 반드시 요구한다."""
        return f"{self.service}/{self.operation}"


# ---------------------------------------------------------------------------
# 레지스트리 — 예측 파이프라인에 실제로 쓰이는 것부터
# ---------------------------------------------------------------------------

RACE_CARD = Endpoint(
    key="race_card",
    title="출주표",
    service="SRVC_OD_API_CRA_RACE_ORGAN",
    operation="TODZ_API_CRA_RACE_ORGAN_I",
    dataset_pk="15107830",
    optional=["meet_nm", "stnd_yr", "week_tcnt"],
    note=(
        "**예측 피처의 근간.** 47개 필드로 경주 전에 확정되는 정보를 거의 다 담는다. "
        "전법 입상 횟수(pre/pas/brk/mrk_win_cnt = 선행·젖히기·추입·마크)가 경정의 "
        "'코스별 연대율' 자리를 차지한다. 기어배수·200m 기록·훈련지·등급 조정 이력, "
        "그리고 최근 3회전 성적이 일차별 착순으로 온다. "
        "**race_ymd 가 여기 있어** 경정처럼 회차↔날짜를 잇는 별도 API 가 필요 없다."
    ),
)

RACE_RANK = Endpoint(
    key="race_rank",
    title="경주결과순위",
    service="SRVC_CRA_RACE_RANK",
    operation="TODZ_CRA_RACE_RANK",
    dataset_pk="15143989",
    optional=["stnd_year", "tms", "day_ord", "race_no", "race_day",
              "racer_no", "racer_nm"],
    note=(
        "선수 단위 착순 — **학습 레이블의 근간**. 파라미터 이름이 다른 API 와 "
        "**어긋난다**(stnd_year·tms·day_ord·race_day ↔ stnd_yr·week_tcnt·"
        "day_tcnt·race_ymd). "
        "**연 단위로 훑으면 안 된다** — 정렬이 고정돼 있지 않아 페이지를 넘기면 "
        "행이 중복되고 누락된다(페이지 크기를 바꾸면 유니크 건수가 달라진다). "
        "회차 단위로 받으면 한 페이지에 들어가 확정적이다."
    ),
)

PAYOFF = Endpoint(
    key="payoff",
    title="배당률",
    service="SRVC_OD_API_CRA_PAYOFF",
    operation="TODZ_API_CRA_PAYOFF_I",
    dataset_pk="15107845",
    optional=["stnd_yr", "week_tcnt", "day_tcnt"],
    note=(
        "승식별 확정배당(단승·연승1·연승2·쌍승·복승·삼복승) + race_ymd. "
        "**경정보다 훨씬 싸다** — 경정은 4개 파라미터가 전부 필수라 경주 하나씩만 "
        "뽑혔는데, 경륜은 일차 단위로 한 번에 온다. "
        "우리 추천 조합이 실제로 얼마를 돌려줬는지 검증하는 근거다."
    ),
)

RACE_RESULT = Endpoint(
    key="race_result",
    title="경주결과",
    service="SRVC_TODZ_CRA_RACE_RESULT",
    operation="TODZ_API_CRA_RACE_RESULT",
    dataset_pk="15107816",
    required=["stnd_yr", "meet_nm", "week_tcnt", "day_tcnt", "race_no"],
    note=(
        "1~3순위 + 승식 배당 6종 + race_ymd. **가장 비싸다** — 경주 하나를 뽑는 데 "
        "파라미터 5개가 전부 필수라, 24년치를 채우려면 경주 수만큼 호출해야 한다 "
        "(경륜장 3곳 × 회차 × 3일차 × 경주). 착순 레이블은 race_rank 로 만들고 "
        "이 API 는 배당 결손을 메우는 용도로만 쓴다."
    ),
)

TMS_SCORE = Endpoint(
    key="tms_score",
    title="회차별 경주득점",
    service="SRVC_OD_API_CRA_TMS_SCR",
    operation="TODZ_API_CRA_TMS_SCR_I",
    dataset_pk="15107820",
    required=["stnd_yr", "week_tcnt"],
    note=(
        "선수의 회차 단위 경주득점(racer_nm·race_scr·meet_nm·day_tcnt). 7필드뿐이라 "
        "얇지만 **회차별 스냅샷이라 시점을 지켜 쓸 수 있다** — 해당 회차 이전 값만 "
        "쓰면 누수 없이 '그때까지의 폼'이 된다. 경정의 '선수 회차별 성적'이 하던 "
        "역할인데, 경륜에는 이 얇은 판밖에 없다."
    ),
)

DOWN_ACCIDENT = Endpoint(
    key="down_accident",
    title="낙차사고",
    service="SRVC_TODZ_CRA_DOWN_ACDNT",
    operation="TODZ_CRA_DOWN_ACDNT",
    dataset_pk="15119695",
    optional=["stnd_year", "tms", "day_ord", "race_no", "down_acdnt_cd"],
    note=(
        "낙차·사고 이력. 경륜에서 낙차는 결과를 직접 뒤집고, 낙차 이력이 있는 선수는 "
        "이후 전법이 소극적으로 바뀌는 경향이 있다. 경정의 FL(실격) 자리에 해당한다. "
        "파라미터 규칙이 race_rank 계열과 같다. 응답 정의 없음 → 프로브 필요."
    ),
)

SANCTION = Endpoint(
    key="sanction",
    title="제재선수 현황",
    service="SRVC_CRA_RACE_SANC",
    operation="TODZ_CRA_RACE_SANC",
    dataset_pk="15139195",
    optional=["racer_id", "racer_nm", "p_kind", "p_johang"],
    note=(
        "제재유형(p_kind)·제재사유(p_johang)별 선수 현황. 출전 정지·경고가 편성에 "
        "그대로 나타난다. 응답 정의 없음 → 프로브 필요."
    ),
)

OPPO_WIN = Endpoint(
    key="oppo_win",
    title="선수 상대전적",
    service="SRVC_OD_API_CRA_OPPO_WIN",
    operation="todz_api_cra_oppo_win_i",
    dataset_pk="15107822",
    optional=["stnd_yr", "racer_nm", "oppo_racer_nm"],
    note=(
        "승/패/무 + **동반입상횟수(same_win_tcnt)**. 경정에서는 표본이 얕아 "
        "콘텐츠용이었지만, **경륜은 라인을 짜고 함께 들어오는 종목이라 동반입상이 "
        "실제 피처가 된다.** 오퍼레이션 세그먼트가 소문자다."
    ),
)

RACER_INFO = Endpoint(
    key="racer_info",
    title="선수정보",
    service="SRVC_CRA_RACER_INFO",
    operation="TODZ_CRA_RACER_INFO",
    dataset_pk="15107844",
    optional=["stnd_yr", "racer_nm", "period_no"],
    note="선수 마스터(기수·신상). 선수 페이지 표시용. 응답 정의 없음 → 프로브 필요.",
)

BKNO_SUM = Endpoint(
    key="bkno_sum",
    title="선수 배번집계",
    service="SRVC_OD_API_CRA_RACER_BKNO_SUM",
    operation="todz_api_racer_bkno_sum_i",
    dataset_pk="15107825",
    optional=["stnd_yr", "racer_nm"],
    note=(
        "1~7번 배번을 몇 번 받았는지. **횟수만 있고 배번별 성적이 없다** — 경정 "
        "출주표의 '코스별 6개월 연대율'에 해당하는 값이 경륜에는 없다는 뜻이라, "
        "배번별 강약은 우리가 결과에서 직접 집계해야 한다. 오퍼레이션이 소문자."
    ),
)

# --- 홈페이지 자료실(SRVC_WEB_CRA_MBR_INFO) — 한 서비스에 오퍼레이션 8개 ---

SCHEDULE = Endpoint(
    key="schedule",
    title="연간경주일정",
    service="SRVC_WEB_CRA_MBR_INFO",
    operation="todz_api_web_schedule",   # 대문자로 부르면 코드 12
    dataset_pk="15107871",
    optional=["schdl_yr", "schdl_mm"],
    note=(
        "**빈 API 다.** 개최일정을 여기서 얻을 생각이었으나, 연도·월을 어떻게 "
        "주든(무조건 포함해서) totalCount 가 0 이다 — 2024·2025·2026 전부. "
        "정상 응답(코드 00)에 자료만 없으므로 권한 문제가 아니다. "
        "**개최일은 출주표의 race_ymd 로 잡는다.** 남겨 두는 것은 나중에 자료가 "
        "채워졌을 때 알아채기 위해서다(수집기가 0건을 기록한다)."
    ),
)

RACE_VIDEO = Endpoint(
    key="race_video",
    title="경주동영상",
    service="SRVC_WEB_CRA_MBR_INFO",
    operation="TODZ_API_CYCLE_RACE_VIDEO",
    dataset_pk="15107871",
    optional=["race_ymd", "stnd_yr", "week_tcnt", "mbr_no", "day_tcnt"],
    note="경주 다시보기 URL. race_ymd 가 있어 회차↔날짜 대조에도 쓸 수 있다.",
)

START_PHOTO = Endpoint(
    key="start_photo",
    title="스타트사진",
    service="SRVC_WEB_CRA_MBR_INFO",
    operation="TODZ_API_WEB_STRT_PHOTO",
    dataset_pk="15107871",
    optional=["race_sn", "stnd_yr", "week_tcnt", "mbr_no", "day_tcnt"],
    note="경주 상세 페이지용. mbr_no(경주장번호) 가 meet_nm 과 어떻게 대응되는지 확인 필요.",
)

NEWS = Endpoint(
    key="news",
    title="경륜뉴스",
    service="SRVC_WEB_CRA_MBR_INFO",
    operation="todz_api_web_cycle_news",   # 대문자로 부르면 코드 12
    dataset_pk="15107871",
    optional=["title_nm"],
    note="콘텐츠용. 248건이지만 2013년 스피돔뉴스 동영상이 대부분이라 최신성이 없다.",
)

# --- 운영정보(SRVC_OD_API_CRA_CYCLE_EXER) — 오퍼레이션 4개 ---

EXERCISE = Endpoint(
    key="exercise",
    title="경주일 조조연습현황",
    service="SRVC_OD_API_CRA_CYCLE_EXER",
    operation="TODZ_API_CRA_CYCLE_EXER_I",
    dataset_pk="15107870",
    required=["stnd_yr", "week_tcnt", "day_tcnt", "race_ymd"],
    note=(
        "경정의 '틸트각'에 해당하는 자리이길 바랐던 API. **기대는 낮춰야 한다** — "
        "선수별이 아니라 급별 인원수 집계(선발·우수·특선 × 트랙/로라/정비)다. "
        "경륜에는 선수 개인의 작전을 드러내는 공개 데이터가 없다. "
        "**race_ymd 는 8자리(20260102)여야 한다** — 출주표가 주는 '2026.01.02' 를 "
        "그대로 넘기면 오류 없이 0건이 온다(NORM_YMD 가 막는다)."
    ),
)

INSPECT = Endpoint(
    key="inspect",
    title="검차 운영정보",
    service="SRVC_OD_API_CRA_CYCLE_EXER",
    operation="TODZ_API_CRA_INSPECT_I",
    dataset_pk="15107870",
    required=["stnd_yr", "week_tcnt"],
    note="회차 단위 점검 횟수 합계. 개별 경주 예측 기여는 거의 없다.",
)

INOUT = Endpoint(
    key="inout",
    title="자전거 보관현황",
    service="SRVC_OD_API_CRA_CYCLE_EXER",
    operation="TODZ_API_CRA_INOUT_I",
    dataset_pk="15107870",
    required=["stnd_yr", "week_tcnt", "day_tcnt"],
    note="보관·출고 대수. 참고용.",
)

CYCLE_PART = Endpoint(
    key="cycle_part",
    title="자전거 부품정보",
    service="SRVC_OD_API_CRA_CYCLE_EXER",
    operation="TODZ_API_CRA_CYCLE_PART_I",
    dataset_pk="15107870",
    optional=["mstr_unit_nm", "salv_unit_nm"],
    note="부품 분류 마스터. 해설용.",
)


REGISTRY: Dict[str, Endpoint] = {
    e.key: e
    for e in (
        # 예측 파이프라인 핵심
        RACE_CARD, RACE_RANK, PAYOFF, RACE_RESULT,
        # 보강
        TMS_SCORE, DOWN_ACCIDENT, SANCTION, OPPO_WIN, RACER_INFO, BKNO_SUM,
        # 편성·콘텐츠
        SCHEDULE, RACE_VIDEO, START_PHOTO, NEWS,
        # 운영
        EXERCISE, INSPECT, INOUT, CYCLE_PART,
    )
}

# 이것들이 없으면 파이프라인이 성립하지 않는다.
#   출주표 = 피처, 경주결과순위 = 레이블, 배당률 = 검증
REQUIRED_KEYS = ["race_card", "race_rank", "payoff"]

# 파라미터 이름 규칙이 두 갈래다. 한쪽 이름으로 통일해 쓰고 호출 직전에 바꾼다.
# (이걸 호출부마다 손으로 맞추면 언젠가 한 곳을 빠뜨리고, 그 API 만 조용히
#  0건을 돌려준다 — 파라미터가 전부 '옵션'이라 오류조차 나지 않는다.)
ALT_PARAM_NAMES = {
    "stnd_yr": "stnd_year",
    "week_tcnt": "tms",
    "day_tcnt": "day_ord",
    "race_ymd": "race_day",
}
ALT_PARAM_KEYS = {"race_rank", "down_accident"}

# **2자리 0채움이 필요한 파라미터.** 오류가 아니라 빈 결과로 실패하므로
# 호출부마다 손으로 맞추면 언젠가 한 곳을 빠뜨리고 그 API 만 조용히 비게 된다.
#
# 실측:
#   race_no    "01" → 1건,   "1" → 0건   (경주결과)
#   week_tcnt  "09" → 448건, "9" → 0건   (출주표)
#   tms        "09" → 182건, "9" → 0건   (경주결과순위)
#
# 회차가 **한 자리일 때만** 터진다는 것이 이 버그의 고약한 점이다. 1~9회차는
# 연초 두 달이라, 회차별로 훑는 수집기를 여름에 만들면 통과하고 이듬해 1월에
# 조용히 빈다. 두 자리 회차만 보고 '되는구나' 하고 넘어가기 딱 좋다.
#
# 반대로 padding 이 해가 되는 API 는 없었다 — 경주결과·회차별득점·배당률·
# 조조연습은 "1"·"01"·1 셋 다 같은 결과를 준다. 그래서 전역으로 건다.
ZERO_PAD_2 = ("race_no", "week_tcnt", "tms")

# 날짜 파라미터는 **8자리 숫자**여야 한다. 이것도 오류 없이 0건으로 실패한다.
# 응답 쪽 날짜 표기가 API 마다 다르기 때문에 반드시 필요하다 — 실측:
#   출주표     race_ymd = "2026.01.02"   (점 구분)
#   배당률     race_ymd = "20260102"     (8자리)
#   경주결과   race_ymd = "0102"         (월일만! 연도는 stnd_yr 에서 온다)
#   경주결과순위 race_day = "20260613"   (8자리)
# 출주표에서 배운 좌표를 조조연습현황에 그대로 넘기면 점 때문에 0건이 된다.
YMD_FIELDS = ("race_ymd", "race_day")


def norm_ymd(value: object, year: object = None) -> str:
    """날짜 표기를 8자리 ``YYYYMMDD`` 로 통일한다.

    월일 4자리만 온 경우(경주결과)는 ``year`` 를 붙여야 완성된다. 연도를 모르면
    4자리 그대로 돌려준다 — 여기서 임의의 연도를 지어내면 그 행은 조용히 다른
    해의 경주가 된다.
    """
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) == 4 and year:
        y = "".join(ch for ch in str(year) if ch.isdigit())
        if len(y) == 4:
            return y + digits
    return digits


def to_api_params(key: str, params: Dict[str, object]) -> Dict[str, object]:
    """내부 표준 파라미터명·표기를 해당 API 가 받는 형태로 바꾼다.

    이름을 **먼저** 바꾸고 0채움을 나중에 건다. 순서를 뒤집으면 ``week_tcnt`` 만
    채워지고 이름이 ``tms`` 로 바뀐 쪽은 그냥 지나가, 경주결과순위가 1~9회차에서
    조용히 빈다.
    """
    out = dict(params)
    for name in YMD_FIELDS:
        v = out.get(name)
        if v not in (None, ""):
            out[name] = norm_ymd(v, out.get("stnd_yr") or out.get("stnd_year"))
    if key in ALT_PARAM_KEYS:
        out = {ALT_PARAM_NAMES.get(k, k): v for k, v in out.items()}
    for name in ZERO_PAD_2:
        v = out.get(name)
        if v not in (None, ""):
            out[name] = f"{int(v):02d}" if str(v).strip().isdigit() else v
    return out


# ---------------------------------------------------------------------------
# 프로브 결과 캐시
# ---------------------------------------------------------------------------

def load_resolved(path: Optional[Path] = None) -> Dict[str, str]:
    p = path or RESOLVED_PATH
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("paths", {})
        except (ValueError, OSError) as e:
            log.warning("확정 경로 캐시를 읽지 못했습니다 (%s): %s", p, e)
    return {}


def save_resolved(paths: Dict[str, str], meta: Optional[dict] = None,
                  path: Optional[Path] = None) -> Path:
    p = path or RESOLVED_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"paths": paths, "meta": meta or {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


def resolve(key: str) -> str:
    """호출 경로. 명세에서 그대로 왔으므로 후보를 두지 않는다."""
    return REGISTRY[key].path
