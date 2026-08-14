"""API 응답을 DB 행으로 정규화한다.

경륜 API 는 사람이 읽을 화면을 그대로 내보낸 필드가 많다. 발주시각이
``12"55`` 로, 200m 기록이 ``11"78`` 로 **같은 구분자에 다른 뜻**으로 오고,
최근 성적은 ``"선발 5-1"`` 처럼 등급·경주번호·착순 셋이 한 칸에 뭉쳐 있다.
이런 값을 파서 없이 그대로 피처에 넣으면 조용히 문자열 범주가 되어, 모델은
``"선발 5-1"`` 과 ``"선발 5-2"`` 를 아무 관계 없는 두 값으로 배운다.

파싱 규칙을 여기 한 곳에 모으는 이유는 같은 값을 두 군데서 다르게 읽는 사고를
막기 위해서다. 원본은 ``raw_json`` 에 그대로 남기므로, 이 규칙이 틀렸다는 게
나중에 드러나도 API 를 다시 때리지 않고 로컬에서 다시 만들 수 있다.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .endpoints import norm_meet, norm_ymd

# 성적표는 배번을 원 문자로 쓴다. ⑦임  섭 / (7-3-4) 처럼 착순·배당 어디에나
# 섞여 나오므로 한 곳에서 숫자로 되돌린다.
CIRCLED = {c: i for i, c in enumerate("①②③④⑤⑥⑦⑧⑨", start=1)}

# 경주 급. 최근 성적 칸의 앞머리(``"선발"``·``"선준"``·``"선결"``)는
# <급 첫 글자> + <라운드 첫 글자> 로 붙어 온다.
GRADE_HEAD = {"선": "선발", "우": "우수", "특": "특선"}
ROUND_HEAD = {"발": "예선", "준": "준결승", "결": "결승"}

# 경륜장 약칭 ↔ 정식명. 최근 성적 칸은 한 글자로만 온다(``"광"``·``"창"``·``"부"``).
VENUE_ABBR = {"광": "광명", "창": "창원", "부": "부산"}


def _s(v: Any) -> str:
    return "" if v is None else str(v).strip()


def to_int(v: Any) -> Optional[int]:
    t = _s(v)
    if not t:
        return None
    m = re.search(r"-?\d+", t)
    return int(m.group()) if m else None


def to_float(v: Any) -> Optional[float]:
    t = _s(v)
    if not t:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    return float(m.group()) if m else None


def race_key(stnd_yr: Any, meet: Any, week: Any, day: Any, race_no: Any) -> Optional[str]:
    """경주 식별자.

    **경륜장을 키에 넣는다.** 광명·창원·부산이 같은 날 같은 회차 같은 경주번호를
    각자 쓰기 때문에, 경정처럼 (연도·회차·일차·경주번호) 만으로 키를 만들면 세
    경주가 한 행에서 서로를 덮어쓴다.

    날짜를 키로 쓰지 않는 이유는 API 마다 날짜 표기가 제각각이고(점 구분·월일만·
    8자리) 아예 없는 곳도 있기 때문이다. 좌표를 키로 두면 어느 API 로 들어온
    조각이든 같은 행에 붙는다.
    """
    yr, wk, dy, rn = (to_int(stnd_yr), to_int(week), to_int(day), to_int(race_no))
    m = norm_meet(meet)
    if None in (yr, wk, dy, rn) or not m:
        return None
    return f"{yr}-{m}-{wk:02d}-{dy}-{rn:02d}"


def coords_from_key(key: str) -> Dict[str, Any]:
    """``"2026-광명-24-1-03"`` → 좌표 컬럼들.

    착순으로만 알게 되는 경주(출주표가 없는 창원·부산)도 races 에 넣어야 하는데
    좌표 컬럼이 NOT NULL 이다. 키에 이미 좌표가 들어 있으므로 되짚어 채운다 —
    출주표가 없다고 그 경주를 통째로 버리면 착순도 함께 사라지고, 그 착순은
    광명 선수의 원정 성적이라 폼 피처로 쓸 값이다.
    """
    parts = key.split("-")
    if len(parts) != 5:
        return {}
    return {"race_key": key, "stnd_yr": int(parts[0]), "meet_nm": parts[1],
            "week_tcnt": int(parts[2]), "day_tcnt": int(parts[3]),
            "race_no": int(parts[4])}


# ---------------------------------------------------------------------------
# 출주표 (경주 전 확정 정보)
# ---------------------------------------------------------------------------

def parse_clock(v: Any) -> Optional[str]:
    """``'12"55'`` → ``"12:55"``. 발주시각이다.

    한 경주의 일곱 명이 모두 같은 값을 갖고 경주번호를 따라 커진다 — 그래서
    시각이라고 판정했다. 같은 ``"`` 구분자를 쓰는 ``rec_200m_scr`` 은 초 단위
    기록이므로 **다른 파서로 읽는다**. 구분자가 같다는 이유로 한 함수에 몰면
    발주시각 12시 55분이 12.55초가 된다.
    """
    t = _s(v)
    m = re.match(r"^(\d{1,2})\D+(\d{2})$", t)
    if not m:
        return t or None
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def parse_seconds(v: Any) -> Optional[float]:
    """``'11"78'`` → ``11.78``. 200m 기록(초)."""
    t = _s(v)
    m = re.match(r"^(\d{1,2})\D+(\d{1,2})$", t)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    return to_float(t)


def parse_past_rank(v: Any) -> Dict[str, Any]:
    """``"선발 5-1"`` → 등급·라운드·경주번호·착순.

    **경주번호가 앞, 착순이 뒤다.** 명세에 설명이 없어 실측으로 확정했다 —
    2026-05-23 강석호 ``"선발 5-1"`` 은 경주결과순위에서 부산 5경주 1착이었고,
    같은 날 홍석헌 ``"선발 3-1"`` 은 광명 3경주 1착, 신호재 ``"선발 3-7"`` 은
    광명 3경주 7착이었다. 뒤집어 읽으면 7착을 1착으로 배우게 된다.

    ``"결  장"`` 은 그 일차에 뛰지 않았다는 뜻이다. 0 이나 7 로 채우면 안 된다 —
    결장은 나쁜 성적이 아니라 성적 없음이다.
    """
    t = _s(v)
    if not t or "결" in t and "장" in t and "-" not in t:
        return {"rank": None, "race_no": None, "grade": None, "round": None,
                "absent": bool(t)}
    head, _, tail = t.partition(" ")
    grade = GRADE_HEAD.get(head[:1]) if head else None
    rnd = ROUND_HEAD.get(head[1:2]) if len(head) > 1 else None
    m = re.search(r"(\d+)\s*-\s*(\d+)", t)
    if not m:
        return {"rank": None, "race_no": None, "grade": grade, "round": rnd,
                "absent": False}
    return {"race_no": int(m.group(1)), "rank": int(m.group(2)),
            "grade": grade, "round": rnd, "absent": False}


def entry_row(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """출주표 한 줄 → entries 행."""
    key = race_key(rec.get("stnd_yr"), rec.get("meet_nm"), rec.get("week_tcnt"),
                   rec.get("day_tcnt"), rec.get("race_no"))
    if not key:
        return None

    row: Dict[str, Any] = {
        "race_key": key,
        "back_no": to_int(rec.get("back_no")),
        "racer_nm": _s(rec.get("racer_nm")) or None,
        "racer_grd": _s(rec.get("racer_grd_cd")) or None,
        "grade_cur": _s(rec.get("racer_grd_cur_cd")) or None,
        "grade_bef": _s(rec.get("racer_grd_bef_cd")) or None,
        "period_no": to_int(rec.get("period_no")),
        "age": to_int(rec.get("racer_age")),
        "trng_plc": _s(rec.get("trng_plc_nm")) or None,
        "color_nm": _s(rec.get("color_nm")) or None,

        "gear_rate": to_float(rec.get("gear_rate")),
        "rec_200m": parse_seconds(rec.get("rec_200m_scr")),

        # 전법별 입상 횟수 — 경륜에서 승부를 가르는 축
        "pre_win_cnt": to_int(rec.get("pre_win_cnt")),
        "pas_win_cnt": to_int(rec.get("pas_win_cnt")),
        "brk_win_cnt": to_int(rec.get("brk_win_cnt")),
        "mrk_win_cnt": to_int(rec.get("mrk_win_cnt")),
        "win_tot_cnt": to_int(rec.get("win_tot_tcnt")),
        "run_day_cnt": to_int(rec.get("run_day_tcnt")),

        "win_rate": to_float(rec.get("win_rate")),
        "high_rate": to_float(rec.get("high_rate")),
        "high_3_rate": to_float(rec.get("high_3_rate")),
        "tot_avg_scr": to_float(rec.get("tot_tms_avg_scr")),
        "area_avg_scr": to_float(rec.get("area_tms3_avg_scr")),
    }

    # 최근 3회전 × 3일차 착순
    ranks: List[int] = []
    for i in (1, 2, 3):
        # 1회전만 필드명이 다르다 (bf1_meet_nm_nm). 명세의 오타를 그대로 받는다.
        meet_field = "bf1_meet_nm_nm" if i == 1 else f"bf{i}_meet_nm"
        row[f"bf{i}_meet"] = VENUE_ABBR.get(_s(rec.get(meet_field)),
                                            _s(rec.get(meet_field))) or None
        row[f"bf{i}_ymd"] = _s(rec.get(f"bf{i}_day1_ymd")) or None
        for d in (1, 2, 3):
            p = parse_past_rank(rec.get(f"bf{i}_day{d}_rank"))
            row[f"bf{i}_d{d}"] = p["rank"]
            if p["rank"]:
                ranks.append(p["rank"])
    if ranks:
        row["bf_avg"] = sum(ranks) / len(ranks)
        row["bf_cnt"] = len(ranks)
    return row


def race_row_from_entry(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """출주표 한 줄에서 경주 단위 메타를 뽑는다."""
    key = race_key(rec.get("stnd_yr"), rec.get("meet_nm"), rec.get("week_tcnt"),
                   rec.get("day_tcnt"), rec.get("race_no"))
    if not key:
        return None
    return {
        "race_key": key,
        "stnd_yr": to_int(rec.get("stnd_yr")),
        "meet_nm": norm_meet(rec.get("meet_nm")),
        "week_tcnt": to_int(rec.get("week_tcnt")),
        "day_tcnt": to_int(rec.get("day_tcnt")),
        "race_no": to_int(rec.get("race_no")),
        "race_ymd": norm_ymd(rec.get("race_ymd")) or None,
        "post_time": parse_clock(rec.get("dptre_tm")),
        "race_grade": _s(rec.get("racer_grd_cd")) or None,
        "race_len": to_int(rec.get("race_len")),
        "round_cnt": to_int(rec.get("round_cnt")),
    }


# ---------------------------------------------------------------------------
# 착순 (경주결과순위)
# ---------------------------------------------------------------------------

def result_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """착순 응답 → results 행.

    배번은 여기 없다. 출주표에만 있으므로 저장 후 ``store.link_result_back_no``
    가 경주 안에서 선수명으로 이어 붙인다. 이어붙이지 못한 행도 남긴다 —
    조용히 버리면 창원·부산 착순(=광명 선수의 원정 성적)이 통째로 사라진다.
    """
    out = []
    for rec in records:
        key = race_key(rec.get("stnd_year") or rec.get("stnd_yr"),
                       rec.get("meet_nm"),
                       rec.get("tms") or rec.get("week_tcnt"),
                       rec.get("day_ord") or rec.get("day_tcnt"),
                       rec.get("race_no"))
        name = _s(rec.get("racer_nm"))
        if not key or not name:
            continue
        out.append({
            "race_key": key,
            "racer_nm": name,
            "racer_no": _s(rec.get("racer_no")) or None,
            # 착순은 실격·기권 때 비거나 숫자가 아닐 수 있다. 숫자만 ord 로 두고
            # 원문은 note 에 남긴다 — 실격을 7착으로 바꿔 세면 안 된다.
            "ord": to_int(rec.get("race_rank")),
            "ord_note": _s(rec.get("race_rank")) or None,
        })
    return out


def race_rows_from_result(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """착순 응답에서 경주 메타(특히 실제 경주일자)를 만든다."""
    seen: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        key = race_key(rec.get("stnd_year"), rec.get("meet_nm"), rec.get("tms"),
                       rec.get("day_ord"), rec.get("race_no"))
        if not key or key in seen:
            continue
        row = coords_from_key(key)
        row["race_ymd"] = norm_ymd(rec.get("race_day")) or None
        seen[key] = row
    return list(seen.values())


# ---------------------------------------------------------------------------
# 배당
# ---------------------------------------------------------------------------

# 배당률 API 의 승식 칸. 이 여섯이 검증에 쓰는 전부다.
PAYOFF_FIELDS = {
    "pool1_val": "단승",
    "pool2_1_val": "연승1",
    "pool2_2_val": "연승2",
    "pool4_val": "쌍승",
    "pool5_val": "복승",
    "pool6_val": "삼복승",
}

# 경주결과 API 는 같은 배당을 조합과 함께 주고, **칸이 둘 더 있다**(pool7·pool8).
# 둘 다 1~3착 조합이 붙어 오는데 배당값은 삼복승과 다르다. 어느 승식인지
# 명세에도 응답에도 적혀 있지 않아 **이름을 지어내지 않았다** — 이름을 잘못
# 붙이면 나중에 그 이름으로 회수율을 계산하게 되고, 틀렸다는 걸 알 길이 없다.
RESULT_POOL_FIELDS = {
    "pool1_val": "단승",
    "pool2_val": "연승",
    "pool4_val": "쌍승",
    "pool5_val": "복승",
    "pool6_val": "삼복승",
    "pool7_val": "pool7",
    "pool8_val": "pool8",
}

# 배당률 API 는 경륜장을 주지 않는다. 출주표가 광명뿐이고 건수도 광명 경주 수와
# 맞아떨어져(2026년 1,568건 ↔ 광명 1,603경주) 광명으로 본다. 창원·부산 배당은
# 공개 API 에 없다 — 그쪽은 애초에 예측 대상이 아니다(출주표가 없다).
PAYOFF_MEET = "광명"


def parse_pool_value(v: Any) -> tuple:
    """``"(7-3-4)4.4"`` → ("7-3-4", 4.4). 숫자만 오면 조합은 None.

    경주결과 API 는 조합과 배당을 한 문자열로 붙여 주고, 배당률 API 는 배당만
    준다. 두 소스를 같은 표에 넣기 위해 여기서 형태를 맞춘다.
    """
    t = _s(v)
    if not t:
        return None, None
    combo = re.search(r"\(([\d\-]+)\)", t)
    if combo:
        return combo.group(1), to_float(t[combo.end():])
    lanes = [str(CIRCLED[c]) for c in t if c in CIRCLED]
    payout = to_float(re.sub(r"[①-⑨\-\s]", "", t))
    return ("-".join(lanes) if lanes else None), payout


def payoff_rows(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """배당률 응답 한 줄 → payoffs 행들 (승식별)."""
    key = race_key(rec.get("stnd_yr"), PAYOFF_MEET, rec.get("week_tcnt"),
                   rec.get("day_tcnt"), rec.get("race_no"))
    if not key:
        return []
    out = []
    for field, pool in PAYOFF_FIELDS.items():
        combo, payout = parse_pool_value(rec.get(field))
        if payout is None:
            continue
        out.append({"race_key": key, "pool": pool, "combo": combo, "payout": payout})
    return out


def result_payoff_rows(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """경주결과 응답 한 줄 → payoffs 행들. 조합(배번)까지 함께 온다.

    연승은 한 칸에 둘이 붙어 온다(``"(7)1.0 (3)1.6"``). 나눠서 연승1·연승2 로
    넣어 배당률 API 와 같은 모양으로 맞춘다.
    """
    key = race_key(rec.get("stnd_yr"), rec.get("meet_nm"), rec.get("week_tcnt"),
                   rec.get("day_tcnt"), rec.get("race_no"))
    if not key:
        return []
    out = []
    for field, pool in RESULT_POOL_FIELDS.items():
        raw = _s(rec.get(field))
        if not raw:
            continue
        if pool == "연승":
            parts = re.findall(r"\(([\d\-]+)\)\s*([\d.]+)", raw)
            for i, (combo, payout) in enumerate(parts[:2], start=1):
                out.append({"race_key": key, "pool": f"연승{i}",
                            "combo": combo, "payout": to_float(payout)})
            continue
        combo, payout = parse_pool_value(raw)
        if payout is None:
            continue
        out.append({"race_key": key, "pool": pool, "combo": combo, "payout": payout})
    return out


def result_top3(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """경주결과의 ``rank1~3`` (``"⑦임  섭"``) → 착순 행. 배번이 함께 온다.

    이름에 정렬용 공백이 박혀 있다(``"임  섭"``). 그대로 두면 착순 API 의
    ``"임섭"`` 과 다른 사람이 된다 — 공백을 걷어 낸다.
    """
    key = race_key(rec.get("stnd_yr"), rec.get("meet_nm"), rec.get("week_tcnt"),
                   rec.get("day_tcnt"), rec.get("race_no"))
    if not key:
        return []
    out = []
    for i in (1, 2, 3):
        t = _s(rec.get(f"rank{i}"))
        if not t:
            continue
        back_no = next((CIRCLED[c] for c in t if c in CIRCLED), None)
        name = re.sub(r"\s+", "", re.sub(r"[①-⑨]", "", t))
        if not name:
            continue
        out.append({"race_key": key, "back_no": back_no, "racer_nm": name, "ord": i})
    return out


def race_row_from_result(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    key = race_key(rec.get("stnd_yr"), rec.get("meet_nm"), rec.get("week_tcnt"),
                   rec.get("day_tcnt"), rec.get("race_no"))
    if not key:
        return None
    row = coords_from_key(key)
    # 경주결과의 race_ymd 는 월일 4자리다. 연도를 붙여야 날짜가 된다.
    row["race_ymd"] = norm_ymd(rec.get("race_ymd"), rec.get("stnd_yr")) or None
    return row


# ---------------------------------------------------------------------------
# 선수·부가 자료
# ---------------------------------------------------------------------------

def racer_year_row(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """선수정보 한 줄 → racer_year 행.

    명세에 응답 정의가 없어 '표시용'으로 낮춰 봤던 API 인데, 실제로는 연도별
    착순 분포까지 있는 집계다. 신인·복귀 선수의 사전값으로 쓴다.
    """
    name = _s(rec.get("racer_nm"))
    yr = to_int(rec.get("stnd_yr"))
    if not name or not yr:
        return None
    row = {"stnd_yr": yr, "racer_nm": name,
           "period_no": to_int(rec.get("period_no")),
           "grade": _s(rec.get("racer_grd_cd")) or None,
           "run_cnt": to_int(rec.get("run_cnt")),
           "run_day_cnt": to_int(rec.get("run_day_tcnt")),
           "win_rate": to_float(rec.get("win_rate")),
           "high_rate": to_float(rec.get("high_rate")),
           "high_3_rate": to_float(rec.get("high_3_rate")),
           "down_cnt": to_int(rec.get("down_po_cnt")),
           "elim_cnt": to_int(rec.get("elim_tcnt")),
           "go_po_cnt": to_int(rec.get("go_po_tcnt"))}
    # 7명 경주인데 8·9착 칸이 있다. 실격·기권 자리로 보이므로 그대로 남긴다 —
    # 없는 셈 치면 '출전 수 = 착순 합' 이 맞지 않는 이유를 나중에 못 찾는다.
    for i in range(1, 10):
        row[f"rank{i}"] = to_int(rec.get(f"rank{i}_tcnt"))
    return row


def tms_score_row(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = _s(rec.get("racer_nm"))
    yr, wk, dy = (to_int(rec.get("stnd_yr")), to_int(rec.get("week_tcnt")),
                  to_int(rec.get("day_tcnt")))
    meet = norm_meet(rec.get("meet_nm"))
    if not (name and yr and wk and dy and meet):
        return None
    return {"stnd_yr": yr, "week_tcnt": wk, "day_tcnt": dy, "meet_nm": meet,
            "racer_nm": name, "race_scr": to_float(rec.get("race_scr"))}


def accident_rows(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """낙차사고 한 줄 → 관련 선수별 행.

    **필드명이 거짓말한다.** ``racer_no1``~``racer_no7`` 의 값은 번호가 아니라
    그 경주 일곱 명의 **이름**이고(배번 순), ``down_indu<N>_yn`` 은 Y/N 이 아니라
    N번 배번 선수가 유발자임을 뜻하는 사유 문자열이다. ``leav<N>_cd`` 는 그
    선수의 후송 여부다.

    경륜장이 없다. 광명이 경주의 8할이라 실용적으로는 광명으로 봐도 되지만,
    그렇게 단정하면 창원 낙차가 광명 경주에 붙는다. 그래서 **경주에 붙이지 않고
    선수 단위 이력으로만 쓴다.**
    """
    yr, wk, dy, rn = (to_int(rec.get("stnd_year")), to_int(rec.get("tms")),
                      to_int(rec.get("day_ord")), to_int(rec.get("race_no")))
    if None in (yr, wk, dy, rn):
        return []
    kind = _s(rec.get("down_acdnt_cd")) or None
    out = []
    for i in range(1, 8):
        name = re.sub(r"\s+", "", _s(rec.get(f"racer_no{i}")))
        if not name:
            continue
        reason = _s(rec.get(f"down_indu{i}_yn")) or None
        leave = _s(rec.get(f"leav{i}_cd")) or None
        # 유발자도 후송자도 아니면 그냥 같은 경주에 있었을 뿐이다. 그것까지
        # 사고 이력으로 세면 한 번의 낙차가 일곱 명의 이력이 된다.
        if not reason and not leave:
            continue
        out.append({"stnd_yr": yr, "week_tcnt": wk, "day_tcnt": dy, "race_no": rn,
                    "racer_nm": name, "kind": kind,
                    "reason": " ".join(x for x in (reason, leave) if x) or None})
    return out


# 제재 사유 문장에 해당 경주가 통째로 박혀 있다: "26년광명제25회2일차15경주"
SANCTION_RACE = re.compile(
    r"(\d{2})년\s*(\S+?)\s*제?(\d+)회\s*(\d+)일차\s*(\d+)경주")


def sanction_row(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = _s(rec.get("racer_nm"))
    kind = _s(rec.get("p_kind"))
    if not name or not kind:
        return None
    reason = _s(rec.get("p_johang"))
    # <BR> 과 개행이 섞인 덩어리다. 화면에 그대로 실으면 태그가 보인다.
    reason = re.sub(r"<[^>]+>", "\n", reason)
    reason = re.sub(r"\s*\n\s*", "\n", reason).strip()
    m = SANCTION_RACE.search(reason)
    race_ref = None
    if m:
        yy, meet, wk, dy, rn = m.groups()
        race_ref = f"20{yy}-{meet}-{int(wk):02d}-{int(dy)}-{int(rn):02d}"
    return {"racer_id": _s(rec.get("racer_id")) or None, "racer_nm": name,
            "kind": kind, "period": _s(rec.get("p_period")) or None,
            "reason": reason or None, "race_ref": race_ref}


def oppo_row(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    a, b = _s(rec.get("racer_nm")), _s(rec.get("oppo_racer_nm"))
    yr = to_int(rec.get("stnd_yr"))
    if not (a and b and yr):
        return None
    return {"stnd_yr": yr, "racer_nm": a, "oppo_nm": b,
            "win_cnt": to_int(rec.get("win_tcnt")),
            "lose_cnt": to_int(rec.get("lose_tcnt")),
            "draw_cnt": to_int(rec.get("draw_tcnt")),
            "same_win_cnt": to_int(rec.get("same_win_tcnt"))}
