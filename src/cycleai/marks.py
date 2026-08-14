"""예상 기호.

기호 규칙은 화면(site)과 검증(verify) 양쪽이 똑같이 써야 한다. 한쪽에만 두면
언젠가 갈리고, 그때 '기호별 실적'은 화면에서 벌어지는 일과 무관한 숫자가 된다.
그래서 규칙은 여기 한 곳에만 둔다.

경륜은 일곱 명이 나온다. 상위 넷에만 기호를 주고 셋은 비운다 — 일곱 중 다섯에
기호가 붙으면 기호가 아무것도 가리지 않는다.
"""

from __future__ import annotations

from typing import Dict, List

# 기본        ◎ ○ ▲ △
# 우세 뚜렷    ★ ○ ▲ △
#
# 자리를 고정하면 기호는 '절대적 약속'이 아니라 **경주 안에서의 상대 순위**가
# 된다. 절대 신호는 ★ 하나가 맡는다.
MARK_SEQUENCE = ["◎", "○", "▲", "△"]
MARK_SEQUENCE_STAR = ["★", "○", "▲", "△"]
MARK_LIMIT = len(MARK_SEQUENCE)

# ★ 기준은 손으로 정하지 않고 과거 기록에서 보정했다.
# (``python -m cycleai.calibrate`` · v1-oos 워크포워드 18,968경주 ·
#  무작위는 1착 14.3% / 2착이내 28.6%, 1순위 전체 평균은 1착 60.4% / 2착이내 79.1%)
#
#   기준   ★출현   ★1착   ★2착이내
#   0.80    50%   72.9%    87.3%
#   0.85    36%   76.2%    89.2%
#   0.90    25%   79.8%    91.1%
#   0.93    19%   81.8%    92.0%   ← 채택
#   0.95    16%   83.0%    92.6%
#   0.96    14%   83.3%    92.7%
#
# **경정에서 쓰던 0.70 을 그대로 가져오면 안 된다.** 경륜은 예측이 잘 되는
# 종목이라 그 기준으로는 ★ 가 전체의 70% 넘게 붙는다 — 일곱 경주 중 다섯에
# 붙는 표시는 아무것도 가리지 않는다.
#
# 0.93 이 곡선의 무릎이다. 더 올리면 1착이 1.5%p 오르는 대신 출현이 19%→14% 로
# 줄어 하루 16경주에서 3개가 2개가 되고, 내리면 흔해져 '우세가 뚜렷'이
# 무색해진다. 19% 면 경주일마다 세 번쯤 나온다 — 드물지도 흔하지도 않다.
MARK_THRESHOLDS = {"star": 0.93}

# 기호 자체가 표기이므로 화면에 이름은 붙이지 않는다. 다만 처음 보는 사람을 위해
# '무엇을 뜻하는가'는 범례로 남긴다 — 이름이 아니라 뜻이다.
MARK_MEANING = {
    "★": "우세가 뚜렷한 축",
    "◎": "축",
    "○": "상대",
    "▲": "복병",
    "△": "참고",
}


def assign_marks(runners: List[Dict]) -> None:
    """예상 기호를 붙인다 (제자리 수정).

    자리 수가 고정이므로 예측 순위대로 배분한다. 1순위가 2착 이내에 들 확률이
    충분히 높으면 ◎ 대신 ★ 를 준다 — 접전 경주와 한 명이 압도하는 경주를 같은
    기호로 적으면, 읽는 쪽은 둘을 구분할 방법이 없다.
    """
    ordered = sorted(runners, key=lambda r: r.get("pred_rank") or 99)
    top = ordered[0] if ordered else None
    star = bool(top and (top.get("p_top2") or 0.0) >= MARK_THRESHOLDS["star"])
    seq = MARK_SEQUENCE_STAR if star else MARK_SEQUENCE

    for i, r in enumerate(ordered):
        mark = seq[i] if i < MARK_LIMIT else ""
        r["mark"] = mark
        r["mark_meaning"] = MARK_MEANING.get(mark, "")
