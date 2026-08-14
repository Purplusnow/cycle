"""도메인 시각은 항상 한국 시간으로 읽는다.

경주일과 발주 시각은 KST 로 발표된다. ``date.today()`` 를 그대로 쓰면 UTC
환경에서만 날짜가 하루 어긋나고, 그 어긋남은 조용하다 — 예상이 하루 늦게
만들어지거나, 이미 끝난 경주가 '아직 발주 전'으로 취급돼 예측이 다시 쓰인다.

한국은 서머타임이 없으므로 고정 오프셋으로 충분하고, tzdata 설치 여부에
의존하지 않는다.
"""

from __future__ import annotations

import datetime as dt

KST = dt.timezone(dt.timedelta(hours=9), "KST")


def now_kst() -> dt.datetime:
    """현재 한국 시각 (naive — DB·API 문자열과 그대로 비교하기 위함)."""
    return dt.datetime.now(dt.timezone.utc).astimezone(KST).replace(tzinfo=None)


def today_kst() -> dt.date:
    """오늘 (한국 기준 경주일)."""
    return now_kst().date()


def recent_dates(n: int = 14, today: dt.date | None = None) -> list[str]:
    """최근 n일을 최신순 ``YYYYMMDD`` 로.

    **개최 요일을 코드에 박지 않는다.** 경륜은 경륜장마다 개최 요일이 다르고
    특별 편성도 붙는다. 경정은 미사리 한 곳이라 '화·수·목' 후보로 좁힐 수
    있었지만, 여기서 같은 짓을 하면 어느 경륜장 하루가 조용히 통째로 빠진다.
    실제 개최일은 연간경주일정 API(``schedule``)와 응답의 ``race_ymd`` 로
    확인한다.
    """
    today = today or today_kst()
    return [(today - dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(n)]
