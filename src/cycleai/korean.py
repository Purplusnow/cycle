"""한글 조사 붙이기.

이걸 안 하면 화면에 '김용진 가 7착' 이나 '선행 으로 앞에 섰다' 가 나간다.
문장 하나가 어색한 것으로 끝나지 않는다 — **자동으로 만든 티가 나는 순간,
옆에 있는 숫자도 대충 만든 것처럼 읽힌다.**

전개 대본(simulate)과 화면(site) 양쪽이 같은 규칙을 써야 하므로 여기 한 곳에만
둔다.
"""

from __future__ import annotations


def josa_of(word: object, pair: str = "이/가") -> str:
    """조사만 돌려준다. 앞말을 태그로 감싸야 할 때 쓴다.

    ``pair`` 는 '받침 있음/받침 없음' 순서다 — ``"이/가"``, ``"은/는"``,
    ``"을/를"``, ``"으로/로"``.
    """
    a, _, b = pair.partition("/")
    s = str(word or "").rstrip()
    if not s:
        return ""
    last = s[-1]
    if not ("가" <= last <= "힣"):
        # 숫자·영문·기호는 받침을 알 수 없다. 흔한 쪽으로 둔다.
        return b
    jong = (ord(last) - 0xAC00) % 28
    # '로/으로' 만 예외다. ㄹ 받침(jong == 8)은 '로' 를 쓴다 — '서울로'.
    if a.endswith("로") and jong == 8:
        return b
    return a if jong else b


def josa(word: object, pair: str = "이/가") -> str:
    """앞말에 조사를 붙여 돌려준다."""
    return f"{str(word or '').rstrip()}{josa_of(word, pair)}"
