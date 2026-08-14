"""공공데이터포털(data.go.kr) 경륜 오픈API 클라이언트.

경륜 API 는 국민체육진흥공단 경주사업총괄본부가 **GW(게이트웨이) 방식**으로
공개한다. 기관코드가 경정과 같은 ``B551014`` 라 규칙도 같고, 그래서 경정에서
겪은 함정이 여기서도 그대로 나온다:

  * 호스트에 **서비스 세그먼트가 포함**된다.
    ``/B551014/SRVC_OD_API_CRA_RACE_ORGAN/TODZ_API_CRA_RACE_ORGAN_I``
    짧은 경로로 부르면 게이트웨이가 서비스를 못 찾아 코드 12 를 돌려준다.
  * 응답 형식 파라미터가 ``_type`` 이 아니라 **``resultType``** 이고 **필수**다.
    빠지면 기본값 XML 로 응답해 JSON 파서가 조용히 빈 결과를 만든다.
  * ``pageNo`` · ``numOfRows`` 도 대부분 **필수**다.

그 밖에 포털 공통의 함정도 여기서 흡수한다:

  * 서비스키가 인코딩본/디코딩본 두 가지로 발급되는데, requests 가 한 번 더
    인코딩하면 ``SERVICE_KEY_IS_NOT_REGISTERED_ERROR`` 가 난다.
  * JSON 을 요청해도 오류 응답만은 XML 로 온다.
  * 정상 응답이어도 items 가 빈 문자열(``""``)로 오는 경우가 있다.
  * item 이 1건일 때 리스트가 아니라 dict 로 온다.
"""

from __future__ import annotations

import logging
import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import requests

log = logging.getLogger(__name__)

BASE = "https://apis.data.go.kr/B551014"

# 게이트웨이가 정상 처리했음을 뜻하는 결과코드들.
# GW 는 "00" 을 쓰지만 일부 오퍼레이션이 "0"/"INFO-0" 을 섞어 쓴다.
OK_CODES = {"00", "0", "INFO-0", "INFO-00", "INFO-000"}

# 재시도해도 의미가 없는(=설정·권한이 틀린) 결과코드.
# 여기에 잘못 넣으면 일시적 오류에 배치가 죽고, 빼먹으면 틀린 설정으로
# 4번씩 두드리며 일일 호출량만 태운다.
FATAL_CODES = {
    "12",   # NO_OPENAPI_SERVICE_ERROR — 호출 URL 이 틀렸다
    "20",   # SERVICE_ACCESS_DENIED / PERMISSION_DENIED — 활용신청 미승인
    "403",  # 미승인 API. GW 는 코드 20 이 아니라 HTTP 403 으로 돌려준다
    "401",
    "22",   # 일일 호출량 초과
    "29",   # 차단된 IP
    "30",   # SERVICE_KEY_IS_NOT_REGISTERED_ERROR
    "31",   # DEADLINE_HAS_EXPIRED_ERROR
    "336",  # 한 번에 1000건 이상 요청 불가
    "340",  # 필수 파라미터 누락
    "601",  # 잘못된 오퍼레이션
}

# 사람이 읽을 수 있는 진단. 코드만 던지면 무엇을 고쳐야 하는지 알 수 없다.
CODE_HINT = {
    "10": "요청 파라미터 값·형식이 올바르지 않다. 명세의 허용값을 확인하라.",
    "12": "호출 URL 이 틀렸다. GW 경로는 /B551014/<서비스명>/<오퍼레이션명> 이다.",
    "20": "활용신청이 승인되지 않았거나 권한이 없다. 포털 마이페이지에서 상태를 확인하라.",
    "22": "오늘 호출량을 다 썼다. 내일 이어서 하거나 트래픽 증설을 신청하라.",
    "23": "초당 호출량 초과. 호출 간격을 늘려야 한다.",
    "30": "등록되지 않은 인증키다. .env 의 KCYCLE_SERVICE_KEY 를 확인하라.",
    "401": "인증 실패. 키가 이 API 에 대해 유효하지 않다.",
    # 실측: 미승인 API 는 오류 코드가 아니라 HTTP 403 으로 돌아온다. 이걸
    # 일시적 오류로 보고 재시도하면 호출량만 태우고 진단도 틀어진다.
    "403": "이 API 는 활용신청이 승인되지 않았다. 포털에서 신청·승인 상태를 확인하라.",
    "31": "인증키 사용 기한이 만료됐다.",
    "340": "필수 요청 파라미터가 빠졌다 (pageNo·numOfRows·resultType 포함).",
}


def redact(text: str) -> str:
    """로그·예외 메시지에서 서비스키를 지운다.

    예외 메시지에 요청 URL 을 그대로 실으면 쿼리스트링의 serviceKey 가 그대로
    남는다. 크론 로그는 사람 눈에 잘 안 띄는 만큼 유출 시 발견도 늦다.
    """
    return re.sub(r"(serviceKey=)[^&\s]+", r"\1<redacted>", str(text))


class KcycleApiError(RuntimeError):
    """게이트웨이가 오류 코드를 반환했을 때."""

    def __init__(self, code: str, msg: str, url: str = ""):
        self.code = str(code)
        self.msg = redact(msg)
        self.url = redact(url)
        hint = CODE_HINT.get(self.code, "")
        super().__init__(
            f"[{self.code}] {self.msg}"
            + (f" — {hint}" if hint else "")
            + (f" ({self.url})" if url else "")
        )

    @property
    def fatal(self) -> bool:
        return self.code in FATAL_CODES


def read_service_key() -> str:
    """서비스키를 환경변수에서, 없으면 .env 에서 읽는다.

    환경변수를 우선하므로 CI 는 Secrets 만 넣으면 되고, 로컬은 .env(chmod 600)만
    두면 매번 export 하지 않아도 된다. 키 값 자체는 어떤 경로로도 로그에 남기지
    않는다.

    경정·경마와 같은 포털 계정이면 키가 같으므로 그쪽 이름도 받아 준다.
    """
    names = ("KCYCLE_SERVICE_KEY", "KBOAT_SERVICE_KEY", "KRA_SERVICE_KEY")
    for name in names:
        key = os.environ.get(name, "").strip()
        if key:
            return key
    for base in (Path.cwd(), Path(__file__).resolve().parents[3]):
        env = base / ".env"
        if not env.is_file():
            continue
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, val = line.partition("=")
            if name.strip() in names:
                return val.strip().strip("'\"")
    return ""


def normalize_service_key(raw: str) -> str:
    """인코딩/디코딩 어느 쪽으로 받아왔든 '디코딩본'으로 통일한다.

    포털은 같은 키를 Encoding/Decoding 두 형태로 보여준다. requests 의 params 에
    넣으면 다시 인코딩되므로, 항상 디코딩본을 넣어야 이중 인코딩을 피할 수 있다.
    """
    key = (raw or "").strip()
    if not key:
        raise ValueError(
            "서비스키가 비어 있습니다. .env 의 KCYCLE_SERVICE_KEY 를 확인하세요."
        )
    if re.search(r"%[0-9A-Fa-f]{2}", key):
        key = urllib.parse.unquote(key)
    return key


def _extract_xml_error(text: str) -> Optional[tuple]:
    """XML 형태 오류 응답에서 (코드, 메시지)를 뽑는다. 오류가 아니면 None."""
    head = text[:400]
    if "<" not in head:
        return None
    code = re.search(r"<returnReasonCode>\s*([^<]+)</returnReasonCode>", text)
    msg = re.search(r"<returnAuthMsg>\s*([^<]+)</returnAuthMsg>", text)
    if code:
        return code.group(1).strip(), (msg.group(1).strip() if msg else "unknown")
    # 서비스가 XML 로 정상 결과를 준 경우일 수 있으므로 resultCode 도 본다.
    code = re.search(r"<resultCode>\s*([^<]+)</resultCode>", text)
    msg = re.search(r"<resultMsg>\s*([^<]+)</resultMsg>", text)
    if code and code.group(1).strip() not in OK_CODES:
        return code.group(1).strip(), (msg.group(1).strip() if msg else "unknown")
    return None


def _as_list(items: Any) -> List[dict]:
    """items 필드를 항상 dict 리스트로 정규화."""
    if items in (None, "", [], {}):
        return []
    if isinstance(items, dict):
        inner = items.get("item", items)
        if isinstance(inner, dict):
            return [inner]
        if isinstance(inner, list):
            return [r for r in inner if isinstance(r, dict)]
        return []
    if isinstance(items, list):
        return [r for r in items if isinstance(r, dict)]
    return []


@dataclass
class KcycleClient:
    service_key: str
    # 연결과 응답을 따로 잡는다. 게이트웨이가 응답하지 않을 때 연결 단계에서
    # 20초씩 붙들리면 재시도까지 겹쳐 배치가 통째로 타임아웃된다.
    connect_timeout: float = 6.0
    timeout: float = 20.0
    max_retries: int = 4
    pause: float = 0.15  # 연속 호출 간격 (코드 23 = 초당 호출량 초과 방지)
    session: requests.Session = field(default_factory=requests.Session)
    _last_call: float = field(default=0.0, repr=False)

    @classmethod
    def from_env(cls, **kw) -> "KcycleClient":
        return cls(service_key=normalize_service_key(read_service_key()), **kw)

    def __post_init__(self):
        self.service_key = normalize_service_key(self.service_key)
        self.session.headers.update(
            {"User-Agent": "cycleai/1.0 (+data.go.kr open api client)"}
        )

    # ---------------------------------------------------------------- 저수준

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_call
        if gap < self.pause:
            time.sleep(self.pause - gap)
        self._last_call = time.monotonic()

    def raw(self, path: str, params: Dict[str, Any]) -> dict:
        """단일 페이지 호출. 성공 시 response.body 딕셔너리를 돌려준다.

        ``path`` 는 ``<서비스명>/<오퍼레이션명>`` 형태다 (BASE 이후 전체).
        """
        url = f"{BASE}/{path.strip('/')}"
        q = {k: v for k, v in params.items() if v not in (None, "")}
        # 이 셋은 GW 에서 사실상 필수다. 호출자가 빠뜨려도 여기서 채워 준다 —
        # 빠지면 코드 340 이거나, 더 나쁘게는 XML 이 돌아와 조용히 0건이 된다.
        q.setdefault("pageNo", 1)
        q.setdefault("numOfRows", 100)
        q["resultType"] = "json"
        q["serviceKey"] = self.service_key

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                r = self.session.get(
                    url, params=q, timeout=(self.connect_timeout, self.timeout)
                )
            except requests.RequestException as e:
                last_exc = e
                time.sleep(1.5 * (attempt + 1))
                continue

            text = r.text or ""
            err = _extract_xml_error(text)
            if err:
                exc = KcycleApiError(err[0], err[1], url)
                if exc.fatal:
                    raise exc
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
                continue

            if r.status_code == 429:
                raise KcycleApiError("22", "호출 한도 초과 (일일 트래픽 제한)", url)
            if r.status_code >= 500:
                last_exc = KcycleApiError(str(r.status_code), "server error", url)
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                # raise_for_status 는 URL 을 그대로 담아 키를 노출한다.
                raise KcycleApiError(str(r.status_code), f"HTTP {r.status_code}", url)

            try:
                data = r.json()
            except ValueError:
                last_exc = KcycleApiError("PARSE", f"JSON 파싱 실패: {text[:200]}", url)
                time.sleep(1.0 * (attempt + 1))
                continue

            resp = data.get("response", data)
            header = resp.get("header", {}) or {}
            code = str(header.get("resultCode", "00"))
            if code not in OK_CODES:
                exc = KcycleApiError(code, str(header.get("resultMsg", "")), url)
                if exc.fatal:
                    raise exc
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
                continue

            return resp.get("body", {}) or {}

        # 네트워크 예외도 KcycleApiError 로 감싸 내보낸다. 호출자가 requests 예외까지
        # 따로 잡아야 한다면, 한 소스의 일시적 실패가 배치 전체를 죽인다.
        if isinstance(last_exc, KcycleApiError):
            raise last_exc
        raise KcycleApiError(
            "NETWORK", f"요청 실패: {type(last_exc).__name__}", url
        ) from last_exc

    # ---------------------------------------------------------------- 고수준

    def fetch(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        rows: int = 500,
        max_pages: int = 200,
    ) -> List[dict]:
        """전 페이지를 순회해 레코드 리스트로 돌려준다."""
        return list(self.iter_pages(path, params, rows=rows, max_pages=max_pages))

    def iter_pages(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        rows: int = 500,
        max_pages: int = 200,
    ) -> Iterator[dict]:
        params = dict(params or {})
        page = 1
        seen = 0
        while page <= max_pages:
            body = self.raw(path, {**params, "pageNo": page, "numOfRows": rows})
            records = _as_list(body.get("items"))
            for rec in records:
                yield rec
            seen += len(records)

            try:
                total = int(body.get("totalCount"))
            except (TypeError, ValueError):
                total = None

            if not records:
                return
            if total is not None and seen >= total:
                return
            if len(records) < rows:
                return
            page += 1
