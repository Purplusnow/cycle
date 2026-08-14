"""엔드포인트 프로브 — 승인 여부 확인과 실제 응답 필드 덤프.

경로는 명세에서 그대로 가져왔으므로 추측할 것이 없다. 대신 확인해야 할 것이
셋 있다.

1. **어느 API 가 실제로 승인됐나.** 미승인은 오류 코드가 아니라 HTTP 403 으로
   돌아오므로, 코드 12(경로 오류)와 구분해 보여준다 — 둘을 뭉뚱그리면
   '키가 안 된다'는 잘못된 결론으로 샌다.
2. **실제 응답 필드.** 경륜 API 12개 중 4개(경주결과순위·낙차사고·제재선수·
   선수정보)는 포털 명세에 응답 정의가 **아예 비어 있다.** 실물을 봐야 파서를
   쓸 수 있다.
3. **회차 좌표를 어떻게 잡나.** 경륜은 경정에 없던 ``meet_nm``(경륜장) 축이
   있고, 경주결과는 그 축까지 포함해 파라미터 5개가 전부 필수다. 출주표에서
   좌표를 배워 나머지에 넘긴다.

    python -m cycleai.kcycle.probe
    python -m cycleai.kcycle.probe --only race_card race_rank
    python -m cycleai.kcycle.probe --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..clock import now_kst, today_kst
from .client import KcycleApiError, KcycleClient, _as_list, redact
from .endpoints import REGISTRY, REQUIRED_KEYS, save_resolved, to_api_params

log = logging.getLogger(__name__)

FIELD_DUMP = Path("config/api_fields.json")

# 회차 좌표를 아직 모를 때 쓰는 탐색 순서. 출주표가 가장 넓게 열려 있고
# (필수 파라미터가 인증·형식뿐) meet_nm·race_ymd·week_tcnt·day_tcnt·race_no 를
# 한 번에 준다 — 좌표를 여기서 얻어 나머지 API 에 넘긴다.
DISCOVERY_KEYS = ["race_card", "race_rank", "payoff"]


def _year_candidates() -> List[str]:
    """올해부터 뒤로. 연초에는 올해 자료가 아직 없을 수 있다."""
    y = today_kst().year
    return [str(y), str(y - 1), str(y - 2)]


def _param_sets(key: str, ctx: Dict[str, object]) -> List[Dict[str, object]]:
    """이 엔드포인트에 시도해 볼 파라미터 조합.

    ``ctx`` 는 앞선 프로브에서 알아낸 실제 좌표
    (stnd_yr / meet_nm / week_tcnt / day_tcnt / race_no / race_ymd / racer_nm).
    """
    ep = REGISTRY[key]
    req, opt = set(ep.required), set(ep.optional)
    sets: List[Dict[str, object]] = []

    def add(d: Dict[str, object]) -> None:
        d = {k: v for k, v in d.items() if v not in (None, "")}
        if d not in sets:
            sets.append(d)

    years = [ctx.get("stnd_yr")] if ctx.get("stnd_yr") else _year_candidates()

    # 경주 하나를 특정해야 하는 API (경주결과). 경정과 달리 meet_nm 까지 필수다.
    if "race_no" in req:
        for y in years:
            if ctx.get("week_tcnt"):
                add({"stnd_yr": y, "meet_nm": ctx.get("meet_nm"),
                     "week_tcnt": ctx["week_tcnt"], "day_tcnt": ctx["day_tcnt"],
                     "race_no": ctx.get("race_no") or 1})
        return sets

    # 경주일자까지 필수인 API (조조연습현황)
    if "race_ymd" in req:
        for y in years:
            if ctx.get("week_tcnt"):
                add({"stnd_yr": y, "week_tcnt": ctx["week_tcnt"],
                     "day_tcnt": ctx["day_tcnt"], "race_ymd": ctx.get("race_ymd")})
        return sets

    # 회차가 필수인 API (회차별 경주득점·검차·자전거 보관)
    if "week_tcnt" in req:
        for y in years:
            if ctx.get("week_tcnt"):
                d = {"stnd_yr": y, "week_tcnt": ctx["week_tcnt"]}
                if "day_tcnt" in req:
                    d["day_tcnt"] = ctx.get("day_tcnt")
                add(d)
            # 좌표를 못 얻었을 때를 위한 보수적 후보 — 회차는 1부터 센다.
            d = {"stnd_yr": y, "week_tcnt": 1}
            if "day_tcnt" in req:
                d["day_tcnt"] = 1
            add(d)
        return sets

    # 연간경주일정 — 연/월 축이 따로다.
    if "schdl_yr" in opt:
        for y in years:
            add({"schdl_yr": y})
        return sets

    # 그 밖 — 좁은 조건부터 넓은 조건으로.
    for y in years:
        if ctx.get("week_tcnt") and {"week_tcnt", "day_tcnt"} & opt:
            add({"stnd_yr": y, "week_tcnt": ctx["week_tcnt"],
                 "day_tcnt": ctx["day_tcnt"] if "day_tcnt" in opt else None})
        if "stnd_yr" in opt or "stnd_year" in opt:
            add({"stnd_yr": y})
    add({})
    return sets


def _learn(ctx: Dict[str, object], records: List[dict]) -> None:
    """응답에서 회차 좌표를 배운다.

    가장 큰 (연도, 회차, 일차)를 고른다 — 최근 회차라야 다른 API 에도 자료가
    있을 가능성이 높다. 정렬 순서를 믿지 않고 값으로 고르는 것은, API 마다
    정렬 기준이 다르고 명시돼 있지도 않기 때문이다.
    """
    def _int(v):
        try:
            return int(str(v).strip())
        except (TypeError, ValueError):
            return None

    best = None
    for r in records:
        yr = _int(r.get("stnd_yr") or r.get("stnd_year"))
        wk = _int(r.get("week_tcnt") or r.get("tms"))
        dy = _int(r.get("day_tcnt") or r.get("day_ord"))
        if yr and wk and dy:
            cand = (yr, wk, dy, _int(r.get("race_no")) or 1,
                    str(r.get("meet_nm") or "").strip(),
                    str(r.get("race_ymd") or r.get("race_day") or "").strip())
            if best is None or cand[:3] > best[:3]:
                best = cand
    if best and not ctx.get("week_tcnt"):
        ctx["stnd_yr"] = str(best[0])
        ctx["week_tcnt"] = best[1]
        ctx["day_tcnt"] = best[2]
        ctx["race_no"] = best[3]
        if best[4]:
            ctx["meet_nm"] = best[4]
        if best[5]:
            ctx["race_ymd"] = best[5]

    for r in records:
        for src, dst in (("racer_nm", "racer_nm"), ("racer_no", "racer_no"),
                         ("meet_nm", "meet_nm")):
            if not ctx.get(dst) and str(r.get(src) or "").strip():
                ctx[dst] = str(r[src]).strip()
        if not ctx.get("race_ymd"):
            v = r.get("race_ymd") or r.get("race_day")
            if v:
                ctx["race_ymd"] = str(v).strip()


def probe_one(client: KcycleClient, key: str, ctx: Dict[str, object]) -> dict:
    ep = REGISTRY[key]
    attempts: List[dict] = []

    for params in _param_sets(key, ctx):
        try:
            body = client.raw(ep.path, to_api_params(key, params))
        except KcycleApiError as e:
            attempts.append({"params": params, "error": f"[{e.code}] {e.msg}"})
            if e.fatal:
                # 권한·경로 문제는 파라미터를 바꿔도 똑같다. 더 두드리면
                # 일일 호출량만 태운다.
                return {"key": key, "ok": False, "fatal": True, "code": e.code,
                        "reason": str(e), "attempts": attempts}
            continue
        except Exception as e:  # noqa: BLE001
            attempts.append({"params": params, "error": redact(repr(e))})
            continue

        records = _as_list(body.get("items"))
        if records:
            _learn(ctx, records)
            return {
                "key": key, "ok": True, "path": ep.path, "sample_params": params,
                "total_count": body.get("totalCount"),
                "fields": sorted(records[0].keys()),
                "sample_record": records[0],
                "attempts": attempts,
            }
        attempts.append({"params": params,
                         "error": f"빈 응답 (totalCount={body.get('totalCount')})"})

    return {"key": key, "ok": False, "fatal": False, "code": "",
            "reason": "승인은 됐으나 레코드가 나오지 않음 (조회 조건 또는 자료 부재)",
            "attempts": attempts}


def probe_zero_pad(client: KcycleClient, ctx: Dict[str, object]) -> Optional[dict]:
    """race_no 가 2자리 0채움을 요구하는지 실측한다.

    경정에서 가장 찾기 어려웠던 실패가 이것이다 — ``"1"`` 을 주면 **오류 없이
    0건**이 온다. 같은 게이트웨이라 경륜도 그럴 것이라 가정했지만, 가정은
    코드에 박아 두면 사실처럼 보인다. 한 번 실제로 물어본다.
    """
    if not ctx.get("week_tcnt"):
        return None
    ep = REGISTRY["race_result"]
    base = {"stnd_yr": ctx["stnd_yr"], "meet_nm": ctx.get("meet_nm"),
            "week_tcnt": ctx["week_tcnt"], "day_tcnt": ctx["day_tcnt"]}
    out = {}
    for label, race_no in (("0채움 '01'", "01"), ("맨숫자 '1'", "1")):
        try:
            body = client.raw(ep.path, {**base, "race_no": race_no})
            out[label] = len(_as_list(body.get("items")))
        except KcycleApiError as e:
            out[label] = f"[{e.code}]"
    return out


def probe_all(client: KcycleClient, keys: Optional[List[str]] = None) -> dict:
    keys = keys or list(REGISTRY)
    # 좌표를 먼저 알아내야 경주결과처럼 좌표가 필수인 API 를 부를 수 있다.
    ordered = [k for k in DISCOVERY_KEYS if k in keys] + \
              [k for k in keys if k not in DISCOVERY_KEYS]
    ctx: Dict[str, object] = {}
    results = {}
    for key in ordered:
        log.info("프로브: %-16s %s", key, REGISTRY[key].title)
        results[key] = probe_one(client, key, ctx)
    results["_context"] = ctx
    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="경륜 오픈API 엔드포인트 프로브")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 으로 출력")
    ap.add_argument("--only", nargs="*", help="특정 엔드포인트 키만 검사")
    ap.add_argument("--zero-pad", action="store_true",
                    help="race_no 0채움 필요 여부를 실측")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        client = KcycleClient.from_env()
    except ValueError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2

    results = probe_all(client, args.only)
    ctx = results.pop("_context", {})

    pad = probe_zero_pad(client, ctx) if args.zero_pad else None

    if args.json:
        print(json.dumps({"results": results, "context": ctx, "zero_pad": pad},
                         ensure_ascii=False, indent=2))
    else:
        ok = [k for k, r in results.items() if r["ok"]]
        denied = [k for k, r in results.items() if r.get("code") in ("20", "401", "403")]
        badpath = [k for k, r in results.items() if r.get("code") == "12"]
        other = [k for k, r in results.items()
                 if not r["ok"] and k not in denied and k not in badpath]

        print(f"\n조회 좌표: {ctx or '(확보 실패)'}\n")
        for key, r in results.items():
            ep = REGISTRY[key]
            if r["ok"]:
                print(f"  ✓ {ep.title:<16} 필드 {len(r['fields']):>2}개  "
                      f"총 {r.get('total_count')}건  {r['sample_params']}")
            else:
                mark = "✗✗" if r.get("fatal") else "· "
                print(f"  {mark} {ep.title:<16} {r['reason']}")

        print(f"\n승인·응답 정상 {len(ok)}개")
        if denied:
            print(f"미승인(HTTP 403) {len(denied)}개: "
                  f"{', '.join(REGISTRY[k].title for k in denied)}")
        if badpath:
            print(f"경로 오류(코드 12) {len(badpath)}개: "
                  f"{', '.join(REGISTRY[k].title for k in badpath)}")
        if other:
            print(f"응답 없음 {len(other)}개: "
                  f"{', '.join(REGISTRY[k].title for k in other)}")
        if pad:
            print(f"\nrace_no 0채움 실측: {pad}")

    resolved = {k: r["path"] for k, r in results.items() if r["ok"]}
    if resolved:
        p = save_resolved(resolved, meta={
            "probed_at": now_kst().isoformat(timespec="seconds"),
            "context": ctx,
            "zero_pad": pad,
        })
        FIELD_DUMP.parent.mkdir(parents=True, exist_ok=True)
        FIELD_DUMP.write_text(json.dumps(
            {k: {"fields": r["fields"], "sample": r["sample_record"],
                 "params": r["sample_params"], "total": r.get("total_count")}
             for k, r in results.items() if r["ok"]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n확정 경로 → {p}\n응답 필드 덤프 → {FIELD_DUMP}")

    missing = [k for k in REQUIRED_KEYS if k not in resolved]
    if missing:
        print(f"\n필수 엔드포인트 미확보: "
              f"{', '.join(REGISTRY[k].title for k in missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
