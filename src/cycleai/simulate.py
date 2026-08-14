"""경주 전개 시뮬레이션 — 경륜의 승부처를 실제로 돌려 본다.

모델은 선수를 한 명씩 독립으로 채점해 확률을 정규화한다. 그러나 경륜은
**혼자 달리는 종목이 아니다.** 앞사람이 바람을 다 맞고 뒷사람은 그 뒤에 숨어
힘을 아낀다. 그래서 '누가 강한가'만으로는 결과가 정해지지 않고, **누가 언제
앞으로 나가느냐**가 정한다. 독립 채점으로는 이 상호작용을 담을 수 없다.

## 무엇을 계산하나 — 바람과 남은 힘의 상충

자전거가 받는 공기저항은 속도의 세제곱에 비례해 출력을 먹는다. 앞사람 뒤에
붙으면(드래프팅) 그 출력이 크게 줄어든다. 그래서 경륜의 전법은 전부 **언제
바람을 맞기 시작할 것인가**의 선택이다.

  * **선행** — 일찍 앞으로 나가 끝까지 버틴다. 위치는 가장 좋지만 마지막
    직선에서 남은 힘이 가장 적다.
  * **젖히기** — 뒤에서 힘을 아끼다 3~4코너에서 바깥으로 크게 돌아 앞선
    선수를 넘어간다. 돌아 가는 만큼 거리를 손해 본다.
  * **추입** — 끝까지 숨어 있다가 마지막 직선에서만 나온다. 힘은 가장 많이
    남지만, 앞이 막히면 나올 자리가 없다.
  * **마크** — 특정 선수의 뒤를 지킨다. 그 선수가 살아야 자기도 산다.

## 어디까지가 사실이고 어디부터가 가정인가

**주로 제원만 사실이다.** 광명 벨로드롬 333.3m, 경주거리 1,691m(5주회)는
출주표가 주는 값이다. 그 밖에 **드래프팅 절감률·바람 노출에 따른 소모율·
스프린트 속도 환산은 전부 가정**이다. 실측 자료를 구하지 못했다.

그래서 이 모듈이 내는 **초 단위 숫자는 신뢰할 수 없다.** 신뢰할 수 있는 것은
부호와 순서다 — 앞에서 끌면 힘이 빨리 준다, 뒤에 붙으면 아낀다, 아낀 힘은
마지막 직선에서 속도가 된다, 바깥으로 돌면 거리를 손해 본다.

그래서 시뮬레이션 결과는 **승률로 쓰지 않는다.** 승률은 학습된 모델이 낸다.
여기서는 모델이 못 주는 것 — 전법 분포, 선행 확률, 라인 구도, 전개 대본 —
만 만든다. 화면에서 둘이 다른 말을 하지 않도록 선수의 기본 기량은 모델 확률에
닻을 내린다(``MODEL_WEIGHT``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .korean import josa

# ── 주로 제원 (출주표 값) ──────────────────────────────────────────
LAP_M = 333.3            # 광명 벨로드롬 1주회
HOME_STRAIGHT = 90.0     # 마지막 직선 (추정)

# ── 가정 상수 ─────────────────────────────────────────────────────
#
# 드래프팅 절감률. 도로 경기 연구에서 바로 뒤 선수의 공기저항이 30~40% 줄어든다는
# 값이 흔히 인용된다. 벨로드롬은 속도가 더 높아 효과가 크지만, 여기서는
# 보수적으로 잡는다. **정확한 값이 아니라 순서를 만드는 것이 목적이다.**
DRAFT_SAVING = (0.00, 0.32, 0.40, 0.43, 0.45, 0.46, 0.46)  # 선두=0, 2번째=32% ...

# 마지막 600m 를 앞에서 끌면 소모하는 힘의 비율. 1.0 이면 스프린트 여력이
# 남지 않는다는 뜻이라, 0.55 는 '선행은 마지막 직선에서 절반쯤 소진된 상태'로
# 본다는 가정이다.
LEAD_COST = 0.55

# 바깥으로 돌아 나갈 때의 추가 거리(m). 젖히기가 치르는 대가다.
WIDE_PENALTY = {"선행": 0.0, "젖히기": 9.0, "추입": 3.0, "마크": 2.0}

# 전법마다 언제 바람을 맞기 시작하는가 (마지막 몇 m 부터).
TACTIC_EXPOSE = {"선행": 700.0, "젖히기": 300.0, "추입": 150.0, "마크": 200.0}

# 모델 확률이 기본 기량을 얼마나 끌고 갈 것인가. 0 이면 시뮬레이션이 모델과
# 무관한 이야기를 하고, 1 이면 시뮬레이션이 모델을 그대로 되풀이한다.
MODEL_WEIGHT = 0.65

N_SIMS = 2000
RNG_SEED = 20260814

TACTICS = ("선행", "젖히기", "추입", "마크")

TACTIC_DESC = {
    "선행": "일찍 앞으로 나가 끝까지 버티는 전법. 위치는 최고지만 마지막 직선에서 남은 힘이 가장 적다.",
    "젖히기": "뒤에서 힘을 아끼다 바깥으로 크게 돌아 넘어가는 전법. 돌아 가는 거리만큼 손해를 본다.",
    "추입": "끝까지 숨어 있다가 마지막 직선에서만 나오는 전법. 힘은 가장 많이 남지만 앞이 막히면 못 나온다.",
    "마크": "특정 선수의 뒤를 지키는 전법. 그 선수가 살아야 자기도 산다.",
}


@dataclass
class Rider:
    back_no: int
    racer_nm: str
    p_win: float
    # 전법 성향 (입상 기준 비율). 없으면 균등.
    tactic_p: Dict[str, float] = field(default_factory=dict)
    sprint: float = 0.0        # 마지막 직선 속도 지표 (클수록 빠름)
    stamina: float = 0.0       # 힘의 총량 (클수록 오래 버틴다)
    trng_plc: Optional[str] = None
    grade: Optional[str] = None


def _norm(vals: Sequence[float]) -> np.ndarray:
    """0~1 로 눕힌다. 전부 같으면 가운데."""
    a = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)
    if np.all(np.isnan(a)):
        return np.full(len(a), 0.5)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if hi - lo < 1e-9:
        return np.full(len(a), 0.5)
    out = (a - lo) / (hi - lo)
    return np.where(np.isnan(out), 0.5, out)


def build_riders(runners: List[Dict]) -> List[Rider]:
    """예측 프레임의 행들을 시뮬레이션 입력으로 옮긴다."""
    riders: List[Rider] = []
    # 200m 기록은 **작을수록 빠르다**. 부호를 뒤집지 않으면 가장 느린 선수가
    # 가장 빠른 것으로 들어간다 — 오류 없이 조용히 뒤집히는 종류의 실수다.
    rec = _norm([-(r.get("rec_200m") or np.nan) for r in runners])
    scr = _norm([r.get("tot_avg_scr") for r in runners])
    run = _norm([r.get("run_day_cnt") for r in runners])

    for i, r in enumerate(runners):
        tp = {}
        for name, col in (("선행", "pre_ratio"), ("젖히기", "pas_ratio"),
                          ("추입", "brk_ratio"), ("마크", "mrk_ratio")):
            v = r.get(col)
            tp[name] = float(v) if v is not None and not _isnan(v) else None
        if all(v is None for v in tp.values()):
            # 입상 이력이 없는 선수(신인)는 성향을 말할 수 없다. 균등으로 둔다.
            tp = {t: 0.25 for t in TACTICS}
        else:
            filled = {k: (v if v is not None else 0.0) for k, v in tp.items()}
            s = sum(filled.values()) or 1.0
            tp = {k: v / s for k, v in filled.items()}

        riders.append(Rider(
            back_no=int(r.get("back_no") or i + 1),
            racer_nm=str(r.get("racer_nm") or "?"),
            p_win=float(r.get("p_win") or r.get("p_win_norm") or 1.0 / len(runners)),
            tactic_p=tp,
            sprint=0.6 * rec[i] + 0.4 * scr[i],
            stamina=0.7 * scr[i] + 0.3 * run[i],
            trng_plc=r.get("trng_plc"),
            grade=r.get("racer_grd"),
        ))
    return riders


def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def lines_of(riders: List[Rider]) -> List[List[int]]:
    """훈련지가 같은 선수를 한 라인으로 묶는다.

    공개 자료에 '오늘 누가 누구와 라인을 짤 것인가'는 없다. 훈련지는 그것의
    **대리 지표일 뿐**이다 — 실제로는 지역·선후배·그날의 판단으로 갈린다.
    화면에서 이것을 '라인'이라고 단정하지 않는 이유다.
    """
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(riders):
        if not r.trng_plc:
            continue
        groups.setdefault(r.trng_plc, []).append(i)
    return [v for v in groups.values() if len(v) > 1]


def _run_once(riders: List[Rider], rng: np.random.Generator) -> Dict:
    """한 판. 전법을 뽑고, 위치를 잡고, 마지막 직선을 계산한다."""
    n = len(riders)

    # 1) 전법 추첨 — 각자의 성향대로.
    tactics = []
    for r in riders:
        p = np.array([r.tactic_p.get(t, 0.25) for t in TACTICS], dtype=float)
        p = np.clip(p, 1e-3, None)
        tactics.append(TACTICS[int(rng.choice(len(TACTICS), p=p / p.sum()))])

    # 2) 타종 시점의 대열. 선행이 앞, 마크·추입이 뒤로 간다.
    #    같은 전법끼리는 기량(모델 확률)과 운으로 가른다.
    order_score = []
    for i, r in enumerate(riders):
        base = {"선행": 3.0, "젖히기": 1.6, "마크": 1.0, "추입": 0.4}[tactics[i]]
        base += MODEL_WEIGHT * r.p_win * n * 0.5
        base += rng.normal(0, 0.45)
        order_score.append(base)
    pos = list(np.argsort(order_score)[::-1])   # 앞에서부터

    # 3) 마크는 자기 라인의 앞사람 바로 뒤로 당겨 붙는다. 라인이 뜻하는 것이
    #    바로 이것이다 — 자리를 다투지 않고 앞사람 뒤를 지킨다.
    for line in lines_of(riders):
        head = min(line, key=lambda i: pos.index(i))
        for m in line:
            if m == head or tactics[m] != "마크":
                continue
            pos.remove(m)
            pos.insert(pos.index(head) + 1, m)

    place = {rider_i: k for k, rider_i in enumerate(pos)}

    # 4) 남은 힘. 바람을 언제부터 맞았는지와 대열 위치로 정한다.
    energy = np.zeros(n)
    for i, r in enumerate(riders):
        k = place[i]
        saving = DRAFT_SAVING[min(k, len(DRAFT_SAVING) - 1)]
        exposed = TACTIC_EXPOSE[tactics[i]]
        # 앞에 있을수록, 일찍 나갈수록 많이 쓴다. stamina 가 그것을 버틴다.
        cost = LEAD_COST * (exposed / 700.0) * (1.0 - saving)
        cost *= (1.25 - 0.5 * r.stamina)
        energy[i] = float(np.clip(1.0 - cost, 0.05, 1.0))

    # 5) 마지막 직선. 앞사람과의 거리를 남은 힘 × 스프린트로 좁힌다.
    gap = np.array([place[i] * 2.6 for i in range(n)])      # 대열 간격(m, 가정)
    wide = np.array([WIDE_PENALTY[tactics[i]] for i in range(n)])
    speed = np.array([
        (0.55 + 0.45 * riders[i].sprint) * (0.45 + 0.55 * energy[i])
        for i in range(n)])
    speed *= (1.0 + MODEL_WEIGHT * 0.25 * (np.array([r.p_win for r in riders]) * n - 1))
    speed *= rng.normal(1.0, 0.06, n)
    speed = np.clip(speed, 0.05, None)

    # 추입은 앞이 막히면 못 나온다. 대열 안쪽에 갇힐 확률을 준다.
    blocked = np.zeros(n, dtype=bool)
    for i in range(n):
        if tactics[i] == "추입" and place[i] >= 3:
            blocked[i] = rng.random() < 0.30

    # 결승선 통과 시각(임의 단위). 남은 거리를 속도로 나눈다.
    dist = HOME_STRAIGHT + gap + wide
    t = dist / speed
    t[blocked] += 0.9

    finish = list(np.argsort(t))
    return {"tactics": tactics, "place": place, "order_bell": pos,
            "energy": energy, "speed": speed, "finish": finish, "t": t}


def simulate(runners: List[Dict], n_sims: int = N_SIMS,
             seed: int = RNG_SEED) -> Optional[Dict]:
    """전개를 여러 번 돌려 분포와 대본을 만든다.

    난수 씨앗을 고정한다. 같은 입력에 같은 전개가 나와야, 발주 전에 화면에
    있던 것과 나중에 다시 본 것이 같다.
    """
    if not runners or len(runners) < 2:
        return None
    riders = build_riders(runners)
    n = len(riders)
    rng = np.random.default_rng(seed)

    win = np.zeros(n)
    lead = np.zeros(n)            # 타종 시점 선두
    tactic_cnt = {t: np.zeros(n) for t in TACTICS}
    top3 = np.zeros(n)
    sample = None

    for s in range(n_sims):
        res = _run_once(riders, rng)
        w = res["finish"][0]
        win[w] += 1
        lead[res["order_bell"][0]] += 1
        for i, t in enumerate(res["tactics"]):
            tactic_cnt[t][i] += 1
        for i in res["finish"][:3]:
            top3[i] += 1
        # 대본은 '가장 흔한 결말' 한 판을 골라 쓴다. 첫 판을 쓰면 드문 전개가
        # 대표 이야기가 될 수 있다.
        if sample is None or w == int(np.argmax(win)):
            sample = res

    win /= n_sims
    lead /= n_sims
    top3 /= n_sims

    anchor = int(np.argmax([r.p_win for r in riders]))
    share = float(win[anchor])

    tactic_dist = []
    for i, r in enumerate(riders):
        d = {t: float(tactic_cnt[t][i] / n_sims) for t in TACTICS}
        tactic_dist.append({
            "back_no": r.back_no, "racer_nm": r.racer_nm,
            "main": max(d, key=d.get), **d,
        })
    # 경주 전체에서 가장 자주 나온 전법
    totals = {t: float(tactic_cnt[t].sum() / (n_sims * n)) for t in TACTICS}
    top_tactic = max(totals, key=totals.get)

    return {
        "n_sims": n_sims,
        "runners": [{
            "back_no": r.back_no, "racer_nm": r.racer_nm,
            "sim_win": float(win[i]), "sim_top3": float(top3[i]),
            "lead_prob": float(lead[i]),
            "main_tactic": tactic_dist[i]["main"],
            "sprint": round(float(r.sprint), 3),
            "stamina": round(float(r.stamina), 3),
        } for i, r in enumerate(riders)],
        "tactic_dist": tactic_dist,
        "tactic_share": totals,
        "top_tactic": top_tactic,
        "top_tactic_desc": TACTIC_DESC.get(top_tactic, ""),
        "lines": [[riders[i].back_no for i in line] for line in lines_of(riders)],
        "confidence": _confidence(share),
        "script": _script(sample, riders) if sample else [],
    }


def _confidence(share: float) -> Dict:
    """반복 계산에서 예상 1순위가 얼마나 자주 이겼는가.

    이것은 승률이 아니라 **전개에 대한 견고함**이다. 같은 기량이라도 전법이
    엇갈리면 결과가 흔들린다는 것을 숫자로 보여 준다.
    """
    if share >= 0.50:
        return {"label": "강승부", "score": round(share * 100),
                "desc": "반복 계산의 절반 이상에서 예상 1순위가 승리했습니다."}
    if share >= 0.32:
        return {"label": "중승부", "score": round(share * 100),
                "desc": "예상 1순위가 우세하나 전개에 따라 갈릴 여지가 있습니다."}
    return {"label": "약승부", "score": round(share * 100),
            "desc": "전개에 따라 순위가 크게 바뀔 수 있습니다."}


def _script(res: Dict, riders: List[Rider]) -> List[Dict]:
    """한 판을 사람이 읽는 세 장면으로 옮긴다.

    조사는 ``korean.josa`` 로 붙인다. '6번 이주영 가 선행 으로' 처럼 나가면
    문장이 어색한 데서 그치지 않고, 옆에 있는 숫자까지 대충 만든 것처럼 읽힌다.
    """
    def name(i: int) -> str:
        return f"{riders[i].back_no}번 {riders[i].racer_nm}"

    bell = res["order_bell"]
    finish = res["finish"]
    energy = res["energy"]
    tactics = res["tactics"]
    front = bell[0]
    fresh = int(np.argmax(energy))

    return [{
        "title": "타종 (남은 1주회)",
        "text": (f"{josa(name(front), '이/가')} "
                 f"{josa(tactics[front], '으로/로')} 앞에 섰다. "
                 f"뒤로 {' · '.join(name(i) for i in bell[1:4])}의 대열."),
        "order": [riders[i].back_no for i in bell],
    }, {
        "title": "3~4코너",
        "text": (f"바람을 맞은 {name(front)}의 남은 힘이 "
                 f"{energy[front] * 100:.0f}%로 떨어졌다. "
                 f"뒤에 숨어 있던 {josa(name(fresh), '은/는')} "
                 f"{energy[fresh] * 100:.0f}%를 남겨 두고 있다."),
        "order": [riders[i].back_no for i in bell],
    }, {
        "title": "결승선",
        "text": (f"{josa(name(finish[0]), '이/가')} 먼저 들어왔다. "
                 f"{name(finish[1])} · {josa(name(finish[2]), '이/가')} "
                 f"뒤를 이었다."),
        "order": [riders[i].back_no for i in finish],
    }]
