"""승률 모델 학습 · 워크포워드 검증 · 추론.

모델 자체는 평범한 그래디언트 부스팅이다. 이 파일에서 중요한 건 **정직한
평가**다. 경주 예측은 그럴듯한 숫자를 만들기는 쉽고 실제로 쓸모 있기는 어렵다.
그래서 둘을 강제한다.

1. **워크포워드 검증** — 시간 순서대로 과거로 학습해 미래를 맞힌다. 무작위
   분할은 같은 경주의 다른 선수가 학습셋에 들어가 성능을 부풀린다.

2. **베이스라인 동시 측정** — 공개 API 에 **발주 전 배당률이 없다.** 단승 배당은
   적중한 것만, 그것도 경주 후에 나온다. 그래서 시장 대신 둘을 쓴다.

   * **득점 최고** — 경륜을 보는 사람이 가장 먼저 보는 숫자다. 출주표의 종합
     평균득점이 가장 높은 선수를 찍는다. 이걸 못 이기면 모델이 배운 것이 없다.
   * **배번 1번 고정** — 경정에서는 이 자리가 1코스였고 그 자체로 강력한
     베이스라인이었다. 경륜에서 같은 것을 재 보는 이유는, **자리가 유리함을
     뜻하지 않는다**는 전제를 숫자로 확인하기 위해서다. 이 값이 1/7(14.3%)
     근처에 머물면 전제가 맞은 것이고, 크게 웃돌면 피처 설계를 다시 봐야 한다.

   두 베이스라인 모두 우리 모델과 **같은 경주**에서 계산하므로 비교가 공정하다.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

from .features import CATEGORICAL, build_training_frame, feature_columns

log = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

MODEL_VERSION = "v1"


def _matrix(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """컬럼 순서를 고정한 입력 행렬. 범주형은 코드로 인코딩."""
    X = df.reindex(columns=cols).copy()
    for c in CATEGORICAL:
        if c in X:
            X[c] = pd.Categorical(X[c].astype(str)).codes.astype(float)
    return X.astype(float)


def normalize_within_race(df: pd.DataFrame, prob_col: str, out_col: str) -> pd.DataFrame:
    """경주 내 확률 합이 1이 되도록 정규화.

    이진 분류기는 선수 단위로 독립 예측하므로 합이 1이 아니다. 경주는 정확히 한
    명만 이기므로 정규화해야 확률로 읽을 수 있다.
    """
    df = df.copy()
    p = pd.to_numeric(df[prob_col], errors="coerce").clip(1e-6, 1 - 1e-6)
    total = p.groupby(df["race_key"]).transform("sum")
    n = df.groupby("race_key")["back_no"].transform("count")
    df[out_col] = (p / total.replace(0, np.nan)).fillna(1.0 / n)
    return df


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

@dataclass
class Eval:
    n_races: int
    n_runners: int
    top1_win: float           # 1순위 추천의 단승 적중률
    top1_top2: float          # 1순위가 2착 이내 (연승)
    top2_quinella: float      # 상위 2명이 실제 1·2착 (순서 무관) = 복승
    top3_trio: float          # 상위 3명이 실제 1·2·3착 (순서 무관) = 삼복승
    top3_has_winner: float
    logloss: float
    auc: float
    roi_win: float            # 1순위 단승 100원 베팅 회수율
    # 베이스라인
    score_win: float          # 종합 평균득점 최고
    score_roi: float
    back1_win: float          # 배번 1번 고정
    back1_roi: float
    coverage_payout: float    # 배당이 있어 회수율을 낼 수 있는 경주 비율


def _hit_rates(df: pd.DataFrame, rank_col: str) -> Tuple[float, float, float, float]:
    """추천 순위 기준 승식 적중률. (단승, 연승, 복승, 삼복승)"""
    top1 = df[df[rank_col] == 1]
    win = float((top1["ord"] == 1).mean()) if len(top1) else float("nan")
    plc = float((top1["ord"] <= 2).mean()) if len(top1) else float("nan")

    top2 = df[df[rank_col] <= 2]
    qnl = float(top2.groupby("race_key")["ord"]
                .apply(lambda s: set(s.dropna()) == {1.0, 2.0}).mean()) if len(top2) else float("nan")
    top3 = df[df[rank_col] <= 3]
    tri = float(top3.groupby("race_key")["ord"]
                .apply(lambda s: set(s.dropna()) == {1.0, 2.0, 3.0}).mean()) if len(top3) else float("nan")
    return win, plc, qnl, tri


def _roi(sub: pd.DataFrame) -> float:
    """배당이 들어온 경주만으로 회수율을 낸다.

    배당 자료가 없는 경주를 '0원 회수'로 세면 회수율이 자료 결손만큼 낮게 나온다.
    """
    bet = sub[sub["win_payout"].notna()]
    if not len(bet):
        return float("nan")
    return float(((bet["ord"] == 1) * bet["win_payout"]).sum() / len(bet))


def evaluate(pred: pd.DataFrame) -> Eval:
    """경주 단위 지표. pred 는 p_win_norm, ord, back_no, win_payout 을 포함."""
    df = pred.copy()
    df["ord"] = pd.to_numeric(df["ord"], errors="coerce")
    df["win_payout"] = pd.to_numeric(df["win_payout"], errors="coerce")

    valid = df.groupby("race_key")["ord"].transform(lambda s: (s == 1).sum() == 1)
    df = df[valid]
    if df.empty:
        raise ValueError("평가 가능한 경주가 없습니다 (1착 기록 부재).")

    df["model_rank"] = df.groupby("race_key")["p_win_norm"].rank(ascending=False,
                                                                method="first")
    races = df["race_key"].nunique()

    win, plc, qnl, tri = _hit_rates(df, "model_rank")
    top3 = df[df["model_rank"] <= 3]
    top3_has_winner = float(top3.groupby("race_key")["ord"]
                            .apply(lambda s: (s == 1).any()).mean())

    y = (df["ord"] == 1).astype(int)
    p = df["p_win_norm"].clip(1e-6, 1 - 1e-6)
    try:
        ll = float(log_loss(y, p, labels=[0, 1]))
        auc = float(roc_auc_score(y, p))
    except ValueError:
        ll, auc = float("nan"), float("nan")

    roi = _roi(df[df["model_rank"] == 1])

    # ── 베이스라인 ───────────────────────────────────────────────
    if "tot_avg_scr" in df:
        df["score_rank"] = df.groupby("race_key")["tot_avg_scr"].rank(
            ascending=False, method="first")
        s1 = df[df["score_rank"] == 1]
        score_win = float((s1["ord"] == 1).mean()) if len(s1) else float("nan")
        score_roi = _roi(s1)
    else:
        score_win = score_roi = float("nan")

    b1 = df[df["back_no"] == 1]
    back1_win = float((b1["ord"] == 1).mean()) if len(b1) else float("nan")
    back1_roi = _roi(b1)

    cov = df[df["win_payout"].notna()]["race_key"].nunique() / races if races else 0.0

    return Eval(
        n_races=int(races), n_runners=int(len(df)),
        top1_win=win, top1_top2=plc, top2_quinella=qnl, top3_trio=tri,
        top3_has_winner=top3_has_winner, logloss=ll, auc=auc, roi_win=roi,
        score_win=score_win, score_roi=score_roi,
        back1_win=back1_win, back1_roi=back1_roi,
        coverage_payout=float(cov),
    )


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------

def fit(df: pd.DataFrame, cols: List[str], seed: int = 42) -> Dict:
    X = _matrix(df, cols)
    # 범주형의 '위치'를 넘겨야 한다. 이름이 아니라 인덱스이므로, cols 순서가
    # 학습과 추론에서 같아야 한다는 전제가 여기에도 걸려 있다.
    cat_idx = [i for i, c in enumerate(cols) if c in CATEGORICAL]
    models: Dict[str, HistGradientBoostingClassifier] = {}
    for target, col in (("win", "y_win"), ("top2", "y_top2"), ("top3", "y_top3")):
        y = pd.to_numeric(df[col], errors="coerce")
        mask = y.notna()
        if mask.sum() < 500 or y[mask].nunique() < 2:
            log.warning("%s 레이블이 부족해 학습을 건너뜁니다 (%d건)", target, int(mask.sum()))
            continue
        m = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=6, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.12,
            n_iter_no_change=30, random_state=seed,
            categorical_features=cat_idx or None,
        )
        m.fit(X[mask], y[mask].astype(int))
        models[target] = m
    return models


def predict_frame(models: Dict, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    X = _matrix(df, cols)
    out = df.copy()
    out["p_win_raw"] = models["win"].predict_proba(X)[:, 1] if "win" in models else np.nan
    out["p_top2_raw"] = models["top2"].predict_proba(X)[:, 1] if "top2" in models else np.nan
    out["p_top3_raw"] = models["top3"].predict_proba(X)[:, 1] if "top3" in models else np.nan
    out = normalize_within_race(out, "p_win_raw", "p_win_norm")

    # 착순 확률은 경주당 자리 수가 정해져 있다 — 2착 이내는 두 자리, 3착 이내는
    # 세 자리. 개별 예측의 합이 그 자리 수가 되도록 맞춰야 확률로 읽을 수 있다.
    n = out.groupby("race_key")["back_no"].transform("count")
    for raw, norm, slots in (("p_top2_raw", "p_top2_norm", 2),
                             ("p_top3_raw", "p_top3_norm", 3)):
        p = pd.to_numeric(out[raw], errors="coerce").clip(1e-6, 1 - 1e-6)
        total = p.groupby(out["race_key"]).transform("sum")
        scale = (np.minimum(slots, n) / total).replace([np.inf, -np.inf], 1)
        out[norm] = (p * scale).clip(0, 1)

    # 세 확률은 **포개진 사건**이다 — 1착이면 2착 이내이고, 2착 이내면 3착
    # 이내다. 그런데 위 정규화는 셋을 각각 따로 맞추므로 순서가 뒤집힐 수 있다.
    # 독립 모델 셋의 산출물을 확률로 읽으려면 이 관계는 지켜져야 한다.
    out["p_top2_norm"] = out[["p_win_norm", "p_top2_norm"]].max(axis=1)
    out["p_top3_norm"] = out[["p_top2_norm", "p_top3_norm"]].max(axis=1)

    out["pred_rank"] = out.groupby("race_key")["p_win_norm"].rank(
        ascending=False, method="first").astype(int)
    return out


def walk_forward(df: pd.DataFrame, cols: List[str], n_folds: int = 5,
                 min_train_races: int = 2000, seed: int = 42
                 ) -> Tuple[List[Eval], pd.DataFrame]:
    """시간순 확장 학습 검증."""
    df = df.sort_values("order_key").reset_index(drop=True)
    keys = df["order_key"].dropna().sort_values().unique()
    if len(keys) < 200:
        raise ValueError("검증에 필요한 경주가 부족합니다.")

    split_points = np.linspace(len(keys) * 0.5, len(keys), n_folds + 1).astype(int)
    evals: List[Eval] = []
    all_preds: List[pd.DataFrame] = []

    for i in range(n_folds):
        cut = keys[split_points[i] - 1]
        end = keys[min(split_points[i + 1] - 1, len(keys) - 1)]
        train = df[df["order_key"] <= cut]
        test = df[(df["order_key"] > cut) & (df["order_key"] <= end)]
        if train["race_key"].nunique() < min_train_races or test.empty:
            log.info("fold %d 건너뜀 (학습 %d경주, 검증 %d행)", i + 1,
                     train["race_key"].nunique(), len(test))
            continue

        models = fit(train, cols, seed)
        if "win" not in models:
            continue
        pred = predict_frame(models, test, cols)
        try:
            ev = evaluate(pred)
        except ValueError as e:
            log.info("fold %d 평가 불가: %s", i + 1, e)
            continue
        evals.append(ev)
        pred["fold"] = i + 1
        all_preds.append(pred)
        log.info("fold %d | 학습 %d경주 | 검증 %d경주 → 단승 %.1f%% "
                 "(득점최고 %.1f%% · 배번1 %.1f%%)", i + 1,
                 train["race_key"].nunique(), ev.n_races,
                 ev.top1_win * 100, ev.score_win * 100, ev.back1_win * 100)

    combined = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return evals, combined


def summarize(evals: List[Eval]) -> Dict:
    if not evals:
        return {}
    keys = [k for k in asdict(evals[0]) if k not in ("n_races", "n_runners")]
    out = {"folds": len(evals), "n_races": sum(e.n_races for e in evals),
           "n_runners": sum(e.n_runners for e in evals)}
    for k in keys:
        vals = [(getattr(e, k), e.n_races) for e in evals if not np.isnan(getattr(e, k))]
        out[k] = (float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))
                  if vals else None)
    return out


def save(models: Dict, cols: List[str], metrics: Dict, path: Path = MODEL_PATH) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "features": cols, "version": MODEL_VERSION}, path)

    # metrics.json 은 학습 지표만의 파일이 아니다. 다른 도구가 각자의 블록을
    # 여기에 남기므로 통째로 덮어쓰지 않고 병합한다.
    blob: Dict = {}
    try:
        blob = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    blob.update(metrics)
    METRICS_PATH.write_text(json.dumps(blob, ensure_ascii=False, indent=2),
                            encoding="utf-8")


def load(path: Path = MODEL_PATH) -> Dict:
    import joblib

    if not path.exists():
        raise FileNotFoundError(
            f"학습된 모델이 없습니다: {path}  (python -m cycleai.model train)")
    return joblib.load(path)


def _fmt(v: Optional[float], pct: bool = True) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  n/a"
    return f"{v * 100:5.1f}%" if pct else f"{v:5.3f}"


def report(s: Dict) -> str:
    if not s:
        return "검증 결과가 없습니다."
    return "\n".join([
        f"검증 경주 {s['n_races']:,}회 / 출주 {s['n_runners']:,}명  (fold {s['folds']})",
        "",
        f"{'지표':<26}{'예측 모델':>10}{'득점 최고':>12}{'배번 1번':>12}",
        "-" * 62,
        f"{'단승 적중률':<26}{_fmt(s.get('top1_win')):>10}"
        f"{_fmt(s.get('score_win')):>12}{_fmt(s.get('back1_win')):>12}",
        f"{'단승 회수율(ROI)':<26}{_fmt(s.get('roi_win')):>10}"
        f"{_fmt(s.get('score_roi')):>12}{_fmt(s.get('back1_roi')):>12}",
        "-" * 62,
        f"{'연승 (1순위 2착 이내)':<26}{_fmt(s.get('top1_top2')):>10}",
        f"{'복승 (상위 2명)':<26}{_fmt(s.get('top2_quinella')):>10}",
        f"{'삼복승 (상위 3명)':<26}{_fmt(s.get('top3_trio')):>10}",
        f"{'상위 3명 안에 1착':<26}{_fmt(s.get('top3_has_winner')):>10}",
        "-" * 62,
        f"로그손실 {_fmt(s.get('logloss'), False)}   AUC {_fmt(s.get('auc'), False)}"
        f"   배당 커버리지 {_fmt(s.get('coverage_payout'))}",
    ])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="경륜 승률 모델 학습/검증")
    ap.add_argument("command", choices=["train", "validate"])
    ap.add_argument("--db", default="data/cycleai.sqlite")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-train-races", type=int, default=2000)
    ap.add_argument("--since-year", type=int, default=None,
                    help="이 해부터만 학습 (오래된 규정 변화를 배제)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    conn = sqlite3.connect(args.db)
    try:
        df = build_training_frame(conn)
    finally:
        conn.close()

    if df.empty:
        print("학습 데이터가 없습니다. 먼저 수집하세요:\n"
              "  python -m cycleai.kcycle.collect backfill", file=sys.stderr)
        return 1
    if args.since_year:
        df = df[df["stnd_yr"] >= args.since_year]

    cols = feature_columns(df)
    log.info("피처 %d개", len(cols))

    evals, _ = walk_forward(df, cols, n_folds=args.folds,
                            min_train_races=args.min_train_races)
    summary = summarize(evals)
    print("\n" + report(summary) + "\n")

    if args.command == "train":
        models = fit(df, cols)
        if "win" not in models:
            print("모델 학습 실패: 레이블 부족", file=sys.stderr)
            return 1
        save(models, cols, {
            "walk_forward": summary,
            "trained_rows": len(df),
            "trained_races": int(df["race_key"].nunique()),
            "date_max": str(df["race_date"].max())[:10],
            "n_features": len(cols),
        })
        print(f"모델 저장 → {MODEL_PATH}\n검증 지표 → {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
