"""예상 생성과 **확정 저장**.

적중률 숫자를 나중에 믿을 수 있으려면, 예측이 결과를 본 뒤에 바뀌지 않았다는
보장이 있어야 한다. 그래서 규칙은 하나다: **발주 시각이 지난 경주의 예측은
절대 다시 쓰지 않는다.** 모델을 새로 학습해도, 피처 파싱을 고쳐도 마찬가지다.

    python -m cycleai.predict upcoming          # 아직 안 달린 경주
    python -m cycleai.predict backfill          # 과거 경주 (워크포워드, 사후)

``upcoming`` 과 ``backfill`` 은 model_version 을 다르게 남긴다. 전자는 발주 전에
만든 **실전 기록**이고 후자는 사후에 만든 **모의 기록**이라, 둘을 한 숫자로
합치면 그 숫자는 아무 것도 뜻하지 않는다. verify 가 둘을 갈라서 집계한다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .clock import now_kst
from .features import build_frame, feature_columns
from .kcycle.store import session, upsert
from .model import MODEL_VERSION, fit, load, predict_frame

log = logging.getLogger(__name__)

# 사후에 만든 모의 예측임을 이름에 박아 둔다. 표에서 실전 기록과 나란히 놓였을
# 때 무엇인지 바로 보이지 않으면 언젠가 섞인다.
OOS_VERSION = f"{MODEL_VERSION}-oos"


def _post_datetime(row) -> Optional[dt.datetime]:
    """경주일자 + 발주시각 → datetime. 하나라도 없으면 None."""
    ymd, hhmm = row.get("race_ymd"), row.get("post_time")
    if not ymd or not hhmm:
        return None
    try:
        return dt.datetime.strptime(f"{str(ymd)[:8]} {str(hhmm)[:5]}", "%Y%m%d %H:%M")
    except ValueError:
        return None


def frozen_keys(conn: sqlite3.Connection, version: str) -> set:
    return {r[0] for r in conn.execute(
        "SELECT DISTINCT race_key FROM predictions WHERE model_version = ?",
        (version,))}


def _write(conn: sqlite3.Connection, pred: pd.DataFrame, version: str) -> int:
    rows = [{
        "race_key": r.race_key, "back_no": int(r.back_no), "racer_nm": r.racer_nm,
        "p_win": float(r.p_win_norm),
        "p_top2": None if pd.isna(r.p_top2_norm) else float(r.p_top2_norm),
        "p_top3": None if pd.isna(r.p_top3_norm) else float(r.p_top3_norm),
        "pred_rank": int(r.pred_rank), "model_version": version,
    } for r in pred.itertuples()]
    n = upsert(conn, "predictions", rows, ["race_key", "back_no", "model_version"])
    conn.commit()
    return n


def predict_upcoming(conn: sqlite3.Connection) -> int:
    """아직 발주하지 않은 경주에 예측을 만든다.

    이미 예측이 있는 경주는 건드리지 않는다. 발주가 지난 경주도 만들지 않는다 —
    출주표는 나중에도 조회되므로, 막지 않으면 '결과를 아는 상태에서 만든 예측'이
    실전 기록에 섞여 들어간다.
    """
    bundle = load()
    models, cols = bundle["models"], bundle["features"]

    df = build_frame(conn, with_labels=False)
    if df.empty:
        log.info("출주표가 없습니다.")
        return 0

    now = now_kst()
    have = frozen_keys(conn, MODEL_VERSION)

    rows = df.to_dict("records")
    keep_keys = set()
    for r in rows:
        if r["race_key"] in have or r.get("has_result"):
            continue
        pt = _post_datetime(r)
        # 발주 시각을 모르면 예측하지 않는다. 모른 채 만들면 그것이 발주 전에
        # 만들어졌다고 주장할 근거가 없다.
        if pt is None or pt <= now:
            continue
        keep_keys.add(r["race_key"])

    if not keep_keys:
        log.info("예측할 경주가 없습니다 (발주 전 미예측 경주 0개).")
        # 예측은 있는데 전개 시뮬레이션만 없는 경우가 있다 — 기능이 나중에
        # 생겼기 때문이다. **아직 발주 전인 경주에 한해서만** 채운다.
        pending = {r["race_key"] for r in rows
                   if r["race_key"] in have and not r.get("has_result")
                   and (_post_datetime(r) or now) > now}
        todo = [k for k in pending if not _has_simulation(conn, k)]
        if todo:
            sub = df[df["race_key"].isin(todo)].copy()
            n = _freeze_simulations(conn, _attach_stored_probs(conn, sub))
            log.info("전개 시뮬레이션 %d경주 추가 저장", n)
        return 0

    target = df[df["race_key"].isin(keep_keys)].copy()
    pred = predict_frame(models, target, cols)
    n = _write(conn, pred, MODEL_VERSION)
    n_sim = _freeze_simulations(conn, pred)
    log.info("예상 %d행 / %d경주 확정 저장 (전개 시뮬레이션 %d경주)",
             n, len(keep_keys), n_sim)
    return n


def _has_simulation(conn: sqlite3.Connection, race_key: str) -> bool:
    return conn.execute("SELECT 1 FROM simulations WHERE race_key=?",
                        (race_key,)).fetchone() is not None


def _attach_stored_probs(conn: sqlite3.Connection, df: pd.DataFrame) -> pd.DataFrame:
    """**저장된** 예측 확률을 붙인다.

    시뮬레이션은 예측 확률에 닻을 내린다. 모델을 다시 돌려 확률을 새로 만들면
    이미 저장된 예상과 어긋난 전개가 나온다 — 화면의 순위와 전개가 다른 말을
    하게 된다. 그래서 DB 에 있는 그 값을 그대로 쓴다.
    """
    stored = pd.read_sql_query(
        "SELECT race_key, back_no, p_win AS p_win_norm FROM predictions "
        "WHERE model_version = ?", conn, params=[MODEL_VERSION])
    return df.drop(columns=["p_win_norm"], errors="ignore").merge(
        stored, on=["race_key", "back_no"], how="inner")


def _freeze_simulations(conn: sqlite3.Connection, pred: pd.DataFrame) -> int:
    """전개 시뮬레이션을 예측과 **같은 시점에** 고정한다.

    나중에 다시 돌리면 그때의 상수로 다른 전개가 나온다. 발주 전에 화면에 있던
    전개와 결과를 본 뒤의 전개가 다르면, 그 전개는 아무것도 증명하지 못한다.
    """
    from .simulate import simulate as run_sim

    done = 0
    for key, g in pred.groupby("race_key"):
        runners = g.to_dict("records")
        for r in runners:
            r["p_win"] = r.get("p_win_norm", r.get("p_win"))
        sim = run_sim(runners)
        if not sim:
            continue
        conn.execute(
            "INSERT INTO simulations(race_key,payload,conf_label,conf_score,"
            "top_tactic,n_sims) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(race_key) DO NOTHING",
            (key, json.dumps(sim, ensure_ascii=False, default=float),
             sim["confidence"]["label"], sim["confidence"]["score"],
             sim["top_tactic"], sim["n_sims"]))
        done += 1
    conn.commit()
    return done


def predict_backfill(conn: sqlite3.Connection, n_folds: int = 5,
                     min_train_races: int = 2000) -> int:
    """과거 경주에 **워크포워드 표본외** 예측을 남긴다.

    각 구간의 예측은 그 구간 이전 자료만으로 학습한 모델이 만든다. 사후에
    만들었다는 점은 변하지 않지만, 적어도 자기 결과를 보고 만든 것은 아니다.
    승식별 회수율처럼 표본이 많아야 의미가 생기는 지표를 지금 당장 보기 위한
    용도이고, 실전 기록이 쌓이면 그쪽으로 대체된다.
    """
    from .features import build_training_frame

    df = build_training_frame(conn)
    if df.empty:
        log.info("학습 데이터가 없습니다.")
        return 0
    cols = feature_columns(df)
    df = df.sort_values("order_key").reset_index(drop=True)
    keys = df["order_key"].dropna().sort_values().unique()
    splits = np.linspace(len(keys) * 0.5, len(keys), n_folds + 1).astype(int)

    total = 0
    for i in range(n_folds):
        cut = keys[splits[i] - 1]
        end = keys[min(splits[i + 1] - 1, len(keys) - 1)]
        train = df[df["order_key"] <= cut]
        test = df[(df["order_key"] > cut) & (df["order_key"] <= end)]
        if train["race_key"].nunique() < min_train_races or test.empty:
            continue
        models = fit(train, cols)
        if "win" not in models:
            continue
        pred = predict_frame(models, test, cols)
        total += _write(conn, pred, OOS_VERSION)
        log.info("fold %d → %d경주 예측 기록", i + 1, test["race_key"].nunique())
    return total


# ---------------------------------------------------------------------------
# 확정 기록의 보존
# ---------------------------------------------------------------------------
#
# 발주 전에 확정 저장한 예상은 **절대 잃으면 안 되는 기록**이다. 적중률이
# 사후 조작이 아니라는 근거가 오직 이것뿐이기 때문이다. 그런데 자동 배포에서
# DB 는 임시 캐시에 있고, 캐시는 만료되거나 지워진다 — 그 순간 지금까지의
# 실전 기록이 통째로 사라진다.
#
# 그래서 확정 기록만 따로 파일로 떠서 저장소에 남긴다. 경주당 7행이라 한 해를
# 다 모아도 몇 MB 다. 수집 자료(수백 MB)와 달리 저장소에 담을 만한 크기다.

FROZEN_PATH = Path("records/frozen.json")


def export_frozen(conn: sqlite3.Connection, path: Path = FROZEN_PATH) -> int:
    """확정 저장한 예상과 전개를 파일로 뜬다."""
    preds = [dict(r) for r in conn.execute(
        "SELECT race_key, back_no, racer_nm, p_win, p_top2, p_top3, pred_rank, "
        "model_version, created_at FROM predictions WHERE model_version = ? "
        "ORDER BY race_key, back_no", (MODEL_VERSION,))]
    sims = [dict(r) for r in conn.execute(
        "SELECT race_key, payload, conf_label, conf_score, top_tactic, n_sims, "
        "created_at FROM simulations ORDER BY race_key")]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"predictions": preds, "simulations": sims},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    return len(preds)


def import_frozen(conn: sqlite3.Connection, path: Path = FROZEN_PATH) -> int:
    """파일의 확정 기록을 DB 로 되돌린다.

    **이미 있는 행은 절대 덮어쓰지 않는다.** 확정 기록은 한 번 쓰이면 바뀌지
    않는 것이고, 되돌리는 과정에서 값이 달라지면 그 기록은 아무것도 증명하지
    못한다. 없는 것만 채운다.
    """
    if not path.exists():
        return 0
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        log.warning("확정 기록 파일을 읽지 못했습니다: %s", path)
        return 0
    n = 0
    for r in blob.get("predictions", []):
        cols = ",".join(r)
        conn.execute(
            f"INSERT INTO predictions({cols}) VALUES({','.join('?' * len(r))}) "
            "ON CONFLICT(race_key, back_no, model_version) DO NOTHING",
            tuple(r.values()))
        n += 1
    for r in blob.get("simulations", []):
        cols = ",".join(r)
        conn.execute(
            f"INSERT INTO simulations({cols}) VALUES({','.join('?' * len(r))}) "
            "ON CONFLICT(race_key) DO NOTHING", tuple(r.values()))
    conn.commit()
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="경륜 예측 생성")
    ap.add_argument("command", choices=["upcoming", "backfill", "export", "import"])
    ap.add_argument("--db", default="data/cycleai.sqlite")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with session(args.db) as conn:
        if args.command == "upcoming":
            try:
                predict_upcoming(conn)
            except FileNotFoundError as e:
                print(f"✗ {e}", file=sys.stderr)
                return 1
        elif args.command == "backfill":
            predict_backfill(conn, n_folds=args.folds)
        elif args.command == "export":
            n = export_frozen(conn)
            print(f"확정 기록 {n}행 → {FROZEN_PATH}")
        elif args.command == "import":
            n = import_frozen(conn)
            print(f"확정 기록 {n}행 복원 (이미 있는 행은 그대로)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
