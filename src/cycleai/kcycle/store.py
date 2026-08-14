"""SQLite 저장소.

경주 단위 자료는 재수집이 잦으므로 모든 쓰기를 upsert 로 처리한다. 원본 응답은
``raw_json`` 에 그대로 남겨서, 나중에 파싱 규칙이 틀렸다는 게 드러나도 API 를
다시 때리지 않고 로컬에서 재정규화할 수 있게 한다 — 경륜 API 는 필드 하나에
값 두세 개를 붙여 주는 곳이 많아(``"선발 5-1"``) 규칙이 바뀔 여지가 특히 크다.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from ..clock import today_kst

DEFAULT_DB = Path("data/cycleai.sqlite")

SCHEMA = """
PRAGMA journal_mode=WAL;

-- 경주 단위 메타.
--
-- 경정과 달리 **경륜장 축이 있다**. 출주표는 광명만 나오지만 착순은 창원·부산도
-- 오므로, 키에 경륜장을 넣지 않으면 같은 날 같은 회차 같은 경주번호가 세 곳에서
-- 겹쳐 서로를 덮어쓴다.
CREATE TABLE IF NOT EXISTS races (
    race_key    TEXT PRIMARY KEY,
    stnd_yr     INTEGER NOT NULL,
    meet_nm     TEXT NOT NULL,
    week_tcnt   INTEGER NOT NULL,
    day_tcnt    INTEGER NOT NULL,
    race_no     INTEGER NOT NULL,
    race_ymd    TEXT,            -- YYYYMMDD 로 통일해서 넣는다
    post_time   TEXT,            -- 발주 시각. 예상을 더 이상 고치지 않는 기준선이다.
    race_grade  TEXT,            -- 선발/우수/특선 — 출전 선수 등급에서 채운다
    race_len    INTEGER,         -- 경주거리(m)
    round_cnt   INTEGER,         -- 주회수
    field_size  INTEGER,
    has_card    INTEGER NOT NULL DEFAULT 0,   -- 출주표(=피처)가 있는가
    has_result  INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_races_ymd   ON races(race_ymd);
CREATE INDEX IF NOT EXISTS idx_races_coord ON races(stnd_yr, week_tcnt, day_tcnt);
CREATE INDEX IF NOT EXISTS idx_races_meet  ON races(meet_nm);

-- 출주표: 경주 전에 확정되는 정보. 피처는 전부 여기서 나온다.
CREATE TABLE IF NOT EXISTS entries (
    race_key    TEXT NOT NULL,
    back_no     INTEGER NOT NULL,     -- 배번 1~7
    racer_nm    TEXT,
    racer_grd   TEXT,                 -- 선발/우수/특선 (경주 급)
    grade_cur   TEXT,                 -- A1~B3 (등급 조정 후)
    grade_bef   TEXT,                 -- 등급 조정 전
    period_no   INTEGER,              -- 기수
    age         INTEGER,
    trng_plc    TEXT,                 -- 훈련지
    color_nm    TEXT,

    gear_rate   REAL,                 -- 기어배수
    rec_200m    REAL,                 -- 200m 기록(초)

    -- 전법별 입상 횟수. 경정의 '코스별 연대율' 자리를 이것이 차지한다.
    pre_win_cnt INTEGER,              -- 선행
    pas_win_cnt INTEGER,              -- 젖히기
    brk_win_cnt INTEGER,              -- 추입
    mrk_win_cnt INTEGER,              -- 마크
    win_tot_cnt INTEGER,              -- 입상 합계
    run_day_cnt INTEGER,              -- 출전 횟수

    win_rate    REAL,
    high_rate   REAL,                 -- 연대율
    high_3_rate REAL,                 -- 삼연대율
    tot_avg_scr REAL,                 -- 종합 평균득점
    area_avg_scr REAL,                -- 광명 평균득점

    -- 최근 3회전 × 3일차 착순. 개별 값을 남긴다 — 요약만 두면 나중에
    -- '최근 1회전만' 같은 다른 창을 못 만든다.
    bf1_meet TEXT, bf1_ymd TEXT, bf1_d1 INTEGER, bf1_d2 INTEGER, bf1_d3 INTEGER,
    bf2_meet TEXT, bf2_ymd TEXT, bf2_d1 INTEGER, bf2_d2 INTEGER, bf2_d3 INTEGER,
    bf3_meet TEXT, bf3_ymd TEXT, bf3_d1 INTEGER, bf3_d2 INTEGER, bf3_d3 INTEGER,
    bf_avg   REAL, bf_cnt INTEGER,

    raw_json    TEXT,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (race_key, back_no),
    FOREIGN KEY (race_key) REFERENCES races(race_key)
);
CREATE INDEX IF NOT EXISTS idx_entries_racer ON entries(racer_nm);

-- 착순. 실격·기권으로 숫자가 아닐 수 있어 원문을 ord_note 에 남긴다.
-- **창원·부산 것도 넣는다** — 광명 선수의 원정 성적이 폼 피처가 되기 때문이다.
CREATE TABLE IF NOT EXISTS results (
    race_key   TEXT NOT NULL,
    racer_nm   TEXT NOT NULL,
    racer_no   TEXT,
    back_no    INTEGER,          -- 출주표에서 이어 붙인다 (착순 API 엔 없다)
    ord        INTEGER,
    ord_note   TEXT,
    raw_json   TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (race_key, racer_nm)
);
CREATE INDEX IF NOT EXISTS idx_results_ord   ON results(ord);
CREATE INDEX IF NOT EXISTS idx_results_racer ON results(racer_nm);

-- 승식별 확정배당. 우리 추천 조합이 실제로 얼마를 돌려줬는지 검증하는 근거다.
CREATE TABLE IF NOT EXISTS payoffs (
    race_key   TEXT NOT NULL,
    pool       TEXT NOT NULL,     -- 단승·연승1·연승2·쌍승·복승·삼복승·삼쌍승
    combo      TEXT,              -- 적중 조합 (경주결과 API 에만 있다)
    payout     REAL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (race_key, pool)
);
CREATE INDEX IF NOT EXISTS idx_payoffs_race ON payoffs(race_key);

-- 예측: 발주 전에 만들어 고정한다. 공개(기록) 후 수정하지 않는 것이
-- 적중률 숫자를 믿을 수 있게 하는 유일한 근거다.
CREATE TABLE IF NOT EXISTS predictions (
    race_key      TEXT NOT NULL,
    back_no       INTEGER NOT NULL,
    racer_nm      TEXT,
    p_win         REAL NOT NULL,
    p_top2        REAL,
    p_top3        REAL,
    pred_rank     INTEGER,
    model_version TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (race_key, back_no, model_version)
);
CREATE INDEX IF NOT EXISTS idx_pred_race ON predictions(race_key);

-- 전개 시뮬레이션. **예상과 함께 확정 저장한다** — 나중에 다시 돌리면 그때의
-- 상수와 난수로 다른 전개가 나와, 발주 전에 화면에 있던 것과 달라진다.
CREATE TABLE IF NOT EXISTS simulations (
    race_key   TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    conf_label TEXT,
    conf_score INTEGER,
    top_tactic TEXT,
    n_sims     INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 선수 연도별 성적 집계(선수정보 API). 착순별 횟수까지 있어 사전값으로 쓴다.
CREATE TABLE IF NOT EXISTS racer_year (
    stnd_yr    INTEGER NOT NULL,
    racer_nm   TEXT NOT NULL,
    period_no  INTEGER,
    grade      TEXT,
    run_cnt    INTEGER, run_day_cnt INTEGER,
    rank1 INTEGER, rank2 INTEGER, rank3 INTEGER, rank4 INTEGER, rank5 INTEGER,
    rank6 INTEGER, rank7 INTEGER, rank8 INTEGER, rank9 INTEGER,
    win_rate REAL, high_rate REAL, high_3_rate REAL,
    down_cnt INTEGER, elim_cnt INTEGER, go_po_cnt INTEGER,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (stnd_yr, racer_nm)
);

-- 회차별 경주득점 스냅샷. 회차 단위라 '그 시점까지의 폼'을 누수 없이 쓸 수 있다.
CREATE TABLE IF NOT EXISTS tms_score (
    stnd_yr   INTEGER NOT NULL,
    week_tcnt INTEGER NOT NULL,
    day_tcnt  INTEGER NOT NULL,
    meet_nm   TEXT NOT NULL,
    racer_nm  TEXT NOT NULL,
    race_scr  REAL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (stnd_yr, week_tcnt, day_tcnt, meet_nm, racer_nm)
);

-- 낙차·사고. 경륜에서 낙차는 결과를 직접 뒤집는다.
CREATE TABLE IF NOT EXISTS accidents (
    stnd_yr   INTEGER NOT NULL,
    week_tcnt INTEGER NOT NULL,
    day_tcnt  INTEGER NOT NULL,
    race_no   INTEGER NOT NULL,
    racer_nm  TEXT NOT NULL,
    kind      TEXT,          -- 낙차/사고
    reason    TEXT,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (stnd_yr, week_tcnt, day_tcnt, race_no, racer_nm)
);
CREATE INDEX IF NOT EXISTS idx_acc_racer ON accidents(racer_nm);

-- 제재선수. 출전 정지가 편성에 그대로 나타난다.
CREATE TABLE IF NOT EXISTS sanctions (
    racer_id  TEXT NOT NULL,
    racer_nm  TEXT NOT NULL,
    kind      TEXT NOT NULL,
    period    TEXT,
    reason    TEXT,
    race_ref  TEXT,          -- 사유 문장에서 뽑아낸 해당 경주
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (racer_nm, kind, reason)
);

-- 상대전적(동반입상 포함). 경륜은 라인을 짜고 함께 들어오는 종목이라
-- 동반입상 횟수가 실제 피처가 된다.
CREATE TABLE IF NOT EXISTS oppo (
    stnd_yr   INTEGER NOT NULL,
    racer_nm  TEXT NOT NULL,
    oppo_nm   TEXT NOT NULL,
    win_cnt   INTEGER, lose_cnt INTEGER, draw_cnt INTEGER, same_win_cnt INTEGER,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (stnd_yr, racer_nm, oppo_nm)
);
CREATE INDEX IF NOT EXISTS idx_oppo_racer ON oppo(racer_nm);

-- 수집 이력 (증분 수집용). 0건이었던 좌표도 '확인 완료'로 남긴다 —
-- 개최가 없던 회차를 매번 다시 묻는 것은 일일 호출량의 낭비다.
CREATE TABLE IF NOT EXISTS fetch_log (
    endpoint   TEXT NOT NULL,
    coord      TEXT NOT NULL,
    n_records  INTEGER NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (endpoint, coord)
);
"""


def connect(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def session(path: Path | str = DEFAULT_DB) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        init_db(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]


def upsert(conn: sqlite3.Connection, table: str, rows: Iterable[Dict[str, Any]],
           key_cols: List[str]) -> int:
    """존재하는 컬럼만 골라 upsert. 스키마에 없는 키는 조용히 버린다.

    **빈 값으로는 덮어쓰지 않는다.** 같은 행을 여러 API 가 조각조각 채운다 —
    배번은 출주표에만, 선수번호는 착순에만 있다. 들어온 값이 비었다고 기존 값을
    지우면 나중 수집이 앞선 수집을 무효로 만든다.
    """
    rows = [r for r in rows if r]
    if not rows:
        return 0
    cols = [c for c in _table_columns(conn, table) if c != "updated_at"]
    usable = [c for c in cols if any(c in r for r in rows)]
    if not usable:
        return 0

    placeholders = ",".join("?" for _ in usable)
    update_cols = [c for c in usable if c not in key_cols]
    set_clause = ", ".join(f"{c}=COALESCE(excluded.{c}, {table}.{c})" for c in update_cols)
    if "updated_at" in _table_columns(conn, table):
        set_clause = (set_clause + ", " if set_clause else "") + "updated_at=datetime('now')"

    sql = (f"INSERT INTO {table} ({','.join(usable)}) VALUES ({placeholders}) "
           f"ON CONFLICT({','.join(key_cols)}) DO UPDATE SET {set_clause}")
    conn.executemany(sql, [tuple(r.get(c) for c in usable) for r in rows])
    return len(rows)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def log_fetch(conn: sqlite3.Connection, endpoint: str, coord: str, n: int) -> None:
    conn.execute(
        "INSERT INTO fetch_log(endpoint,coord,n_records) VALUES(?,?,?) "
        "ON CONFLICT(endpoint,coord) DO UPDATE SET "
        "n_records=excluded.n_records, fetched_at=datetime('now')",
        (endpoint, coord, n),
    )


def already_fetched(conn: sqlite3.Connection, endpoint: str, coord: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM fetch_log WHERE endpoint=? AND coord=?", (endpoint, coord)
    ).fetchone()
    return row is not None


def mark_flags(conn: sqlite3.Connection) -> Dict[str, int]:
    """출주표·착순 유무 플래그와 출주 인원을 채운다."""
    out = {}
    cur = conn.execute(
        "UPDATE races SET has_result=1 WHERE has_result=0 AND race_key IN "
        "(SELECT race_key FROM results WHERE ord IS NOT NULL)")
    out["has_result"] = cur.rowcount or 0
    cur = conn.execute(
        "UPDATE races SET has_card=1 WHERE has_card=0 AND race_key IN "
        "(SELECT DISTINCT race_key FROM entries)")
    out["has_card"] = cur.rowcount or 0
    cur = conn.execute(
        "UPDATE races SET field_size = ("
        "  SELECT COUNT(*) FROM entries e WHERE e.race_key = races.race_key) "
        "WHERE field_size IS NULL OR field_size = 0")
    out["field_size"] = cur.rowcount or 0
    return out


def link_result_back_no(conn: sqlite3.Connection) -> int:
    """착순 행에 배번을 이어 붙인다.

    착순 API 는 배번을 주지 않는다(선수번호·선수명·착순뿐). 배번은 출주표에만
    있으므로 **경주 안에서 선수명으로 잇는다**. 한 경주에 일곱 명뿐이라 동명이인
    충돌은 사실상 없다 — 위험한 것은 경주를 가로지르는 이름 집계 쪽이고, 그건
    선수번호로 따로 푼다.
    """
    cur = conn.execute(
        "UPDATE results SET back_no = ("
        "  SELECT e.back_no FROM entries e "
        "  WHERE e.race_key = results.race_key AND e.racer_nm = results.racer_nm) "
        "WHERE back_no IS NULL")
    return cur.rowcount or 0


def prune_raw_json(conn: sqlite3.Connection, keep_days: int = 365) -> int:
    """오래된 행의 ``raw_json`` 을 비운다.

    raw_json 의 가치는 파싱 규칙을 고치는 초기에 집중되는 반면 용량은 계속
    늘어난다. 최근 구간만 남겨 DB 를 옮길 수 있는 크기로 유지한다.
    """
    cutoff = (today_kst() - dt.timedelta(days=keep_days)).strftime("%Y%m%d")
    total = 0
    for table in ("entries", "results"):
        cur = conn.execute(
            f"UPDATE {table} SET raw_json = NULL WHERE raw_json IS NOT NULL AND race_key IN "
            f"(SELECT race_key FROM races WHERE race_ymd IS NOT NULL AND race_ymd < ?)",
            (cutoff,))
        total += cur.rowcount or 0
    conn.commit()
    conn.execute("VACUUM")
    return total


def counts(conn: sqlite3.Connection) -> Dict[str, int]:
    out = {}
    for t in ("races", "entries", "results", "payoffs", "predictions",
              "simulations", "racer_year", "tms_score", "accidents",
              "sanctions", "oppo"):
        try:
            out[t] = conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
        except sqlite3.Error:
            out[t] = 0
    return out
