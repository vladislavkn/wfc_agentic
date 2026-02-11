"""
Single source of truth for all swarm state via SQLite.

Fixes:
  #4  — SQLite handles concurrent access with WAL mode
  #6  — No separate files to drift out of sync
  #15 — Token/time tracking built into schema
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Enums & dataclasses (used as in-memory transfer objects, NOT the source of truth)
# ---------------------------------------------------------------------------

class BranchStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused_awaiting_results"
    CLOSED = "closed"
    ABANDONED = "abandoned"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED_CV = "passed_cv"
    FAILED_CV = "failed_cv"
    FAILED_ERROR = "failed_error"
    SUBMITTED = "submitted"


@dataclass
class Branch:
    name: str
    status: BranchStatus
    hypothesis: str
    best_cv_score: float = 0.0
    submission_score: Optional[float] = None
    submitted_at: Optional[str] = None
    notes: str = ""
    created_at: str = ""


@dataclass
class Experiment:
    experiment_id: str
    branch_name: str
    hypothesis: str
    config_summary: str
    status: ExperimentStatus
    fold_scores: Optional[list[float]] = None
    fold_std: Optional[float] = None
    mean_score: Optional[float] = None
    error_log: Optional[str] = None
    parent_experiment_id: Optional[str] = None
    code_path: Optional[str] = None
    training_time_s: Optional[float] = None
    created_at: str = ""


@dataclass
class IterationMetrics:
    iteration: int
    agent_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS branches (
    name            TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'active',
    hypothesis      TEXT NOT NULL,
    best_cv_score   REAL NOT NULL DEFAULT 0.0,
    submission_score REAL,
    submitted_at    TEXT,
    notes           TEXT DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id       TEXT PRIMARY KEY,
    branch_name         TEXT NOT NULL REFERENCES branches(name),
    hypothesis          TEXT NOT NULL,
    config_summary      TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending',
    fold_scores         TEXT,          -- JSON array
    fold_std            REAL,
    mean_score          REAL,
    error_log           TEXT,
    parent_experiment_id TEXT,
    code_path           TEXT,
    training_time_s     REAL,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS iteration_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration       INTEGER NOT NULL,
    agent_name      TEXT NOT NULL,
    prompt_tokens   INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    wall_time_s     REAL DEFAULT 0.0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategist_advice (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration       INTEGER NOT NULL,
    advice          TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SwarmDB:
    """Thread-safe SQLite wrapper.  One instance per process."""

    def __init__(self, db_path: str = "experiments/swarm.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()

    # -- connection per thread ------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self):
        conn = self._conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    @contextmanager
    def _tx(self):
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # -- branches -------------------------------------------------------
    def create_branch(self, name: str, hypothesis: str) -> Branch:
        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO branches (name, hypothesis, created_at) VALUES (?, ?, ?)",
                (name, hypothesis, now),
            )
        return Branch(name=name, status=BranchStatus.ACTIVE,
                      hypothesis=hypothesis, created_at=now)

    def get_branch(self, name: str) -> Optional[Branch]:
        row = self._conn().execute(
            "SELECT * FROM branches WHERE name = ?", (name,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_branch(row)

    def list_branches(self, status: Optional[BranchStatus] = None) -> list[Branch]:
        if status:
            rows = self._conn().execute(
                "SELECT * FROM branches WHERE status = ? ORDER BY created_at",
                (status.value,),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM branches ORDER BY created_at"
            ).fetchall()
        return [self._row_to_branch(r) for r in rows]

    def update_branch(self, name: str, **kwargs):
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [name]
        with self._tx() as conn:
            conn.execute(f"UPDATE branches SET {sets} WHERE name = ?", vals)

    def _row_to_branch(self, row) -> Branch:
        return Branch(
            name=row["name"], status=BranchStatus(row["status"]),
            hypothesis=row["hypothesis"], best_cv_score=row["best_cv_score"],
            submission_score=row["submission_score"],
            submitted_at=row["submitted_at"], notes=row["notes"] or "",
            created_at=row["created_at"],
        )

    # -- experiments ----------------------------------------------------
    def create_experiment(self, exp: Experiment) -> Experiment:
        exp.created_at = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO experiments
                   (experiment_id, branch_name, hypothesis, config_summary,
                    status, parent_experiment_id, code_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (exp.experiment_id, exp.branch_name, exp.hypothesis,
                 exp.config_summary, exp.status.value,
                 exp.parent_experiment_id, exp.code_path, exp.created_at),
            )
        return exp

    def update_experiment(self, experiment_id: str, **kwargs):
        if "fold_scores" in kwargs and isinstance(kwargs["fold_scores"], list):
            kwargs["fold_scores"] = json.dumps(kwargs["fold_scores"])
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [experiment_id]
        with self._tx() as conn:
            conn.execute(
                f"UPDATE experiments SET {sets} WHERE experiment_id = ?", vals
            )

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        row = self._conn().execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return self._row_to_experiment(row) if row else None

    def list_experiments(self, branch_name: Optional[str] = None) -> list[Experiment]:
        if branch_name:
            rows = self._conn().execute(
                "SELECT * FROM experiments WHERE branch_name = ? ORDER BY created_at",
                (branch_name,),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM experiments ORDER BY created_at"
            ).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def count_stagnant(self, branch_name: str, threshold: float = 0.01) -> int:
        """Count consecutive recent experiments without meaningful improvement."""
        exps = self.list_experiments(branch_name)
        scored = [e for e in exps if e.mean_score is not None]
        if len(scored) < 2:
            return 0
        best_so_far = 0.0
        stagnant = 0
        for e in scored:
            if e.mean_score > best_so_far + threshold:
                best_so_far = e.mean_score
                stagnant = 0
            else:
                stagnant += 1
        return stagnant

    def _row_to_experiment(self, row) -> Experiment:
        fs = row["fold_scores"]
        fold_scores = json.loads(fs) if fs else None
        return Experiment(
            experiment_id=row["experiment_id"],
            branch_name=row["branch_name"],
            hypothesis=row["hypothesis"],
            config_summary=row["config_summary"],
            status=ExperimentStatus(row["status"]),
            fold_scores=fold_scores,
            fold_std=row["fold_std"],
            mean_score=row["mean_score"],
            error_log=row["error_log"],
            parent_experiment_id=row["parent_experiment_id"],
            code_path=row["code_path"],
            training_time_s=row["training_time_s"],
            created_at=row["created_at"],
        )

    # -- metrics (fix #15) ----------------------------------------------
    def record_metrics(self, m: IterationMetrics):
        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO iteration_metrics
                   (iteration, agent_name, prompt_tokens, completion_tokens,
                    wall_time_s, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (m.iteration, m.agent_name, m.prompt_tokens,
                 m.completion_tokens, m.wall_time_s, now),
            )

    def total_tokens(self) -> dict:
        row = self._conn().execute(
            """SELECT COALESCE(SUM(prompt_tokens),0) as p,
                      COALESCE(SUM(completion_tokens),0) as c
               FROM iteration_metrics"""
        ).fetchone()
        return {"prompt": row["p"], "completion": row["c"],
                "total": row["p"] + row["c"]}

    # -- strategist advice (fix #9) -------------------------------------
    def save_advice(self, iteration: int, advice: str):
        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO strategist_advice (iteration, advice, created_at) VALUES (?, ?, ?)",
                (iteration, advice, now),
            )

    def latest_advice(self) -> Optional[str]:
        row = self._conn().execute(
            "SELECT advice FROM strategist_advice ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["advice"] if row else None

    # -- kv store for misc state ----------------------------------------
    def kv_set(self, key: str, value: str):
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                (key, value),
            )

    def kv_get(self, key: str, default: str = "") -> str:
        row = self._conn().execute(
            "SELECT value FROM kv WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default


# ---------------------------------------------------------------------------
# In-memory context object passed to agents via RunContext
# ---------------------------------------------------------------------------

@dataclass
class SwarmContext:
    """Passed to Runner.run(context=...).  Tools read db through this."""
    db: SwarmDB
    current_branch: Optional[str] = None
    iteration: int = 0
    target_score: float = 0.3
    cv_reject_threshold: float = 0.10
    cv_unreliable_std: float = 0.05
    stagnation_limit: int = 3
    strategist_frequency: int = 5
    max_training_timeout_s: int = 1800  # 30 min hard kill
    max_retries_per_experiment: int = 1
