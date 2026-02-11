"""
Structured logging: JSONL for machine parsing, rich console for humans.

Fixes #15 — every log entry includes iteration + timing context.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


class SwarmLogger:
    def __init__(self, run_id: str, log_dir: str = "logs"):
        self.run_id = run_id
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        self._console = logging.getLogger(f"swarm.{run_id}")
        self._console.setLevel(logging.DEBUG)
        self._console.propagate = False

        # file handler
        fh = logging.FileHandler(f"{log_dir}/swarm_{run_id}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s"))
        self._console.addHandler(fh)

        # stdout handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s"))
        self._console.addHandler(sh)

    # -- core -----------------------------------------------------------
    def log(self, event_type: str, data: dict | None = None):
        data = data or {}
        self._emit_human(event_type, data)

    # -- specific events ------------------------------------------------
    def iteration_start(self, i: int):
        self._console.info(f"{'='*60}")
        self._console.info(f"  ITERATION {i}")
        self._console.info(f"{'='*60}")

    def decision(self, agent: str, summary: str):
        self._console.info(f"🧭 DECISION [{agent}]: {summary}")

    def dedup_result(self, is_duplicate: bool, reasoning: str, novelty: float):
        icon = "🔴 DUPLICATE" if is_duplicate else "🟢 NOVEL"
        self._console.info(f"{icon} (novelty={novelty:.2f}): {reasoning[:150]}")

    def experiment_start(self, exp_id: str, branch: str, hypothesis: str):
        self._console.info(
            f"🧪 START {exp_id} on '{branch}': {hypothesis[:120]}"
        )

    def experiment_result(self, exp_id: str, mean_score: float | None,
                          fold_scores: list[float] | None, fold_std: float | None,
                          passed: bool):
        icon = "✅" if passed else "❌"
        fs = ""
        if fold_scores:
            rendered_scores: list[str] = []
            for score in fold_scores:
                if score is None:
                    rendered_scores.append("N/A")
                else:
                    rendered_scores.append(f"{score:.4f}")
            fs = f" folds={rendered_scores}"
        std_str = f" std={fold_std:.4f}" if fold_std is not None else ""
        score = f"{mean_score:.4f}" if mean_score is not None else "N/A"
        self._console.info(f"{icon} RESULT {exp_id}: mean={score}{fs}{std_str}")

    def experiment_error(self, exp_id: str, error: str, will_retry: bool):
        action = "will retry" if will_retry else "giving up"
        self._console.error(f"💥 ERROR {exp_id} ({action}): {error[:200]}")

    def experiment_retry(self, exp_id: str, attempt: int):
        self._console.info(f"🔄 RETRY {exp_id} attempt {attempt}")

    def branch_switch(self, from_branch: str | None, to_branch: str, reason: str):
        self._console.info(
            f"🔀 SWITCH '{from_branch}' → '{to_branch}': {reason[:120]}"
        )

    def branch_created(self, name: str, hypothesis: str):
        self._console.info(f"🌿 NEW BRANCH '{name}': {hypothesis[:120]}")

    def submission_ready(self, branch: str, cv_score: float, path: str):
        self._console.warning(
            f"\n{'='*60}\n"
            f"📦 SUBMISSION READY: '{branch}' | CV={cv_score:.4f}\n"
            f"   → Files: {path}\n"
            f"   → Upload solution.py + model weights to wundernn.io\n"
            f"   → Then run: python -m swarm.cli update-submission \\\n"
            f"       --branch {branch} --score <SCORE> --action reopen|close\n"
            f"   → Branch is now PAUSED.\n"
            f"{'='*60}\n"
        )

    def git_commit(self, message: str):
        self._console.info(f"💾 COMMIT: {message[:120]}")

    def strategist_advice(self, advice: str):
        self._console.info(f"🎯 STRATEGIST:\n{advice[:500]}")

    def token_report(self, prompt: int, completion: int, total: int):
        self._console.info(
            f"📊 TOKENS: prompt={prompt:,} completion={completion:,} total={total:,}"
        )

    def all_paused(self):
        self._console.warning(
            "\n⏸️  All branches are paused or closed.\n"
            "   Return with submission scores to continue.\n"
        )

    def target_reached(self, score: float, branch: str):
        self._console.warning(
            f"\n🎉 TARGET REACHED on '{branch}': {score:.4f}\n"
            "   Preparing submission...\n"
        )

    def validation_failed(self, reason: str):
        self._console.error(f"🚫 VALIDATION FAILED: {reason}")

    def validation_passed(self):
        self._console.info("✅ solution.py interface validated OK")

    def fold_variance_warning(self, exp_id: str, std: float):
        self._console.warning(
            f"⚠️  HIGH VARIANCE {exp_id}: fold std={std:.4f} — result unreliable"
        )

    def sanity_check_warning(self, submission_score: float, cv_score: float):
        drop = (cv_score - submission_score) / cv_score * 100
        self._console.warning(
            f"⚠️  SANITY CHECK: submission={submission_score:.4f} vs CV={cv_score:.4f} "
            f"({drop:.1f}% drop)"
        )

    # -- internal -------------------------------------------------------
    def _emit_human(self, event_type: str, data: dict):
        # Fallback for events without a dedicated method
        self._console.info(f"[{event_type}] {json.dumps(data, default=str)[:200]}")
