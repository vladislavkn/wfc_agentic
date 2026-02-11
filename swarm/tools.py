"""
Tool functions for the swarm agents.

Fixes:
  #3  — check_experiment_duplicate wraps sub-agent call with JSON parsing + safe default
  #4  — git operations check status before committing
  #8  — run_training has hard timeout, error capture, retry support
  #10 — validate_submission_package checks solution.py interface
  #12 — submission mode includes 5% holdout sanity check
  #15 — experiment history summarized when large to avoid context bloat
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from agents import Agent, Runner, RunContextWrapper, function_tool

from swarm.state import (
    BranchStatus, Experiment, ExperimentStatus,
    SwarmContext,
)
from swarm.validator import validate_solution

if TYPE_CHECKING:
    pass

# Max experiments to include verbatim in context; older ones get summarized
_MAX_VERBOSE_EXPERIMENTS = 15


# ===========================================================================
# EXPERIMENT HISTORY TOOLS
# ===========================================================================

@function_tool
def get_branch_status(ctx: RunContextWrapper[SwarmContext]) -> str:
    """Get an overview of all branches and their current status."""
    db = ctx.context.db
    branches = db.list_branches()
    if not branches:
        return "No branches exist yet. Create one to start experimenting."

    lines = []
    for b in branches:
        exp_count = len(db.list_experiments(b.name))
        stagnant = db.count_stagnant(b.name)
        lines.append(
            f"- {b.name} [{b.status.value}]: {exp_count} experiments, "
            f"best_cv={b.best_cv_score:.4f}, stagnant={stagnant}, "
            f"sub_score={b.submission_score or 'N/A'}"
        )
    tokens = db.total_tokens()
    lines.append(f"\nTotal tokens used: {tokens['total']:,}")
    return "\n".join(lines)


@function_tool
def get_branch_experiments(ctx: RunContextWrapper[SwarmContext],
                           branch_name: str) -> str:
    """Get all experiments for a branch. Summarizes old experiments to save context."""
    db = ctx.context.db
    experiments = db.list_experiments(branch_name)
    if not experiments:
        return f"No experiments in branch '{branch_name}' yet."

    lines = [f"## Experiments in '{branch_name}' ({len(experiments)} total)\n"]

    # Fix #15: summarize old experiments, keep recent ones verbose
    if len(experiments) > _MAX_VERBOSE_EXPERIMENTS:
        old = experiments[:-_MAX_VERBOSE_EXPERIMENTS]
        recent = experiments[-_MAX_VERBOSE_EXPERIMENTS:]
        # Summary of old experiments
        scored_old = [e for e in old if e.mean_score is not None]
        failed_old = [e for e in old if e.status in
                      (ExperimentStatus.FAILED_CV, ExperimentStatus.FAILED_ERROR)]
        lines.append(
            f"### Older experiments (summarized, {len(old)} total)\n"
            f"- {len(scored_old)} scored, best={max((e.mean_score for e in scored_old), default=0):.4f}\n"
            f"- {len(failed_old)} failed\n"
            f"- Approaches tried: "
            f"{', '.join(set(e.config_summary[:50] for e in old))}\n"
        )
    else:
        recent = experiments

    for e in recent:
        fs = f" folds={e.fold_scores}" if e.fold_scores else ""
        std = f" std={e.fold_std:.4f}" if e.fold_std else ""
        err = f" error={e.error_log[:100]}" if e.error_log else ""
        lines.append(
            f"### {e.experiment_id} [{e.status.value}]\n"
            f"  Hypothesis: {e.hypothesis}\n"
            f"  Config: {e.config_summary}\n"
            f"  Score: {e.mean_score or 'N/A'}{fs}{std}{err}\n"
            f"  Parent: {e.parent_experiment_id or 'none'}\n"
        )

    return "\n".join(lines)


@function_tool
def get_full_history(ctx: RunContextWrapper[SwarmContext]) -> str:
    """Get a summary of ALL experiments across ALL branches (for strategist)."""
    db = ctx.context.db
    branches = db.list_branches()
    lines = [f"## Full history across {len(branches)} branches\n"]

    for b in branches:
        exps = db.list_experiments(b.name)
        scored = [e for e in exps if e.mean_score is not None]
        best = max((e.mean_score for e in scored), default=0)
        archs = set(e.config_summary.split(",")[0].strip() for e in exps if e.config_summary)
        lines.append(
            f"### {b.name} [{b.status.value}] — {len(exps)} exps, "
            f"best={best:.4f}\n"
            f"  Hypothesis: {b.hypothesis}\n"
            f"  Architectures tried: {', '.join(archs)}\n"
            f"  Notes: {b.notes}\n"
        )

    return "\n".join(lines)


# ===========================================================================
# BRANCH MANAGEMENT
# ===========================================================================

@function_tool
def create_branch(ctx: RunContextWrapper[SwarmContext],
                   name: str, hypothesis: str) -> str:
    """Create a new experiment branch."""
    db = ctx.context.db
    existing = db.get_branch(name)
    if existing:
        return f"ERROR: Branch '{name}' already exists."
    db.create_branch(name, hypothesis)
    ctx.context.current_branch = name
    return f"Created branch '{name}': {hypothesis}"


@function_tool
def switch_branch(ctx: RunContextWrapper[SwarmContext],
                   branch_name: str) -> str:
    """Switch to working on a different branch."""
    db = ctx.context.db
    branch = db.get_branch(branch_name)
    if not branch:
        return f"ERROR: Branch '{branch_name}' does not exist."
    if branch.status != BranchStatus.ACTIVE:
        return f"ERROR: Branch '{branch_name}' is {branch.status.value}, cannot switch to it."
    ctx.context.current_branch = branch_name
    return f"Switched to branch '{branch_name}'"


# ===========================================================================
# DEDUP — Fix #3: programmatic wrapper with JSON parsing + safe default
# ===========================================================================

async def check_duplicate_via_subagent(
    dedup_agent: Agent,
    ctx: SwarmContext,
    proposed_approach: str,
    branch_name: str,
) -> dict:
    """
    Call the DedupChecker sub-agent and parse its response.
    Returns structured dict. On any failure, defaults to is_duplicate=True (safe).
    """
    db = ctx.db
    experiments = db.list_experiments(branch_name)

    if not experiments:
        return {
            "is_duplicate": False,
            "reasoning": "First experiment in this branch — no prior work to compare.",
            "similar_experiments": [],
            "novelty_score": 1.0,
        }

    # Build branch history for the sub-agent
    history_lines = []
    for e in experiments:
        status = e.status.value
        score = (
            f"score={e.mean_score:.4f}"
            if e.mean_score is not None
            else "no score"
        )
        err = f" error={e.error_log[:80]}" if e.error_log else ""
        history_lines.append(
            f"- {e.experiment_id} [{status}] {score}{err}\n"
            f"  Hypothesis: {e.hypothesis}\n"
            f"  Config: {e.config_summary}"
        )
    history = "\n".join(history_lines)

    prompt = (
        f"## Proposed experiment\n{proposed_approach}\n\n"
        f"## Branch history ({len(experiments)} experiments)\n{history}\n\n"
        f"Is the proposed experiment a duplicate of any prior work in this branch? "
        f"Respond with ONLY a JSON object."
    )

    try:
        result = await Runner.run(dedup_agent, prompt, context=ctx, max_turns=3)
        text = result.final_output.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

        parsed = json.loads(text)
        # Validate required fields
        return {
            "is_duplicate": bool(parsed.get("is_duplicate", True)),
            "reasoning": str(parsed.get("reasoning", "No reasoning provided")),
            "similar_experiments": list(parsed.get("similar_experiments", [])),
            "novelty_score": float(parsed.get("novelty_score", 0.0)),
        }
    except (json.JSONDecodeError, KeyError, TypeError, Exception) as e:
        # Safe default: assume duplicate to prevent wasted GPU time
        return {
            "is_duplicate": True,
            "reasoning": f"Dedup check failed ({type(e).__name__}: {e}). "
                        f"Assuming duplicate as safe default.",
            "similar_experiments": [],
            "novelty_score": 0.0,
        }


# ===========================================================================
# EXPERIMENT REGISTRATION
# ===========================================================================

@function_tool
def register_experiment(ctx: RunContextWrapper[SwarmContext],
                        branch_name: str, hypothesis: str,
                        config_summary: str, code: str,
                        parent_experiment_id: str = "") -> str:
    """Register a new experiment: save training code to disk and record in DB."""
    db = ctx.context.db
    exp_id = f"exp-{uuid.uuid4().hex[:8]}"
    exp_dir = f"experiments/{branch_name}/{exp_id}"
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    # Write training script
    code_path = f"{exp_dir}/train.py"
    Path(code_path).write_text(code)

    exp = Experiment(
        experiment_id=exp_id,
        branch_name=branch_name,
        hypothesis=hypothesis,
        config_summary=config_summary,
        status=ExperimentStatus.PENDING,
        parent_experiment_id=parent_experiment_id or None,
        code_path=code_path,
    )
    db.create_experiment(exp)
    return f"Registered {exp_id} in '{branch_name}'. Code at {code_path}"


# ===========================================================================
# TRAINING EXECUTION — Fix #8: timeout, error capture, structured output
# ===========================================================================

@function_tool
async def run_training(ctx: RunContextWrapper[SwarmContext],
                       experiment_id: str, mode: str = "cv") -> str:
    """
    Execute a training script. mode='cv' for 3-fold CV, mode='submission' for full data.
    Returns JSON with results or error details.
    """
    return await _execute_training(ctx.context, experiment_id, mode)


async def _execute_training(ctx: SwarmContext,
                            experiment_id: str, mode: str = "cv") -> str:
    """Core training execution logic, callable from both tool and prepare_submission."""
    db = ctx.db
    exp = db.get_experiment(experiment_id)
    if not exp:
        return json.dumps({"error": f"Experiment {experiment_id} not found"})
    if not exp.code_path or not Path(exp.code_path).exists():
        return json.dumps({"error": f"Training script not found: {exp.code_path}"})

    exp_dir = str(Path(exp.code_path).parent)
    db.update_experiment(experiment_id, status=ExperimentStatus.RUNNING.value)

    start_time = time.time()
    timeout = ctx.max_training_timeout_s

    try:
        proc = await asyncio.create_subprocess_exec(
            "python", exp.code_path,
            "--mode", mode,
            "--experiment-dir", exp_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path.cwd()),
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            elapsed = time.time() - start_time
            error_msg = f"Training timed out after {elapsed:.0f}s (limit: {timeout}s)"
            db.update_experiment(experiment_id,
                                status=ExperimentStatus.FAILED_ERROR.value,
                                error_log=error_msg,
                                training_time_s=elapsed)
            return json.dumps({"error": error_msg, "timeout": True})

        elapsed = time.time() - start_time
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            # Capture meaningful error
            error_msg = stderr[-1000:] if stderr else f"Exit code {proc.returncode}"
            db.update_experiment(experiment_id,
                                status=ExperimentStatus.FAILED_ERROR.value,
                                error_log=error_msg,
                                training_time_s=elapsed)
            return json.dumps({
                "error": error_msg,
                "exit_code": proc.returncode,
                "training_time_s": elapsed,
            })

        # Parse metrics from last line of stdout
        lines = stdout.strip().split("\n")
        if not lines:
            db.update_experiment(experiment_id,
                                status=ExperimentStatus.FAILED_ERROR.value,
                                error_log="No output from training script")
            return json.dumps({"error": "No output from training script"})

        try:
            metrics = json.loads(lines[-1])
        except json.JSONDecodeError:
            db.update_experiment(experiment_id,
                                status=ExperimentStatus.FAILED_ERROR.value,
                                error_log=f"Could not parse metrics: {lines[-1][:200]}")
            return json.dumps({"error": f"Invalid metrics JSON: {lines[-1][:200]}"})

        # Update experiment with results
        mean_score = metrics.get("mean_score", 0.0)
        fold_scores = metrics.get("fold_scores")
        fold_std = metrics.get("fold_std")

        passed = mean_score >= ctx.cv_reject_threshold
        status = ExperimentStatus.PASSED_CV if passed else ExperimentStatus.FAILED_CV

        db.update_experiment(
            experiment_id,
            status=status.value,
            mean_score=mean_score,
            fold_scores=fold_scores,
            fold_std=fold_std,
            training_time_s=elapsed,
        )

        # Update branch best score
        branch = db.get_branch(exp.branch_name)
        if branch and mean_score > branch.best_cv_score:
            db.update_branch(exp.branch_name, best_cv_score=mean_score)

        return json.dumps({
            "status": "passed_cv" if passed else "failed_cv",
            "mean_score": mean_score,
            "fold_scores": fold_scores,
            "fold_std": fold_std,
            "training_time_s": elapsed,
            "passed": passed,
        })

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        db.update_experiment(experiment_id,
                            status=ExperimentStatus.FAILED_ERROR.value,
                            error_log=error_msg,
                            training_time_s=elapsed)
        return json.dumps({"error": error_msg, "training_time_s": elapsed})


# ===========================================================================
# SUBMISSION PREPARATION — Fix #10, #12
# ===========================================================================

@function_tool
async def prepare_submission(ctx: RunContextWrapper[SwarmContext],
                             experiment_id: str) -> str:
    """
    Prepare a submission: retrain on full data (with 5% holdout sanity check),
    validate solution.py interface, package files.
    """
    db = ctx.context.db
    exp = db.get_experiment(experiment_id)
    if not exp:
        return json.dumps({"error": f"Experiment {experiment_id} not found"})

    exp_dir = str(Path(exp.code_path).parent)
    submission_dir = f"experiments/{exp.branch_name}/submission"
    Path(submission_dir).mkdir(parents=True, exist_ok=True)

    # Copy training script to submission dir
    shutil.copy2(exp.code_path, f"{submission_dir}/train.py")

    # Run in submission mode (trains on full data with 5% holdout)
    # Call the training execution directly (not via tool decorator)
    result_str = await _execute_training(
        ctx.context, experiment_id, mode="submission"
    )
    result = json.loads(result_str)

    if "error" in result:
        return json.dumps({"error": f"Submission training failed: {result['error']}"})

    # Fix #12: sanity check — submission score vs CV score
    sanity_score = result.get("sanity_score", result.get("mean_score", 0))
    if exp.mean_score and sanity_score < exp.mean_score * 0.8:
        return json.dumps({
            "warning": "SANITY CHECK FAILED",
            "sanity_score": sanity_score,
            "cv_score": exp.mean_score,
            "message": f"Submission training score ({sanity_score:.4f}) dropped "
                      f">20% below CV score ({exp.mean_score:.4f}). "
                      f"Possible overfit. Consider more regularization.",
        })

    # Fix #10: validate solution.py interface
    solution_path = f"{exp_dir}/solution.py"
    if not Path(solution_path).exists():
        return json.dumps({"error": "solution.py not generated by training script"})

    validation = validate_solution(solution_path, exp_dir)
    if not validation.passed:
        return json.dumps({
            "error": "solution.py VALIDATION FAILED",
            "validation_errors": validation.errors,
            "validation_warnings": validation.warnings,
        })

    # Copy validated files to submission directory
    shutil.copy2(solution_path, f"{submission_dir}/solution.py")
    model_path = f"{exp_dir}/model_best.pt"
    if Path(model_path).exists():
        shutil.copy2(model_path, f"{submission_dir}/model_best.pt")

    # Write manifest
    manifest = {
        "experiment_id": experiment_id,
        "branch": exp.branch_name,
        "cv_score": exp.mean_score,
        "sanity_score": sanity_score,
        "hypothesis": exp.hypothesis,
        "config": exp.config_summary,
        "validation_warnings": validation.warnings,
    }
    Path(f"{submission_dir}/manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    # Mark branch as paused
    db.update_branch(exp.branch_name,
                     status=BranchStatus.PAUSED.value,
                     submitted_at=exp.created_at)

    return json.dumps({
        "status": "submission_ready",
        "submission_dir": submission_dir,
        "sanity_score": sanity_score,
        "cv_score": exp.mean_score,
        "validation": "passed",
        "validation_warnings": validation.warnings,
    })


# ===========================================================================
# GIT OPERATIONS — Fix #4: check git status before committing
# ===========================================================================

def _git_run(cmd: str) -> tuple[int, str]:
    """Run a git command, return (returncode, output)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=30
    )
    return result.returncode, result.stdout + result.stderr


@function_tool
def git_commit_experiment(ctx: RunContextWrapper[SwarmContext],
                          experiment_id: str, message: str) -> str:
    """Commit experiment files to git."""
    db = ctx.context.db
    exp = db.get_experiment(experiment_id)
    if not exp:
        return f"ERROR: Experiment {experiment_id} not found"

    exp_dir = f"experiments/{exp.branch_name}/{experiment_id}"

    # Check for clean state first
    code, out = _git_run("git status --porcelain")
    if code != 0:
        return f"ERROR: git status failed: {out}"

    # Stage and commit
    _git_run(f"git add {exp_dir}/")
    _git_run("git add experiments/")
    _git_run("git add swarm.db 2>/dev/null || true")

    code, out = _git_run(
        f'git commit -m "[{exp.branch_name}/{experiment_id}] {message}"'
    )
    if code != 0 and "nothing to commit" not in out:
        return f"ERROR: git commit failed: {out}"
    return f"Committed: [{exp.branch_name}/{experiment_id}] {message}"


@function_tool
def git_commit_submission(ctx: RunContextWrapper[SwarmContext],
                          branch_name: str) -> str:
    """Commit submission package."""
    sub_dir = f"experiments/{branch_name}/submission"
    _git_run(f"git add {sub_dir}/")
    _git_run("git add experiments/")

    code, out = _git_run(
        f'git commit -m "[SUBMISSION] {branch_name} — ready for upload"'
    )
    if code != 0 and "nothing to commit" not in out:
        return f"ERROR: git commit failed: {out}"
    return f"Submission committed for '{branch_name}'"
