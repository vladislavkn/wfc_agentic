"""
Main swarm loop.

Architecture: NO handoff chains.  Each agent is called sequentially via
separate Runner.run() calls with appropriate max_turns.

    for each iteration:
        1. [deterministic] check stagnation, force branch switch if needed
        2. [Strategist]    every N iterations, review + advise
        3. [Orchestrator]  select branch
        4. [Planner]       propose experiment
        5. [dedup check]   programmatic sub-agent call with JSON parsing
        6. [Coder]         generate training code
        7. [Evaluator]     run 3-fold CV, commit results
        8. [analyze]       check if submission threshold reached
        9. [retry]         if training errored, feed error to Coder for fix

Fixes:
  #1  — sequential Runner.run(), no handoff chain
  #2  — Context7 warmup after entering context manager
  #4  — file lock at startup
  #7  — separate max_turns per agent
  #8  — retry loop for training errors
  #9  — strategist advice injected into orchestrator prompt
  #11 — deterministic stagnation-based branch switching
  #15 — token + time tracking per iteration
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from swarm.agents import Runner, set_tracing_disabled

from swarm.agents import AgentSuite
from swarm.logger import SwarmLogger
from swarm.state import (
    BranchStatus, ExperimentStatus, IterationMetrics, SwarmContext, SwarmDB,
)
from swarm.tools import check_duplicate_via_subagent


# ---------------------------------------------------------------------------
# File lock (fix #4)
# ---------------------------------------------------------------------------
_LOCK_FILE = "experiments/.swarm.lock"


def acquire_lock() -> int:
    Path(_LOCK_FILE).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        print("ERROR: Another swarm instance is already running.", file=sys.stderr)
        sys.exit(1)
    return fd


def release_lock(fd: int):
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


# ---------------------------------------------------------------------------
# Parse agent JSON output safely
# ---------------------------------------------------------------------------
def parse_agent_json(text: str) -> dict:
    """Extract JSON from agent output, tolerating markdown fences and preamble."""
    text = text.strip()
    # Remove markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue
    # Try direct parse
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_swarm(
    vllm_url: str = "http://localhost:8000/v1",
    model_name: str = "GPT-OSS-120B",
    target_score: float = 0.3,
    max_iterations: int = 100,
    db_path: str = "experiments/swarm.db",
):
    set_tracing_disabled(True)  # vLLM doesn't support OpenAI tracing

    lock_fd = acquire_lock()
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = SwarmLogger(run_id)

    db = SwarmDB(db_path)
    ctx = SwarmContext(
        db=db,
        target_score=target_score,
        cv_reject_threshold=0.10,
    )

    suite = AgentSuite(vllm_url, model_name)

    # Fix #2: warm up Context7 MCP
    logger.log("startup", {"message": "Starting Context7 MCP server..."})
    async with suite.context7:
        # Warm-up: force tool discovery before any agent uses it
        try:
            tools = await suite.context7.list_tools()
            logger.log("startup", {
                "message": f"Context7 ready with {len(tools)} tools"
            })
        except Exception as e:
            logger.log("warning", {
                "message": f"Context7 warmup failed: {e}. DocLookup will use fallback."
            })

        try:
            await _main_loop(ctx, suite, logger, max_iterations)
        except KeyboardInterrupt:
            logger.log("shutdown", {"message": "Interrupted by user"})
        except Exception as e:
            logger.log("crash", {"error": f"{type(e).__name__}: {e}"})
            raise
        finally:
            tokens = db.total_tokens()
            logger.token_report(
                tokens["prompt"], tokens["completion"], tokens["total"])
            release_lock(lock_fd)


async def _main_loop(
    ctx: SwarmContext,
    suite: AgentSuite,
    logger: SwarmLogger,
    max_iterations: int,
):
    for i in range(max_iterations):
        ctx.iteration = i
        iter_start = time.time()
        logger.iteration_start(i)

        # ---------------------------------------------------------------
        # 0. Check if all branches are paused/closed
        # ---------------------------------------------------------------
        active_branches = ctx.db.list_branches(BranchStatus.ACTIVE)
        all_branches = ctx.db.list_branches()

        if not active_branches and all_branches:
            logger.all_paused()
            return

        # ---------------------------------------------------------------
        # Fix #11: deterministic stagnation check before asking orchestrator
        # ---------------------------------------------------------------
        if ctx.current_branch:
            stagnant_count = ctx.db.count_stagnant(
                ctx.current_branch, threshold=0.01
            )
            if stagnant_count >= ctx.stagnation_limit:
                logger.decision(
                    "System",
                    f"Branch '{ctx.current_branch}' stagnated "
                    f"({stagnant_count} experiments without improvement). "
                    f"Forcing branch switch."
                )
                ctx.db.update_branch(
                    ctx.current_branch,
                    notes=ctx.db.get_branch(ctx.current_branch).notes
                    + f" | Stagnated at iter {i}",
                )
                ctx.current_branch = None  # force orchestrator to pick new

        # ---------------------------------------------------------------
        # 1. Strategist (fix #9: every N iterations, advice injected below)
        # ---------------------------------------------------------------
        strategist_text = ""
        if i > 0 and i % ctx.strategist_frequency == 0:
            logger.decision("System", "Running strategist review...")
            try:
                strat_result = await Runner.run(
                    suite.strategist,
                    f"Review full history. Current best global score: "
                    f"{_best_global(ctx):.4f}. Target: {ctx.target_score}.",
                    context=ctx,
                    max_turns=5,
                )
                strategist_text = strat_result.final_output
                ctx.db.save_advice(i, strategist_text)
                logger.strategist_advice(strategist_text)
            except Exception as e:
                logger.log("error", {
                    "agent": "Strategist", "error": str(e)
                })

        # ---------------------------------------------------------------
        # 2. Orchestrator: select branch
        # ---------------------------------------------------------------
        latest_advice = ctx.db.latest_advice() or "No strategist advice yet."
        orch_prompt = (
            f"Iteration {i}. "
            f"Active branches: {[b.name for b in active_branches]}. "
            f"Best global score: {_best_global(ctx):.4f}. "
            f"Target: {ctx.target_score}.\n"
        )
        if ctx.current_branch:
            orch_prompt += f"Currently on branch '{ctx.current_branch}'.\n"
        # Fix #9: inject strategist advice
        orch_prompt += f"\nStrategist's latest advice:\n{latest_advice}\n"
        orch_prompt += "\nSelect a branch or create a new one. Output JSON only."

        try:
            orch_result = await Runner.run(
                suite.orchestrator, orch_prompt, context=ctx, max_turns=8,
            )
            orch_decision = parse_agent_json(orch_result.final_output)
            _record_metrics(ctx, i, "Orchestrator", iter_start)
        except Exception as e:
            logger.log("error", {"agent": "Orchestrator", "error": str(e)})
            continue

        # Process orchestrator decision
        action = orch_decision.get("action", "")
        if action == "create_branch":
            name = orch_decision.get("name", f"branch-{i}")
            hypothesis = orch_decision.get("hypothesis", "New approach")
            ctx.db.create_branch(name, hypothesis)
            ctx.current_branch = name
            logger.branch_created(name, hypothesis)
        elif action == "work_on_branch":
            new_branch = orch_decision.get("branch", "")
            if new_branch and new_branch != ctx.current_branch:
                logger.branch_switch(
                    ctx.current_branch, new_branch,
                    orch_decision.get("reasoning", "")
                )
                ctx.current_branch = new_branch
        else:
            # Fallback: pick first active branch or create one
            if active_branches:
                ctx.current_branch = active_branches[0].name
            else:
                ctx.db.create_branch(
                    f"auto-{i}", "Auto-created initial branch")
                ctx.current_branch = f"auto-{i}"
                logger.branch_created(ctx.current_branch, "Auto-created")

        logger.decision("Orchestrator", json.dumps(
            orch_decision, default=str)[:200])

        if not ctx.current_branch:
            continue

        # ---------------------------------------------------------------
        # 3. Planner: propose experiment
        # ---------------------------------------------------------------
        branch_exps = ctx.db.list_experiments(ctx.current_branch)
        branch = ctx.db.get_branch(ctx.current_branch)
        exp_history = _format_experiments_for_planner(branch_exps)

        planner_prompt = (
            f"Branch: '{ctx.current_branch}'\n"
            f"Branch hypothesis: {branch.hypothesis if branch else 'N/A'}\n"
            f"Target score: {ctx.target_score}\n"
            f"Current best in branch: {branch.best_cv_score if branch else 0:.4f}\n"
            f"Best global: {_best_global(ctx):.4f}\n\n"
            f"## Experiment history ({len(branch_exps)} experiments)\n"
            f"{exp_history}\n\n"
            f"Propose the NEXT experiment. Output JSON only."
        )

        try:
            plan_result = await Runner.run(
                suite.planner, planner_prompt, context=ctx, max_turns=5,
            )
            plan = parse_agent_json(plan_result.final_output)
            _record_metrics(ctx, i, "Planner", iter_start)
        except Exception as e:
            logger.log("error", {"agent": "Planner", "error": str(e)})
            continue

        if not plan.get("hypothesis"):
            logger.log("error", {"agent": "Planner",
                       "message": "Empty plan, skipping"})
            continue

        # ---------------------------------------------------------------
        # 4. Dedup check (fix #3: programmatic, not LLM-reads-LLM)
        # ---------------------------------------------------------------
        approach_desc = plan.get(
            "approach_description",
            f"{plan.get('hypothesis', '')} | {plan.get('config_summary', '')}"
        )
        dedup_result = await check_duplicate_via_subagent(
            suite.dedup_checker, ctx, approach_desc, ctx.current_branch
        )

        logger.dedup_result(
            dedup_result["is_duplicate"],
            dedup_result["reasoning"],
            dedup_result["novelty_score"],
        )

        if dedup_result["is_duplicate"]:
            logger.decision(
                "System",
                f"Planner's proposal is a duplicate. Skipping and re-planning next iteration."
            )
            continue

        # ---------------------------------------------------------------
        # 5. Coder: generate training code
        # ---------------------------------------------------------------
        coder_prompt = (
            f"Generate a training script for this experiment:\n\n"
            f"Branch: {ctx.current_branch}\n"
            f"Hypothesis: {plan.get('hypothesis', '')}\n"
            f"Config: {plan.get('config_summary', '')}\n"
            f"Modifications from parent: {plan.get('modifications_from_parent', 'N/A')}\n"
            f"Parent experiment: {plan.get('parent_experiment_id', 'none')}\n\n"
            f"Modify the baseline template according to the spec above. "
            f"Then call register_experiment to save it."
        )

        try:
            code_result = await Runner.run(
                suite.coder, coder_prompt, context=ctx, max_turns=10,
            )
            _record_metrics(ctx, i, "Coder", iter_start)
        except Exception as e:
            logger.log("error", {"agent": "Coder", "error": str(e)})
            continue

        # Find the experiment ID from the coder's output
        exp_id = _extract_experiment_id(code_result.final_output, ctx)
        if not exp_id:
            logger.log("error", {
                "agent": "Coder",
                "message": "Could not find registered experiment ID",
            })
            continue

        logger.experiment_start(
            exp_id, ctx.current_branch, plan.get("hypothesis", "")
        )

        # ---------------------------------------------------------------
        # 6. Evaluator: run 3-fold CV (fix #8: with retry)
        # ---------------------------------------------------------------
        eval_output = await _run_evaluation_with_retry(
            suite, ctx, logger, exp_id, plan, i, iter_start
        )

        if not eval_output:
            continue

        # ---------------------------------------------------------------
        # 7. Analyze results
        # ---------------------------------------------------------------
        mean_score = eval_output.get("mean_score", 0.0)
        fold_scores = eval_output.get("fold_scores")
        fold_std = eval_output.get("fold_std", 0.0)
        passed = eval_output.get("passed", False)

        logger.experiment_result(
            exp_id, mean_score, fold_scores, fold_std, passed)

        # Fix #5: warn on high fold variance
        if fold_std and fold_std > ctx.cv_unreliable_std:
            logger.fold_variance_warning(exp_id, fold_std)

        # Check if submission threshold reached
        if passed and mean_score >= ctx.target_score:
            logger.target_reached(mean_score, ctx.current_branch)
            await _handle_submission(suite, ctx, logger, exp_id)

        # Track iteration time
        elapsed = time.time() - iter_start
        logger.log("iteration_end", {
            "iteration": i, "elapsed_s": f"{elapsed:.1f}",
            "experiment": exp_id, "score": mean_score,
        })


# ---------------------------------------------------------------------------
# Evaluation with retry (fix #8)
# ---------------------------------------------------------------------------

async def _run_evaluation_with_retry(
    suite: AgentSuite,
    ctx: SwarmContext,
    logger: SwarmLogger,
    exp_id: str,
    plan: dict,
    iteration: int,
    iter_start: float,
) -> dict | None:
    """Run evaluation, retry once if training errors (not CV failure)."""
    for attempt in range(1 + ctx.max_retries_per_experiment):
        if attempt > 0:
            logger.experiment_retry(exp_id, attempt + 1)

        eval_prompt = (
            f"Run experiment '{exp_id}' with mode='cv'. "
            f"Then commit the results with git_commit_experiment."
        )

        try:
            eval_result = await Runner.run(
                suite.evaluator, eval_prompt, context=ctx, max_turns=8,
            )
            _record_metrics(ctx, iteration, "Evaluator", iter_start)
            eval_output = parse_agent_json(eval_result.final_output)
        except Exception as e:
            logger.log("error", {"agent": "Evaluator", "error": str(e)})
            return None

        # Check if it was a training error (not a low-score CV failure)
        exp = ctx.db.get_experiment(exp_id)
        if exp and exp.status == ExperimentStatus.FAILED_ERROR and attempt < ctx.max_retries_per_experiment:
            # Feed error back to Coder for a fix
            error_msg = exp.error_log or "Unknown error"
            logger.experiment_error(exp_id, error_msg, will_retry=True)

            fix_prompt = (
                f"The training script for experiment '{exp_id}' failed with this error:\n"
                f"```\n{error_msg[-500:]}\n```\n\n"
                f"Original config: {plan.get('config_summary', '')}\n"
                f"Fix the issue and register a NEW experiment with the corrected code. "
                f"Common fixes: reduce batch size for OOM, fix import errors, "
                f"ensure all tensors are on the same device."
            )
            try:
                fix_result = await Runner.run(
                    suite.coder, fix_prompt, context=ctx, max_turns=10,
                )
                new_exp_id = _extract_experiment_id(
                    fix_result.final_output, ctx)
                if new_exp_id:
                    exp_id = new_exp_id
                    continue  # retry with fixed code
            except Exception as e:
                logger.experiment_error(exp_id, str(e), will_retry=False)
                return None
        else:
            # Either passed, failed CV (not an error), or no retries left
            if exp and exp.status == ExperimentStatus.FAILED_ERROR:
                logger.experiment_error(
                    exp_id, exp.error_log or "Unknown", will_retry=False
                )
            return eval_output

    return None


# ---------------------------------------------------------------------------
# Submission handling
# ---------------------------------------------------------------------------

async def _handle_submission(
    suite: AgentSuite,
    ctx: SwarmContext,
    logger: SwarmLogger,
    exp_id: str,
):
    """Prepare and commit a submission package."""
    sub_prompt = (
        f"Prepare submission for experiment '{exp_id}'. "
        f"This will retrain on full data with a 5% sanity holdout."
    )
    try:
        sub_result = await Runner.run(
            suite.evaluator,
            sub_prompt,
            context=ctx,
            max_turns=10,
        )
        sub_output = parse_agent_json(sub_result.final_output)
    except Exception as e:
        logger.log("error", {"agent": "Evaluator",
                   "message": f"Submission failed: {e}"})
        return

    if sub_output.get("warning") == "SANITY CHECK FAILED":
        exp = ctx.db.get_experiment(exp_id)
        logger.sanity_check_warning(
            sub_output.get("sanity_score", 0),
            exp.mean_score if exp else 0,
        )
        return

    if sub_output.get("status") == "submission_ready":
        exp = ctx.db.get_experiment(exp_id)
        branch = ctx.current_branch or ""
        sub_dir = sub_output.get(
            "submission_dir", f"experiments/{branch}/submission")

        # Git commit submission
        try:
            await Runner.run(
                suite.evaluator,
                f"Call git_commit_submission for branch '{branch}'.",
                context=ctx,
                max_turns=3,
            )
        except Exception:
            pass

        logger.submission_ready(branch, exp.mean_score if exp else 0, sub_dir)
        logger.git_commit(f"[SUBMISSION] {branch}")
        ctx.current_branch = None  # force orchestrator to pick another


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_global(ctx: SwarmContext) -> float:
    branches = ctx.db.list_branches()
    return max((b.best_cv_score for b in branches), default=0.0)


def _format_experiments_for_planner(experiments: list) -> str:
    """Format experiment list, summarizing old ones (fix #15)."""
    if not experiments:
        return "No experiments yet."

    lines = []
    max_verbose = 15

    if len(experiments) > max_verbose:
        old = experiments[:-max_verbose]
        recent = experiments[-max_verbose:]
        scored = [e for e in old if e.mean_score is not None]
        best_old = max((e.mean_score for e in scored), default=0)
        lines.append(
            f"[{len(old)} older experiments summarized: "
            f"best_score={best_old:.4f}, "
            f"approaches: {', '.join(set(e.config_summary[:40] for e in old))}]\n"
        )
    else:
        recent = experiments

    for e in recent:
        score = f"score={e.mean_score:.4f}" if e.mean_score else "no score"
        std = f" std={e.fold_std:.4f}" if e.fold_std else ""
        err = f" ERROR: {e.error_log[:80]}" if e.error_log else ""
        lines.append(
            f"- {e.experiment_id} [{e.status.value}] {score}{std}{err}\n"
            f"  Hypothesis: {e.hypothesis}\n"
            f"  Config: {e.config_summary}"
        )

    return "\n".join(lines)


def _extract_experiment_id(text: str, ctx: SwarmContext) -> str | None:
    """Find the most recently created experiment ID from coder output or DB."""
    # Try to find exp-XXXXXXXX pattern in text
    import re
    matches = re.findall(r"exp-[a-f0-9]{8}", text)
    if matches:
        return matches[-1]

    # Fallback: get the latest experiment from DB
    all_exps = ctx.db.list_experiments(ctx.current_branch)
    if all_exps:
        return all_exps[-1].experiment_id

    return None


def _record_metrics(ctx: SwarmContext, iteration: int, agent: str, start: float):
    """Record timing metrics for an agent call."""
    elapsed = time.time() - start
    ctx.db.record_metrics(IterationMetrics(
        iteration=iteration,
        agent_name=agent,
        wall_time_s=elapsed,
        # Note: actual token counts would require intercepting the API response.
        # For vLLM, you'd parse the usage field from the response.
        # This is a placeholder — enhance by wrapping the model client.
        prompt_tokens=0,
        completion_tokens=0,
    ))
