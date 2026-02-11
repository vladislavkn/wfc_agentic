"""
Agent definitions.

All agents use GPT-OSS-120B via vLLM.  Sub-agents (DedupChecker, DocLookup)
are short-lived and invoked via Runner.run() or .as_tool() to preserve
parent context.

Fixes:
  #1  — No handoff chains; main.py calls each agent sequentially
  #3  — DedupChecker is wrapped by check_duplicate_via_subagent() in tools.py
  #9  — Strategist advice injected into Orchestrator prompt in main loop
"""

from __future__ import annotations

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.mcp import MCPServerStdio

from swarm.state import SwarmContext
from swarm.tools import (
    create_branch,
    get_branch_experiments,
    get_branch_status,
    get_full_history,
    git_commit_experiment,
    git_commit_submission,
    prepare_submission,
    register_experiment,
    run_training,
    switch_branch,
)


# ---------------------------------------------------------------------------
# Model & MCP setup
# ---------------------------------------------------------------------------

def build_model(base_url: str = "http://localhost:8000/v1",
                model_name: str = "GPT-OSS-120B") -> OpenAIChatCompletionsModel:
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


def build_context7() -> MCPServerStdio:
    return MCPServerStdio(
        name="context7",
        params={
            "command": "npx",
            "args": ["-y", "@upstash/context7-mcp@latest"],
        },
        cache_tools_list=True,
    )


# ---------------------------------------------------------------------------
# Read baseline template (fix #14 — coder gets working template)
# ---------------------------------------------------------------------------
_BASELINE_TEMPLATE_PATH = "swarm/templates/baseline_train.py"


def _read_baseline_template() -> str:
    try:
        from pathlib import Path
        return Path(_BASELINE_TEMPLATE_PATH).read_text()
    except FileNotFoundError:
        return "(baseline template not found — write training code from scratch)"


# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------

def build_dedup_checker(model: OpenAIChatCompletionsModel) -> Agent[SwarmContext]:
    return Agent[SwarmContext](
        name="DedupChecker",
        model=model,
        instructions=(
            "You are a duplication detector for ML experiments on the "
            "Wunderfund Predictorium challenge (LOB data, weighted Pearson scoring).\n\n"
            "You receive a proposed experiment and the complete history of experiments "
            "in the current branch.\n\n"
            "An experiment is a DUPLICATE if:\n"
            "- The core model architecture is the same AND the key hyperparameters are "
            "substantially similar (e.g., same GRU 256 hidden vs 264 hidden)\n"
            "- The feature engineering approach is identical even if worded differently\n"
            "- The only change is cosmetic (variable names, code style)\n\n"
            "An experiment is NOT a duplicate if:\n"
            "- Same architecture but meaningfully different config "
            "(e.g., GRU 256→512 hidden with dropout added)\n"
            "- Genuinely new technique on existing architecture "
            "(e.g., adding attention to GRU)\n"
            "- Prior experiment failed due to a bug, this one fixes it "
            "(must be explicitly stated as the reason)\n\n"
            "Respond with ONLY a JSON object, no markdown fences, no preamble:\n"
            '{"is_duplicate": true/false, "reasoning": "...", '
            '"similar_experiments": ["exp-xxx"], "novelty_score": 0.0-1.0}'
        ),
    )


def build_doc_lookup(model: OpenAIChatCompletionsModel,
                     context7: MCPServerStdio) -> Agent[SwarmContext]:
    return Agent[SwarmContext](
        name="DocLookup",
        model=model,
        instructions=(
            "You look up library documentation using the Context7 tools.\n\n"
            "Steps:\n"
            "1. Use resolve-library-id to find the library\n"
            "2. Use get-library-docs with a focused topic query\n"
            "3. Return ONLY the relevant API signature + minimal code example\n\n"
            "Keep responses under 500 tokens. No explanations beyond the code."
        ),
        mcp_servers=[context7],
    )


# ---------------------------------------------------------------------------
# Main agents
# ---------------------------------------------------------------------------

def build_orchestrator(model: OpenAIChatCompletionsModel) -> Agent[SwarmContext]:
    """
    NOTE: The orchestrator does NOT use handoffs.  The main loop calls
    each agent sequentially (fix #1).  The orchestrator only decides
    which branch to work on and what strategy to follow.
    """
    return Agent[SwarmContext](
        name="Orchestrator",
        model=model,
        instructions=(
            "You coordinate an ML experiment swarm targeting a weighted Pearson "
            "correlation score on the Wunderfund Predictorium challenge.\n\n"
            "Your ONLY job each iteration is to:\n"
            "1. Review all branches via get_branch_status\n"
            "2. Select which ACTIVE branch to work on next (or create a new one)\n"
            "3. Output your decision as a JSON object:\n"
            '   {"action": "work_on_branch", "branch": "<name>", "reasoning": "..."}\n'
            '   {"action": "create_branch", "name": "<name>", "hypothesis": "...", '
            '"reasoning": "..."}\n\n'
            "Rules:\n"
            "- Never select a branch with status 'paused_awaiting_results' or 'closed'\n"
            "- If all active branches are stagnating, create a new one\n"
            "- Consider the strategist's latest advice if provided\n"
            "- Output ONLY the JSON decision, nothing else"
        ),
        tools=[get_branch_status, create_branch, switch_branch],
    )


def build_planner(model: OpenAIChatCompletionsModel) -> Agent[SwarmContext]:
    """
    The planner proposes experiments. Dedup checking happens in the main loop
    via check_duplicate_via_subagent() before the planner's proposal is accepted.
    """
    return Agent[SwarmContext](
        name="Planner",
        model=model,
        instructions=(
            "You plan the next experiment for a given branch on the Wunderfund "
            "Predictorium challenge.\n\n"
            "You will receive:\n"
            "- The current branch name and its full experiment history\n"
            "- The target score and current best score\n\n"
            "Your job:\n"
            "1. Review ALL experiments in this branch (provided to you)\n"
            "2. Propose the next experiment that is DIFFERENT from all prior work\n"
            "3. Output a JSON object:\n"
            '   {"hypothesis": "what you expect to improve and why",\n'
            '    "config_summary": "architecture, key hyperparams, features",\n'
            '    "approach_description": "detailed natural language description '
            'of the full approach for duplication checking",\n'
            '    "parent_experiment_id": "exp-xxx or empty if new approach",\n'
            '    "modifications_from_parent": "what changed and why"}\n\n'
            "Challenge details:\n"
            "- Dataset: LOB data, 32 features (bid/ask prices p0-p11, volumes v0-v11, "
            "trade deltas dp0-dp3/dv0-dv3)\n"
            "- Targets: t0, t1 (price movement indicators)\n"
            "- Scoring: weighted Pearson correlation (weight = abs(target))\n"
            "- Steps 0-98 warmup (not scored), 99-999 scored\n"
            "- 1000 timesteps per sequence, predictions per-timestep\n"
            "- GPU: RTX 5070 Ti (16GB VRAM)\n\n"
            "Strategy priorities:\n"
            "- Transformer, GRU, LSTM, Mamba-2, CNN+RNN hybrids\n"
            "- Feature engineering: bid-ask spread, volume imbalance, order flow\n"
            "- Loss function aligned with weighted Pearson metric\n"
            "- Ensemble diverse models once 2+ score >0.2\n\n"
            "Output ONLY the JSON, no markdown fences."
        ),
        tools=[get_branch_experiments],
    )


def build_coder(model: OpenAIChatCompletionsModel,
                doc_lookup: Agent[SwarmContext]) -> Agent[SwarmContext]:
    baseline = _read_baseline_template()
    return Agent[SwarmContext](
        name="Coder",
        model=model,
        instructions=(
            "You generate complete, runnable PyTorch training scripts for the "
            "Wunderfund Predictorium challenge.\n\n"
            "CRITICAL: You MUST modify the baseline template below rather than "
            "writing from scratch. This ensures correct data loading, evaluation, "
            "and solution.py generation.\n\n"
            "## Baseline template\n"
            f"```python\n{baseline}\n```\n\n"
            "## What to modify\n"
            "- CONFIG dict: change hyperparameters as specified\n"
            "- BaselineModel class: replace with the specified architecture\n"
            "- Feature engineering: add preprocessing in PredictoriumDataset\n"
            "- Loss function: modify WeightedMSELoss if needed\n"
            "- generate_solution_py: update to match your model architecture\n\n"
            "## What NOT to modify\n"
            "- The CLI interface (--mode cv/submission)\n"
            "- The output format (last line = JSON)\n"
            "- The evaluate() function\n"
            "- The 5% holdout in submission mode\n\n"
            "## Hardware constraints (RTX 5070 Ti = 16GB VRAM)\n"
            "- Transformer: ≤4 heads, ≤256 hidden, ≤4 layers, batch ≤128\n"
            "- GRU/LSTM: ≤512 hidden, ≤3 layers, batch ≤128\n"
            "- Always use mixed precision (torch.amp)\n\n"
            "Use the lookup_docs tool for any API you're unsure about.\n\n"
            "After generating the code, call register_experiment with:\n"
            "- branch_name, hypothesis, config_summary from the plan\n"
            "- The COMPLETE training script as the code parameter\n"
            "- parent_experiment_id if this refines a prior experiment\n\n"
            "Output the experiment_id after registration."
        ),
        tools=[
            register_experiment,
            doc_lookup.as_tool(
                tool_name="lookup_docs",
                tool_description=(
                    "Look up current API docs for PyTorch, Mamba, NumPy, etc. "
                    "Ask specific questions like 'torch.nn.TransformerEncoder API "
                    "with custom positional encoding' or 'Mamba-2 SSM layer usage'"
                ),
            ),
        ],
    )


def build_evaluator(model: OpenAIChatCompletionsModel) -> Agent[SwarmContext]:
    return Agent[SwarmContext](
        name="Evaluator",
        model=model,
        instructions=(
            "You execute experiments and interpret results.\n\n"
            "For each experiment:\n"
            "1. Call run_training with the experiment_id and mode='cv'\n"
            "2. Interpret the results JSON\n"
            "3. Call git_commit_experiment with a descriptive message\n\n"
            "Your output should be a JSON summary:\n"
            '{"experiment_id": "...", "passed": true/false, '
            '"mean_score": 0.XX, "fold_scores": [...], "fold_std": 0.XX, '
            '"analysis": "what the results suggest for next steps"}\n\n'
            "If training fails with an error, include the error in your analysis "
            "and suggest what the Coder should fix.\n\n"
            "Output ONLY the JSON, no markdown fences."
        ),
        tools=[run_training, git_commit_experiment],
    )


def build_strategist(model: OpenAIChatCompletionsModel) -> Agent[SwarmContext]:
    return Agent[SwarmContext](
        name="Strategist",
        model=model,
        instructions=(
            "You review the full experiment history across ALL branches every few "
            "iterations and propose high-level strategic shifts.\n\n"
            "Consider:\n"
            "- Which architecture families have been tried? Which haven't?\n"
            "- Are we stuck in local optima (many refinements, no score improvement)?\n"
            "- Should we ensemble top models from different branches?\n"
            "- Are there unexplored feature engineering ideas?\n"
            "- Is the loss function properly aligned with weighted Pearson?\n\n"
            "Predictorium strategies to consider:\n"
            "- Temporal Fusion Transformer, GRU, LSTM, Mamba-2, 1D-CNN+RNN\n"
            "- Features: bid-ask spread, volume imbalance, order flow, mid-price velocity\n"
            "- Loss: weighted MSE where weight = abs(target)\n"
            "- Ensemble: average predictions from top 2-3 diverse models\n\n"
            "Output a ranked list of 2-3 recommendations with reasoning.\n"
            "Be specific and actionable — the Orchestrator will read this."
        ),
        tools=[get_full_history],
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class AgentSuite:
    """Holds all agent instances. Created once in main.py."""

    def __init__(self, vllm_url: str = "http://localhost:8000/v1",
                 model_name: str = "GPT-OSS-120B"):
        self.model = build_model(vllm_url, model_name)
        self.context7 = build_context7()

        self.dedup_checker = build_dedup_checker(self.model)
        self.doc_lookup = build_doc_lookup(self.model, self.context7)

        self.orchestrator = build_orchestrator(self.model)
        self.planner = build_planner(self.model)
        self.coder = build_coder(self.model, self.doc_lookup)
        self.evaluator = build_evaluator(self.model)
        self.strategist = build_strategist(self.model)
