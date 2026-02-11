# Predictorium Agent Swarm

A self-evolving ML experiment swarm targeting 0.3+ weighted Pearson correlation
on the [Wunderfund Predictorium](https://wundernn.io/predictorium) challenge.

## Architecture

```
Sequential loop (no handoff chains):

  ┌─ Deterministic stagnation check ──┐
  │                                    │
  │  Strategist (every 5 iters)        │
  │       ↓                            │
  │  Orchestrator → select branch      │
  │       ↓                            │
  │  Planner → propose experiment      │
  │       ↓                            │
  │  DedupChecker (sub-agent) → novel? │
  │       ↓                            │
  │  Coder → generate training code    │
  │       ↓                            │
  │  Evaluator → 3-fold CV on 30%      │
  │       ↓                            │
  │  Analyze → submit if ≥ 0.3         │
  │       ↓                            │
  │  Git commit                        │
  └────────────────────────────────────┘
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt
npm install -g @upstash/context7-mcp@latest

# 2. Place train.parquet and valid.parquet in project root

# 3. Prepare data (30% CV subset + full merged dataset)
python data/prepare_data.py

# 4. Initialize git
git init && git add . && git commit -m "Initial setup"

# 5. Start vLLM with GPT-OSS-120B (separate terminal)

# 6. Run the swarm
python -m swarm.cli run --url http://localhost:8000/v1 --model GPT-OSS-120B
```

## Commands

```bash
# Run the swarm
python -m swarm.cli run [--url URL] [--model MODEL] [--target 0.3]

# Check status
python -m swarm.cli status

# After uploading a submission to wundernn.io:
python -m swarm.cli update-submission --branch <NAME> --score <SCORE> --action reopen|close
```

## How It Works

1. **Orchestrator** picks a branch (or creates one)
2. **Planner** reviews all prior experiments in the branch, proposes a novel approach
3. **DedupChecker** (sub-agent) semantically verifies the proposal isn't a repeat
4. **Coder** generates training code by modifying the baseline template
5. **Evaluator** runs 3-fold CV on 30% subset (~3-5 min per experiment)
6. If score ≥ 0.3: train on full data, validate solution.py, package submission, pause branch
7. If stagnating: force switch to another branch after 3 experiments without improvement

## Key Design Decisions

- **SQLite** as single source of truth (no file drift)
- **Sequential execution** — no parallel GPU jobs
- **Semantic dedup** via sub-agent, not hash-based
- **Baseline template** — Coder modifies working code instead of writing from scratch
- **Deterministic branch switching** — hard-coded stagnation threshold
- **5% holdout sanity check** on submission training
- **solution.py interface validation** before packaging

## File Structure

```
swarm/
├── cli.py          # Entry point (run, status, update-submission)
├── main.py         # Main loop with sequential agent calls
├── agents.py       # Agent definitions + AgentSuite factory
├── tools.py        # All tool functions (training, git, dedup, submission)
├── state.py        # SQLite DB + SwarmContext dataclass
├── logger.py       # Structured logging (console + file)
├── validator.py    # solution.py interface checker
└── templates/
    └── baseline_train.py  # Working GRU baseline template
```
