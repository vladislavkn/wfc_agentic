"""
CLI entry point.

Usage:
    python -m swarm.cli run [--url URL] [--model MODEL] [--target 0.3]
    python -m swarm.cli update-submission --branch NAME --score 0.28 --action reopen|close
    python -m swarm.cli status
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from swarm.state import BranchStatus, SwarmDB


def main():
    parser = argparse.ArgumentParser(
        prog="swarm", description="Predictorium Agent Swarm")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = sub.add_parser("run", help="Start the swarm loop")
    run_p.add_argument("--url", default=None,
                       help="Optional OpenAI-compatible base URL override")
    run_p.add_argument("--model", default="gpt-5.2",
                       help="Model name (defaults to OpenAI)")
    run_p.add_argument("--target", type=float, default=0.3,
                       help="Target weighted Pearson score")
    run_p.add_argument("--max-iterations", type=int, default=100)
    run_p.add_argument("--db", default="experiments/swarm.db",
                       help="SQLite database path")

    # --- update-submission ---
    up_p = sub.add_parser("update-submission",
                          help="Update a branch with submission results")
    up_p.add_argument("--branch", required=True, help="Branch name")
    up_p.add_argument("--score", type=float, required=True,
                      help="Actual submission score from wundernn.io")
    up_p.add_argument("--action", choices=["reopen", "close"], required=True,
                      help="Reopen branch for refinement or close it")
    up_p.add_argument("--db", default="experiments/swarm.db")

    # --- status ---
    st_p = sub.add_parser("status", help="Show current swarm status")
    st_p.add_argument("--db", default="experiments/swarm.db")

    args = parser.parse_args()

    if args.command == "run":
        from swarm.main import run_swarm
        asyncio.run(run_swarm(
            vllm_url=args.url,
            model_name=args.model,
            target_score=args.target,
            max_iterations=args.max_iterations,
            db_path=args.db,
        ))

    elif args.command == "update-submission":
        _update_submission(args)

    elif args.command == "status":
        _show_status(args)


def _update_submission(args):
    db = SwarmDB(args.db)
    branch = db.get_branch(args.branch)

    if not branch:
        print(f"ERROR: Branch '{args.branch}' not found.", file=sys.stderr)
        sys.exit(1)

    if branch.status != BranchStatus.PAUSED:
        print(f"WARNING: Branch '{args.branch}' is {branch.status.value}, "
              f"not paused_awaiting_results.", file=sys.stderr)

    old_cv = branch.best_cv_score
    db.update_branch(
        args.branch,
        submission_score=args.score,
        status=(BranchStatus.ACTIVE.value if args.action == "reopen"
                else BranchStatus.CLOSED.value),
        notes=branch.notes + f" | Submission: {args.score}, {args.action}d",
    )

    print(f"Updated branch '{args.branch}':")
    print(f"  CV score:         {old_cv:.4f}")
    print(f"  Submission score: {args.score:.4f}")
    print(f"  Action:           {args.action}")
    print(
        f"  New status:       {'active' if args.action == 'reopen' else 'closed'}")

    if args.action == "reopen":
        drop = (old_cv - args.score) / old_cv * 100 if old_cv > 0 else 0
        if drop > 10:
            print(f"\n  ⚠️  Score dropped {drop:.1f}% from CV. "
                  f"Consider more regularization or larger CV subset.")
        print("\n  Run 'python -m swarm.cli run' to continue the swarm.")


def _show_status(args):
    db = SwarmDB(args.db)
    branches = db.list_branches()
    tokens = db.total_tokens()

    print(f"\n{'='*60}")
    print("  SWARM STATUS")
    print(f"{'='*60}")
    print(f"  Total tokens: {tokens['total']:,}")
    print(f"  Branches: {len(branches)}")
    print()

    for b in branches:
        exps = db.list_experiments(b.name)
        stagnant = db.count_stagnant(b.name)
        scored = [e for e in exps if e.mean_score is not None]
        icon = {
            BranchStatus.ACTIVE: "🟢",
            BranchStatus.PAUSED: "⏸️ ",
            BranchStatus.CLOSED: "🔴",
            BranchStatus.ABANDONED: "⚫",
        }.get(b.status, "❓")

        print(f"  {icon} {b.name} [{b.status.value}]")
        print(f"     Hypothesis: {b.hypothesis[:80]}")
        print(f"     Experiments: {len(exps)} ({len(scored)} scored)")
        print(f"     Best CV: {b.best_cv_score:.4f}")
        if b.submission_score is not None:
            print(f"     Submission: {b.submission_score:.4f}")
        print(f"     Stagnation: {stagnant}")
        if b.notes:
            print(f"     Notes: {b.notes[:100]}")
        print()

    # Latest advice
    advice = db.latest_advice()
    if advice:
        print("  Latest strategist advice:")
        for line in advice.split("\n")[:5]:
            print(f"    {line}")
        print()


if __name__ == "__main__":
    main()
