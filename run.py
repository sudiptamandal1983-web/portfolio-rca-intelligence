"""
run.py — CLI entry point. Thin wrapper around Orchestrator.

Usage:
    python3 run.py                         # template narratives, no email
    python3 run.py --llm                   # LLM narratives, no eval gate
    python3 run.py --llm --eval            # LLM narratives + eval quality gate
    python3 run.py --llm --eval --email    # full pipeline
    python3 run.py --email-dry-run         # render email, don't send
    python3 run.py --dimensions regional vintage_risk
    python3 run.py --z 2.5 --min-sample 200

Email env vars:
    export GMAIL_USER="you@gmail.com"
    export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
    export RISK_TEAM_EMAILS="alice@bank.com,bob@bank.com"
    export PORTFOLIO_MGR_EMAILS="cfo@bank.com"

LLM env var:
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

import argparse
import sys
from agents import Orchestrator, OrchestratorConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Banking Portfolio RCA Pipeline")
    parser.add_argument("--db",             default="data/portfolio.db")
    parser.add_argument("--out",            default="reports")
    parser.add_argument("--llm",            action="store_true",
                        help="Use LLM for narrative generation")
    parser.add_argument("--eval",           action="store_true",
                        help="Score LLM narratives, fall back to template if below threshold")
    parser.add_argument("--eval-threshold", type=float, default=3.5,
                        help="Min avg score (1-5) to keep LLM narrative (default: 3.5)")
    parser.add_argument("--email",          action="store_true")
    parser.add_argument("--email-dry-run",  action="store_true")
    parser.add_argument("--dimensions",     nargs="+", default=None)
    parser.add_argument("--z",              type=float, default=2.0)
    parser.add_argument("--min-sample",     type=int,   default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    config = OrchestratorConfig(
        db_path          = args.db,
        report_dir       = args.out,
        dimensions       = args.dimensions,
        min_sample_size  = args.min_sample,
        z_threshold      = args.z,
        use_llm          = args.llm,
        use_eval         = args.eval,
        eval_threshold   = args.eval_threshold,
        send_email       = args.email,
        email_dry_run    = getattr(args, "email_dry_run", False),
    )

    result = Orchestrator(config).run()
    sys.exit(0 if result.bundles or not result.agent_statuses else 1)


if __name__ == "__main__":
    main()
