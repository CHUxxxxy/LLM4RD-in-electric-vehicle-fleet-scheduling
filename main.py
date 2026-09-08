#!/usr/bin/env python
"""LLM4RD 命令行入口。"""
import argparse
import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="LLM4RD: LLM-driven reward design for MAPPO EV fleet scheduling"
    )
    parser.add_argument(
        "--excel",
        required=True,
        help="Order Excel path (columns: SHour, STime, SLat, SLon, ELat, ELon, TD)",
    )
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key (prefer .env, do not commit)")
    parser.add_argument("--model", default=None, help="Chat model name")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small search budget for a smoke test",
    )
    args = parser.parse_args()

    if args.api_base:
        os.environ["LLM4RD_API_BASE"] = args.api_base
    if args.api_key:
        os.environ["LLM4RD_API_KEY"] = args.api_key
    if args.model:
        os.environ["LLM4RD_MODEL"] = args.model

    if args.quick:
        os.environ["LLM4RD_QUICK"] = "1"
        os.environ.setdefault("LLM4RD_TRAINING_STEPS", "50000")
        os.environ.setdefault("LLM4RD_ITERATIONS", "2")
        os.environ.setdefault("LLM4RD_CANDIDATES", "2")

    if not os.path.isfile(args.excel):
        print(f"Error: Excel file not found: {args.excel}")
        sys.exit(1)

    from LLM4RD.config import require_llm_credentials
    try:
        require_llm_credentials()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    from LLM4RD.pipeline.evolution import run_llm4rd

    print(f"Data file: {args.excel}")
    result = run_llm4rd(args.excel)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
