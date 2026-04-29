#!/usr/bin/env python3
"""Command-line interface for the Job Finder agent.

Usage
-----
    python main.py --resume path/to/resume.txt \\
                   --field "software engineering" \\
                   --location "San Francisco, CA" \\
                   --min-salary 120000 \\
                   --max-salary 180000

Environment variables required:
    OPENAI_API_KEY   – OpenAI API key
    TAVILY_API_KEY   – Tavily search API key (or use DuckDuckGo fallback)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from job_finder.agent import find_jobs  # noqa: E402 – after dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find jobs matching your resume and compensation requirements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resume",
        required=True,
        help="Path to a plain-text file containing the resume, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--field",
        required=True,
        help='Target job field, e.g. "data science" or "backend engineering".',
    )
    parser.add_argument(
        "--location",
        default="anywhere",
        help='Preferred work location, e.g. "New York, NY" or "remote". Default: anywhere.',
    )
    parser.add_argument(
        "--min-salary",
        type=float,
        default=None,
        metavar="USD",
        help="Minimum acceptable annual salary in USD.",
    )
    parser.add_argument(
        "--max-salary",
        type=float,
        default=None,
        metavar="USD",
        help="Maximum desired annual salary in USD.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model to use (default: OPENAI_MODEL env var or gpt-4o-mini).",
    )
    parser.add_argument(
        "--output",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format. 'pretty' prints a human-readable summary; 'json' dumps raw JSON.",
    )
    return parser.parse_args(argv)


def _read_resume(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _print_pretty(jobs: list[dict]) -> None:
    if not jobs:
        print("\nNo suitable jobs found. Try broadening your search criteria.")
        return

    print(f"\n{'='*70}")
    print(f"  Found {len(jobs)} suitable job(s)")
    print(f"{'='*70}\n")
    for i, job in enumerate(jobs, 1):
        score = job.get("fit_score", "N/A")
        print(f"#{i}  [{score}/10]  {job.get('title', 'N/A')}  –  {job.get('company', 'N/A')}")
        print(f"     Location   : {job.get('location', 'N/A')}")
        print(f"     Salary     : {job.get('salary_range') or 'Not listed'}")
        print(f"     URL        : {job.get('url', 'N/A')}")
        print(f"     Summary    : {job.get('description_summary', '')}")
        print(f"     Fit reason : {job.get('fit_reasoning', '')}")
        print()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set. Please configure it in .env or the shell.")
        return 1

    # Read resume
    try:
        resume_text = _read_resume(args.resume)
    except FileNotFoundError:
        logger.error("Resume file not found: %s", args.resume)
        return 1

    if not resume_text.strip():
        logger.error("Resume file is empty.")
        return 1

    logger.info(
        "Starting job search | field=%s | location=%s",
        args.field,
        args.location,
    )

    jobs = find_jobs(
        resume_text=resume_text,
        job_field=args.field,
        location=args.location,
        compensation_min=args.min_salary,
        compensation_max=args.max_salary,
        model=args.model,
    )

    if args.output == "json":
        print(json.dumps(jobs, indent=2))
    else:
        _print_pretty(jobs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
