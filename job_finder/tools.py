"""LangChain tools used by the Job Finder agent."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory store that the search agent writes to so the evaluation node can
# read the collected listings without parsing conversation history.
# ---------------------------------------------------------------------------
_collected_jobs: list[dict[str, Any]] = []


def clear_collected_jobs() -> None:
    """Reset the in-memory job store (useful between runs / tests)."""
    _collected_jobs.clear()


def get_collected_jobs() -> list[dict[str, Any]]:
    """Return the jobs saved by the search agent."""
    return list(_collected_jobs)


@tool
def save_job_listings(jobs_json: str) -> str:
    """Persist a JSON list of job listings collected during web search.

    Args:
        jobs_json: A JSON-encoded list where every element is an object with
            at minimum the keys: title, company, location, url,
            salary_range (empty string if unknown), description_summary.

    Returns:
        A confirmation message with the count of jobs saved.
    """
    try:
        jobs: list[dict[str, Any]] = json.loads(jobs_json)
        if not isinstance(jobs, list):
            return "Error: jobs_json must be a JSON array."
    except json.JSONDecodeError as exc:
        return f"Error parsing jobs_json: {exc}"

    required_keys = {"title", "company", "location", "url", "description_summary"}
    validated: list[dict[str, Any]] = []
    for i, job in enumerate(jobs):
        missing = required_keys - set(job.keys())
        if missing:
            logger.warning("Job %d missing keys %s – skipping", i, missing)
            continue
        job.setdefault("salary_range", "")
        validated.append(job)

    _collected_jobs.extend(validated)
    return f"Saved {len(validated)} job listing(s). Total collected: {len(_collected_jobs)}."


def build_search_tools() -> list:
    """Return the list of tools available to the search agent.

    Attempts to use TavilySearchResults (requires TAVILY_API_KEY).
    Falls back to DuckDuckGoSearchRun if Tavily is unavailable.
    """
    tools: list = []

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        tools.append(TavilySearchResults(max_results=5))
        logger.debug("Using Tavily as the web-search backend.")
    except (ImportError, ModuleNotFoundError):
        try:
            from langchain_community.tools import DuckDuckGoSearchRun

            tools.append(DuckDuckGoSearchRun())
            logger.debug("Tavily unavailable; falling back to DuckDuckGo.")
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "No web-search backend available. "
                "Install tavily-python and set TAVILY_API_KEY, or install "
                "duckduckgo-search."
            )

    tools.append(save_job_listings)
    return tools
