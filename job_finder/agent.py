"""Core LangGraph agent for the Job Finder application.

Graph layout
------------

    START
      │
      ▼
  extract_profile        ← Summarises the resume into a compact profile
      │
      ▼
  search_agent           ← ReAct loop: searches web & saves job listings
      │
      ▼
  evaluate_jobs          ← Scores each listing against resume + comp. req.
      │
      ▼
    END
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from .prompts import (
    COMPENSATION_GUIDANCE_TEMPLATE,
    EVALUATION_SYSTEM_PROMPT,
    NO_COMPENSATION_GUIDANCE,
    SEARCH_AGENT_SYSTEM_PROMPT,
)
from .state import JobFinderState, JobListing
from .tools import build_search_tools, clear_collected_jobs, get_collected_jobs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_llm(model: Optional[str] = None) -> ChatOpenAI:
    """Instantiate the language model from environment / argument."""
    resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=resolved_model, temperature=0)


def _compensation_guidance(
    comp_min: Optional[float], comp_max: Optional[float]
) -> str:
    if comp_min is None and comp_max is None:
        return NO_COMPENSATION_GUIDANCE
    lo = f"${comp_min:,.0f}" if comp_min is not None else "unspecified"
    hi = f"${comp_max:,.0f}" if comp_max is not None else "unspecified"
    return COMPENSATION_GUIDANCE_TEMPLATE.format(min_salary=lo, max_salary=hi)


# ---------------------------------------------------------------------------
# Node: extract_profile
# ---------------------------------------------------------------------------

def extract_profile(state: JobFinderState) -> dict[str, Any]:
    """Summarise the resume into a short profile for use in prompts."""
    llm = _build_llm()
    system = (
        "You are a resume parser. Extract a concise (≤150 words) professional "
        "profile from the resume below. Include: current/target role, years of "
        "experience, top 5-8 technical/domain skills, education level, and any "
        "notable achievements. Output plain text only – no headers or bullets."
    )
    response = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=state["resume_text"])]
    )
    resume_summary = response.content

    # Build the first human message that kick-starts the search agent
    comp_guidance = _compensation_guidance(
        state.get("compensation_min"), state.get("compensation_max")
    )
    search_prompt = SEARCH_AGENT_SYSTEM_PROMPT.format(
        job_field=state["job_field"],
        location=state.get("location", "anywhere"),
        compensation_guidance=comp_guidance,
        resume_summary=resume_summary,
    )

    return {
        "messages": [HumanMessage(content=search_prompt)],
        "search_complete": False,
        "found_jobs": [],
        "suitable_jobs": [],
    }


# ---------------------------------------------------------------------------
# Node: search_agent  (built at graph-compile time)
# ---------------------------------------------------------------------------

def _build_search_agent_node(model: Optional[str] = None):
    """Return a callable node that runs the ReAct search loop.

    The LLM and ReAct agent are created lazily on the first invocation so
    that constructing the graph does not require API keys to be present.
    """
    _cache: dict[str, Any] = {}

    def search_node(state: JobFinderState) -> dict[str, Any]:
        if "agent" not in _cache:
            _cache["llm"] = _build_llm(model)
            _cache["tools"] = build_search_tools()
            _cache["agent"] = create_react_agent(_cache["llm"], _cache["tools"])

        clear_collected_jobs()
        result = _cache["agent"].invoke({"messages": state["messages"]})
        jobs = get_collected_jobs()
        return {
            "messages": result["messages"],
            "found_jobs": jobs,
            "search_complete": True,
        }

    return search_node


# ---------------------------------------------------------------------------
# Node: evaluate_jobs
# ---------------------------------------------------------------------------

def evaluate_jobs(state: JobFinderState) -> dict[str, Any]:
    """Score each found job listing and filter out poor matches."""
    found = state.get("found_jobs", [])
    if not found:
        logger.warning("No job listings to evaluate.")
        return {"suitable_jobs": []}

    llm = _build_llm()
    comp_guidance = _compensation_guidance(
        state.get("compensation_min"), state.get("compensation_max")
    )
    system_prompt = EVALUATION_SYSTEM_PROMPT.format(
        compensation_guidance=comp_guidance,
        resume_text=state["resume_text"],
    )

    listings_json = json.dumps(found, indent=2)
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Here are the job listings to evaluate:\n\n{listings_json}\n\n"
                    "Return the augmented JSON array."
                )
            ),
        ]
    )

    raw = response.content.strip()
    # Strip optional markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        evaluated: list[dict[str, Any]] = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("LLM returned non-JSON during evaluation: %s", raw[:300])
        evaluated = found  # fall back: return un-scored listings

    # Filter to fit_score >= 6 and sort descending
    suitable: list[JobListing] = []
    for job in evaluated:
        score = job.get("fit_score")
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0
        job["fit_score"] = score
        job.setdefault("fit_reasoning", "")
        if score >= 6:
            suitable.append(job)  # type: ignore[arg-type]

    suitable.sort(key=lambda j: j.get("fit_score", 0), reverse=True)
    return {"suitable_jobs": suitable}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(model: Optional[str] = None) -> StateGraph:
    """Compile and return the job-finder StateGraph."""
    graph = StateGraph(JobFinderState)

    search_node = _build_search_agent_node(model)

    graph.add_node("extract_profile", extract_profile)
    graph.add_node("search_agent", search_node)
    graph.add_node("evaluate_jobs", evaluate_jobs)

    graph.add_edge(START, "extract_profile")
    graph.add_edge("extract_profile", "search_agent")
    graph.add_edge("search_agent", "evaluate_jobs")
    graph.add_edge("evaluate_jobs", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def find_jobs(
    resume_text: str,
    job_field: str,
    location: str = "anywhere",
    compensation_min: Optional[float] = None,
    compensation_max: Optional[float] = None,
    model: Optional[str] = None,
) -> list[JobListing]:
    """High-level entry point: run the full pipeline and return suitable jobs.

    Args:
        resume_text: Full text of the candidate's resume.
        job_field: Target job field, e.g. "data science" or "product management".
        location: Preferred work location or "remote" / "anywhere".
        compensation_min: Minimum annual salary in USD (optional).
        compensation_max: Maximum annual salary in USD (optional).
        model: OpenAI model name override (default: value of OPENAI_MODEL env
            var, or "gpt-4o-mini").

    Returns:
        A list of :class:`JobListing` dicts sorted by fit score descending.
    """
    app = build_graph(model)
    initial_state: JobFinderState = {
        "resume_text": resume_text,
        "job_field": job_field,
        "location": location,
        "compensation_min": compensation_min,
        "compensation_max": compensation_max,
        "messages": [],
        "found_jobs": [],
        "suitable_jobs": [],
        "search_complete": False,
    }
    final_state = app.invoke(initial_state)
    return final_state.get("suitable_jobs", [])
