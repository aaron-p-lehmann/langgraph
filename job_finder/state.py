"""State definitions for the Job Finder agent."""

from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class JobListing(TypedDict):
    """A single job listing discovered and evaluated by the agent."""

    title: str
    company: str
    location: str
    url: str
    salary_range: str
    description_summary: str
    fit_score: int          # 1-10 scale set by the evaluation step
    fit_reasoning: str


class JobFinderState(TypedDict):
    """Full agent state threaded through every node in the graph.

    Attributes:
        resume_text: Raw text of the candidate's resume.
        job_field: Target job field / industry (e.g. "software engineering").
        location: Preferred work location or "remote".
        compensation_min: Minimum acceptable annual salary (USD, optional).
        compensation_max: Maximum desired annual salary (USD, optional).
        messages: Conversation history used by the ReAct agent.
        found_jobs: Accumulated raw job listings collected during search.
        suitable_jobs: Filtered / ranked listings that meet the criteria.
        search_complete: Flag set when the search phase is finished.
    """

    resume_text: str
    job_field: str
    location: str
    compensation_min: Optional[float]
    compensation_max: Optional[float]
    messages: Annotated[list[BaseMessage], add_messages]
    found_jobs: list[JobListing]
    suitable_jobs: list[JobListing]
    search_complete: bool
