"""Unit tests for the Job Finder agent.

These tests mock all external calls (OpenAI, Tavily) so they can run without
real API keys.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESUME = """
Jane Smith
Senior Software Engineer

Summary:
8 years of experience building distributed backend systems in Python and Go.
Led teams of 3-6 engineers. Strong background in microservices, Kubernetes,
and cloud-native architecture (AWS, GCP).

Skills: Python, Go, Kubernetes, Docker, PostgreSQL, Redis, gRPC, REST APIs,
        CI/CD (GitHub Actions), Terraform.

Experience:
- Senior Software Engineer, Acme Corp (2020–present)
  Designed high-throughput data pipeline processing 10M events/day.
- Software Engineer, Startup Inc. (2016–2020)
  Built microservices powering a SaaS platform with 50k users.

Education: B.S. Computer Science, State University, 2016
"""

SAMPLE_JOBS = [
    {
        "title": "Senior Backend Engineer",
        "company": "TechCorp",
        "location": "Remote",
        "url": "https://techcorp.com/jobs/1",
        "salary_range": "$140,000 – $170,000",
        "description_summary": "Python/Go microservices role at a growing SaaS company.",
    },
    {
        "title": "Junior Frontend Developer",
        "company": "WebAgency",
        "location": "New York, NY",
        "url": "https://webagency.com/jobs/2",
        "salary_range": "$65,000 – $85,000",
        "description_summary": "React/CSS role for a creative agency.",
    },
]

EVALUATED_JOBS = [
    {**SAMPLE_JOBS[0], "fit_score": 9, "fit_reasoning": "Strong Python/Go match."},
    {**SAMPLE_JOBS[1], "fit_score": 2, "fit_reasoning": "Frontend role mismatches backend profile."},
]


# ---------------------------------------------------------------------------
# Tests: tools
# ---------------------------------------------------------------------------

class TestSaveJobListings:
    def setup_method(self):
        from job_finder.tools import clear_collected_jobs
        clear_collected_jobs()

    def test_saves_valid_jobs(self):
        from job_finder.tools import get_collected_jobs, save_job_listings

        result = save_job_listings.invoke(json.dumps(SAMPLE_JOBS))
        assert "2 job listing(s)" in result
        collected = get_collected_jobs()
        assert len(collected) == 2
        assert collected[0]["title"] == "Senior Backend Engineer"

    def test_rejects_non_list_json(self):
        from job_finder.tools import save_job_listings

        result = save_job_listings.invoke('{"title": "bad"}')
        assert "Error" in result

    def test_rejects_invalid_json(self):
        from job_finder.tools import save_job_listings

        result = save_job_listings.invoke("not json at all")
        assert "Error" in result

    def test_skips_jobs_missing_required_keys(self):
        from job_finder.tools import get_collected_jobs, save_job_listings

        incomplete = [{"title": "No URL job", "company": "X"}]
        result = save_job_listings.invoke(json.dumps(incomplete))
        assert "0 job listing(s)" in result
        assert get_collected_jobs() == []

    def test_defaults_salary_range_to_empty_string(self):
        from job_finder.tools import get_collected_jobs, save_job_listings

        jobs_without_salary = [
            {k: v for k, v in SAMPLE_JOBS[0].items() if k != "salary_range"}
        ]
        save_job_listings.invoke(json.dumps(jobs_without_salary))
        collected = get_collected_jobs()
        assert collected[0]["salary_range"] == ""

    def test_clear_collected_jobs(self):
        from job_finder.tools import clear_collected_jobs, get_collected_jobs, save_job_listings

        save_job_listings.invoke(json.dumps(SAMPLE_JOBS))
        assert len(get_collected_jobs()) == 2
        clear_collected_jobs()
        assert get_collected_jobs() == []

    def test_accumulates_across_calls(self):
        from job_finder.tools import get_collected_jobs, save_job_listings

        save_job_listings.invoke(json.dumps([SAMPLE_JOBS[0]]))
        save_job_listings.invoke(json.dumps([SAMPLE_JOBS[1]]))
        assert len(get_collected_jobs()) == 2


# ---------------------------------------------------------------------------
# Tests: prompts helpers
# ---------------------------------------------------------------------------

class TestCompensationGuidance:
    def test_no_compensation(self):
        from job_finder.agent import _compensation_guidance

        result = _compensation_guidance(None, None)
        assert "not specified" in result.lower() or "has not" in result.lower()

    def test_both_bounds(self):
        from job_finder.agent import _compensation_guidance

        result = _compensation_guidance(100_000, 150_000)
        assert "$100,000" in result
        assert "$150,000" in result

    def test_min_only(self):
        from job_finder.agent import _compensation_guidance

        result = _compensation_guidance(80_000, None)
        assert "$80,000" in result
        assert "unspecified" in result

    def test_max_only(self):
        from job_finder.agent import _compensation_guidance

        result = _compensation_guidance(None, 200_000)
        assert "$200,000" in result
        assert "unspecified" in result


# ---------------------------------------------------------------------------
# Tests: build_graph structure
# ---------------------------------------------------------------------------

class TestGraphStructure:
    def test_graph_compiles(self):
        """Graph should compile without errors."""
        from job_finder.agent import build_graph

        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        from job_finder.agent import build_graph

        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "extract_profile" in node_names
        assert "search_agent" in node_names
        assert "evaluate_jobs" in node_names


# ---------------------------------------------------------------------------
# Tests: evaluate_jobs node (mocked LLM)
# ---------------------------------------------------------------------------

class TestEvaluateJobsNode:
    def _make_state(self, found_jobs=None, comp_min=None, comp_max=None):
        return {
            "resume_text": SAMPLE_RESUME,
            "job_field": "software engineering",
            "location": "remote",
            "compensation_min": comp_min,
            "compensation_max": comp_max,
            "messages": [],
            "found_jobs": found_jobs or [],
            "suitable_jobs": [],
            "search_complete": True,
        }

    def test_returns_empty_suitable_jobs_when_no_found_jobs(self):
        from job_finder.agent import evaluate_jobs

        result = evaluate_jobs(self._make_state(found_jobs=[]))
        assert result["suitable_jobs"] == []

    @patch("job_finder.agent._build_llm")
    def test_filters_low_scoring_jobs(self, mock_build_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps(EVALUATED_JOBS)
        )
        mock_build_llm.return_value = mock_llm

        from job_finder.agent import evaluate_jobs

        result = evaluate_jobs(self._make_state(found_jobs=SAMPLE_JOBS))
        suitable = result["suitable_jobs"]
        # Only score-9 job should pass the >= 6 threshold
        assert len(suitable) == 1
        assert suitable[0]["title"] == "Senior Backend Engineer"
        assert suitable[0]["fit_score"] == 9

    @patch("job_finder.agent._build_llm")
    def test_sorts_by_fit_score_descending(self, mock_build_llm):
        multi_jobs = [
            {**SAMPLE_JOBS[0], "fit_score": 7, "fit_reasoning": "Good match"},
            {**SAMPLE_JOBS[1], "fit_score": 9, "fit_reasoning": "Excellent match"},
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(multi_jobs))
        mock_build_llm.return_value = mock_llm

        from job_finder.agent import evaluate_jobs

        result = evaluate_jobs(self._make_state(found_jobs=SAMPLE_JOBS))
        scores = [j["fit_score"] for j in result["suitable_jobs"]]
        assert scores == sorted(scores, reverse=True)

    @patch("job_finder.agent._build_llm")
    def test_handles_llm_returning_markdown_fenced_json(self, mock_build_llm):
        fenced = "```json\n" + json.dumps(EVALUATED_JOBS) + "\n```"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=fenced)
        mock_build_llm.return_value = mock_llm

        from job_finder.agent import evaluate_jobs

        result = evaluate_jobs(self._make_state(found_jobs=SAMPLE_JOBS))
        assert len(result["suitable_jobs"]) == 1

    @patch("job_finder.agent._build_llm")
    def test_fallback_on_invalid_json_from_llm(self, mock_build_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="This is not JSON at all.")
        mock_build_llm.return_value = mock_llm

        from job_finder.agent import evaluate_jobs

        # Should not raise; falls back to returning un-scored listings
        result = evaluate_jobs(self._make_state(found_jobs=SAMPLE_JOBS))
        # fit_score defaults to 0 for items without it, so suitable_jobs may be empty
        assert isinstance(result["suitable_jobs"], list)


# ---------------------------------------------------------------------------
# Tests: extract_profile node (mocked LLM)
# ---------------------------------------------------------------------------

class TestExtractProfileNode:
    @patch("job_finder.agent._build_llm")
    def test_builds_initial_message(self, mock_build_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Jane Smith – Senior SWE with 8 years Python/Go.")
        mock_build_llm.return_value = mock_llm

        from job_finder.agent import extract_profile

        state = {
            "resume_text": SAMPLE_RESUME,
            "job_field": "software engineering",
            "location": "remote",
            "compensation_min": 120_000.0,
            "compensation_max": 180_000.0,
            "messages": [],
            "found_jobs": [],
            "suitable_jobs": [],
            "search_complete": False,
        }
        result = extract_profile(state)

        assert len(result["messages"]) == 1
        assert not result["search_complete"]
        assert result["found_jobs"] == []
        # The message should mention the job field
        msg_content = result["messages"][0].content
        assert "software engineering" in msg_content.lower()


# ---------------------------------------------------------------------------
# Tests: CLI (main.py)
# ---------------------------------------------------------------------------

class TestCLI:
    def test_missing_openai_key_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        resume_file = tmp_path / "resume.txt"
        resume_file.write_text(SAMPLE_RESUME)

        from main import main

        rc = main(["--resume", str(resume_file), "--field", "engineering"])
        assert rc == 1

    def test_missing_resume_file_returns_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from main import main

        rc = main(["--resume", "/nonexistent/resume.txt", "--field", "engineering"])
        assert rc == 1

    def test_pretty_print_empty_results(self, capsys):
        from main import _print_pretty

        _print_pretty([])
        captured = capsys.readouterr()
        assert "No suitable jobs" in captured.out

    def test_pretty_print_with_jobs(self, capsys):
        from main import _print_pretty

        jobs = [
            {
                "title": "Engineer",
                "company": "ACME",
                "location": "Remote",
                "url": "https://example.com",
                "salary_range": "$100k",
                "description_summary": "A great job.",
                "fit_score": 8,
                "fit_reasoning": "Great match.",
            }
        ]
        _print_pretty(jobs)
        captured = capsys.readouterr()
        assert "Engineer" in captured.out
        assert "ACME" in captured.out
        assert "8/10" in captured.out

    def test_json_output_format(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        resume_file = tmp_path / "resume.txt"
        resume_file.write_text(SAMPLE_RESUME)

        mock_jobs = [
            {
                "title": "Engineer",
                "company": "ACME",
                "location": "Remote",
                "url": "https://example.com",
                "salary_range": "$100k",
                "description_summary": "A great job.",
                "fit_score": 8,
                "fit_reasoning": "Great match.",
            }
        ]

        with patch("main.find_jobs", return_value=mock_jobs):
            from main import main

            rc = main(
                [
                    "--resume", str(resume_file),
                    "--field", "engineering",
                    "--output", "json",
                ]
            )

        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert parsed[0]["title"] == "Engineer"
