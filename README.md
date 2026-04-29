# langgraph – Job Finder Agent

A **LangGraph** agent that searches the internet to find available jobs in a
field that are suitable for a particular resume and compensation requirements.

---

## Overview

The agent runs a three-stage pipeline:

```
START
  │
  ▼
extract_profile   ← Summarises the resume into a compact professional profile
  │
  ▼
search_agent      ← ReAct loop: runs web searches & collects job listings
  │
  ▼
evaluate_jobs     ← Scores each listing (1-10) against the resume & comp. req.
  │
  ▼
END               ← Returns listings with fit_score ≥ 6, sorted best-first
```

**Key features**

| Feature | Detail |
|---|---|
| Resume parsing | LLM extracts skills, seniority, education, achievements |
| Internet search | Tavily (primary) or DuckDuckGo (fallback) |
| Compensation filter | Min/max salary window applied during evaluation |
| Fit scoring | Each listing scored 1-10 with a written explanation |
| CLI & Python API | Use from the shell or import `find_jobs()` directly |

---

## Quick start

### 1 – Clone & install dependencies

```bash
git clone https://github.com/aaron-p-lehmann/langgraph
cd langgraph
pip install -r requirements.txt
```

### 2 – Configure API keys

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY and TAVILY_API_KEY
```

Get your keys from:
- **OpenAI**: <https://platform.openai.com/api-keys>
- **Tavily**: <https://tavily.com/> (free tier available)

### 3 – Run the agent

```bash
python main.py \
  --resume path/to/my_resume.txt \
  --field "software engineering" \
  --location "San Francisco, CA" \
  --min-salary 130000 \
  --max-salary 200000
```

Example output:

```
======================================================================
  Found 5 suitable job(s)
======================================================================

#1  [9/10]  Staff Software Engineer  –  Stripe
     Location   : Remote / San Francisco, CA
     Salary     : $165,000 – $195,000
     URL        : https://stripe.com/jobs/listing/staff-software-engineer/...
     Summary    : High-scale distributed systems role on the Payments Platform.
     Fit reason : 8+ years Python/Go experience matches requirements exactly.
...
```

---

## CLI reference

```
usage: main.py [-h] --resume RESUME --field FIELD [--location LOCATION]
               [--min-salary USD] [--max-salary USD] [--model MODEL]
               [--output {pretty,json}]

options:
  --resume     Path to a plain-text resume file, or '-' for stdin  (required)
  --field      Target job field, e.g. "data science"               (required)
  --location   Preferred work location or "remote"      (default: anywhere)
  --min-salary Minimum acceptable annual salary in USD
  --max-salary Maximum desired annual salary in USD
  --model      OpenAI model override                    (default: gpt-4o-mini)
  --output     pretty (human-readable) or json          (default: pretty)
```

---

## Python API

```python
from job_finder.agent import find_jobs

resume = open("resume.txt").read()

jobs = find_jobs(
    resume_text=resume,
    job_field="data science",
    location="remote",
    compensation_min=100_000,
    compensation_max=160_000,
)

for job in jobs:
    print(f"[{job['fit_score']}/10] {job['title']} at {job['company']}")
    print(f"  {job['url']}")
    print(f"  {job['fit_reasoning']}")
```

### `find_jobs()` parameters

| Parameter | Type | Description |
|---|---|---|
| `resume_text` | `str` | Full text of the candidate's resume |
| `job_field` | `str` | Target job field / industry |
| `location` | `str` | Preferred location or `"remote"` / `"anywhere"` |
| `compensation_min` | `float \| None` | Minimum annual salary in USD |
| `compensation_max` | `float \| None` | Maximum annual salary in USD |
| `model` | `str \| None` | OpenAI model name override |

Returns a `list[JobListing]` sorted by `fit_score` descending. Each
`JobListing` is a `TypedDict` with the keys:

```
title, company, location, url, salary_range,
description_summary, fit_score, fit_reasoning
```

---

## Project layout

```
langgraph/
├── job_finder/
│   ├── __init__.py      – Package declaration
│   ├── agent.py         – LangGraph StateGraph, nodes, find_jobs()
│   ├── prompts.py       – System prompts for each node
│   ├── state.py         – JobFinderState / JobListing TypedDicts
│   └── tools.py         – save_job_listings tool + search backend wiring
├── tests/
│   └── test_agent.py    – 24 unit tests (no API keys required)
├── main.py              – CLI entry point
├── requirements.txt     – Python dependencies
└── .env.example         – Example environment variable file
```

---

## Running tests

```bash
pytest tests/ -v
```

All 24 tests run without real API keys by mocking the LLM and web-search
backends.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `TAVILY_API_KEY` | ✅* | Tavily search API key (*falls back to DuckDuckGo) |
| `OPENAI_MODEL` | ❌ | Override default model (default: `gpt-4o-mini`) |

---

## License

[MIT](LICENSE)
