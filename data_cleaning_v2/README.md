---
title: Customer Records Cleaning OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Customer Records Cleaning OpenEnv

An **OpenEnv-compliant** RL environment where AI agents learn to clean messy customer records — one of the most common and high-value real-world data tasks.

---

## Problem Overview

Real businesses maintain customer databases with records like:

```
name            email                  phone        date_of_birth  status
──────────────────────────────────────────────────────────────────────────
rahul sharma    rahul.sharma@gmail.com 9876543210   1990-05-15     Active   ← name not title case
PRIYA PATEL     priya.patel@yahoo.com  98765-43211  15/08/1992     active   ← phone, date, status wrong
Amit Singh      amit.singhgmail.com    9123456789   1988-11-22     Active   ← email missing @
```

This environment trains agents to fix all these issues automatically.

---

## Action Space

| Action | Description |
|--------|-------------|
| `normalize_name` | Convert name to Title Case |
| `normalize_email` | Lowercase + validate email |
| `normalize_phone` | Strip to 10 digits |
| `normalize_date` | Convert DD/MM/YYYY → YYYY-MM-DD |
| `normalize_status` | Standardize to Active/Inactive/Pending |
| `normalize_city` | Title-case city name |
| `fill_missing` | Fill null field with correct value |
| `mark_duplicate` | Flag duplicate row |
| `mark_invalid` | Flag unfixable row |
| `submit` | End episode |

---

## Observation Space

```json
{
  "task_id": "task_2",
  "task_description": "...",
  "dataset": [
    {
      "_row_index": 1,
      "name": "PRIYA PATEL",
      "email": "priya.patel@yahoo.com",
      "phone": "98765-43211",
      "date_of_birth": "15/08/1992",
      "status": "active",
      "_errors": {
        "name": ["not title case"],
        "phone": ["not 10 digits"],
        "date_of_birth": ["wrong format (need YYYY-MM-DD)"],
        "status": ["invalid value: 'active'"]
      }
    }
  ],
  "score": 0.412,
  "schema_errors": 8,
  "step_count": 3,
  "max_steps": 40,
  "done": false,
  "message": "Normalized phone row 1: '98765-43211' → '9876543211'."
}
```

---

## Reward Function

| Component | Weight | Description |
|-----------|--------|-------------|
| Field correctness | 50% | Matches ground truth CSV |
| Schema validity | 25% | Passes business rules |
| Duplicate removal | 25% | Correct duplicates marked |
| Invalid action penalty | -0.05 | Per invalid step |
| Over-cleaning penalty | -0.10 | Changed a correct value |

---

## Tasks

### Task 1 — Name & Status Cleaning *(easy)*
- 5 rows, fix name casing and status normalization
- Max steps: 20

### Task 2 — Email, Phone & Date Cleaning *(medium)*
- 10 rows, fix emails, phone formats, date formats + names
- Max steps: 40

### Task 3 — Full Pipeline *(hard)*
- 17 rows, all issue types + 2 duplicate rows
- Max steps: 80

---

## Business Rules

| Field | Rule |
|-------|------|
| `name` | Title Case, letters only |
| `email` | Must contain exactly one `@` |
| `phone` | Exactly 10 digits |
| `date_of_birth` | YYYY-MM-DD format |
| `signup_date` | YYYY-MM-DD format |
| `status` | One of: Active, Inactive, Pending |
| `city` | Title Case |

---

## Setup & Usage

### Local
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
python demo.py
```

### Docker
```bash
docker build -t customer-cleaning-env .
docker run -p 7860:7860 customer-cleaning-env
```

### Run Baseline Inference
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
export ENV_URL="http://localhost:7860"
python inference.py
```

---

## Project Structure

```
├── inference.py          Baseline agent (START/STEP/END logs)
├── demo.py               End-to-end demo
├── app.py                FastAPI server
├── tasks.py              Task definitions (loads from CSV)
├── openenv.yaml          OpenEnv metadata
├── Dockerfile            Container
├── requirements.txt      Dependencies
├── README.md             This file
├── env/
│   ├── cleaning_env.py   Environment logic
│   ├── rules.py          Business rules & validation
│   └── scorer.py         Ground truth scoring
├── data/
│   ├── noisy_dataset.csv         Raw dirty data
│   └── cleaned_ground_truth.csv  Expected clean output
└── app/
    └── hf_space_demo.py  HF Spaces entry point
```

---

## Baseline Scores (gpt-4o-mini, temperature=0)

| Task | Difficulty | Score |
|------|-----------|-------|
| task_1 | easy | ~0.90 |
| task_2 | medium | ~0.75 |
| task_3 | hard | ~0.65 |
