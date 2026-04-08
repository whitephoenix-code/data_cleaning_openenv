"""
app.py — FastAPI server for the Customer Records Cleaning OpenEnv.

Endpoints
─────────
  GET  /          → environment info
  POST /reset     → start a new episode  (session_id in body)
  POST /step      → take an action       (session_id in body)
  GET  /state     → current episode state (session_id query param)
  GET  /health    → liveness check
  GET  /tasks     → list all tasks
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional

from tasks import ALL_TASKS
from env.cleaning_env import DataCleaningEnv

app = FastAPI(
    title="Customer Records Cleaning OpenEnv",
    description=(
        "An OpenEnv-compliant RL environment where AI agents clean messy "
        "customer records: names, emails, phone numbers, dates, cities, "
        "status fields, and duplicate rows."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIX: per-session envs instead of a single shared global instance
_sessions: Dict[str, DataCleaningEnv] = {}

def _get_env(session_id: str) -> DataCleaningEnv:
    if session_id not in _sessions:
        _sessions[session_id] = DataCleaningEnv(tasks=ALL_TASKS)
    return _sessions[session_id]


class ResetRequest(BaseModel):
    task_id:    Optional[str] = "task_1"
    session_id: Optional[str] = "default"


class StepRequest(BaseModel):
    action_type: str
    row_index:   Optional[int] = None
    column:      Optional[str] = None
    value:       Optional[Any] = None
    session_id:  Optional[str] = "default"


@app.get("/")
def root():
    return {
        "name":    "Customer Records Cleaning OpenEnv",
        "version": "1.0.0",
        "tasks":   list(ALL_TASKS.keys()),
        "difficulties": {t: ALL_TASKS[t]["difficulty"] for t in ALL_TASKS},
        "action_types": [
            "fix_value", "fill_missing", "normalize_name", "normalize_email",
            "normalize_phone", "normalize_date", "normalize_status",
            "normalize_city", "mark_duplicate", "mark_invalid",
            "flag_outlier", "submit",
        ],
        "endpoints": {
            "reset": "POST /reset",
            "step":  "POST /step",
            "state": "GET  /state",
        },
    }


@app.post("/reset")
def reset(request: ResetRequest = None):
    task_id    = "task_1"
    session_id = "default"
    if request:
        task_id    = request.task_id or "task_1"
        session_id = request.session_id or "default"
    env = _get_env(session_id)
    return env.reset(task_id=task_id)


@app.post("/step")
def step(action: StepRequest):
    session_id = action.session_id or "default"
    env = _get_env(session_id)
    return env.step(action.model_dump(exclude={"session_id"}))


@app.get("/state")
def state(session_id: str = Query(default="default")):
    return _get_env(session_id).state()


@app.get("/tasks")
def tasks():
    return {
        tid: {
            "name":        t["name"],
            "difficulty":  t["difficulty"],
            "description": t["description"],
            "max_steps":   t["max_steps"],
            "rows":        len(t["dirty"]),
        }
        for tid, t in ALL_TASKS.items()
    }


@app.get("/health")
def health():
    return {"status": "ok"}
