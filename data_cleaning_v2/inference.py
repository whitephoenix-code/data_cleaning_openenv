"""
inference.py — Baseline agent for Customer Records Cleaning OpenEnv.

Environment variables
──────────────────────
  API_BASE_URL      LLM API base URL          (default: https://api.openai.com/v1)
  MODEL_NAME        Model identifier          (default: gpt-4o-mini)
  HF_TOKEN          API key                   (NO default — must be set)
  ENV_URL           Environment server URL    (default: http://localhost:7860)
  LOCAL_IMAGE_NAME  Docker image name         (optional)
"""

import json, os, re, sys, time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ─── Config ────────────────────────────────────────────────────────────────
# Checklist: API_BASE_URL and MODEL_NAME have defaults; HF_TOKEN does NOT.
API_BASE_URL     = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN         = os.getenv("HF_TOKEN")                        # NO default
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")                # optional
ENV_URL          = os.getenv("ENV_URL", "http://localhost:7860")

TEMPERATURE       = 0.0
MAX_TOKENS        = 300
SUCCESS_THRESHOLD = 0.85   # aligned with baseline scores (task_1 ~0.90, task_2 ~0.75, task_3 ~0.65)

# All LLM calls use the OpenAI client configured via these variables
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

TASK_META = {
    "task_1": {"difficulty": "easy",   "max_steps": 20},
    "task_2": {"difficulty": "medium", "max_steps": 40},
    "task_3": {"difficulty": "hard",   "max_steps": 80},
}

# ─── Structured logging (START / STEP / END) ───────────────────────────────

def log_start(task, env, model):
    print(json.dumps({"type": "START", "task": task, "env": env, "model": model}), flush=True)

def log_step(step, action, reward, done, error=None):
    p = {"type": "STEP", "step": step, "action": action,
         "reward": round(reward, 4), "done": done}
    if error:
        p["error"] = error
    print(json.dumps(p), flush=True)

def log_end(success, steps, score, rewards):
    print(json.dumps({
        "type": "END", "success": success, "steps": steps,
        "score": round(score, 4),
        "rewards": [round(r, 4) for r in rewards],
    }), flush=True)

# ─── HTTP helpers ──────────────────────────────────────────────────────────

def env_post(endpoint, data=None):
    r = requests.post(f"{ENV_URL}{endpoint}", json=data or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_get(endpoint):
    r = requests.get(f"{ENV_URL}{endpoint}", timeout=30)
    r.raise_for_status()
    return r.json()

# ─── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a data cleaning agent. Fix messy customer records using JSON actions.

AVAILABLE ACTIONS (reply with ONE JSON object only, no markdown):
{"action_type": "normalize_name",   "row_index": N}
{"action_type": "normalize_email",  "row_index": N}
{"action_type": "fix_value",        "row_index": N, "column": "email",         "value": "correct@email.com"}
{"action_type": "normalize_phone",  "row_index": N}
{"action_type": "normalize_date",   "row_index": N, "column": "date_of_birth"}
{"action_type": "normalize_date",   "row_index": N, "column": "signup_date"}
{"action_type": "normalize_status", "row_index": N}
{"action_type": "normalize_city",   "row_index": N}
{"action_type": "fill_missing",     "row_index": N, "column": "X", "value": "Y"}
{"action_type": "mark_duplicate",   "row_index": N}
{"action_type": "mark_invalid",     "row_index": N}
{"action_type": "submit"}

RULES:
- Use "_row_index" (NOT "id") as row_index.
- Only fix rows that have "_errors". Skip rows with empty "_errors".
- Fix ONE issue per step.
- submit when ALL rows are clean."""

# ─── Deterministic action logic ────────────────────────────────────────────

def _digits_only(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r'\D', '', value)


def _find_content_duplicates(dataset: List[Dict]) -> List[int]:
    """
    Return row_indexes of content-duplicate rows (same phone digits as an
    earlier row). Duplicate rows have different id values, so we match by phone.
    """
    seen_phones: set = set()
    dup_indexes: List[int] = []
    for row in dataset:
        if row.get("_duplicate") or row.get("_invalid"):
            continue
        phone = _digits_only(row.get("phone", ""))
        ri    = row["_row_index"]
        if phone and phone in seen_phones:
            dup_indexes.append(ri)
        else:
            seen_phones.add(phone)
    return dup_indexes


def _fix_email(email: str, ri: int) -> Optional[Dict]:
    """Return a fix_value action for a broken email, or None if unsure."""
    if not isinstance(email, str):
        return None
    # No @ at all — insert before domain keyword
    if "@" not in email:
        # e.g. "amit.singhgmail.com" → "amit.singh@gmail.com"
        for domain in ("gmail", "yahoo", "hotmail", "outlook"):
            if domain in email:
                fixed = email.replace(domain, f"@{domain}")
                if fixed.count("@") == 1:
                    return {"action_type": "fix_value", "row_index": ri,
                            "column": "email", "value": fixed}
    # Double @ — e.g. "deepak@joshi@gmail.com" → "deepak.joshi@gmail.com"
    if email.count("@") > 1:
        parts = email.split("@")
        # parts[0] = local prefix, parts[1] = middle word, parts[2] = domain
        fixed = f"{parts[0]}.{parts[1]}@{parts[2]}"
        if fixed.count("@") == 1:
            return {"action_type": "fix_value", "row_index": ri,
                    "column": "email", "value": fixed}
    return None


def deterministic_action(obs: Dict) -> Optional[Dict]:
    """
    Fully rule-based action selection. Covers ALL known error patterns.
    Returns None only for edge cases that need LLM fallback.
    """
    dataset = obs.get("dataset", [])

    # ── 1. Mark content-duplicate rows first ─────────────────
    dup_indexes = _find_content_duplicates(dataset)
    if dup_indexes:
        return {"action_type": "mark_duplicate", "row_index": dup_indexes[0]}

    # ── 2. Mark Ghost User / outlier rows as invalid ──────────
    for row in dataset:
        if row.get("_duplicate") or row.get("_invalid"):
            continue
        outliers = row.get("_outliers", {})
        # signup year in the future AND the date is already YYYY-MM-DD
        # (normalize_date won't fix it — mark the row invalid instead)
        sig = str(row.get("signup_date", ""))
        if outliers.get("signup_date") and re.match(r'^\d{4}-\d{2}-\d{2}$', sig):
            return {"action_type": "mark_invalid", "row_index": row["_row_index"]}

    # ── 3. Fix errors row by row in priority order ────────────
    rows_with_errors = [
        r for r in dataset
        if r.get("_errors") and not r.get("_duplicate") and not r.get("_invalid")
    ]

    if not rows_with_errors:
        return {"action_type": "submit"}

    for row in rows_with_errors:
        ri     = row["_row_index"]
        errors = row.get("_errors", {})

        if "name" in errors:
            return {"action_type": "normalize_name", "row_index": ri}

        if "email" in errors:
            action = _fix_email(row.get("email", ""), ri)
            if action:
                return action
            # _fix_email couldn't handle it — skip to LLM for this row only
            return None

        if "phone" in errors:
            return {"action_type": "normalize_phone", "row_index": ri}

        if "date_of_birth" in errors:
            return {"action_type": "normalize_date", "row_index": ri,
                    "column": "date_of_birth"}

        if "signup_date" in errors:
            return {"action_type": "normalize_date", "row_index": ri,
                    "column": "signup_date"}

        if "status" in errors:
            return {"action_type": "normalize_status", "row_index": ri}

        if "city" in errors:
            return {"action_type": "normalize_city", "row_index": ri}

    return {"action_type": "submit"}


def llm_action(obs: Dict, history: List[Dict]) -> Dict:
    """LLM fallback for edge cases the deterministic layer couldn't handle."""
    dataset = obs.get("dataset", [])
    max_row_index = len(dataset) - 1  # valid range: 0..max_row_index

    rows_with_errors = [
        r for r in dataset
        if r.get("_errors") and not r.get("_duplicate") and not r.get("_invalid")
    ]
    user_content = (
        f"TASK: {obs['task_description']}\n\n"
        f"ROWS WITH ERRORS:\n{json.dumps(rows_with_errors[:5], indent=2)}\n\n"
        f"Score: {obs['score']:.3f} | "
        f"Schema errors: {obs['schema_errors']} | "
        f"Step: {obs['step_count']}/{obs['max_steps']}\n"
        f"Last message: {obs['message']}\n\n"
        "Fix the FIRST error in the FIRST row. ONE JSON action only."
    )
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history[-6:]
        + [{"role": "user", "content": user_content}]
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"):
                    raw = part
                    break
        action = json.loads(raw)
        # Guard: reject hallucinated out-of-range row indexes
        ri = action.get("row_index")
        if ri is not None and not (0 <= ri <= max_row_index):
            print(f"[DEBUG] LLM hallucinated row_index {ri} (valid: 0–{max_row_index}), submitting instead.", flush=True)
            return {"action_type": "submit"}
        history.append({"role": "assistant", "content": json.dumps(action)})
        return action
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return {"action_type": "submit"}


def get_action(obs: Dict, history: List[Dict]) -> Dict:
    """Try deterministic first; fall back to LLM only if needed."""
    action = deterministic_action(obs)
    if action is not None:
        return action
    return llm_action(obs, history)

# ─── Task runner ───────────────────────────────────────────────────────────

def run_task(task_id: str) -> float:
    meta = TASK_META[task_id]
    log_start(
        task=f"customer_cleaning_{task_id}",
        env="customer-cleaning-env",
        model=MODEL_NAME,
    )

    session_id  = f"{task_id}_{int(__import__('time').time())}"
    obs         = env_post("/reset", {"task_id": task_id, "session_id": session_id})
    history:    List[Dict] = []
    rewards:    List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    try:
        for step in range(1, meta["max_steps"] + 1):
            if obs.get("done"):
                break

            action = get_action(obs, history)
            action["session_id"] = session_id
            history.append({"role": "user", "content": f"[step {step}]"})

            try:
                result     = env_post("/step", action)
                obs        = result["observation"]
                reward_val = result["reward"]["score"]
                done       = result["done"]
                action_ok  = result["info"].get("action_success", True)
                err        = None if action_ok else "invalid_action"

                if not action_ok:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"[DEBUG] {consecutive_failures} consecutive failures — submitting to end episode.", flush=True)
                        env_post("/step", {"action_type": "submit", "session_id": session_id})
                        log_step(step=step, action=action, reward=reward_val, done=True, error=err)
                        break
                else:
                    consecutive_failures = 0

                rewards.append(reward_val)
                steps_taken = step
                score       = reward_val

                log_step(step=step, action=action, reward=reward_val, done=done, error=err)

                if done:
                    break
            except Exception as e:
                log_step(step=step, action=action, reward=0.0, done=False, error=str(e))
                break

        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print(f"[DEBUG] Customer Records Cleaning OpenEnv", flush=True)
    print(f"[DEBUG] Model: {MODEL_NAME} | Server: {ENV_URL}", flush=True)

    try:
        env_get("/health")
        print("[DEBUG] Server health: OK", flush=True)
    except Exception as e:
        print(f"[DEBUG] ERROR: Server not reachable — {e}", flush=True)
        sys.exit(1)

    all_scores: Dict[str, float] = {}
    t_start = time.time()

    for task_id in ["task_1", "task_2", "task_3"]:
        print(f"\n[DEBUG] ── {task_id} ({TASK_META[task_id]['difficulty']}) ──", flush=True)
        try:
            all_scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"[DEBUG] ERROR in {task_id}: {e}", flush=True)
            all_scores[task_id] = 0.0

    elapsed = time.time() - t_start
    avg     = sum(all_scores.values()) / len(all_scores)

    print(f"\n[DEBUG] ══════════════════════", flush=True)
    print(f"[DEBUG] FINAL RESULTS",          flush=True)
    for tid, sc in all_scores.items():
        print(f"[DEBUG]   {tid}: {sc:.3f}",  flush=True)
    print(f"[DEBUG] Average : {avg:.3f}",    flush=True)
    print(f"[DEBUG] Elapsed : {elapsed:.1f}s", flush=True)
    print(f"[DEBUG] ══════════════════════", flush=True)


if __name__ == "__main__":
    main()
