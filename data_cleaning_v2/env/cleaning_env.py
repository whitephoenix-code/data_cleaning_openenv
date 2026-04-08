"""
cleaning_env.py — Core DataCleaningEnv with full action/reward logic.

Actions supported:
  fill_missing     — fill a null field
  fix_value        — directly set any field to a new value
  normalize_name   — title-case a name
  normalize_email  — lowercase + validate email
  normalize_phone  — strip to 10 digits
  normalize_date   — convert to YYYY-MM-DD
  normalize_status — standardize status value
  normalize_city   — title-case city name
  mark_duplicate   — flag duplicate row
  mark_invalid     — flag row as invalid/unfixable
  flag_outlier     — flag statistical outlier row
  submit           — end episode
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

from env.rules import (
    normalize_name, normalize_phone, normalize_date,
    normalize_status, is_valid_email, validate_row,
    count_errors, is_outlier
)
from env.scorer import score_dataset, SCORED_FIELDS


class DataCleaningEnv:

    def __init__(self, tasks: Dict[str, Any]) -> None:
        self._tasks      = tasks
        self._task_id    = None
        self._dataset    = None
        self._original   = None
        self._truth      = None
        self._step       = 0
        self._done       = False
        self._prev_score = 0.0
        self._max_steps  = 50

    # ─── Public API ───────────────────────────────────────────

    def reset(self, task_id: str = "task_1") -> Dict[str, Any]:
        task             = self._tasks.get(task_id, self._tasks["task_1"])
        self._task_id    = task_id
        self._dataset    = copy.deepcopy(task["dirty"])
        self._original   = copy.deepcopy(task["dirty"])
        self._truth      = copy.deepcopy(task["clean"])
        self._step       = 0
        self._done       = False
        self._prev_score = 0.0
        self._max_steps  = task.get("max_steps", 50)
        return self._obs("Episode started. Analyse the dataset and fix all issues.")

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self._done:
            obs = self._obs("Episode done — call reset() to start over.")
            return self._result(obs, 0.0, 0.0, True, {"error": "episode_done"})

        self._step += 1
        atype = action.get("action_type", "")

        if atype == "submit" or self._step >= self._max_steps:
            return self._finish("Agent submitted." if atype == "submit" else "Max steps reached.")

        prev_errors = self._total_errors()
        msg, ok     = self._execute(action)
        new_errors  = self._total_errors()
        scores      = score_dataset(self._dataset, self._truth, self._original)
        new_score   = scores["final_score"]

        # FIX: compute delta BEFORE updating self._prev_score
        if not ok:
            shaped_delta = -0.05
        elif new_errors < prev_errors:
            shaped_delta = new_score - self._prev_score
        elif new_errors > prev_errors:
            shaped_delta = -0.1
        else:
            shaped_delta = -0.01

        self._prev_score = new_score

        if new_errors == 0:
            self._done = True
            msg += " Dataset is perfectly clean!"

        obs = self._obs(msg)
        return self._result(obs, new_score, shaped_delta, self._done, {
            "action_success":  ok,
            "errors_before":   prev_errors,
            "errors_after":    new_errors,
            "delta":           round(shaped_delta, 4),
            "score_breakdown": scores,
        })

    def state(self) -> Dict[str, Any]:
        if self._task_id is None:
            return {"status": "not_started"}
        scores = score_dataset(self._dataset, self._truth, self._original)
        return {
            "task_id":      self._task_id,
            "step":         self._step,
            "max_steps":    self._max_steps,
            "done":         self._done,
            "score":        scores["final_score"],
            "field_score":  scores["field_score"],
            "schema_score": scores["schema_score"],
            "dup_score":    scores["dup_score"],
            "schema_errors":scores["schema_errors"],
            "rows":         len(self._dataset),
        }

    # ─── Action executor ──────────────────────────────────────

    def _execute(self, action: Dict[str, Any]) -> Tuple[str, bool]:
        atype = action.get("action_type")
        ri    = action.get("row_index")
        col   = action.get("column")
        val   = action.get("value")

        if ri is None:
            return "Error: row_index required.", False
        if not (0 <= ri < len(self._dataset)):
            return f"Error: row_index {ri} out of range.", False

        row = self._dataset[ri]

        if atype == "fix_value":
            if not col:
                return "Error: column required.", False
            if val is None:
                return "Error: value required.", False
            old = row.get(col)
            row[col] = val
            return f"Fixed row {ri}[{col}]: '{old}' → '{val}'.", True

        elif atype == "fill_missing":
            if not col:
                return "Error: column required.", False
            current = row.get(col)
            if current is not None and str(current).strip() != "":
                return f"Row {ri}[{col}] already has value '{current}' — use fix_value instead.", False
            # Validate value against business rules before filling
            from env.rules import (
                is_valid_name, is_valid_email, is_valid_phone,
                is_valid_date, is_valid_status
            )
            validators = {
                "name":          is_valid_name,
                "email":         is_valid_email,
                "phone":         is_valid_phone,
                "date_of_birth": is_valid_date,
                "signup_date":   is_valid_date,
                "status":        is_valid_status,
            }
            if col in validators and val is not None and not validators[col](val):
                return f"Error: value '{val}' fails business rule for column '{col}'.", False
            row[col] = val
            return f"Filled row {ri}[{col}] = '{val}'.", True

        elif atype == "normalize_name":
            old = row.get("name")
            new = normalize_name(old)
            if new is None:
                return f"Cannot normalize name: '{old}'.", False
            if new == old:
                return f"Name already correct: '{old}'.", False
            row["name"] = new
            return f"Normalized name row {ri}: '{old}' → '{new}'.", True

        elif atype == "normalize_email":
            old = row.get("email", "")
            if not isinstance(old, str):
                return "Email is not a string.", False
            new = old.strip().lower()
            if not is_valid_email(new):
                return f"Email still invalid after normalize: '{new}'.", False
            if new == old:
                return "Email already correct.", False
            row["email"] = new
            return f"Normalized email row {ri}: '{old}' → '{new}'.", True

        elif atype == "normalize_phone":
            old = row.get("phone", "")
            new = normalize_phone(old)
            if new is None:
                return f"Cannot normalize phone '{old}' to 10 digits.", False
            if new == old:
                return "Phone already correct.", False
            row["phone"] = new
            return f"Normalized phone row {ri}: '{old}' → '{new}'.", True

        elif atype == "normalize_date":
            if not col:
                col = "date_of_birth"
            old = row.get(col, "")
            new = normalize_date(old)
            if new is None:
                return f"Cannot parse date '{old}'.", False
            if new == old:
                return f"Date already correct: '{old}'.", False
            row[col] = new
            return f"Normalized date row {ri}[{col}]: '{old}' → '{new}'.", True

        elif atype == "normalize_status":
            old = row.get("status", "")
            new = normalize_status(old)
            if new is None:
                return f"Cannot normalize status '{old}'.", False
            if new == old:
                return "Status already correct.", False
            row["status"] = new
            return f"Normalized status row {ri}: '{old}' → '{new}'.", True

        elif atype == "normalize_city":
            old = row.get("city", "")
            if not isinstance(old, str):
                return "City is not a string.", False
            new = old.strip().title()
            if new == old:
                return "City already correct.", False
            row["city"] = new
            return f"Normalized city row {ri}: '{old}' → '{new}'.", True

        elif atype == "mark_duplicate":
            if row.get("_duplicate"):
                return f"Row {ri} already marked duplicate.", False
            row["_duplicate"] = True
            return f"Marked row {ri} as duplicate.", True

        elif atype == "mark_invalid":
            if row.get("_invalid"):
                return f"Row {ri} already marked invalid.", False
            row["_invalid"] = True
            return f"Marked row {ri} as invalid.", True

        elif atype == "flag_outlier":
            outliers = is_outlier(row)
            if not outliers:
                return f"Row {ri} has no detectable outliers.", False
            if row.get("_outlier"):
                return f"Row {ri} already flagged as outlier.", False
            row["_outlier"] = True
            row["_outlier_reasons"] = outliers
            return f"Flagged row {ri} as outlier: {outliers}.", True

        else:
            return (
                f"Unknown action '{atype}'. Valid: fix_value, fill_missing, normalize_name, "
                "normalize_email, normalize_phone, normalize_date, normalize_status, "
                "normalize_city, mark_duplicate, mark_invalid, flag_outlier, submit.",
                False,
            )

    # ─── Helpers ──────────────────────────────────────────────

    def _obs(self, message: str) -> Dict[str, Any]:
        task   = self._tasks.get(self._task_id, {})
        scores = score_dataset(self._dataset, self._truth, self._original)

        indexed = []
        for i, row in enumerate(self._dataset):
            r = {"_row_index": i}
            r.update(row)
            errs     = validate_row(row)
            outliers = is_outlier(row)
            r["_errors"] = {k: v for k, v in errs.items()} if errs else {}
            if outliers and not row.get("_outlier"):
                r["_outliers"] = outliers
            indexed.append(r)

        return {
            "task_id":          self._task_id,
            "task_name":        task.get("name", ""),
            "task_description": task.get("description", ""),
            "dataset":          indexed,
            "columns":          SCORED_FIELDS,
            "score":            scores["final_score"],
            "field_score":      scores["field_score"],
            "schema_score":     scores["schema_score"],
            "schema_errors":    scores["schema_errors"],
            "step_count":       self._step,
            "max_steps":        self._max_steps,
            "done":             self._done,
            "message":          message,
        }

    def _result(self, obs, score, delta, done, info) -> Dict[str, Any]:
        return {
            "observation": obs,
            "reward": {
                "score":   score,
                "delta":   round(delta, 4),   # FIX: now uses correct pre-update delta
                "message": obs["message"],
            },
            "done": done,
            "info": info,
        }

    def _finish(self, reason: str) -> Dict[str, Any]:
        self._done  = True
        scores      = score_dataset(self._dataset, self._truth, self._original)
        score       = scores["final_score"]
        delta       = score - self._prev_score
        self._prev_score = score
        msg         = f"{reason} Final score: {score:.3f}"
        obs         = self._obs(msg)
        return self._result(obs, score, delta, True, {
            "final_score":     score,
            "score_breakdown": scores,
            "reason":          reason,
        })

    def _total_errors(self) -> int:
        return sum(
            count_errors(r) for r in self._dataset
            if not r.get("_duplicate") and not r.get("_invalid")
        )
