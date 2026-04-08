"""
scorer.py — Scores agent-cleaned dataset against ground truth.

Scoring is multi-dimensional:
  - Field-level correctness  (matches ground truth)
  - Schema validity           (passes business rules)
  - Duplicate removal         (duplicates correctly marked)
  - Final combined score      (0.0 - 1.0)
"""

import re
from typing import Any, Dict, List

from env.rules import validate_row, count_errors

SCORED_FIELDS = ["name", "email", "phone", "date_of_birth", "city", "signup_date", "status"]


def score_field(agent_val: Any, truth_val: Any, original_val: Any) -> float:
    """
    1.0  = matches ground truth
    0.0  = wrong value
   -0.3  = agent changed a correct value (over-cleaning)
    """
    is_correct  = (agent_val == truth_val)
    was_correct = (original_val == truth_val)

    if is_correct:
        return 1.0
    if was_correct and agent_val != original_val:
        return -0.3
    return 0.0


def score_row(agent_row, truth_row, original_row, fields) -> float:
    if not fields:
        return 0.0
    total = sum(
        score_field(agent_row.get(f), truth_row.get(f), original_row.get(f))
        for f in fields
    )
    return max(0.0, total / len(fields))


def _phone_digits(value: Any) -> str:
    """Return only digits from a phone string."""
    if not isinstance(value, str):
        return ""
    return re.sub(r'\D', '', value)


def _count_true_duplicates(dataset: List[Dict[str, Any]]) -> int:
    """
    FIX: detect content-based duplicates (same phone digits + lowercased name),
    NOT id-based. The duplicate rows in the dataset have different id values
    from their originals, so id matching always returned 0.
    """
    seen: set = set()
    dups = 0
    for row in dataset:
        phone_key = _phone_digits(row.get("phone", ""))
        name_key  = str(row.get("name", "")).strip().lower()
        key = (name_key, phone_key)
        if phone_key and key in seen:
            dups += 1
        else:
            seen.add(key)
    return dups  # 0 if no duplicates — dup_score will be 1.0 when none expected


def score_dataset(
    agent_dataset:    List[Dict[str, Any]],
    truth_dataset:    List[Dict[str, Any]],
    original_dataset: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Returns full scoring breakdown + final score."""

    # Filter rows the agent marked as duplicate or invalid
    agent_clean = [r for r in agent_dataset if not r.get("_duplicate") and not r.get("_invalid")]

    # ── Field correctness ────────────────────────────────────
    # Build id-keyed lookups to avoid positional index drift when rows are
    # marked as duplicate/invalid and filtered out of agent_clean.
    agent_by_id   = {r.get("id"): r for r in agent_clean}
    original_by_id = {r.get("id"): r for r in original_dataset}

    field_scores = []
    for truth_row in truth_dataset:
        oid      = truth_row.get("id")
        agent_row = agent_by_id.get(oid, {})
        orig_row  = original_by_id.get(oid, {})
        field_scores.append(score_row(agent_row, truth_row, orig_row, SCORED_FIELDS))

    field_score = sum(field_scores) / len(field_scores) if field_scores else 0.0

    # ── Schema validity ──────────────────────────────────────
    schema_errors = sum(count_errors(r) for r in agent_clean)
    total_fields  = len(agent_clean) * len(SCORED_FIELDS)
    schema_score  = max(0.0, 1.0 - schema_errors / max(total_fields, 1))

    # ── Duplicate removal ─────────────────────────────────────
    marked_dups   = sum(1 for r in agent_dataset if r.get("_duplicate"))
    true_dups     = _count_true_duplicates(original_dataset)
    if true_dups == 0:
        dup_score = 1.0 if marked_dups == 0 else 0.0  # penalise false positives
    else:
        dup_score = min(1.0, marked_dups / true_dups)

    # ── Final weighted score ──────────────────────────────────
    final = round(min(max(
        0.50 * field_score +
        0.25 * schema_score +
        0.25 * dup_score,
        0.0), 1.0), 4)

    return {
        "final_score":   final,
        "field_score":   round(field_score,  4),
        "schema_score":  round(schema_score, 4),
        "dup_score":     round(dup_score,    4),
        "schema_errors": schema_errors,
    }
