"""
tasks.py — Load tasks from CSV files and define 3 difficulty levels.

Task 1 (Easy)   : Fix names + status normalization (5 rows)
Task 2 (Medium) : Fix emails + phones + dates (10 rows)  
Task 3 (Hard)   : Full pipeline — all issue types (15 rows + duplicates)
"""

import csv
import copy
import os
from typing import Any, Dict, List


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_csv(filename: str) -> List[Dict[str, Any]]:
    path = os.path.join(DATA_DIR, filename)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            r = dict(row)
            r["id"] = int(r["id"])
            rows.append(r)
    return rows


def _build_tasks() -> Dict[str, Any]:
    dirty_all = _load_csv("noisy_dataset.csv")
    clean_all = _load_csv("cleaned_ground_truth.csv")

    # ── Task 1: Easy — rows 0-4, only name + status issues ──
    task_1_dirty = copy.deepcopy(dirty_all[:5])
    task_1_clean = copy.deepcopy(clean_all[:5])

    # ── Task 2: Medium — rows 0-9, email + phone + date ──
    task_2_dirty = copy.deepcopy(dirty_all[:10])
    task_2_clean = copy.deepcopy(clean_all[:10])

    # ── Task 3: Hard — all 17 rows (15 unique + 2 duplicates) ──
    task_3_dirty = copy.deepcopy(dirty_all)
    task_3_clean = copy.deepcopy(clean_all)

    return {
        "task_1": {
            "id":          "task_1",
            "name":        "Customer Name & Status Cleaning",
            "difficulty":  "easy",
            "description": (
                "Clean a 5-row customer dataset. Fix issues: "
                "1) NAMES: Some names are all-lowercase or ALL-CAPS — normalize to Title Case. "
                "2) STATUS: Values like 'active', 'INACTIVE' must be standardized to "
                "'Active', 'Inactive', or 'Pending'. "
                "Use normalize_name and normalize_status actions. Call submit when done."
            ),
            "dirty":       task_1_dirty,
            "clean":       task_1_clean,
            "max_steps":   20,
        },
        "task_2": {
            "id":          "task_2",
            "name":        "Email, Phone & Date Format Cleaning",
            "difficulty":  "medium",
            "description": (
                "Clean a 10-row customer dataset. Fix issues: "
                "1) EMAILS: Invalid emails (missing @, extra @) — fix with normalize_email or fill_missing. "
                "2) PHONES: Format like '98765-43211' → normalize to 10 digits using normalize_phone. "
                "3) DATES: Format like '15/08/1992' → convert to YYYY-MM-DD using normalize_date. "
                "   Apply normalize_date to both date_of_birth and signup_date columns. "
                "4) NAMES: Some names need normalization to Title Case. "
                "Call submit when done."
            ),
            "dirty":       task_2_dirty,
            "clean":       task_2_clean,
            "max_steps":   40,
        },
        "task_3": {
            "id":          "task_3",
            "name":        "Full Customer Records Pipeline",
            "difficulty":  "hard",
            "description": (
                "Clean a full 17-row customer dataset with ALL issue types. "
                "1) NAMES: Normalize all non-title-case names. "
                "2) EMAILS: Fix invalid email formats (missing @, double @). "
                "3) PHONES: Normalize all phones to 10 digits (remove dashes/spaces). "
                "4) DATES: Convert all DD/MM/YYYY dates to YYYY-MM-DD format. "
                "   Fix both date_of_birth and signup_date columns. "
                "5) STATUS: Normalize all inconsistent status values. "
                "6) CITIES: Normalize city names to Title Case. "
                "7) DUPLICATES: Rows 15 and 16 are duplicates of rows 0 and 2. "
                "   Mark them with mark_duplicate. "
                "Check each row's _errors field to see what needs fixing. "
                "Call submit when all _errors are empty."
            ),
            "dirty":       task_3_dirty,
            "clean":       task_3_clean,
            "max_steps":   80,
        },
    }


# Load once at import time
ALL_TASKS = _build_tasks()
