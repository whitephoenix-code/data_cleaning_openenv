"""
demo.py — End-to-end demo of the Customer Records Cleaning environment.

Shows sample input, actions taken, and cleaned output.
Run: python demo.py
"""

import json
from tasks import ALL_TASKS
from env.cleaning_env import DataCleaningEnv
from env.rules import validate_row

def print_table(rows, title=""):
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    fields = ["id", "name", "email", "phone", "date_of_birth", "status"]
    header = f"{'id':<4} {'name':<18} {'email':<28} {'phone':<12} {'dob':<12} {'status':<10}"
    print(header)
    print("-" * 90)
    for row in rows:
        errors = validate_row(row)
        flag   = " ❌" if errors else " ✅"
        print(
            f"{str(row.get('id','')):<4} "
            f"{str(row.get('name','')):<18} "
            f"{str(row.get('email','')):<28} "
            f"{str(row.get('phone','')):<12} "
            f"{str(row.get('date_of_birth','')):<12} "
            f"{str(row.get('status','')):<10}"
            f"{flag}"
        )

def main():
    print("\n🧹 Customer Records Cleaning OpenEnv — DEMO")
    print("=" * 60)

    env = DataCleaningEnv(tasks=ALL_TASKS)

    # ── Task 1 Demo ──────────────────────────────────────────
    obs = env.reset("task_1")
    print_table(ALL_TASKS["task_1"]["dirty"], "TASK 1 — Dirty Data (Before)")

    # Apply fixes manually for demo
    actions = [
        {"action_type": "normalize_name",   "row_index": 0},
        {"action_type": "normalize_status", "row_index": 1},
        {"action_type": "normalize_name",   "row_index": 1},
        {"action_type": "normalize_name",   "row_index": 3},
        {"action_type": "normalize_status", "row_index": 3},
        {"action_type": "normalize_name",   "row_index": 5},  # will be skipped (out of range for task_1)
        {"action_type": "submit"},
    ]

    print("\n📋 Actions taken:")
    for a in actions:
        result = env.step(a)
        msg    = result["observation"]["message"]
        score  = result["reward"]["score"]
        print(f"  {a['action_type']:<20} → score: {score:.3f}  | {msg[:50]}")
        if result["done"]:
            break

    print_table(env._dataset, "TASK 1 — Cleaned Data (After)")
    print(f"\n✅ Final score: {result['reward']['score']:.3f}")

    # ── Issue summary ────────────────────────────────────────
    print("\n\n📊 ALL TASKS SUMMARY")
    print("=" * 60)
    for tid, task in ALL_TASKS.items():
        dirty = task["dirty"]
        errors = sum(
            1 for row in dirty
            if validate_row(row)
        )
        print(f"  {tid} ({task['difficulty']:<6}): {len(dirty)} rows, "
              f"{errors} rows with errors, max_steps={task['max_steps']}")

    print("\n🚀 Server running at http://localhost:7860")
    print("📖 API docs at    http://localhost:7860/docs")
    print("🎯 Run inference: python inference.py\n")

if __name__ == "__main__":
    main()
