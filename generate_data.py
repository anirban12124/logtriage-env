"""
Data generation script for LogTriage.

Generates and saves:
  - data/logs/{easy,medium,hard}_log.json
  - data/ground_truth/{easy,medium,hard}_gt.json

Also runs a 5-iteration determinism check and prints the SHA-256
hash of each dataset to confirm reproducibility.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import hashlib
from src.log_generator import generate_logs
from src.tasks import TASKS

BASE = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE, "data", "logs")
GT_DIR  = os.path.join(BASE, "data", "ground_truth")

def sha256_of(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def generate_and_save_all():
    task_map = {
        "task_easy":   "easy",
        "task_medium": "medium",
        "task_hard":   "hard",
    }

    for task_id, name in task_map.items():
        print(f"\n{'='*60}")
        print(f"Generating: {task_id}")
        print(f"{'='*60}")

        logs = generate_logs(task_id)
        task = TASKS[task_id]
        gt   = task["ground_truth"]

        # ── Save logs ────────────────────────────────────────────
        log_path = os.path.join(LOG_DIR, f"{name}_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(logs)} log entries → {log_path}")

        # ── Save ground truth ────────────────────────────────────
        gt_path = os.path.join(GT_DIR, f"{name}_gt.json")
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)
        print(f"  Saved ground truth ({len(gt['annotations'])} annotations) → {gt_path}")

        # ── Determinism check (5 regenerations) ──────────────────
        h0 = sha256_of(logs)
        print(f"  SHA-256: {h0[:16]}…")
        ok = True
        for i in range(2, 7):
            hi = sha256_of(generate_logs(task_id))
            if hi != h0:
                print(f"  ❌ Run {i} DIFFERS — non-determinism detected!")
                ok = False
                break
        if ok:
            print(f"  ✅ 5-run determinism check passed")

        # ── Print sample logs ─────────────────────────────────────
        print(f"\n  First 3 log entries:")
        for log in logs[:3]:
            msg = log['message'][:80]
            print(f"    [{log['id']}] {log['severity']:<5} | {log['service']:<20} | {msg}")

        print(f"\n  Ground-truth logs:")
        gt_ann = gt["annotations"]
        for log_id, cat in gt_ann.items():
            # Find the matching log entry
            entry = next((l for l in logs if l["id"] == log_id), None)
            if entry:
                msg = entry['message'][:70]
                print(f"    {log_id} [{cat}] → {msg}")

        if gt.get("correlations"):
            print(f"\n  Correlations ({len(gt['correlations'])} pairs):")
            for pair in gt["correlations"]:
                print(f"    {pair[0]} → {pair[1]}")

        print(f"\n  Severity: {gt['severity']}")
        print(f"  Key findings: {gt['key_findings']}")

    print(f"\n{'='*60}")
    print("All datasets generated and saved successfully.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_and_save_all()
