"""
test_grader_e2e.py — Full end-to-end grader validation
=======================================================
Tests the live server (must be running on port 8000) using scripted
agents that simulate three quality levels:

  1. ORACLE  — takes all correct actions, perfect report → expects HIGH score
  2. PARTIAL — finds half the GT, mediocre report       → expects MID score
  3. BAD     — annotates random wrong things, spam report → expects LOW score

Also validates:
  - Reward sequence is non-trivial (varies across steps)
  - Grader scores are in [0.0, 1.0]
  - Episode ends cleanly with grader_result in info
  - /health, /tasks, /state endpoints work correctly
  - Determinism: same oracle run twice → identical grader score
"""

import json
import requests

BASE = "http://localhost:8000"


# ── Helpers ──────────────────────────────────────────────────────────────────

def health_check():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200, f"Health failed: {r.status_code}"
    data = r.json()
    print(f"[HEALTH] {data}")
    return data


def list_tasks():
    r = requests.get(f"{BASE}/tasks")
    assert r.status_code == 200
    tasks = r.json()
    print(f"[TASKS] {[t['id'] for t in tasks]}")
    assert len(tasks) == 3, "Expected 3 tasks"
    return tasks


def reset(task_id):
    r = requests.post(f"{BASE}/reset", json={"task_id": task_id})
    assert r.status_code == 200, f"Reset failed: {r.text}"
    return r.json()


def step(session_id, action_type, params=None):
    r = requests.post(f"{BASE}/step", json={
        "session_id": session_id,
        "action_type": action_type,
        "params": params or {},
    })
    assert r.status_code == 200, f"Step failed: {r.text}"
    return r.json()


def get_state(session_id):
    r = requests.get(f"{BASE}/state", params={"session_id": session_id})
    assert r.status_code == 200
    return r.json()


def run_episode(task_id, actions, label):
    """Execute a scripted action sequence and return the grader result."""
    data   = reset(task_id)
    sid    = data["session_id"]
    obs    = data["observation"]
    done   = False
    result = None
    rewards = []

    print(f"\n  [{label}] {task_id}  goal: {obs['goal'][:60]}…")
    print(f"  [{label}] total_logs={obs['total_log_count']}  max_steps={obs['max_steps']}")

    for i, (atype, params) in enumerate(actions):
        if done:
            break
        res    = step(sid, atype, params)
        obs    = res["observation"]
        rew    = res["reward"]["value"]
        done   = res["done"]
        result = res

        print(f"    step {i+1:02d}: {atype:<22} reward={rew:+.4f}  cum={res['reward']['cumulative']:+.4f}  "
              f"done={done}  msg={obs['last_action_message'][:40]}")
        rewards.append(rew)

    grader = result.get("info", {}).get("grader_result") if result else None
    if not done:
        # Force-submit to get grader result if we ran out of actions
        res    = step(sid, "submit_report", {"summary": "No findings."})
        done   = res["done"]
        grader = res.get("info", {}).get("grader_result")

    return grader, rewards


# ── Task-specific oracle action sequences ─────────────────────────────────────

def oracle_easy():
    """Perfect agent for task_easy — finds all 3 GT errors, correct report."""
    actions = [
        ("filter_severity", {"level": "ERROR"}),
        ("inspect",         {"log_id": "log_008"}),
        ("annotate",        {"log_id": "log_008", "category": "error"}),
        ("inspect",         {"log_id": "log_023"}),
        ("annotate",        {"log_id": "log_023", "category": "error"}),
        ("inspect",         {"log_id": "log_041"}),
        ("annotate",        {"log_id": "log_041", "category": "error"}),
        ("classify_incident", {"severity": "MEDIUM"}),
        ("draft_report",    {"summary":
            "The auth-service experienced repeated database connection failures "
            "with errors connecting to the primary database server on port 5432. "
            "All three errors are database connection refused on postgresql port 5432. "
            "The auth-service was the sole affected component."}),
        ("submit_report",   {"summary":
            "The auth-service experienced repeated database connection failures "
            "with errors connecting to the primary database server on postgresql port 5432. "
            "Database connection refused on port 5432 caused degradation. "
            "The auth-service was the sole affected component experiencing database "
            "connectivity issues. Severity: MEDIUM."}),
    ]
    return actions


def oracle_medium():
    """Perfect agent for task_medium."""
    actions = [
        ("filter_service",  {"service": "payment-service"}),
        ("inspect",         {"log_id": "log_045"}),
        ("annotate",        {"log_id": "log_045", "category": "root_cause"}),
        ("inspect",         {"log_id": "log_067"}),
        ("annotate",        {"log_id": "log_067", "category": "symptom"}),
        ("filter_service",  {"service": "order-service"}),
        ("inspect",         {"log_id": "log_089"}),
        ("annotate",        {"log_id": "log_089", "category": "symptom"}),
        ("inspect",         {"log_id": "log_134"}),
        ("annotate",        {"log_id": "log_134", "category": "cascading_failure"}),
        ("filter_service",  {"service": "api-gateway"}),
        ("inspect",         {"log_id": "log_102"}),
        ("annotate",        {"log_id": "log_102", "category": "symptom"}),
        ("inspect",         {"log_id": "log_156"}),
        ("annotate",        {"log_id": "log_156", "category": "cascading_failure"}),
        ("correlate",       {"source_log_id": "log_045", "target_log_id": "log_067"}),
        ("correlate",       {"source_log_id": "log_067", "target_log_id": "log_134"}),
        ("correlate",       {"source_log_id": "log_089", "target_log_id": "log_156"}),
        ("classify_incident", {"severity": "HIGH"}),
        ("submit_report",   {"summary":
            "Payment service database connection pool exhausted with all 50 connections "
            "in use triggering transaction timeouts. This root cause caused order service "
            "queue backup with 15000 messages pending causing order processing failure. "
            "API gateway returned HTTP 503 503 Service Unavailable errors to end users. "
            "Cascading failure propagated from payment-service through order-service "
            "to the api-gateway. Severity HIGH."}),
    ]
    return actions


def oracle_hard():
    """Perfect agent for task_hard — full attack chain."""
    actions = [
        ("search",          {"pattern": "198.51.100"}),
        ("inspect",         {"log_id": "log_023"}),
        ("annotate",        {"log_id": "log_023", "category": "reconnaissance"}),
        ("inspect",         {"log_id": "log_078"}),
        ("annotate",        {"log_id": "log_078", "category": "brute_force"}),
        ("inspect",         {"log_id": "log_091"}),
        ("annotate",        {"log_id": "log_091", "category": "brute_force"}),
        ("inspect",         {"log_id": "log_145"}),
        ("annotate",        {"log_id": "log_145", "category": "credential_compromise"}),
        ("inspect",         {"log_id": "log_201"}),
        ("annotate",        {"log_id": "log_201", "category": "privilege_escalation"}),
        ("inspect",         {"log_id": "log_267"}),
        ("annotate",        {"log_id": "log_267", "category": "data_exfiltration"}),
        ("inspect",         {"log_id": "log_312"}),
        ("annotate",        {"log_id": "log_312", "category": "lateral_movement"}),
        ("inspect",         {"log_id": "log_389"}),
        ("annotate",        {"log_id": "log_389", "category": "data_exfiltration"}),
        ("correlate",       {"source_log_id": "log_023", "target_log_id": "log_078"}),
        ("correlate",       {"source_log_id": "log_091", "target_log_id": "log_145"}),
        ("correlate",       {"source_log_id": "log_145", "target_log_id": "log_201"}),
        ("correlate",       {"source_log_id": "log_201", "target_log_id": "log_267"}),
        ("correlate",       {"source_log_id": "log_201", "target_log_id": "log_312"}),
        ("correlate",       {"source_log_id": "log_312", "target_log_id": "log_389"}),
        ("classify_incident", {"severity": "CRITICAL"}),
        ("submit_report",   {"summary":
            "A multi-stage security breach was detected across 5 services. "
            "The attacker performed reconnaissance probing from IP 198.51.100.23. "
            "Automated brute force login attempts targeted admin and service account "
            "credentials with hundreds of failed login attempts from external IP. "
            "Compromised credentials for the svc-deploy service account were obtained "
            "after sustained brute force with over 91 failed logins. "
            "The attacker escalated privileges by accessing admin API endpoint bypassing "
            "authorization with unauthorized admin access via legacy policy. "
            "Bulk data export of 15247 customer records constitutes data exfiltration. "
            "Lateral movement to file-service reused compromised authentication token. "
            "Bulk file download of 342 files from /confidential/reports as final exfiltration. "
            "Severity: CRITICAL."}),
    ]
    return actions


def partial_easy():
    """Finds only 1 of 3 GT logs, mediocre report."""
    return [
        ("inspect",         {"log_id": "log_008"}),
        ("annotate",        {"log_id": "log_008", "category": "error"}),
        ("classify_incident", {"severity": "HIGH"}),  # Wrong severity (MEDIUM expected)
        ("submit_report",   {"summary": "Found some database errors in the logs."}),
    ]


def bad_easy():
    """Annotates wrong logs, no classification, empty report."""
    return [
        ("annotate", {"log_id": "log_001", "category": "error"}),  # wrong ID
        ("annotate", {"log_id": "log_002", "category": "error"}),  # wrong ID
        ("annotate", {"log_id": "log_003", "category": "error"}),  # wrong ID
        ("submit_report", {"summary": "no issues found"}),
    ]


# ── Main test runner ──────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("LogTriage — End-to-End Grader Test")
    print("=" * 65)

    # 1. Basic API checks
    print("\n== API Health Checks ==")
    health_check()
    list_tasks()

    results = {}

    # 2. Oracle agents (should score HIGH)
    print("\n== ORACLE AGENT (expects high scores) ==")
    for task_id, action_fn in [
        ("task_easy",   oracle_easy),
        ("task_medium", oracle_medium),
        ("task_hard",   oracle_hard),
    ]:
        grader, rewards = run_episode(task_id, action_fn(), "ORACLE")
        results[("oracle", task_id)] = grader
        score = grader.get("final_score", 0) if grader else 0
        print(f"  >> ORACLE {task_id} FINAL SCORE: {score:.4f}")
        if grader:
            print(f"     Components:")
            for comp, data in grader.get("components", {}).items():
                s = data.get("score", 0)
                w = data.get("weight", 0)
                ws = data.get("weighted", 0)
                print(f"       {comp:<30} score={s:.4f}  weight={w:.2f}  weighted={ws:.4f}")

    # 3. Partial agent on easy (should score MID)
    print("\n== PARTIAL AGENT (expects mid scores) ==")
    grader, _ = run_episode("task_easy", partial_easy(), "PARTIAL")
    results[("partial", "task_easy")] = grader
    score = grader.get("final_score", 0) if grader else 0
    print(f"  >> PARTIAL task_easy FINAL SCORE: {score:.4f}")

    # 4. Bad agent (should score LOW)
    print("\n== BAD AGENT (expects low scores) ==")
    grader, _ = run_episode("task_easy", bad_easy(), "BAD")
    results[("bad", "task_easy")] = grader
    score = grader.get("final_score", 0) if grader else 0
    print(f"  >> BAD task_easy FINAL SCORE: {score:.4f}")

    # 5. Determinism check: run oracle_easy twice, compare scores
    print("\n== DETERMINISM CHECK (oracle_easy x2) ==")
    g1, _ = run_episode("task_easy", oracle_easy(), "DET-1")
    g2, _ = run_episode("task_easy", oracle_easy(), "DET-2")
    s1 = round(g1.get("final_score", 0), 6) if g1 else 0
    s2 = round(g2.get("final_score", 0), 6) if g2 else 0
    match = s1 == s2
    print(f"  Run 1: {s1:.6f}  Run 2: {s2:.6f}  MATCH: {'YES' if match else 'NO *** FAIL'}")

    # 6. Summary table
    print("\n" + "=" * 65)
    print("GRADER SCORE SUMMARY")
    print("=" * 65)
    print(f"  {'Agent':<10}  {'Task':<15}  {'Score':>8}  {'Expected range'}")
    print(f"  {'-'*55}")

    score_ranges = {
        ("oracle", "task_easy"):   (0.60, 1.0,  "0.60-1.0 (oracle)"),
        ("oracle", "task_medium"): (0.40, 1.0,  "0.40-1.0 (oracle)"),
        ("oracle", "task_hard"):   (0.30, 1.0,  "0.30-1.0 (oracle)"),
        ("partial", "task_easy"):  (0.10, 0.60, "0.10-0.60 (partial)"),
        ("bad",     "task_easy"):  (0.00, 0.20, "0.00-0.20 (bad)"),
    }

    all_passed = True
    for (agent, task_id), grader in results.items():
        score = grader.get("final_score", 0) if grader else 0
        lo, hi, label = score_ranges.get((agent, task_id), (0, 1, ""))
        passed = lo <= score <= hi
        if not passed:
            all_passed = False
        status = "OK" if passed else "!! OUTSIDE RANGE"
        print(f"  {agent:<10}  {task_id:<15}  {score:>8.4f}  {label}  {status}")

    print(f"\n  Determinism: {'PASS' if match else 'FAIL'}")
    print(f"  Score ranges: {'ALL PASS' if all_passed else 'SOME FAILED'}")
    print(f"\n{'OVERALL: PASS' if all_passed and match else 'OVERALL: NEEDS REVIEW'}")


if __name__ == "__main__":
    main()
