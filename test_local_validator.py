"""Test the LOCAL server (both app.py endpoints) for score range compliance."""
import json
import subprocess
import time
import requests
import sys
import os

# Start local server
print("Starting local server on port 9999...")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9999"],
    cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(4)

BASE = "http://localhost:9999"
all_ok = True

try:
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        print(f"\n=== {task_id} ===")
        
        r = requests.post(f"{BASE}/reset", json={"task_id": task_id}, timeout=10)
        sid = r.json()["session_id"]
        
        # 3 noop + submit
        for i in range(3):
            r = requests.post(f"{BASE}/step", json={
                "session_id": sid, "action_type": "noop", "params": {}
            }, timeout=10)
            data = r.json()
            # Check reward
            rew = data.get("reward", {})
            if isinstance(rew, dict):
                for k, v in rew.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        if v <= 0.0 or v >= 1.0:
                            print(f"  *** NON-DONE reward.{k} = {v} OUT OF RANGE!")
                            all_ok = False
                    elif isinstance(v, dict):
                        for ik, iv in v.items():
                            if isinstance(iv, (int, float)) and not isinstance(iv, bool):
                                if iv <= 0.0 or iv >= 1.0:
                                    print(f"  *** NON-DONE reward.{k}.{ik} = {iv} OUT OF RANGE!")
                                    all_ok = False
        
        # Submit
        r = requests.post(f"{BASE}/step", json={
            "session_id": sid,
            "action_type": "submit_report",
            "params": {"summary": "Test report for validation."}
        }, timeout=10)
        final = r.json()
        
        done = final.get("done")
        score = final.get("score")
        task_score = final.get("task_score")
        final_score = final.get("final_score")
        
        print(f"  done={done}")
        print(f"  score={score}")
        print(f"  task_score={task_score}")
        print(f"  final_score={final_score}")
        
        # Check top-level score keys exist
        for key in ("score", "task_score", "final_score"):
            val = final.get(key)
            if val is None:
                print(f"  *** MISSING top-level '{key}'!")
                all_ok = False
            elif val <= 0.0 or val >= 1.0:
                print(f"  *** {key} = {val} OUT OF RANGE!")
                all_ok = False
        
        # Deep check reward
        rew = final.get("reward", {})
        if isinstance(rew, dict):
            for k, v in rew.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if v <= 0.0 or v >= 1.0:
                        print(f"  *** reward.{k} = {v} OUT OF RANGE!")
                        all_ok = False
                elif isinstance(v, dict):
                    for ik, iv in v.items():
                        if isinstance(iv, (int, float)) and not isinstance(iv, bool):
                            if iv <= 0.0 or iv >= 1.0:
                                print(f"  *** reward.{k}.{ik} = {iv} OUT OF RANGE!")
                                all_ok = False
        
        # Check grader scores
        gr = final.get("info", {}).get("grader_result", {})
        for key in ("score", "final_score"):
            val = gr.get(key)
            if val is not None and (val <= 0.0 or val >= 1.0):
                print(f"  *** grader.{key} = {val} OUT OF RANGE!")
                all_ok = False
        for comp_name, comp_data in gr.get("components", {}).items():
            if isinstance(comp_data, dict):
                s = comp_data.get("score")
                if s is not None and (s <= 0.0 or s >= 1.0):
                    print(f"  *** component {comp_name}.score = {s} OUT OF RANGE!")
                    all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("✅ ALL SCORES OK — every value is strictly in (0, 1)")
    else:
        print("❌ FOUND ISSUES — see above")
    print(f"{'='*60}")

finally:
    proc.terminate()
    proc.wait()
