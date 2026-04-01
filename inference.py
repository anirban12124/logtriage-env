"""
LogTriage Inference Script
==========================
Baseline agent that runs an LLM against the LogTriage environment.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    SPACE_URL      (Optional) URL of the LogTriage HF Space.
                   Defaults to http://localhost:7860
"""

import os
import re
import json
import time
import textwrap
from typing import List, Tuple, Dict, Any

import requests
from openai import OpenAI


# ─── Configuration ───────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
SPACE_URL = os.getenv("SPACE_URL", "http://localhost:8000")

TEMPERATURE = 0.2
MAX_TOKENS = 400
FALLBACK_ACTION = ("noop", {})


# ─── Environment Client ─────────────────────────────────────────

class LogTriageClient:
    """HTTP client wrapping the LogTriage FastAPI environment."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id = None

    def reset(self, task_id: str) -> dict:
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["session_id"]
        return data

    def step(self, action_type: str, params: dict = None) -> dict:
        resp = requests.post(
            f"{self.base_url}/step",
            json={
                "session_id": self.session_id,
                "action_type": action_type,
                "params": params or {},
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        resp = requests.get(
            f"{self.base_url}/state",
            params={"session_id": self.session_id},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()


# ─── System Prompt ───────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert SRE investigating system logs during an incident.
You interact with a log investigation environment via JSON actions.

Reply with ONLY a JSON object. No explanations, no markdown.

Format: {"action_type": "<type>", "params": {<params>}}

Available actions:

NAVIGATION:
  {"action_type": "search", "params": {"pattern": "error keyword"}}
  {"action_type": "filter_severity", "params": {"level": "ERROR"}}
  {"action_type": "filter_service", "params": {"service": "auth-service"}}
  {"action_type": "filter_time_range", "params": {"start": "2024-01-15T09:00:00Z", "end": "2024-01-15T10:00:00Z"}}
  {"action_type": "clear_filters", "params": {}}
  {"action_type": "scroll", "params": {"direction": "down"}}

INVESTIGATION:
  {"action_type": "inspect", "params": {"log_id": "log_042"}}
  {"action_type": "annotate", "params": {"log_id": "log_042", "category": "error"}}
  {"action_type": "correlate", "params": {"source_log_id": "log_012", "target_log_id": "log_042"}}

CONCLUSION:
  {"action_type": "classify_incident", "params": {"severity": "HIGH"}}
  {"action_type": "draft_report", "params": {"summary": "your analysis..."}}
  {"action_type": "submit_report", "params": {"summary": "final report..."}}

Categories for annotation:
  Infrastructure: error, root_cause, symptom, cascading_failure, warning
  Security: reconnaissance, brute_force, credential_compromise,
    privilege_escalation, lateral_movement, data_exfiltration, persistence

Investigation strategy:
  1. Explore: search and filter to find relevant logs
  2. Investigate: inspect suspicious entries, annotate anomalies
  3. Correlate: link cause → effect (source caused target)
  4. Conclude: classify severity and submit detailed report

Your report should describe:
  - Root cause of the incident
  - Affected services and impact
  - Timeline of events
  - Severity justification
""").strip()


# ─── Observation Formatting ──────────────────────────────────────

def format_observation(obs: dict, history: List[str]) -> str:
    """Format observation for LLM context window."""
    parts = []

    # Goal and progress
    parts.append(f"GOAL: {obs['goal']}")
    parts.append(f"Step: {obs['step_number']}/{obs['max_steps']}")
    parts.append(f"Last action: {obs['last_action_message']}")

    if obs.get("draft_feedback"):
        parts.append(f"DRAFT FEEDBACK: {obs['draft_feedback']}")

    # Dashboard
    parts.append(f"\nSEVERITY COUNTS: {json.dumps(obs['severity_counts'])}")
    parts.append(f"SERVICES: {', '.join(obs['available_services'])}")
    parts.append(f"TOTAL LOGS: {obs['total_log_count']}")
    parts.append(f"FILTERS: {json.dumps(obs['current_filters']) if obs['current_filters'] else 'none'}")
    parts.append(f"PAGE: {obs['current_page'] + 1}/{obs['total_pages']}")

    # Current logs
    parts.append(f"\nVISIBLE LOGS ({len(obs['visible_logs'])} entries):")
    for log in obs["visible_logs"]:
        meta_str = ""
        if log.get("metadata"):
            meta_items = [f"{k}={v}" for k, v in list(log["metadata"].items())[:3]]
            meta_str = f" [{', '.join(meta_items)}]"
        parts.append(
            f"  {log['id']} | {log['timestamp']} | {log['service']} | "
            f"{log['severity']} | {log['message'][:120]}{meta_str}"
        )

    # Inspected log
    if obs.get("inspected_log"):
        il = obs["inspected_log"]
        parts.append(f"\nINSPECTED LOG ({il['id']}):")
        parts.append(f"  Full message: {il['message']}")
        parts.append(f"  Metadata: {json.dumps(il.get('metadata', {}))}")

    # Agent's work so far
    parts.append(f"\nYOUR WORK SO FAR:")
    parts.append(f"  Annotations: {obs['annotations_count']} "
                 f"(categories: {json.dumps(obs['annotations_by_category'])})")
    if obs["recent_annotations"]:
        for ann in obs["recent_annotations"]:
            parts.append(f"    - {ann['log_id']}: {ann['category']}")
    parts.append(f"  Correlations: {obs['correlations_count']}")
    if obs["recent_correlations"]:
        for corr in obs["recent_correlations"]:
            parts.append(f"    - {corr[0]} → {corr[1]}")
    parts.append(f"  Severity classified: {obs.get('severity_classified', 'not yet')}")
    if obs.get("current_report_draft"):
        draft_preview = obs["current_report_draft"][:100]
        parts.append(f"  Draft report: {draft_preview}...")

    # History
    if history:
        parts.append(f"\nRECENT HISTORY:")
        for h in history[-5:]:
            parts.append(f"  {h}")

    return "\n".join(parts)


# ─── Action Parsing ──────────────────────────────────────────────

def parse_agent_action(response_text: str) -> Tuple[str, dict]:
    """Parse LLM response into (action_type, params)."""
    if not response_text:
        return FALLBACK_ACTION

    text = response_text.strip()

    # Try direct JSON parse
    try:
        action = json.loads(text)
        if "action_type" in action:
            return action["action_type"], action.get("params", {})
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting JSON from code block
    code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_match:
        try:
            action = json.loads(code_match.group(1))
            if "action_type" in action:
                return action["action_type"], action.get("params", {})
        except (json.JSONDecodeError, TypeError):
            pass

    # Try finding any JSON object with action_type
    json_match = re.search(r'\{[^{}]*"action_type"\s*:\s*"[^"]+?"[^{}]*\}', text)
    if json_match:
        try:
            action = json.loads(json_match.group(0))
            if "action_type" in action:
                return action["action_type"], action.get("params", {})
        except (json.JSONDecodeError, TypeError):
            pass

    # Try to find action_type keyword anywhere
    type_match = re.search(r'"action_type"\s*:\s*"(\w+)"', text)
    if type_match:
        action_type = type_match.group(1)
        # Try to extract params
        params = {}
        params_match = re.search(r'"params"\s*:\s*(\{[^{}]*\})', text)
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                pass
        return action_type, params

    return FALLBACK_ACTION


# ─── Main Agent Loop ─────────────────────────────────────────────

def run_task(
    env: LogTriageClient,
    task_id: str,
    llm_client: OpenAI,
    model_name: str,
) -> dict:
    """Run a single task and return the grader result."""

    print(f"\n  Resetting environment for {task_id}...")
    result = env.reset(task_id)
    obs = result["observation"]
    max_steps = obs["max_steps"]

    print(f"  Goal: {obs['goal'][:80]}...")
    print(f"  Max steps: {max_steps}")
    print(f"  Total logs: {obs['total_log_count']}")

    history: List[str] = []
    grader_result = None

    for step in range(1, max_steps + 1):
        # Format observation for LLM
        user_prompt = format_observation(obs, history)

        # Call LLM
        try:
            completion = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  Step {step}: LLM call failed ({exc}). Using noop.")
            response_text = ""

        # Parse action
        action_type, params = parse_agent_action(response_text)
        params_str = json.dumps(params) if params else "{}"
        print(f"  Step {step}: {action_type}({params_str})")

        # Execute step
        step_result = env.step(action_type, params)
        obs = step_result["observation"]
        reward = step_result["reward"]
        done = step_result["done"]

        # Record history
        reward_val = reward.get("value", 0) if isinstance(reward, dict) else 0
        history.append(
            f"Step {step}: {action_type} → reward={reward_val:+.3f} "
            f"| {obs.get('last_action_message', '')[:50]}"
        )

        print(f"         reward={reward_val:+.3f} | done={done}")

        if done:
            grader_result = step_result.get("info", {}).get("grader_result")
            break

    if grader_result is None:
        # Shouldn't happen, but handle gracefully
        grader_result = step_result.get("info", {}).get("grader_result", {
            "task_id": task_id,
            "final_score": 0.0,
            "components": {},
        })

    return grader_result


def wait_for_server(env: LogTriageClient, max_wait: int = 60):
    """Wait for the environment server to be ready."""
    print(f"Waiting for server at {env.base_url}...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            health = env.health()
            print(f"Server ready: {health}")
            return True
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"Server not ready after {max_wait}s")


def main():
    """Run all 3 tasks and report scores."""

    print("=" * 60)
    print("LogTriage — Baseline Inference")
    print("=" * 60)

    # Validate configuration
    if not MODEL_NAME:
        raise ValueError("MODEL_NAME environment variable is required.")
    if not API_KEY:
        raise ValueError("HF_TOKEN or API_KEY environment variable is required.")

    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Space URL: {SPACE_URL}")

    # Initialize clients
    env = LogTriageClient(SPACE_URL)
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Wait for server
    wait_for_server(env)

    # Run all tasks
    task_ids = ["task_easy", "task_medium", "task_hard"]
    results: Dict[str, Any] = {}

    for task_id in task_ids:
        print(f"\n{'=' * 60}")
        print(f"TASK: {task_id}")
        print(f"{'=' * 60}")

        try:
            grader_result = run_task(env, task_id, llm_client, MODEL_NAME)
            results[task_id] = grader_result

            score = grader_result.get("final_score", 0.0)
            print(f"\n  FINAL SCORE: {score:.4f}")

            components = grader_result.get("components", {})
            if components:
                print(f"  {'Component':<30} {'Score':>8} {'Weight':>8} {'Weighted':>8}")
                print(f"  {'-'*54}")
                for name, comp in components.items():
                    if isinstance(comp, dict):
                        print(f"  {name:<30} {comp.get('score', 0):>8.4f} "
                              f"{comp.get('weight', 0):>8.2%} "
                              f"{comp.get('weighted', 0):>8.4f}")
        except Exception as exc:
            print(f"\n  ERROR running {task_id}: {exc}")
            results[task_id] = {
                "task_id": task_id,
                "final_score": 0.0,
                "components": {},
                "error": str(exc),
            }

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    scores = []
    for task_id in task_ids:
        score = results[task_id].get("final_score", 0.0)
        scores.append(score)
        print(f"  {task_id:<20} {score:.4f}")

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':<20} {avg_score:.4f}")
    print(f"\nInference complete.")


if __name__ == "__main__":
    main()