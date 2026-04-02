"""
LogTriage Inference Script
==========================
Baseline agent that runs an LLM against the LogTriage environment.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    SPACE_URL      (Optional) URL of the LogTriage HF Space.
                   Defaults to http://localhost:8000
"""

import os
import re
import json
import time
import textwrap
from typing import List, Tuple, Dict, Any

import requests
from openai import OpenAI


# ─── .env Loader ─────────────────────────────────────────────────

try:
    with open('.env') as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                if _line.lower().startswith('export '):
                    _line = _line[7:]
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"\''))
except FileNotFoundError:
    pass


# ─── Configuration ───────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
SPACE_URL = os.getenv("SPACE_URL", "http://localhost:8000")

TEMPERATURE = 0.1          # Lower for more deterministic JSON output
MAX_TOKENS = 350
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


# ─── Task-Specific Kickstart Sequences (1F) ─────────────────────

KICKSTART_ACTIONS: Dict[str, List[Tuple[str, dict]]] = {
    "task_easy": [
        ("filter_severity", {"level": "ERROR"}),
    ],
    "task_medium": [
        ("filter_severity", {"level": "ERROR"}),
        ("search", {"pattern": "timeout"}),
    ],
    "task_hard": [
        ("search", {"pattern": "failed login"}),
        ("filter_severity", {"level": "WARN"}),
    ],
}

# Minimum annotations required before allowing report submission per task
MIN_ANNOTATIONS: Dict[str, int] = {
    "task_easy": 2,
    "task_medium": 3,
    "task_hard": 4,
}


# ─── System Prompt (1B + 1C) ────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert SRE investigating system logs during an incident.
You interact with a log investigation environment via JSON actions.

OUTPUT FORMAT: You MUST reply with ONLY a single raw JSON object.
No markdown, no code blocks, no explanations, no text before or after.

Format: {"action_type": "<type>", "params": {<params>}}

EXAMPLES:
{"action_type": "search", "params": {"pattern": "connection refused"}}
{"action_type": "filter_severity", "params": {"level": "ERROR"}}
{"action_type": "filter_service", "params": {"service": "payment-service"}}
{"action_type": "annotate", "params": {"log_id": "log_042", "category": "error"}}
{"action_type": "correlate", "params": {"source_log_id": "log_012", "target_log_id": "log_042"}}
{"action_type": "classify_incident", "params": {"severity": "HIGH"}}
{"action_type": "submit_report", "params": {"summary": "Root cause: ... Impact: ... Timeline: ..."}}

Available actions:

NAVIGATION:
  search(pattern) - Search logs by keyword. Use this FIRST to find relevant logs.
  filter_severity(level) - Filter by DEBUG/INFO/WARN/ERROR/FATAL.
  filter_service(service) - Filter by service name.
  clear_filters() - Remove all filters.
  scroll(direction) - Scroll "up" or "down" through log pages.

INVESTIGATION:
  inspect(log_id) - View full details of a specific log entry.
  annotate(log_id, category) - Mark a log with a category. YOU MUST USE THIS.
  correlate(source_log_id, target_log_id) - Link cause → effect. YOU MUST USE THIS.

CONCLUSION:
  classify_incident(severity) - Set severity: LOW/MEDIUM/HIGH/CRITICAL.
  draft_report(summary) - Save a draft report (does NOT end episode).
  submit_report(summary) - Submit final report (ENDS episode).

Categories for annotation:
  Infrastructure: error, root_cause, symptom, cascading_failure, warning
  Security: reconnaissance, brute_force, credential_compromise,
    privilege_escalation, lateral_movement, data_exfiltration, persistence

=== MANDATORY RULES (YOUR SCORE DEPENDS ON THESE) ===

1. You MUST annotate EVERY suspicious log you find using the "annotate" action.
   Each annotation needs a log_id and a category. If you skip annotations,
   your score will be ZERO on investigation components (60% of your grade).

2. You MUST correlate related events using the "correlate" action.
   Link the root cause log to each symptom log (source_log_id caused target_log_id).
   Without correlations, you lose 30% of your grade.

3. Do NOT submit a report until you have:
   - Annotated at least 2-3 suspicious logs
   - Created at least 1 correlation between related logs
   - Classified severity
   Submitting a report without annotations = very low score.

4. Do NOT stop after finding just ONE error. Keep searching for ALL related errors
   across ALL services. Use filter_service to check each service.

=== INVESTIGATION STRATEGY ===

  Step 1: Search and filter to find ERROR and WARN logs.
  Step 2: For EACH suspicious log you see, use "annotate" to mark it.
  Step 3: After annotating 2+ logs, use "correlate" to link related ones.
  Step 4: Check other services with filter_service for related issues.
  Step 5: Classify severity.
  Step 6: Submit a detailed report referencing your findings.

Your report MUST include:
  - Root cause of the incident (mention specific log IDs)
  - Affected services and impact
  - Timeline: use "first", "then", "subsequently", "leading to", "finally"
  - Severity justification (why you chose that level)

CRITICAL: Output ONLY valid JSON. No other text whatsoever.""").strip()


# ─── Phase-Aware User Prompt (1B) ───────────────────────────────

def get_phase_hint(step: int, max_steps: int, obs: dict, task_id: str = "") -> str:
    """Generate phase-specific guidance based on progress."""
    ratio = step / max_steps
    ann_count = obs.get("annotations_count", 0)
    corr_count = obs.get("correlations_count", 0)
    sev = obs.get("severity_classified")
    has_report = bool(obs.get("current_report_draft"))
    min_ann = MIN_ANNOTATIONS.get(task_id, 2)

    if ratio < 0.25:
        return (
            "PHASE: EXPLORATION. Search and filter to find suspicious logs. "
            "Look at ERROR and WARN entries. Once you see a suspicious log, "
            "use 'annotate' to mark it immediately."
        )
    elif ratio < 0.6:
        parts = ["PHASE: INVESTIGATION."]
        if ann_count == 0:
            parts.append(
                f"URGENT: You have ZERO annotations! You MUST annotate suspicious "
                f"logs NOW. Look at the visible logs and annotate any ERROR/WARN "
                f"entries. Use: {{\"action_type\": \"annotate\", \"params\": "
                f"{{\"log_id\": \"<id>\", \"category\": \"error\"}}}}")
        elif ann_count < min_ann:
            parts.append(
                f"You have {ann_count} annotations but need at least {min_ann}. "
                f"Keep annotating suspicious logs. Also search other services.")
        else:
            parts.append(f"Good: {ann_count} annotations. Now create correlations between related logs.")
        if corr_count == 0 and ann_count >= 2:
            parts.append(
                f"You have {ann_count} annotations but ZERO correlations. "
                f"Use 'correlate' to link related events NOW.")
        return " ".join(parts)
    else:
        parts = ["PHASE: CONCLUSION."]
        if ann_count == 0:
            parts.append(
                f"CRITICAL: You have ZERO annotations and are running out of steps! "
                f"Annotate the ERROR logs you can see RIGHT NOW before doing anything else.")
        elif ann_count < min_ann:
            parts.append(f"You have {ann_count}/{min_ann} minimum annotations. Annotate more if possible.")
        if corr_count == 0 and ann_count >= 2:
            parts.append(f"WARNING: {ann_count} annotations but 0 correlations. Correlate related logs NOW.")
        if not sev:
            parts.append("You have NOT classified severity — do it NOW.")
        if ann_count >= min_ann and corr_count > 0 and sev and not has_report:
            parts.append("You have annotations, correlations, and severity. Submit your report NOW.")
        elif not has_report and ann_count >= min_ann:
            parts.append("Submit your report soon — include log IDs and causal chain.")
        remaining = max_steps - step
        parts.append(f"({remaining} steps remaining)")
        return " ".join(parts)


# ─── Compact Observation Formatting (1A) ─────────────────────────

def format_observation(obs: dict, history: List[str], step: int) -> str:
    """Format observation compactly for small LLM context windows."""
    parts = []

    # Goal and progress
    parts.append(f"GOAL: {obs['goal']}")
    parts.append(f"Step: {obs['step_number']}/{obs['max_steps']}")

    # Phase hint (task_id extracted from goal heuristic or passed via obs)
    parts.append(get_phase_hint(step, obs['max_steps'], obs, obs.get('_task_id', '')))

    # Last action feedback
    parts.append(f"Last action: {obs['last_action_message']}")

    if obs.get("draft_feedback"):
        parts.append(f"DRAFT FEEDBACK: {obs['draft_feedback']}")

    # Dashboard (compact)
    parts.append(f"\nSEVERITY COUNTS: {json.dumps(obs['severity_counts'])}")
    parts.append(f"SERVICES: {', '.join(obs['available_services'])}")
    parts.append(f"TOTAL LOGS: {obs['total_log_count']}")
    if obs['current_filters']:
        parts.append(f"FILTERS: {json.dumps(obs['current_filters'])}")
    else:
        parts.append("FILTERS: none")
    parts.append(f"PAGE: {obs['current_page'] + 1}/{obs['total_pages']}")

    # Visible logs — compact format, max 15 entries, truncated to 80 chars
    visible = obs.get("visible_logs", [])[:15]
    parts.append(f"\nVISIBLE LOGS ({len(visible)} entries):")
    for log in visible:
        msg = log['message'][:80]
        parts.append(
            f"  {log['id']} | {log['severity']} | {log['service']} | {msg}"
        )

    # Inspected log (full, if present)
    if obs.get("inspected_log"):
        il = obs["inspected_log"]
        parts.append(f"\nINSPECTED LOG ({il['id']}):")
        parts.append(f"  Message: {il['message']}")
        meta = il.get('metadata', {})
        if meta:
            parts.append(f"  Metadata: {json.dumps(meta)}")

    # Agent's work summary (compact)
    parts.append(f"\nYOUR WORK:")
    parts.append(f"  Annotations: {obs['annotations_count']}")
    if obs.get("recent_annotations"):
        anns = [f"{a['log_id']}:{a['category']}" for a in obs["recent_annotations"]]
        parts.append(f"  Recent: {', '.join(anns)}")
    parts.append(f"  Correlations: {obs['correlations_count']}")
    if obs.get("recent_correlations"):
        corrs = [f"{c[0]}→{c[1]}" for c in obs["recent_correlations"]]
        parts.append(f"  Recent: {', '.join(corrs)}")
    parts.append(f"  Severity: {obs.get('severity_classified') or 'NOT SET'}")
    if obs.get("current_report_draft"):
        parts.append(f"  Report draft: saved ({len(obs['current_report_draft'])} chars)")

    # History — last 3 only
    if history:
        parts.append(f"\nRECENT HISTORY:")
        for h in history[-3:]:
            parts.append(f"  {h}")

    # Final enforcement
    parts.append(
        "\nRespond with ONLY a JSON object: "
        '{"action_type": "<type>", "params": {<params>}}'
    )

    return "\n".join(parts)


# ─── Action Parsing with Retry (1D) ─────────────────────────────

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
        params = {}
        params_match = re.search(r'"params"\s*:\s*(\{[^{}]*\})', text)
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                pass
        return action_type, params

    return FALLBACK_ACTION


# ─── Phase-Aware Fallback (1E) ───────────────────────────────────

def get_intelligent_fallback(
    step: int, max_steps: int, obs: dict
) -> Tuple[str, dict]:
    """Return a useful fallback action based on current phase."""
    ratio = step / max_steps
    ann_count = obs.get("annotations_count", 0)
    sev = obs.get("severity_classified")
    has_report = bool(obs.get("current_report_draft"))

    # Final steps: force report submission
    if ratio > 0.85 and not has_report:
        report = _build_auto_report(obs)
        return ("submit_report", {"summary": report})

    if ratio > 0.85 and not sev:
        return ("classify_incident", {"severity": "HIGH"})

    # Early steps: try to filter
    if ratio < 0.3:
        return ("filter_severity", {"level": "ERROR"})

    # Mid steps: scroll to see more logs
    if ratio < 0.7:
        return ("scroll", {"direction": "down"})

    # Late steps: classify if needed, then report
    if not sev:
        return ("classify_incident", {"severity": "HIGH"})

    if not has_report:
        report = _build_auto_report(obs)
        return ("submit_report", {"summary": report})

    return FALLBACK_ACTION


# ─── Auto-Report Generation (1G) ─────────────────────────────────

def _build_auto_report(obs: dict) -> str:
    """Build a structured report from the agent's current work."""
    parts = []
    parts.append("Incident Report")
    parts.append("")

    # Root cause section
    goal = obs.get("goal", "")
    parts.append(f"Summary: Investigation of system incident.")
    parts.append("")

    # Findings from annotations
    annotations = obs.get("recent_annotations", [])
    if annotations:
        parts.append("Root Cause and Findings:")
        for ann in annotations:
            parts.append(f"  - {ann['log_id']}: categorized as {ann['category']}")
        parts.append("")

    # Categories found
    by_cat = obs.get("annotations_by_category", {})
    if by_cat:
        cats = ", ".join(f"{k} ({v})" for k, v in by_cat.items())
        parts.append(f"Categories identified: {cats}")
        parts.append("")

    # Correlations
    correlations = obs.get("recent_correlations", [])
    if correlations:
        parts.append("Timeline and Causal Chain:")
        for i, corr in enumerate(correlations):
            if i == 0:
                parts.append(f"  First, {corr[0]} triggered the incident.")
            parts.append(f"  Then, {corr[0]} led to {corr[1]}.")
        parts.append("")

    # Services affected
    services = obs.get("available_services", [])
    if services:
        parts.append(f"Affected services: {', '.join(services)}")
        parts.append("")

    # Severity
    sev = obs.get("severity_classified")
    if sev:
        parts.append(f"Severity: {sev}")
        parts.append(f"Severity justification: Based on the scope of affected "
                     f"services and the nature of the identified issues, "
                     f"this incident is classified as {sev}.")
    else:
        parts.append("Severity: HIGH")
        parts.append("Severity justification: Multiple errors detected across "
                     "services indicating significant operational impact.")

    # Impact
    ann_count = obs.get("annotations_count", 0)
    parts.append("")
    parts.append(f"Impact: {ann_count} anomalous log entries identified. "
                 f"The incident affected service availability and may have "
                 f"caused user-facing degradation.")

    return "\n".join(parts)


# ─── LLM Call with Retry (1D) ────────────────────────────────────

def call_llm_with_retry(
    llm_client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[str, dict]:
    """Call LLM and parse action. Retry once on parse failure."""

    # First attempt
    try:
        completion = llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response_text = completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"    LLM call failed ({exc}). Using fallback.")
        return FALLBACK_ACTION

    # Parse first response
    action_type, params = parse_agent_action(response_text)
    if action_type != "noop" or response_text.strip().startswith("{"):
        return action_type, params

    # Retry with aggressive JSON enforcement prompt
    retry_prompt = (
        'Your previous response was not valid JSON. '
        'Return ONLY a raw JSON object like: '
        '{"action_type": "search", "params": {"pattern": "error"}} '
        'No other text. Just the JSON.'
    )
    try:
        completion = llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": retry_prompt},
            ],
            temperature=0.0,  # Maximum determinism for retry
            max_tokens=200,
        )
        retry_text = completion.choices[0].message.content or ""
        action_type2, params2 = parse_agent_action(retry_text)
        if action_type2 != "noop":
            return action_type2, params2
    except Exception:
        pass

    # Return whatever we got from the first attempt
    return action_type, params


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
    step_result = None

    # ─── Kickstart phase (1F): execute pre-planned actions ───
    kickstart = KICKSTART_ACTIONS.get(task_id, [])
    step = 0

    for ks_action_type, ks_params in kickstart:
        step += 1
        params_str = json.dumps(ks_params) if ks_params else "{}"
        print(f"  Step {step}: [KICKSTART] {ks_action_type}({params_str})")

        step_result = env.step(ks_action_type, ks_params)
        obs = step_result["observation"]
        reward = step_result["reward"]
        done = step_result["done"]

        reward_val = reward.get("value", 0) if isinstance(reward, dict) else 0
        history.append(
            f"Step {step}: {ks_action_type} → reward={reward_val:+.3f} "
            f"| {obs.get('last_action_message', '')[:50]}"
        )
        print(f"         reward={reward_val:+.3f} | done={done}")

        if done:
            grader_result = step_result.get("info", {}).get("grader_result")
            break

    # ─── LLM-driven phase ───
    min_ann = MIN_ANNOTATIONS.get(task_id, 2)

    if not (step_result and step_result.get("done", False)):
        for llm_step in range(step + 1, max_steps + 1):
            step = llm_step

            # Inject task_id into obs for phase hint
            obs['_task_id'] = task_id

            # ─── Auto-report guard (1G): force report in final 2 steps ───
            if step >= max_steps - 1:
                sev = obs.get("severity_classified")
                has_report = bool(obs.get("current_report_draft"))

                if not sev:
                    action_type, params = "classify_incident", {"severity": "HIGH"}
                    print(f"  Step {step}: [AUTO] classify_incident(HIGH)")
                elif not has_report:
                    report = _build_auto_report(obs)
                    action_type, params = "submit_report", {"summary": report}
                    print(f"  Step {step}: [AUTO] submit_report({len(report)} chars)")
                else:
                    action_type, params = "submit_report", {
                        "summary": obs.get("current_report_draft", "")
                    }
                    print(f"  Step {step}: [AUTO] submit_report(existing draft)")
            else:
                # Normal LLM-driven step
                user_prompt = format_observation(obs, history, step)
                action_type, params = call_llm_with_retry(
                    llm_client, model_name, SYSTEM_PROMPT, user_prompt
                )

                # If LLM returned noop, use intelligent fallback (1E)
                if action_type == "noop":
                    action_type, params = get_intelligent_fallback(
                        step, max_steps, obs
                    )

                # ─── Annotation Gate: block premature reports ───
                ann_count = obs.get("annotations_count", 0)
                ratio = step / max_steps
                if action_type in ("submit_report", "draft_report") and ann_count < min_ann and ratio < 0.85:
                    # Redirect: the model is trying to report without annotating
                    visible = obs.get("visible_logs", [])
                    error_logs = [l for l in visible if l.get("severity") in ("ERROR", "WARN", "FATAL")]
                    if error_logs:
                        target = error_logs[0]
                        cat = "error"
                        if target.get("severity") == "WARN":
                            cat = "warning"
                        action_type = "annotate"
                        params = {"log_id": target["id"], "category": cat}
                        print(f"  Step {step}: [GATE] Blocked report (only {ann_count}/{min_ann} annotations). "
                              f"Forced annotate({target['id']}, {cat})")
                    else:
                        # No error logs visible — scroll to find some
                        action_type = "scroll"
                        params = {"direction": "down"}
                        print(f"  Step {step}: [GATE] Blocked report, scrolling to find logs")
                else:
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

    if grader_result is None and step_result is not None:
        grader_result = step_result.get("info", {}).get("grader_result", {
            "task_id": task_id,
            "final_score": 0.0,
            "components": {},
        })

    if grader_result is None:
        grader_result = {
            "task_id": task_id,
            "final_score": 0.0,
            "components": {},
        }

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