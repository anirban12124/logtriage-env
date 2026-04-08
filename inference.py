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
import sys
import time
import textwrap
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

import requests
from openai import OpenAI


# ─── Structured Stdout Logging ───────────────────────────────────

def _emit(tag: str, payload: dict):
    """Emit a structured log line: [TAG] {json}

    The OpenEnv evaluation pipeline parses these markers from stdout.
    - [START]  — beginning of a task episode
    - [STEP]   — after each agent step
    - [END]    — task episode complete with final scores
    """
    line = json.dumps(payload, default=str)
    print(f"[{tag}] {line}", flush=True)


# ─── .env Loader ─────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATHS = [
    os.path.join(_SCRIPT_DIR, '.env'),   # same dir as inference.py
    os.path.join(os.getcwd(), '.env'),    # current working directory
]
for _env_path in _ENV_PATHS:
    try:
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    if _line.lower().startswith('export '):
                        _line = _line[7:]
                    _k, _v = _line.split('=', 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"\''))
        break  # stop after first .env found
    except FileNotFoundError:
        continue


# ─── Configuration ───────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B")
SPACE_URL = os.getenv("SPACE_URL", "https://anix12-logtriage-openenv.hf.space")

TEMPERATURE = 0.1          # Lower for more deterministic JSON output
MAX_TOKENS = 200           # JSON actions are typically 50-100 tokens
LLM_TIMEOUT = 30           # seconds — hard timeout per LLM API call
FALLBACK_ACTION = ("noop", {})

# ─── Time Budget ─────────────────────────────────────────────────

GLOBAL_TIMEOUT = 1080      # 18 minutes total (2 min buffer for setup/grading)
TASK_TIME_LIMITS = {       # per-task time budgets in seconds
    "task_easy": 180,      # 3 minutes
    "task_medium": 300,    # 5 minutes
    "task_hard": 600,      # 10 minutes
}

# ─── Agent Step Caps (independent of server max_steps) ───────────

AGENT_MAX_STEPS = {
    "task_easy": 15,
    "task_medium": 20,
    "task_hard": 25,
}


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


# ─── Task-Specific Kickstart Sequences ───────────────────────────

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


# ─── Task-Specific System Prompts ────────────────────────────────

_PROMPT_CORE = textwrap.dedent("""\
You are an SRE investigating logs. Reply with ONLY a raw JSON object.
No markdown, no explanation. Format: {"action_type": "<type>", "params": {<params>}}

Actions:
  search(pattern) | filter_severity(level) | filter_service(service)
  clear_filters() | scroll(direction=up/down)
  inspect(log_id) | annotate(log_id, category) | correlate(source_log_id, target_log_id)
  classify_incident(severity=LOW/MEDIUM/HIGH/CRITICAL)
  submit_report(summary)

Rules:
- Annotate EVERY suspicious log before submitting.
- Correlate related logs (source=cause, target=effect).
- Classify severity before submitting.
- source_log_id=earlier/cause, target_log_id=later/effect.
- NEVER annotate the same log_id twice.
- NEVER correlate the same pair twice.

Output ONLY valid JSON.""").strip()

TASK_PROMPTS: Dict[str, str] = {
    "task_easy": _PROMPT_CORE + "\n\n" + textwrap.dedent("""\
TASK: Find database connection errors in auth-service logs.
Categories: error, warning
Strategy: filter ERROR → annotate each error → classify MEDIUM → submit report.
Search terms: 'connection refused', 'port 5432', 'database'.""").strip(),

    "task_medium": _PROMPT_CORE + "\n\n" + textwrap.dedent("""\
TASK: Trace cascading failure across payment-service, order-service, api-gateway.
Categories: root_cause, symptom, cascading_failure
Strategy: check payment-service → annotate root cause → check order-service → check api-gateway → correlate cause→effect → classify HIGH → submit.
Correlation: root_cause→symptom→cascading_failure (earlier→later).""").strip(),

    "task_hard": _PROMPT_CORE + "\n\n" + textwrap.dedent("""\
TASK: Investigate security breach across auth-service, api-gateway, user-service, file-service, audit-log.
Categories: reconnaissance, brute_force, credential_compromise, privilege_escalation, lateral_movement, data_exfiltration
Attack chain: reconnaissance→brute_force→credential_compromise→privilege_escalation→lateral_movement→data_exfiltration.
Strategy: search failed logins → annotate each attack stage → correlate chain → classify CRITICAL → submit.""").strip(),
}

# Fallback for unknown task IDs
SYSTEM_PROMPT = _PROMPT_CORE


# ─── Phase-Aware User Prompt ─────────────────────────────────────

def get_phase_hint(step: int, max_steps: int, obs: dict, task_id: str = "", time_remaining: float = 999) -> str:
    """Short, directive phase hint."""
    ratio = step / max_steps
    ann = obs.get("annotations_count", 0)
    corr = obs.get("correlations_count", 0)
    sev = obs.get("severity_classified")
    remaining = max_steps - step
    min_ann = MIN_ANNOTATIONS.get(task_id, 2)

    # Time pressure override
    if time_remaining < 30:
        return f"TIME CRITICAL: Submit report NOW. ({remaining} steps left)"

    if ratio < 0.3:
        return f"EXPLORE: Find and annotate suspicious logs. ({remaining} steps left)"
    elif ratio < 0.6:
        if ann == 0:
            return f"URGENT: Annotate ERROR/WARN logs NOW. ({remaining} left)"
        if ann < min_ann:
            return f"Have {ann}/{min_ann} annotations. Annotate more. ({remaining} left)"
        if corr == 0:
            return f"Have {ann} annotations, 0 correlations. Correlate NOW. ({remaining} left)"
        return f"Good progress ({ann} ann, {corr} corr). Continue. ({remaining} left)"
    else:
        if ann == 0:
            return f"CRITICAL: 0 annotations! Annotate visible ERROR logs NOW. ({remaining} left)"
        if not sev:
            return f"Classify severity NOW. ({remaining} left)"
        if corr == 0 and ann >= 2:
            return f"Correlate your {ann} annotations NOW, then submit. ({remaining} left)"
        return f"Submit your report NOW with log IDs and timeline. ({remaining} left)"


# ─── Compact Observation Formatting ──────────────────────────────

_CAUSE_CATEGORIES = frozenset({
    "root_cause", "reconnaissance", "brute_force", "credential_compromise",
})
_EFFECT_CATEGORIES = frozenset({
    "symptom", "cascading_failure", "privilege_escalation",
    "lateral_movement", "data_exfiltration",
})


def format_observation(
    obs: dict,
    history: List[str],
    step: int,
    annotated_ids: set = None,
    correlated_pairs: set = None,
    stale_filter_flag: bool = False,
    time_remaining: float = 999,
) -> str:
    """Compact observation for small models. Targets ~400 tokens."""
    p = []
    task_id = obs.get('_task_id', '')

    # Step progress + phase hint (1 line)
    p.append(f"Step {obs['step_number']}/{obs['max_steps']}. {get_phase_hint(step, obs['max_steps'], obs, task_id, time_remaining)}")

    # Feedback from last action
    p.append(f"Last: {obs['last_action_message'][:60]}")

    # Stale filter warning
    if stale_filter_flag:
        p.append("STALE FILTER: Use clear_filters() or search() with a DIFFERENT term.")

    # State tracking
    if annotated_ids:
        p.append(f"ALREADY_ANNOTATED: {', '.join(sorted(annotated_ids))}")
    if correlated_pairs:
        pairs_str = ', '.join(f"{s}→{t}" for s, t in sorted(correlated_pairs))
        p.append(f"ALREADY_CORRELATED: {pairs_str}")

    # Correlation direction hint
    if annotated_ids:
        ann_data = obs.get("recent_annotations", [])
        cat_map = {a['log_id']: a['category'] for a in ann_data}
        causes  = [lid for lid, cat in cat_map.items() if cat in _CAUSE_CATEGORIES]
        effects = [lid for lid, cat in cat_map.items() if cat in _EFFECT_CATEGORIES]
        if causes and effects:
            p.append(f"CORRELATE HINT: Causes={causes}, Effects={effects}. source=cause, target=effect.")

    # Dashboard — 1 line
    filters = json.dumps(obs['current_filters']) if obs['current_filters'] else 'none'
    p.append(f"Logs: {obs['total_log_count']} | Page {obs['current_page']+1}/{obs['total_pages']} | Filters: {filters}")

    # Visible logs — compact, max 8, marked [DONE] if already annotated
    visible = obs.get("visible_logs", [])[:8]
    if visible:
        p.append(f"LOGS ({len(visible)}):")
        for log in visible:
            already = " [DONE]" if (annotated_ids and log['id'] in annotated_ids) else ""
            p.append(f"  {log['id']}|{log['severity']}|{log['service']}|{log['message'][:55]}{already}")

    # Inspected log (only if present)
    if obs.get("inspected_log"):
        il = obs["inspected_log"]
        p.append(f"INSPECTED {il['id']}: {il['message'][:120]}")

    # Work summary — 1 line
    ann = obs['annotations_count']
    corr = obs['correlations_count']
    sev = obs.get('severity_classified') or '-'
    p.append(f"Work: {ann} ann, {corr} corr, severity={sev}")

    # History — last 1 only
    if history:
        p.append(f"History: {history[-1]}")

    # JSON reminder
    p.append('Reply JSON only. Do NOT re-annotate [DONE] logs.')

    return "\n".join(p)


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
        params = {}
        params_match = re.search(r'"params"\s*:\s*(\{[^{}]*\})', text)
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                pass
        return action_type, params

    return FALLBACK_ACTION


# ─── Phase-Aware Fallback ────────────────────────────────────────

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


# ─── Auto-Report Generation ──────────────────────────────────────

def _build_auto_report(obs: dict) -> str:
    """Build a structured report from the agent's current work."""
    parts = []
    parts.append("Incident Report")
    parts.append("")

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


# ─── LLM Call (No Retry — use fallback on failure) ───────────────

def call_llm(
    llm_client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[str, dict]:
    """Call LLM and parse action. No retry — use fallback on parse failure."""

    try:
        completion = llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=LLM_TIMEOUT,
        )
        response_text = completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"    LLM call failed ({exc}). Using fallback.")
        return FALLBACK_ACTION

    action_type, params = parse_agent_action(response_text)
    return action_type, params


# ─── Main Agent Loop ─────────────────────────────────────────────

def run_task(
    env: LogTriageClient,
    task_id: str,
    llm_client: OpenAI,
    model_name: str,
    global_start: float,
) -> dict:
    """Run a single task and return the grader result."""

    task_start = time.time()
    task_time_limit = TASK_TIME_LIMITS.get(task_id, 300)
    agent_max = AGENT_MAX_STEPS.get(task_id, 15)

    print(f"\n  Resetting environment for {task_id}...")
    result = env.reset(task_id)
    obs = result["observation"]
    max_steps = obs["max_steps"]  # server's max (ceiling)

    # Agent uses the lesser of its own cap and the server's max
    effective_max = min(agent_max, max_steps)

    # ─── [START] structured log ───
    _emit("START", {
        "task_id": task_id,
        "model_name": model_name,
        "max_steps": effective_max,
        "total_logs": obs["total_log_count"],
        "goal": obs["goal"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    print(f"  Goal: {obs['goal'][:80]}...")
    print(f"  Max steps: {effective_max} (server: {max_steps})")
    print(f"  Total logs: {obs['total_log_count']}")
    print(f"  Time budget: {task_time_limit}s")

    history: List[str] = []
    grader_result = None
    step_result = None

    # State tracking
    annotated_ids: set = set()
    correlated_pairs: set = set()
    consecutive_scroll_count: int = 0
    last_filter_log_count: int = -1
    stale_filter_flag: bool = False

    # ─── Kickstart phase: execute pre-planned actions ───
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
        cumulative = reward.get("cumulative", 0) if isinstance(reward, dict) else 0
        history.append(
            f"Step {step}: {ks_action_type} → r={reward_val:+.3f}"
        )
        print(f"         reward={reward_val:+.3f} | done={done}")

        _emit("STEP", {
            "task_id": task_id,
            "step": step,
            "action_type": ks_action_type,
            "params": ks_params,
            "reward": reward_val,
            "cumulative_reward": cumulative,
            "done": done,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if done:
            grader_result = step_result.get("info", {}).get("grader_result")
            break

    # ─── LLM-driven phase ───
    min_ann = MIN_ANNOTATIONS.get(task_id, 2)

    if not (step_result and step_result.get("done", False)):
        for llm_step in range(step + 1, effective_max + 1):
            step = llm_step

            # ─── Time budget checks ───
            elapsed_task = time.time() - task_start
            elapsed_global = time.time() - global_start
            task_time_remaining = task_time_limit - elapsed_task
            global_time_remaining = GLOBAL_TIMEOUT - elapsed_global

            time_remaining = min(task_time_remaining, global_time_remaining)

            if time_remaining < 5:
                print(f"  Step {step}: [TIMEOUT] Time budget exhausted. Forcing submit.")
                sev = obs.get("severity_classified")
                if not sev:
                    env.step("classify_incident", {"severity": "HIGH"})
                report = _build_auto_report(obs)
                step_result = env.step("submit_report", {"summary": report})
                obs = step_result["observation"]
                reward = step_result["reward"]
                done = step_result["done"]
                grader_result = step_result.get("info", {}).get("grader_result")
                _emit("STEP", {
                    "task_id": task_id, "step": step,
                    "action_type": "submit_report", "params": {},
                    "reward": reward.get("value", 0) if isinstance(reward, dict) else 0,
                    "cumulative_reward": reward.get("cumulative", 0) if isinstance(reward, dict) else 0,
                    "done": done,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                break

            # Inject task_id into obs for phase hint
            obs['_task_id'] = task_id

            # ─── Auto-report guard: force report in final 3 steps ───
            if step >= effective_max - 2:
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

            # ─── Time pressure: force wrap-up if >70% of time used ───
            elif time_remaining < task_time_limit * 0.3:
                ann_count = obs.get("annotations_count", 0)
                sev = obs.get("severity_classified")
                if ann_count >= 2 and not sev:
                    action_type, params = "classify_incident", {"severity": "HIGH"}
                    print(f"  Step {step}: [TIME-PRESSURE] classify_incident(HIGH)")
                elif ann_count >= 2 and sev:
                    report = _build_auto_report(obs)
                    action_type, params = "submit_report", {"summary": report}
                    print(f"  Step {step}: [TIME-PRESSURE] submit_report")
                else:
                    # Still need annotations, use LLM but be quick
                    user_prompt = format_observation(
                        obs, history, step,
                        annotated_ids=annotated_ids,
                        correlated_pairs=correlated_pairs,
                        stale_filter_flag=stale_filter_flag,
                        time_remaining=time_remaining,
                    )
                    task_prompt = TASK_PROMPTS.get(task_id, SYSTEM_PROMPT)
                    action_type, params = call_llm(
                        llm_client, model_name, task_prompt, user_prompt
                    )
                    if action_type == "noop":
                        action_type, params = get_intelligent_fallback(step, effective_max, obs)
            else:
                # Normal LLM-driven step
                user_prompt = format_observation(
                    obs, history, step,
                    annotated_ids=annotated_ids,
                    correlated_pairs=correlated_pairs,
                    stale_filter_flag=stale_filter_flag,
                    time_remaining=time_remaining,
                )
                task_prompt = TASK_PROMPTS.get(task_id, SYSTEM_PROMPT)
                action_type, params = call_llm(
                    llm_client, model_name, task_prompt, user_prompt
                )

                # Reset stale flag
                stale_filter_flag = False

                # If LLM returned noop, use intelligent fallback
                if action_type == "noop":
                    action_type, params = get_intelligent_fallback(
                        step, effective_max, obs
                    )

                # ─── Scroll Spam Guard: max 2 consecutive scrolls ───
                if action_type == "scroll":
                    consecutive_scroll_count += 1
                    if consecutive_scroll_count > 2:
                        _spam_searches = {
                            "task_easy":   "error",
                            "task_medium": "timeout",
                            "task_hard":   "failed login",
                        }
                        fallback_pattern = _spam_searches.get(task_id, "error")
                        print(f"  Step {step}: [ANTI-SPAM] Forcing search('{fallback_pattern}').")
                        action_type = "search"
                        params = {"pattern": fallback_pattern}
                        consecutive_scroll_count = 0
                else:
                    consecutive_scroll_count = 0

                # ─── Deduplication Guard ───
                if action_type == "annotate":
                    log_id = params.get("log_id", "")
                    if log_id in annotated_ids:
                        print(f"  Step {step}: [DEDUP] Blocked duplicate annotate({log_id}).")
                        visible = obs.get("visible_logs", [])
                        unannotated_errors = [
                            l for l in visible
                            if l['id'] not in annotated_ids
                            and l.get('severity') in ('ERROR', 'WARN', 'FATAL')
                        ]
                        if unannotated_errors:
                            target = unannotated_errors[0]
                            cat = params.get("category", "error")
                            action_type = "annotate"
                            params = {"log_id": target["id"], "category": cat}
                            print(f"           → Redirected to annotate({target['id']})")
                        else:
                            action_type = "scroll"
                            params = {"direction": "down"}
                            consecutive_scroll_count += 1

                elif action_type == "correlate":
                    pair = (params.get("source_log_id", ""), params.get("target_log_id", ""))
                    if pair in correlated_pairs:
                        print(f"  Step {step}: [DEDUP] Blocked duplicate correlate{pair}.")
                        action_type = "scroll"
                        params = {"direction": "down"}
                        consecutive_scroll_count += 1

                # ─── Annotation Gate: block premature reports ───
                ann_count = obs.get("annotations_count", 0)
                ratio = step / effective_max
                if action_type in ("submit_report", "draft_report") and ann_count < min_ann and ratio < 0.80:
                    visible = obs.get("visible_logs", [])
                    error_logs = [
                        l for l in visible
                        if l.get("severity") in ("ERROR", "WARN", "FATAL")
                        and l['id'] not in annotated_ids
                    ]
                    if error_logs:
                        target = error_logs[0]
                        cat = "error"
                        if target.get("severity") == "WARN":
                            cat = "warning"
                        action_type = "annotate"
                        params = {"log_id": target["id"], "category": cat}
                        print(f"  Step {step}: [GATE] Blocked report ({ann_count}/{min_ann} ann). "
                              f"Forced annotate({target['id']})")
                    else:
                        action_type = "scroll"
                        params = {"direction": "down"}
                        consecutive_scroll_count += 1
                        print(f"  Step {step}: [GATE] Blocked report, scrolling")
                else:
                    params_str = json.dumps(params) if params else "{}"
                    print(f"  Step {step}: {action_type}({params_str})")

            # Execute step
            step_result = env.step(action_type, params)
            obs = step_result["observation"]
            reward = step_result["reward"]
            done = step_result["done"]

            # ─── Update state tracking ───
            if action_type == "annotate" and obs.get("last_action_success", True):
                annotated_ids.add(params.get("log_id", ""))
            elif action_type == "correlate" and obs.get("last_action_success", True):
                correlated_pairs.add(
                    (params.get("source_log_id", ""), params.get("target_log_id", ""))
                )
            elif action_type in ("filter_severity", "filter_service",
                                 "filter_time_range", "search"):
                new_count = obs.get("total_log_count", 0)
                if new_count == last_filter_log_count and new_count > 0:
                    stale_filter_flag = True
                else:
                    stale_filter_flag = False
                    last_filter_log_count = new_count

            # Record history
            reward_val = reward.get("value", 0) if isinstance(reward, dict) else 0
            cumulative = reward.get("cumulative", 0) if isinstance(reward, dict) else 0
            history.append(
                f"Step {step}: {action_type} → r={reward_val:+.3f}"
            )
            print(f"         reward={reward_val:+.3f} | done={done}")

            _emit("STEP", {
                "task_id": task_id,
                "step": step,
                "action_type": action_type,
                "params": params,
                "reward": reward_val,
                "cumulative_reward": cumulative,
                "done": done,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            if done:
                grader_result = step_result.get("info", {}).get("grader_result")
                break

    if grader_result is None and step_result is not None:
        grader_result = step_result.get("info", {}).get("grader_result", {
            "task_id": task_id,
            "final_score": 0.001,
            "components": {},
        })

    if grader_result is None:
        grader_result = {
            "task_id": task_id,
            "final_score": 0.001,
            "components": {},
        }

    elapsed = time.time() - task_start
    _emit("END", {
        "task_id": task_id,
        "final_score": grader_result.get("final_score", 0.001) if grader_result else 0.001,
        "components": grader_result.get("components", {}) if grader_result else {},
        "steps_used": step,
        "max_steps": effective_max,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    print(f"\n  Task completed in {elapsed:.1f}s ({step} steps)")
    return grader_result


def wait_for_server(env: LogTriageClient, max_wait: int = 120):
    """Wait for the environment server to be ready."""
    print(f"Waiting for server at {env.base_url}...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            health = env.health()
            print(f"Server ready: {health}")
            return True
        except Exception as exc:
            elapsed = int(time.time() - start)
            print(f"  Server not ready yet ({elapsed}s elapsed): {exc}")
            time.sleep(3)
    print(f"WARNING: Server not ready after {max_wait}s. Attempting to continue anyway...")
    return False


def main():
    """Run all 3 tasks and report scores."""

    print("=" * 60)
    print("LogTriage — Baseline Inference")
    print("=" * 60)

    # Validate configuration (warn but do not crash)
    if not MODEL_NAME:
        print("WARNING: MODEL_NAME not set. Defaulting to 'Qwen/Qwen3.5-9B'.")
    if not API_KEY:
        print("WARNING: HF_TOKEN / API_KEY not set. LLM calls may fail.")

    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Space URL: {SPACE_URL}")
    print(f"Global timeout: {GLOBAL_TIMEOUT}s")

    # Initialize clients
    env = LogTriageClient(SPACE_URL)
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Wait for server
    wait_for_server(env)

    global_start = time.time()

    # Run all tasks
    task_ids = ["task_easy", "task_medium", "task_hard"]
    results: Dict[str, Any] = {}

    for task_id in task_ids:
        # Check global time budget before starting next task
        elapsed_global = time.time() - global_start
        if elapsed_global > GLOBAL_TIMEOUT - 30:
            print(f"\n  GLOBAL TIMEOUT: Skipping {task_id} ({elapsed_global:.0f}s elapsed)")
            results[task_id] = {
                "task_id": task_id,
                "final_score": 0.001,
                "components": {},
                "error": "Skipped due to global timeout",
            }
            continue

        print(f"\n{'=' * 60}")
        print(f"TASK: {task_id}")
        print(f"{'=' * 60}")

        try:
            grader_result = run_task(env, task_id, llm_client, MODEL_NAME, global_start)
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
                "final_score": 0.001,
                "components": {},
                "error": str(exc),
            }

    # Summary
    total_elapsed = time.time() - global_start
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    scores = []
    for task_id in task_ids:
        score = results[task_id].get("final_score", 0.001)
        scores.append(score)
        print(f"  {task_id:<20} {score:.4f}")

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':<20} {avg_score:.4f}")
    print(f"\nInference complete in {total_elapsed:.1f}s.")


if __name__ == "__main__":
    main()