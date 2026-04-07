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


# ─── Task-Specific System Prompts ────────────────────────────────
#
# Compact prompts (~300-400 tokens each) to avoid hallucination on
# small models (9B).  Each task gets ONLY its relevant categories,
# services, and a short strategy.

_PROMPT_CORE = textwrap.dedent("""\
You are an SRE investigating logs. Reply with ONLY a raw JSON object.
No markdown, no explanation. Format: {"action_type": "<type>", "params": {<params>}}

Actions:
  search(pattern) | filter_severity(level) | filter_service(service)
  clear_filters() | scroll(direction=up/down)
  inspect(log_id) | annotate(log_id, category) | correlate(source_log_id, target_log_id)
  classify_incident(severity=LOW/MEDIUM/HIGH/CRITICAL)
  draft_report(summary) | submit_report(summary)

Rules:
- Annotate EVERY suspicious log before submitting.
- Correlate related logs (source caused target).
- Classify severity before submitting.
- Report must mention log IDs, root cause, affected services, and timeline.

Rules of Engagement (CRITICAL — follow these EXACTLY):
1. NEVER annotate the same log_id twice. If you see a log_id in ALREADY_ANNOTATED, skip it.
2. NEVER correlate the same pair twice. If you see a pair in ALREADY_CORRELATED, skip it.
3. Before calling annotate, CHECK the ALREADY_ANNOTATED list. If the log_id is there, DO NOT annotate it — move to a NEW log.
4. After annotating a log, MOVE ON. Use search, scroll, or filter_service to find the NEXT unannotated log.
5. You must annotate at least 3 different logs before submitting your report.
6. If you have already annotated all visible suspicious logs, use scroll(down) or search to find more.
7. NEVER use scroll more than 2 times in a row. If you need more logs, use search with a specific term or filter_service.
8. CORRELATION DIRECTION IS STRICT: source_log_id MUST be the earlier/cause event; target_log_id MUST be the later/effect event. Timeline flows forward: cause→effect, earlier→later. NEVER reverse this.
9. If a filter or search reveals no new suspicious logs (STALE FILTER warning), immediately use clear_filters() or a different search term.

Completion Criteria (you MUST satisfy ALL of these before calling submit_report):
- You have annotated at least 3 UNIQUE suspicious logs from different timestamps.
- You have searched for DOWNSTREAM CONSEQUENCES of each finding. Do not stop at the first anomaly.
  If you find an error in service A, you MUST search service B and service C for follow-on effects.
- You have correlated at least one source→target pair linking a cause log to an effect log.
- You have classified the incident severity.
- Assume every anomaly is part of a chain until you have searched for evidence proving it ended.
  Do not submit until you have actively looked for what happened NEXT after each suspicious event.

Output ONLY valid JSON.""").strip()

TASK_PROMPTS: Dict[str, str] = {
    "task_easy": _PROMPT_CORE + "\n\n" + textwrap.dedent("""\
TASK: Find database connection errors in auth-service logs.
Categories: error, warning
Services: auth-service
Strategy: filter ERROR logs → annotate each error → classify severity → submit report.
Remember: annotate each UNIQUE log_id ONCE, then move on. If filter shows same results twice, use clear_filters() or search a different term.

Completion Criteria (satisfy ALL before submitting):
- Found and annotated ALL database connection error logs (search 'connection refused', 'port 5432', 'database').
- Verified whether errors are isolated or recurring (search by timestamp range or scroll).
- Classified severity as MEDIUM or HIGH based on error frequency.""").strip(),

    "task_medium": _PROMPT_CORE + "\n\n" + textwrap.dedent("""\
TASK: Trace a cascading failure across 3 services.
Categories: root_cause, symptom, cascading_failure
Services: payment-service, order-service, api-gateway
Strategy: check payment-service first → annotate root cause → check other services → correlate cause→effect → classify severity → submit report.
Correlation direction: root_cause log → symptom log → cascading_failure log (ALWAYS earlier cause → later effect).
Remember: annotate each UNIQUE log_id ONCE, then move to the NEXT service. Never re-annotate.

Completion Criteria (satisfy ALL before submitting):
- Found the ROOT CAUSE in payment-service. If you found a DB pool error, MUST search order-service for queue backup symptoms.
- Found SYMPTOMS in order-service. If you found queue backup, MUST search api-gateway for 503/timeout errors.
- Found CASCADING FAILURES in api-gateway. Do NOT stop at step 1 — the failure chain spans all 3 services.
- Correlated across services: payment→order and order→gateway.
- Assume the failure propagated to ALL 3 services until you have searched each one.""").strip(),

    "task_hard": _PROMPT_CORE + "\n\n" + textwrap.dedent("""\
TASK: Investigate a multi-stage security breach across 5 services.
Categories: reconnaissance, brute_force, credential_compromise, privilege_escalation, lateral_movement, data_exfiltration
Services: auth-service, api-gateway, user-service, file-service, audit-log
Attack chain direction: reconnaissance → brute_force → credential_compromise → privilege_escalation → lateral_movement → data_exfiltration.
Correlation direction: earlier attack stage log → later attack stage log. source=cause, target=effect.
Strategy: search for suspicious IPs and failed logins → annotate each attack stage → correlate the attack chain → classify CRITICAL → submit report.
Remember: annotate each UNIQUE log_id ONCE, then move to the NEXT attack stage. Never re-annotate.

Completion Criteria (satisfy ALL before submitting):
- Found brute force logs? MUST search auth-service for successful logins from the same IP (credential_compromise).
- Found compromised credentials? MUST search api-gateway or user-service for privilege escalation attempts.
- Found privilege escalation? MUST search file-service and audit-log for lateral movement and data exfiltration.
- Assume the attack SUCCEEDED at every stage until you find evidence that it stopped.
- Do NOT submit until you have searched for the FULL attack chain: recon → brute_force → compromise → escalation → lateral_movement → exfiltration.
- Each discovered stage requires a follow-up search for the NEXT stage.""").strip(),
}

# Fallback for unknown task IDs
SYSTEM_PROMPT = _PROMPT_CORE


# ─── Phase-Aware User Prompt (1B) ───────────────────────────────

def get_phase_hint(step: int, max_steps: int, obs: dict, task_id: str = "") -> str:
    """Short, directive phase hint — kept small for 9B models."""
    ratio = step / max_steps
    ann = obs.get("annotations_count", 0)
    corr = obs.get("correlations_count", 0)
    sev = obs.get("severity_classified")
    remaining = max_steps - step
    min_ann = MIN_ANNOTATIONS.get(task_id, 2)

    if ratio < 0.3:
        return f"EXPLORE: Find and annotate suspicious logs. ({remaining} steps left)"
    elif ratio < 0.65:
        if ann == 0:
            return f"URGENT: Annotate ERROR/WARN logs NOW. ({remaining} left)"
        if ann < min_ann:
            return f"Have {ann}/{min_ann} annotations. Annotate more. ({remaining} left)"
        if corr == 0:
            return f"Have {ann} annotations, 0 correlations. Correlate NOW. ({remaining} left)"
        return f"Good progress ({ann} ann, {corr} corr). Continue investigating. ({remaining} left)"
    else:
        if ann == 0:
            return f"CRITICAL: 0 annotations! Annotate visible ERROR logs NOW. ({remaining} left)"
        if not sev:
            return f"Classify severity NOW. ({remaining} left)"
        if corr == 0 and ann >= 2:
            return f"Correlate your {ann} annotations NOW, then submit. ({remaining} left)"
        return f"Submit your report NOW with log IDs and timeline. ({remaining} left)"


# ─── Compact Observation Formatting (1A) ─────────────────────────

# Category sets used to derive correlation direction hints
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
) -> str:
    """Compact observation for small models. Targets ~500 tokens.

    Includes ALREADY_ANNOTATED, ALREADY_CORRELATED, STALE FILTER warnings,
    and CORRELATE HIΝTs so the LLM avoids loops and correlates causally.
    """
    p = []
    task_id = obs.get('_task_id', '')

    # Step progress + phase hint (1 line)
    p.append(f"Step {obs['step_number']}/{obs['max_steps']}. {get_phase_hint(step, obs['max_steps'], obs, task_id)}")

    # Feedback from last action
    p.append(f"Last: {obs['last_action_message'][:80]}")
    if obs.get("draft_feedback"):
        p.append(f"Feedback: {obs['draft_feedback']}")

    # ─── Stale filter warning ───
    if stale_filter_flag:
        p.append(
            "STALE FILTER: Your last filter/search revealed no new suspicious logs. "
            "You MUST use clear_filters() or search() with a DIFFERENT term now."
        )

    # ─── State tracking: show what has already been done ───
    if annotated_ids:
        p.append(f"ALREADY_ANNOTATED (do NOT annotate these again): {', '.join(sorted(annotated_ids))}")
    if correlated_pairs:
        pairs_str = ', '.join(f"{s}\u2192{t}" for s, t in sorted(correlated_pairs))
        p.append(f"ALREADY_CORRELATED (do NOT correlate these again): {pairs_str}")

    # ─── Correlation direction hint (derived from what's been annotated) ───
    if annotated_ids:
        ann_data = obs.get("recent_annotations", [])
        # Build full map from env's recent_annotations
        cat_map = {a['log_id']: a['category'] for a in ann_data}
        causes  = [lid for lid, cat in cat_map.items() if cat in _CAUSE_CATEGORIES]
        effects = [lid for lid, cat in cat_map.items() if cat in _EFFECT_CATEGORIES]
        if causes and effects:
            p.append(
                f"CORRELATE HINT (source=cause→target=effect): "
                f"Causes={causes}, Effects={effects}. "
                f"Always put the cause log as source_log_id and the effect log as target_log_id."
            )

    # Dashboard — 1 line
    filters = json.dumps(obs['current_filters']) if obs['current_filters'] else 'none'
    p.append(f"Logs: {obs['total_log_count']} | Page {obs['current_page']+1}/{obs['total_pages']} | Filters: {filters}")

    # Visible logs — compact, max 12, marked [DONE] if already annotated
    visible = obs.get("visible_logs", [])[:12]
    if visible:
        p.append(f"LOGS ({len(visible)}):")
        for log in visible:
            already = " [DONE]" if (annotated_ids and log['id'] in annotated_ids) else ""
            p.append(f"  {log['id']}|{log['severity']}|{log['service']}|{log['message'][:65]}{already}")

    # Inspected log (only if present)
    if obs.get("inspected_log"):
        il = obs["inspected_log"]
        p.append(f"INSPECTED {il['id']}: {il['message'][:150]}")
        if il.get('metadata'):
            p.append(f"  Meta: {json.dumps(il['metadata'])}")

    # Work summary — 1-2 lines
    ann = obs['annotations_count']
    corr = obs['correlations_count']
    sev = obs.get('severity_classified') or '-'
    work = f"Work: {ann} annotations, {corr} correlations, severity={sev}"
    if obs.get("recent_annotations"):
        recent = ' '.join(f"{a['log_id']}:{a['category']}" for a in obs["recent_annotations"][-3:])
        work += f" | Recent: {recent}"
    p.append(work)

    # History — last 2 only
    if history:
        p.append(f"History: {' | '.join(history[-2:])}")

    # JSON reminder
    p.append('Reply with JSON only. Do NOT re-annotate [DONE] logs. source_log_id=cause, target_log_id=effect.')

    return "\n".join(p)


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

    # ─── [START] structured log ───
    _emit("START", {
        "task_id": task_id,
        "model_name": model_name,
        "max_steps": max_steps,
        "total_logs": obs["total_log_count"],
        "goal": obs["goal"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    print(f"  Goal: {obs['goal'][:80]}...")
    print(f"  Max steps: {max_steps}")
    print(f"  Total logs: {obs['total_log_count']}")

    history: List[str] = []
    grader_result = None
    step_result = None

    # ─── State tracking: prevent loops and spam ───
    annotated_ids: set = set()       # log_ids we have already annotated
    correlated_pairs: set = set()    # (source, target) pairs already correlated
    consecutive_scroll_count: int = 0   # scroll spam guard
    last_filter_log_count: int = -1     # stale filter detection
    stale_filter_flag: bool = False     # injected into observation when filter is stale

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
        cumulative = reward.get("cumulative", 0) if isinstance(reward, dict) else 0
        history.append(
            f"Step {step}: {ks_action_type} → reward={reward_val:+.3f} "
            f"| {obs.get('last_action_message', '')[:50]}"
        )
        print(f"         reward={reward_val:+.3f} | done={done}")

        # ─── [STEP] structured log ───
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
                user_prompt = format_observation(
                    obs, history, step,
                    annotated_ids=annotated_ids,
                    correlated_pairs=correlated_pairs,
                    stale_filter_flag=stale_filter_flag,
                )
                task_prompt = TASK_PROMPTS.get(task_id, SYSTEM_PROMPT)
                action_type, params = call_llm_with_retry(
                    llm_client, model_name, task_prompt, user_prompt
                )

                # Reset stale flag once we've handed it to the LLM this step
                stale_filter_flag = False

                # If LLM returned noop, use intelligent fallback (1E)
                if action_type == "noop":
                    action_type, params = get_intelligent_fallback(
                        step, max_steps, obs
                    )

                # ─── Scroll Spam Guard: max 2 consecutive scrolls ───
                if action_type == "scroll":
                    consecutive_scroll_count += 1
                    if consecutive_scroll_count > 2:
                        # Force tool rotation — search with a task-relevant term
                        _spam_searches = {
                            "task_easy":   "error",
                            "task_medium": "timeout",
                            "task_hard":   "failed login",
                        }
                        fallback_pattern = _spam_searches.get(task_id, "error")
                        print(
                            f"  Step {step}: [ANTI-SPAM] Blocked scroll #{consecutive_scroll_count} in a row. "
                            f"Forcing search('{fallback_pattern}')."
                        )
                        action_type = "search"
                        params = {"pattern": fallback_pattern}
                        consecutive_scroll_count = 0
                else:
                    consecutive_scroll_count = 0  # reset on any non-scroll action

                # ─── Deduplication Guard: intercept duplicate annotations/correlations ───
                if action_type == "annotate":
                    log_id = params.get("log_id", "")
                    if log_id in annotated_ids:
                        # BLOCKED: duplicate annotation. Redirect to a useful action.
                        print(f"  Step {step}: [DEDUP] Blocked duplicate annotate({log_id}). Finding new log.")
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
                            print(f"           \u2192 Redirected to annotate({target['id']}, {cat})")
                        else:
                            action_type = "scroll"
                            params = {"direction": "down"}
                            consecutive_scroll_count += 1
                            print(f"           \u2192 Redirected to scroll(down)")

                elif action_type == "correlate":
                    pair = (params.get("source_log_id", ""), params.get("target_log_id", ""))
                    if pair in correlated_pairs:
                        print(f"  Step {step}: [DEDUP] Blocked duplicate correlate{pair}. Scrolling.")
                        action_type = "scroll"
                        params = {"direction": "down"}
                        consecutive_scroll_count += 1

                # ─── Annotation Gate: block premature reports ───
                ann_count = obs.get("annotations_count", 0)
                ratio = step / max_steps
                if action_type in ("submit_report", "draft_report") and ann_count < min_ann and ratio < 0.85:
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
                        print(f"  Step {step}: [GATE] Blocked report (only {ann_count}/{min_ann} annotations). "
                              f"Forced annotate({target['id']}, {cat})")
                    else:
                        action_type = "scroll"
                        params = {"direction": "down"}
                        consecutive_scroll_count += 1
                        print(f"  Step {step}: [GATE] Blocked report, scrolling to find logs")
                else:
                    params_str = json.dumps(params) if params else "{}"
                    print(f"  Step {step}: {action_type}({params_str})")

            # Execute step
            step_result = env.step(action_type, params)
            obs = step_result["observation"]
            reward = step_result["reward"]
            done = step_result["done"]

            # ─── Update state tracking after successful action ───
            if action_type == "annotate" and obs.get("last_action_success", True):
                annotated_ids.add(params.get("log_id", ""))
            elif action_type == "correlate" and obs.get("last_action_success", True):
                correlated_pairs.add(
                    (params.get("source_log_id", ""), params.get("target_log_id", ""))
                )
            elif action_type in ("filter_severity", "filter_service",
                                 "filter_time_range", "search"):
                # ─── Stale filter detection ───
                new_count = obs.get("total_log_count", 0)
                if new_count == last_filter_log_count and new_count > 0:
                    stale_filter_flag = True
                    print(f"           [STALE-FILTER] Same log count ({new_count}) as before. "
                          f"Will warn agent next step.")
                else:
                    stale_filter_flag = False
                    last_filter_log_count = new_count

            # Record history
            reward_val = reward.get("value", 0) if isinstance(reward, dict) else 0
            cumulative = reward.get("cumulative", 0) if isinstance(reward, dict) else 0
            history.append(
                f"Step {step}: {action_type} → reward={reward_val:+.3f} "
                f"| {obs.get('last_action_message', '')[:50]}"
            )
            print(f"         reward={reward_val:+.3f} | done={done}")

            # ─── [STEP] structured log ───
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
            "final_score": 0.0,
            "components": {},
        })

    if grader_result is None:
        grader_result = {
            "task_id": task_id,
            "final_score": 0.0,
            "components": {},
        }

    # ─── [END] structured log ───
    _emit("END", {
        "task_id": task_id,
        "final_score": grader_result.get("final_score", 0.0) if grader_result else 0.0,
        "components": grader_result.get("components", {}) if grader_result else {},
        "steps_used": step,
        "max_steps": max_steps,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

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