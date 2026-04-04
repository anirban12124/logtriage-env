# LogTriage — SRE Log Investigation Environment

An OpenEnv-compliant AI agent evaluation environment that simulates
real-world Site Reliability Engineer (SRE) log investigation during incidents.

## Overview

Every tech company has SREs who investigate system logs during outages
and security incidents. This environment simulates that process:
the agent reads logs, identifies anomalies, traces causal chains,
classifies severity, and produces an incident report.

The environment tests 7 cognitive dimensions:
- **Information Retrieval** — searching and filtering effectively
- **Pattern Recognition** — spotting anomalies in noisy data
- **Causal Reasoning** — tracing root cause → symptoms
- **Cross-referencing** — linking events across services
- **Prioritization** — focusing on what matters
- **Synthesis** — summarizing findings coherently
- **Classification** — assessing severity accurately

## Quick Start

```bash
# Build and run locally
docker build -t logtriage .
docker run -p 8000:8000 logtriage

# Or run directly (requires Python 3.12+)
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000

# In another terminal, run the baseline agent
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-token"
export SPACE_URL="http://localhost:8000"
python inference.py
```

## Environment Description

LogTriage presents three incident scenarios of increasing difficulty.
The agent interacts with a paginated log viewer via a JSON action API,
investigating logs across one or more cloud services.

### Scenarios

| Task | Difficulty | Logs | Services | Max Steps | Description |
|------|-----------|------|----------|-----------|-------------|
| `task_easy` | Easy | 50 | 1 (auth-service) | 15 | Database connection failures in a single service |
| `task_medium` | Medium | 200 | 3 (payment, order, gateway) | 25 | Cascading service failure across three services |
| `task_hard` | Hard | 500 | 5 (auth, gateway, user, file, audit) | 40 | Multi-stage security breach hidden in noisy logs |

## Action Space

The agent interacts via JSON actions: `{"action_type": "<type>", "params": {<params>}}`.

### Navigation Actions

| Action | Parameters | Description |
|--------|-----------|-------------|
| `search` | `pattern: str` | Search logs by keyword (BM25-ranked results) |
| `filter_severity` | `level: str` | Filter by `DEBUG`, `INFO`, `WARN`, `ERROR`, or `FATAL` |
| `filter_service` | `service: str` | Filter by service name |
| `filter_time_range` | `start: str, end: str` | Filter by timestamp range |
| `clear_filters` | — | Remove all active filters |
| `scroll` | `direction: str` | Scroll `"up"` or `"down"` through log pages |

### Investigation Actions

| Action | Parameters | Description |
|--------|-----------|-------------|
| `inspect` | `log_id: str` | View full details of a log entry (untruncated) |
| `annotate` | `log_id: str, category: str` | Mark a log with a category label |
| `correlate` | `source_log_id: str, target_log_id: str` | Link a cause → effect relationship |

### Conclusion Actions

| Action | Parameters | Description |
|--------|-----------|-------------|
| `classify_incident` | `severity: str` | Set severity: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` |
| `draft_report` | `summary: str` | Save a draft report (does NOT end episode) |
| `submit_report` | `summary: str` | Submit final report (ends episode, triggers grading) |
| `noop` | — | Do nothing (wastes a step) |

## Observation Space

Each step returns an observation object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Current task identifier |
| `goal` | `str` | Human-readable investigation goal |
| `step_number` | `int` | Current step (1-indexed) |
| `max_steps` | `int` | Maximum allowed steps |
| `visible_logs` | `List[LogEntry]` | Current page of logs (max 20, messages truncated to 200 chars) |
| `total_log_count` | `int` | Total logs matching current filters |
| `current_page` / `total_pages` | `int` | Pagination state |
| `severity_counts` | `Dict[str, int]` | Count of logs per severity level |
| `available_services` | `List[str]` | Services in this task |
| `current_filters` | `Dict` | Active filters |
| `annotations_count` | `int` | Number of annotations made |
| `recent_annotations` | `List` | Last 5 annotations |
| `annotations_by_category` | `Dict[str, int]` | Annotations grouped by category |
| `correlations_count` | `int` | Number of correlations made |
| `recent_correlations` | `List` | Last 3 correlations |
| `severity_classified` | `str?` | Incident severity if set |
| `current_report_draft` | `str?` | Saved draft report |
| `inspected_log` | `LogEntry?` | Full log details if inspected |
| `last_action_success` | `bool` | Whether last action succeeded |
| `last_action_message` | `str` | Feedback message |
| `draft_feedback` | `str?` | Feedback on draft report quality |

## Category Taxonomy

### Infrastructure Categories
| Category | Description |
|----------|-------------|
| `error` | A system error or failure in normal operation |
| `root_cause` | The original underlying cause that triggered the incident |
| `symptom` | A visible effect or consequence of an underlying problem |
| `cascading_failure` | A failure in one system that propagates to dependent systems |
| `warning` | An anomaly that hasn't caused failure yet |

### Security Categories
| Category | Description |
|----------|-------------|
| `reconnaissance` | Initial probing or scanning to gather information about target |
| `brute_force` | Repeated automated attempts to guess credentials |
| `credential_compromise` | Successful unauthorized access using stolen credentials |
| `privilege_escalation` | Gaining higher access permissions than authorized |
| `lateral_movement` | Moving from one compromised system to another within network |
| `data_exfiltration` | Unauthorized extraction or theft of data from the system |
| `persistence` | Establishing ongoing unauthorized access to maintain foothold |

## Reward Design

The environment provides **dense per-step rewards** (range: -0.15 to +0.40) to guide agent behavior:

- **Positive signal** for correct annotations, correlations, accurate severity classification, and informative reports
- **Negative signal** for wrong annotations, spam, repetitive actions, and premature report submission
- **Dynamic scaling** based on difficulty, timing, coverage progress, and investigation quality
- **Anti-exploit guards** prevent gaming through annotation spam, immediate submission, or repetitive actions
- **Milestone bonuses** for reaching coverage thresholds (25%, 50%, 75%, 100%)

## Grading System

At episode end, a **10-component grader** produces a final score in **[0.0, 1.0]**:

| Component | What it Measures |
|-----------|-----------------|
| Annotation Precision | Fraction of agent annotations that are correct |
| Annotation Recall | Fraction of ground-truth anomalies found |
| Annotation Quality | Semantic similarity of assigned categories to ground truth |
| Correlation Precision | Fraction of agent correlations that are correct |
| Correlation Recall | Fraction of ground-truth correlations found (with transitive credit) |
| Chain Reconstruction | How well the agent reconstructed the causal chain (path, coverage, direction) |
| Severity Classification | Distance between agent and ground-truth severity levels |
| Report Completeness | Semantic coverage of key findings (MiniLM sentence embeddings) |
| Report Coherence | Report quality: length adequacy, temporal ordering, causal language, structure, specificity |
| Investigation Efficiency | Quality-adjusted step efficiency |

Component weights vary by task difficulty (e.g., chain reconstruction is weighted more heavily on the hard task).

## Setup Instructions

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | The API endpoint for the LLM |
| `MODEL_NAME` | Yes | The model identifier to use for inference |
| `HF_TOKEN` | Yes | Your Hugging Face / API key |
| `SPACE_URL` | No | LogTriage server URL (default: `http://localhost:8000`) |

### Running Locally

```bash
# 1. Start the environment server
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000

# 2. Run inference in another terminal
python inference.py
```

### Running with Docker

```bash
docker build -t logtriage .
docker run -p 8000:8000 logtriage
```

### Running Against HF Space

```bash
export SPACE_URL="https://your-space-name.hf.space"
python inference.py
```

## API Reference

### `GET /health`
Returns server status. No authentication required.
```json
{"status": "ok", "version": "3.0", "sessions_active": 0}
```

### `GET /tasks`
Lists all available tasks.
```json
[
  {"id": "task_easy", "name": "Database Connection Failures",
   "difficulty": "easy", "max_steps": 15, "log_count": 50,
   "services": ["auth-service"]},
  ...
]
```

### `POST /reset`
Starts a new episode. Request: `{"task_id": "task_easy"}`

Returns: `{"session_id": "...", "observation": {...}, "info": {...}}`

### `POST /step`
Executes an action. Request:
```json
{
  "session_id": "...",
  "action_type": "search",
  "params": {"pattern": "connection"}
}
```
Returns: `{"observation": {...}, "reward": {...}, "done": false, "info": {...}}`

### `GET /state?session_id=...`
Returns current session state (annotations, correlations, severity, report, filters).

## Technical Details

### ML Models

| Model | Size | Purpose | Deterministic |
|-------|------|---------|---------------|
| all-MiniLM-L6-v2 | ~80MB | Semantic similarity, report evaluation | Yes (CPU only) |
| BM25 (rank-bm25) | ~2MB | Information retrieval, search ranking | Yes |

All models are **bundled in the Docker image** (no network required at runtime).
Embeddings are precomputed at build time for category descriptions and task findings.

### Resource Requirements

- **CPU**: 2 vCPU sufficient (all ML is CPU-only, ~10ms per step)
- **Memory**: ~250MB total (~90MB models + ~50MB per session + runtime overhead)
- **Startup**: ~3 seconds for model loading
- **Inference budget**: <20 minutes for all 3 tasks

### Determinism

- Fixed random seeds per task (SHA-256 hash of task ID)
- CPU-only ML inference (no GPU float variance)
- Precomputed embedding references
- All scores rounded to 4 decimal places

## Inference Script

The `inference.py` file is the baseline agent. It:
1. Uses the **OpenAI Client** for all LLM calls (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`)
2. Emits structured stdout logs: `[START]`, `[STEP]`, `[END]` (JSON format)
3. Runs all 3 tasks sequentially and reports scores
4. Includes phase-aware prompting, kickstart sequences, annotation gates, and auto-report fallbacks

## License

MIT