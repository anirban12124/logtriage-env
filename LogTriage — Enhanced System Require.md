LogTriage — Enhanced System Requirements Specification & Design Document
Version 3.0 FINAL
Table of Contents
text
1.  Executive Summary
2.  Stakeholder Requirements
3.  Environment Design Philosophy
4.  Task Design
5.  Synthetic Log Generation Design
6.  Enhanced Adaptive Reward System
7.  Grader System (10 Dimensions)
8.  Session Management & API Contract
9.  Error Handling & Edge Cases
10. ML Infrastructure, Determinism & Risk Management
11. Observation Size Control & Context Management
12. Architecture & Deployment Design
13. Inference Script Design
14. Documentation Requirements (README)
15. Score Calibration Strategy
16. Implementation Plan
17. Pre-Submission Validation Checklist
18. Decision Log

1. Executive Summary
1.1 What Are We Building?
An AI agent evaluation environment that simulates the real-world job of a Site Reliability Engineer (SRE) investigating system logs during an incident. The agent must read logs, find problems, connect related events, determine severity, and write an incident report.

The environment fully complies with the OpenEnv specification and is deployed as a containerized Hugging Face Space tagged with openenv.

1.2 Why This Domain?
Factor	Reasoning
Real job	Every tech company has SREs who do this daily during outages.
Under-served	No existing OpenEnv benchmark for log investigation.
Rich decision-making	Not just pattern matching — requires reasoning, prioritization, and synthesis.
Natural difficulty scaling	Single-service error → multi-service cascade → hidden security breach.
Clear ground truth	We plant known issues in synthetic logs → deterministic grading.
7 cognitive dimensions	Information retrieval, pattern recognition, causal reasoning, cross-referencing, prioritization, synthesis, classification.
1.3 Core User Journey
text
Agent receives a goal: "Investigate these logs and produce an incident report"
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
         EXPLORE             IDENTIFY          SYNTHESIZE
         ─────────           ──────────        ───────────
         Search logs         Annotate          Correlate events
         Filter by           errors &          Classify severity
         service/            anomalies         Draft report
         severity/                             Submit incident
         time range                            report
         Inspect entries
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ▼
                          SUBMIT REPORT
                    Agent graded on accuracy,
                    completeness, efficiency
1.4 Theoretical Maximum Scores
text
Perfect 1.0 is NOT expected or required.
Semantic similarity rarely hits 1.0 exactly, and
heuristic coherence scoring has natural ceilings.

  Easy:   ~0.97
  Medium: ~0.94
  Hard:   ~0.90




2. Stakeholder Requirements
2.1 Who Benefits?
Stakeholder	What They Get
AI researchers	A benchmark to test reasoning, information retrieval, and synthesis capabilities.
Agent developers	A structured environment with dense rewards to train and fine-tune agents.
Competition judges	A novel, well-scoped, real-world environment that meets all OpenEnv criteria.
SRE teams (future)	Foundation for evaluating AI-assisted incident response tools.
2.2 Success Criteria
text
1. An agent with zero domain knowledge can still score > 0 on easy task.
   (Environment gives enough signal to make progress.)

2. A frontier model (GPT-4 class) baseline scores:
     Easy:   0.60 – 0.85
     Medium: 0.30 – 0.55
     Hard:   0.10 – 0.35

3. A random agent should score near 0.
   (Graders genuinely measure skill, not luck.)

4. All competition requirements pass automated validation.
   (openenv validate, Dockerfile builds, HF Space responds,
    inference.py reproduces scores.)

5. Same agent + same task = same score. Always.
   (Full determinism across container restarts.)
3. Environment Design Philosophy



3.1 Observation Space
What the agent sees at any moment:

text
┌─────────────────────────────────────────────────────────┐
│                    OBSERVATION                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  TASK BRIEFING                                          │
│    task_id: "task_hard"                                 │
│    goal: "Investigate a potential security breach..."    │
│                                                         │
│  CURRENT LOG PAGE (truncated for context efficiency)    │
│    visible_logs: List[LogEntry]  (up to 20 entries)     │
│    Each entry:                                          │
│      id, timestamp, service, severity,                  │
│      message (truncated to 200 chars),                  │
│      metadata: {ip, user_id, request_id, ...}           │
│                                                         │
│  INSPECTED LOG (if inspect() was last action)           │
│    inspected_log: full untruncated LogEntry or None     │
│                                                         │
│  DASHBOARD SUMMARY                                      │
│    total_log_count: 500                                 │
│    severity_counts: {"ERROR": 12, "WARN": 45, ...}     │
│    available_services: ["auth-service", ...]            │
│    current_page: 3                                      │
│    total_pages: 25                                      │
│                                                         │
│  AGENT'S WORK SUMMARY (compressed, not full dump)       │
│    annotations_count: 5                                 │
│    recent_annotations: [last 5 entries]                  │
│    annotations_by_category: {"error": 2, ...}           │
│    correlations_count: 3                                │
│    recent_correlations: [last 3 pairs]                  │
│    current_report_draft: str or None                    │
│    severity_classified: str or None                     │
│                                                         │
│  SYSTEM STATE                                           │
│    step_number: 12                                      │
│    max_steps: 40                                        │
│    current_filters: {"severity": "ERROR", ...}          │
│    last_action_success: True                            │
│    last_action_message: "OK"                            │
│                                                         │
│  FEEDBACK SIGNALS                                       │
│    precision_warning: str or None                       │
│      e.g., "Low annotation accuracy detected."          │
│    draft_feedback: str or None                          │
│      e.g., "Report covers 2/6 key areas."              │
│                                                         │
└─────────────────────────────────────────────────────────┘

TARGET OBSERVATION SIZE: < 4,000 characters (~1,000 tokens)
3.2 Action Space — Complete Definition
text
┌───────────────────────────────────────────────────────────────────────┐
│                         ACTION SPACE                                  │
├───────────┬──────────────────────────────┬────────────────────────────┤
│ CATEGORY  │ ACTION                       │ PARAMS & BEHAVIOR          │
├───────────┼──────────────────────────────┼────────────────────────────┤
│           │ search(pattern)              │ pattern: str               │
│           │                              │ Keyword search across all  │
│           │                              │ log messages. Updates view │
│           │                              │ to matching logs.          │
│           │                              │ Empty string = show all.   │
│           ├──────────────────────────────┼────────────────────────────┤
│           │ filter_severity(level)       │ level: str                 │
│           │                              │ Valid: DEBUG/INFO/WARN/    │
│           │                              │ ERROR/FATAL                │
│ NAVIGATION│                              │ Filters stack (AND logic). │
│           ├──────────────────────────────┼────────────────────────────┤
│           │ filter_service(name)         │ service: str               │
│           │                              │ Must be in available list. │
│           │                              │ Filters stack (AND logic). │
│           ├──────────────────────────────┼────────────────────────────┤
│           │ filter_time_range(start,end) │ start: str, end: str       │
│           │                              │ ISO format timestamps.     │
│           │                              │ Filters stack (AND logic). │
│           ├──────────────────────────────┼────────────────────────────┤
│           │ clear_filters()              │ No params.                 │
│           │                              │ Removes ALL active filters.│
│           ├──────────────────────────────┼────────────────────────────┤
│           │ clear_filter(filter_type)    │ filter_type: str           │
│           │                              │ Valid: severity/service/   │
│           │                              │ time_range/search          │
│           │                              │ Removes one filter only.   │
│           ├──────────────────────────────┼────────────────────────────┤
│           │ scroll(direction)            │ direction: "up" or "down"  │
│           │                              │ Moves log page view.       │
│           │                              │ Clamps at boundaries.      │
├───────────┼──────────────────────────────┼────────────────────────────┤
│           │ inspect(log_id)              │ log_id: str                │
│ INSPECT   │                              │ Returns FULL untruncated   │
│           │                              │ log entry with all         │
│           │                              │ metadata and stack traces. │
├───────────┼──────────────────────────────┼────────────────────────────┤
│           │ annotate(log_id, category)   │ log_id: str, category: str │
│           │                              │ Categories from predefined │
│           │                              │ taxonomy (Section 3.3).    │
│INVESTIGATE│                              │ Unknown categories         │
│           │                              │ accepted but score low.    │
│           ├──────────────────────────────┼────────────────────────────┤
│           │ correlate(source, target)    │ source_log_id: str,        │
│           │                              │ target_log_id: str         │
│           │                              │ DIRECTED: source CAUSED    │
│           │                              │ or preceded target.        │
│           │                              │ Self-correlation rejected. │
├───────────┼──────────────────────────────┼────────────────────────────┤
│           │ classify_incident(severity)  │ severity: str              │
│           │                              │ Valid: LOW/MEDIUM/HIGH/    │
│           │                              │ CRITICAL                   │
│           ├──────────────────────────────┼────────────────────────────┤
│           │ draft_report(summary)        │ summary: str               │
│ CONCLUDE  │                              │ Saves report. Does NOT end │
│           │                              │ episode. Returns feedback  │
│           │                              │ on coverage. Overwrites    │
│           │                              │ previous draft.            │
│           ├──────────────────────────────┼────────────────────────────┤
│           │ submit_report(summary)       │ summary: str               │
│           │                              │ Ends episode. Final grade. │
│           │                              │ REQUIRES min 3 steps.      │
│           │                              │ Before step 3 auto-        │
│           │                              │ converts to draft_report.  │
├───────────┼──────────────────────────────┼────────────────────────────┤
│ OTHER     │ noop()                       │ No params. Does nothing.   │
│           │                              │ Step still increments.     │
└───────────┴──────────────────────────────┴────────────────────────────┘

TOTAL ACTIONS: 15 distinct action types
3.3 Category Taxonomy — Complete Definition
text
INFRASTRUCTURE CATEGORIES (used in Easy + Medium tasks)
───────────────────────────────────────────────────────
ID                   Description (used for ML embedding)
──────────────────────────────────────────────────────────────
error                "A system error or failure in normal operation"
root_cause           "The original underlying cause that triggered
                      the incident"
symptom              "A visible effect or consequence of an
                      underlying problem"
cascading_failure    "A failure in one system that propagates to
                      dependent systems"
warning              "An anomaly that hasn't caused failure yet but
                      indicates potential problems"

SECURITY CATEGORIES (used in Hard task)
───────────────────────────────────────
ID                     Description (used for ML embedding)
──────────────────────────────────────────────────────────────
reconnaissance         "Initial probing or scanning to gather
                        information about the target system"
brute_force            "Repeated automated attempts to guess
                        credentials or bypass authentication"
credential_compromise  "Successful unauthorized access using stolen
                        or guessed credentials"
privilege_escalation   "Gaining higher-level access permissions than
                        originally authorized"
lateral_movement       "Moving from one compromised system to another
                        within the network"
data_exfiltration      "Unauthorized extraction or theft of data from
                        the system"
persistence            "Establishing ongoing unauthorized access to
                        maintain a foothold in the system"

TOTAL: 12 predefined categories.

PRECOMPUTED SIMILARITY MATRIX (at build time):

              root  symp  casc  error warn  reco  brut  cred  priv  exfi  late  pers
root_cause    1.00  0.68  0.55  0.52  0.45  0.20  0.22  0.25  0.28  0.15  0.18  0.22
symptom       0.68  1.00  0.72  0.65  0.58  0.18  0.20  0.22  0.20  0.15  0.16  0.18
cascading     0.55  0.72  1.00  0.58  0.50  0.22  0.18  0.20  0.25  0.18  0.30  0.20
error         0.52  0.65  0.58  1.00  0.70  0.15  0.18  0.20  0.18  0.12  0.15  0.16
warning       0.45  0.58  0.50  0.70  1.00  0.20  0.22  0.18  0.15  0.10  0.12  0.15
recon         0.20  0.18  0.22  0.15  0.20  1.00  0.62  0.55  0.45  0.40  0.50  0.52
brute_force   0.22  0.20  0.18  0.18  0.22  0.62  1.00  0.72  0.50  0.30  0.35  0.40
credential    0.25  0.22  0.20  0.20  0.18  0.55  0.72  1.00  0.65  0.45  0.48  0.55
privilege     0.28  0.20  0.25  0.18  0.15  0.45  0.50  0.65  1.00  0.52  0.55  0.48
exfiltration  0.15  0.15  0.18  0.12  0.10  0.40  0.30  0.45  0.52  1.00  0.58  0.42
lateral       0.18  0.16  0.30  0.15  0.12  0.50  0.35  0.48  0.55  0.58  1.00  0.50
persistence   0.22  0.18  0.20  0.16  0.15  0.52  0.40  0.55  0.48  0.42  0.50  1.00

STORED AS CONSTANT NUMPY ARRAY — no runtime computation needed.

UNKNOWN CATEGORY HANDLING:
  If agent submits a category not in the 12 predefined:
    → Action succeeds (no crash)
    → The category description text is embedded using MiniLM
    → Compared against all 12 known descriptions
    → Best cosine similarity match determines quality score
    → Naturally penalized if dissimilar to correct answer
    → Self-correcting via the embedding distance system
3.4 Key Design Decisions
Decision	Choice	Rationale
Log pagination	20 entries per page	Forces strategic navigation.
Log truncation	200 chars default	Keeps observation under 4,000 chars. Agent uses inspect() for full entries.
No raw bulk access	API only	Simulates real investigation. Prevents context-window abuse.
Predefined categories	12 fixed + unknown accepted	Deterministic grading with ML flexibility for creative answers.
Directed correlations	Agent specifies cause → effect	Tests deeper causal reasoning.
Two-stage reports	draft_report + submit_report	Avoids accidental episode termination. Gives feedback loop.
Submit minimum	Step 3 or later	Prevents accidental instant submission.
Filter stacking	AND logic with clear_filters()	Matches real log tools.
Step budget	Varies by difficulty (15/25/40)	Creates time pressure.
Observation compression	Summarized annotations and correlations	Prevents unbounded observation growth.


4. Task Design
4.1 Task 1: Database Connection Failures (EASY)
text
SCENARIO
────────
The auth-service is experiencing intermittent database connection
failures. Logs contain routine INFO messages mixed with connection
errors to the primary database.

DATA (50 entries, 1 service)
────────────────────────────
  ~40 normal INFO/DEBUG (logins, token refreshes, health checks)
  ~7  WARN entries (slow queries, high connection pool usage)
  ~3  ERROR entries (java.sql.SQLException: Connection refused)
  ~0  distractors

  Signal ratio: 3/50 = 6.0%

GROUND TRUTH
────────────
  annotations:
    log_008: "error"     — Connection refused to db-primary:5432
    log_023: "error"     — Connection refused to db-primary:5432
    log_041: "error"     — Connection refused to db-primary:5432

  correlations: []       — None needed

  severity: "MEDIUM"

  key_findings:
    "database connection refused"
    "auth-service"
    "port 5432"

  key_finding_descriptions (for semantic matching):
    "The auth-service experienced database connection failures
     with errors connecting to the primary database server"
    "Connection refused errors occurred on PostgreSQL port 5432"
    "The auth-service was the affected component experiencing
     intermittent connectivity issues"

AGENT OBJECTIVE
───────────────
  1. Find the 3 ERROR entries about database connections
  2. Annotate each as "error"
  3. Classify severity as MEDIUM
  4. Submit report mentioning database connection failures

WHY IT'S EASY
─────────────
  • Single service
  • Obvious ERROR-level messages
  • Small dataset (50 logs, 3 pages)
  • No correlations needed
  • No distractors

EXPECTED BASELINE SCORES
────────────────────────
  Frontier model: 0.70 – 0.90
  Basic agent:    0.40 – 0.65
  Random agent:   0.00 – 0.05

MAX STEPS: 15
4.2 Task 2: Cascading Service Failure (MEDIUM)
text
SCENARIO
────────
A payment processing timeout triggered a chain reaction:
payment-service database connection pool exhaustion →
order-service queue backup → api-gateway 503 errors.

DATA (200 entries, 3 services)
──────────────────────────────
  Services: api-gateway, order-service, payment-service
  ~150 normal operational entries across all services
  ~30  WARN entries (elevated latency, retries)
  ~15  ERROR/FATAL entries (timeouts, failures)
  ~5   distractors (DNS timeout, disk warning, TLS cert expiry)

  Of the 15 ERROR/FATAL, only 6 are ground truth.
  Signal ratio: 6/200 = 3.0%

GROUND TRUTH
────────────
  annotations:
    log_045: "root_cause"        — payment-service db pool exhausted
    log_067: "symptom"           — payment-service transaction timeout
    log_089: "symptom"           — payment-service request queue full
    log_102: "symptom"           — order-service timeout waiting for payment
    log_134: "cascading_failure" — order-service message queue backup
    log_156: "cascading_failure" — api-gateway 503 Service Unavailable

  correlations (directed: cause → effect):
    ["log_045", "log_067"]  — db pool → transaction timeout
    ["log_067", "log_134"]  — payment timeout → order queue backup
    ["log_089", "log_156"]  — payment queue full → gateway 503

  severity: "HIGH"

  key_findings:
    "payment-service timeout"
    "order-service queue backup"
    "api-gateway 503 errors"
    "database connection pool exhausted"

  key_finding_descriptions:
    "Payment service experienced timeouts due to database failures"
    "Order service message queue backed up as payment processing stalled"
    "API gateway returned 503 errors to end users"
    "Root cause was database connection pool exhaustion in payment service"

AGENT OBJECTIVE
───────────────
  1. Find 6 annotatable entries across 3 services
  2. Correctly categorize root_cause vs symptom vs cascading_failure
  3. Identify 3 directed correlation pairs (cause → effect)
  4. Classify severity as HIGH
  5. Submit report covering all 4 key findings

WHY IT'S MEDIUM
───────────────
  • Multiple services requiring cross-referencing
  • Must reason about causality chains
  • Larger dataset (200 logs, 10 pages)
  • 5 distractor errors
  • Must distinguish root cause from symptoms

EXPECTED BASELINE SCORES
────────────────────────
  Frontier model: 0.40 – 0.60
  Basic agent:    0.15 – 0.35
  Random agent:   0.00 – 0.05

MAX STEPS: 25
4.3 Task 3: Security Breach Investigation (HARD)
text
SCENARIO
────────
An attacker performed a multi-stage breach:
reconnaissance → brute force → credential compromise →
privilege escalation → lateral movement → data exfiltration.

DATA (500 entries, 5 services)
──────────────────────────────
  Services: auth-service, api-gateway, user-service,
            file-service, audit-log
  ~400 normal entries (routine ops, health checks, traffic)
  ~50  WARN entries with subtle anomalies
  ~42  assorted ERROR entries
  ~8   actual attack chain entries (ground truth)
  ~15  distractor errors (genuine but unrelated: DNS timeout,
       TLS cert warning, disk space, GC pause, connection resets)

  Signal ratio: 8/500 = 1.6%

GROUND TRUTH
────────────
  annotations:
    log_023: "reconnaissance"        — Port scanning from external IP
    log_078: "brute_force"           — 47 failed logins for 'admin'
    log_091: "brute_force"           — 23 failed logins for 'svc-account'
    log_145: "credential_compromise" — Successful login after 91 failures
    log_201: "privilege_escalation"  — Non-admin accessed admin API
    log_267: "data_exfiltration"     — Bulk download of user records
    log_312: "lateral_movement"      — Auth token used on file-service
    log_389: "data_exfiltration"     — Sensitive files exported via API

  correlations (directed: cause → effect):
    ["log_023", "log_078"]  — recon → brute force
    ["log_091", "log_145"]  — brute force → credential compromise
    ["log_145", "log_201"]  — compromise → privilege escalation
    ["log_201", "log_267"]  — escalation → data exfiltration
    ["log_201", "log_312"]  — escalation → lateral movement
    ["log_312", "log_389"]  — lateral movement → exfiltration

  ATTACK CHAIN (DAG):
    log_023 → log_078
    log_091 → log_145 → log_201 → log_267
                               → log_312 → log_389

  severity: "CRITICAL"

  key_findings:
    "brute force login attempts"
    "privilege escalation"
    "data exfiltration"
    "lateral movement"
    "compromised credentials"
    "admin API access"

  key_finding_descriptions:
    "Automated brute force attacks targeted admin and service
     account credentials from an external IP address"
    "The attacker escalated privileges by accessing admin API
     endpoints with a compromised non-admin account"
    "Sensitive data including customer records was exfiltrated
     through bulk download operations"
    "The attacker moved laterally from auth-service to
     file-service using a compromised authentication token"
    "Credentials for a service account were compromised after
     sustained brute force attempts"
    "Administrative API endpoints were accessed by an account
     without admin privileges indicating authorization bypass"

AGENT OBJECTIVE
───────────────
  1. Find 8 attack-related entries among 500 logs
  2. Categorize each attack stage correctly
  3. Build the correlation chain (6 directed pairs)
  4. Classify severity as CRITICAL
  5. Submit detailed security incident report covering 6 key findings
  6. NOT annotate distractors as attack indicators

WHY IT'S HARD
─────────────
  • Large dataset (500 logs, 25 pages)
  • 5 services to investigate
  • Subtle signals buried in noise (1.6% signal ratio)
  • 15 distractor errors that look like real issues
  • False positives are penalized
  • Must reconstruct temporal attack chain DAG
  • Requires security domain reasoning
  • Step budget is tight relative to search space

EXPECTED BASELINE SCORES
────────────────────────
  Frontier model: 0.25 – 0.45
  Basic agent:    0.05 – 0.15
  Random agent:   0.00 – 0.02

MAX STEPS: 40


5. Synthetic Log Generation Design
5.1 Why Synthetic?
text
Real logs:
  ✗ Contain PII and sensitive data
  ✗ Require licensing and data agreements
  ✗ Cannot control ground truth precisely
  ✗ May not have clean incident boundaries

Synthetic logs:
  ✓ We control exactly what is planted
  ✓ Ground truth is 100% known → deterministic grading
  ✓ No privacy concerns
  ✓ Reproducible (seeded random generation)
  ✓ Difficulty is precisely tunable
5.2 Log Entry Schema
text
INFRASTRUCTURE LOG (Easy/Medium tasks):
┌──────────────────────────────────────────────────────────┐
│ 2024-01-15T09:23:41.127Z [http-nio-8080-exec-3]         │
│ ERROR c.a.auth.DatabaseConnectionPool                     │
│ - Failed to acquire connection to db-primary:5432         │
│   java.sql.SQLException: Connection refused               │
│     at org.postgresql.Driver.connect(Driver.java:282)     │
│   Retry attempt 3/3 failed. Circuit breaker OPEN.         │
│   metadata: {                                             │
│     "request_id": "req-a3f8c2",                          │
│     "service": "auth-service",                            │
│     "host": "prod-auth-02",                               │
│     "db_host": "db-primary.internal",                     │
│     "pool_active": 48, "pool_max": 50                    │
│   }                                                       │
└──────────────────────────────────────────────────────────┘

SECURITY LOG (Hard task):
┌──────────────────────────────────────────────────────────┐
│ 2024-01-15T02:14:33.891Z [http-nio-8080-exec-7]         │
│ WARN c.a.auth.LoginController                            │
│ - Failed login attempt for user 'admin'                  │
│   metadata: {                                             │
│     "source_ip": "198.51.100.23",                        │
│     "user_agent": "python-requests/2.28.0",              │
│     "attempt_count": 47,                                  │
│     "geo": "RO",                                          │
│     "request_id": "req-b7d912"                           │
│   }                                                       │
└──────────────────────────────────────────────────────────┘

NORMAL BACKGROUND LOG:
┌──────────────────────────────────────────────────────────┐
│ 2024-01-15T09:15:02.334Z [http-nio-8080-exec-1]         │
│ INFO c.a.auth.AuthService                                │
│ - User login successful                                  │
│   metadata: {                                             │
│     "user_id": "u_44821",                                │
│     "source_ip": "10.0.12.55",                           │
│     "session_id": "sess-d92fc1",                         │
│     "request_id": "req-11ae82"                           │
│   }                                                       │
└──────────────────────────────────────────────────────────┘

DISTRACTOR LOG:
┌──────────────────────────────────────────────────────────┐
│ 2024-01-15T09:18:55.772Z [scheduled-task-3]              │
│ WARN c.a.infra.DnsResolver                               │
│ - DNS resolution timeout for metrics.internal (retried OK)│
│   metadata: {                                             │
│     "resolver": "10.0.0.2",                              │
│     "query_type": "A",                                    │
│     "retry_count": 1,                                     │
│     "resolved_after_ms": 2340                            │
│   }                                                       │
└──────────────────────────────────────────────────────────┘
5.3 Log Field Specification
text
FIELD          TYPE      FORMAT                        EXAMPLE
─────────────────────────────────────────────────────────────────────
id             str       "log_{NNN}"                   "log_042"
timestamp      str       ISO-8601 with milliseconds    "2024-01-15T09:23:41.127Z"
service        str       kebab-case real names         "auth-service"
severity       str       DEBUG/INFO/WARN/ERROR/FATAL   "ERROR"
logger         str       Java package notation         "c.a.auth.DatabaseConnectionPool"
thread         str       Tomcat thread pool format     "[http-nio-8080-exec-3]"
message        str       Human-readable message        "Failed to acquire connection..."
stack_trace    str|null  Java stack trace or null      "java.sql.SQLException: ..."
metadata       dict      Service-specific key-values   {"request_id": "req-a3f8c2", ...}
5.4 Timestamp Generation Rules
text
Base time: 2024-01-15T00:00:00.000Z (fixed for reproducibility)

NORMAL LOG INTERVALS:
  Minimum gap: 2 seconds
  Maximum gap: 30 seconds
  Distribution: exponential (lambda=0.15) + 2s floor
  Jitter: ±500ms added to each timestamp

INCIDENT LOG CLUSTERING:
  Easy:   Errors clustered within a 15-minute window
  Medium: Cascading events span a 30-minute window
  Hard:   Attack chain spans 2-4 hours

TIMESTAMP CONSTRAINTS:
  NO two logs share the exact same timestamp.
  Timestamps are MONOTONICALLY INCREASING within each service.
  Cross-service timestamps may interleave naturally.
5.5 Severity Distribution Per Task
text
                 DEBUG   INFO    WARN    ERROR   FATAL   TOTAL
Easy (50):         5      30       7       8       0       50
  GT errors:                               3
  Distractor:                              0

Medium (200):     10     105      30      50       5      200
  GT errors:                              6*
  Distractor:                             5

Hard (500):       25     280      50      42       3      500
  GT errors:                    8*
  Distractor:                            15

*Note: GT entries may be at WARN level (not just ERROR),
 especially in the hard task where attack signals are subtle.
5.6 Noise vs Signal Design
text
SIGNAL RATIO BY TASK:
  Easy:   3/50   = 6.0%  (high signal)
  Medium: 6/200  = 3.0%  (moderate signal)
  Hard:   8/500  = 1.6%  (needle in haystack)

DISTRACTOR DESIGN:
  Easy:   0 distractors (clean environment)
  Medium: 5 distractors (some unrelated WARN/ERROR)
  Hard:   15 distractors (many unrelated errors)

DISTRACTOR TEMPLATES:
  "DNS resolution timeout for metrics.internal (retried OK)"
  "Disk usage at 82% on /var/log partition"
  "TLS certificate expiring in 14 days for api.example.com"
  "JVM garbage collection pause: 450ms (threshold: 500ms)"
  "Connection reset by peer: 10.0.5.22:3306"
  "Slow query detected: 2.3s for SELECT on users table"
  "Health check timeout for downstream cache-service"
  "Thread pool exhaustion warning: 95% utilization"
  "Rate limit threshold reached: 980/1000 requests"
  "Backup job completed with 2 warnings"
  "Memory usage elevated: 78% of heap allocated"
  "Log rotation failed: permission denied on /var/log/app.log"
  "Stale NFS handle detected for shared mount"
  "SMTP connection timeout for alert-mailer"
  "Config reload triggered: 3 properties changed"

These look like real operational issues but are NOT part of
the incident under investigation.
5.7 Reproducibility Guarantee
text
SEEDING STRATEGY:
  import hashlib

  def task_seed(task_id: str) -> int:
      return int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)

  random.seed(task_seed(task_id))
  numpy.random.seed(task_seed(task_id))

  Same task_id → same seed → same logs → same ground truth positions

GROUND TRUTH LOG ID ASSIGNMENT:
  IDs assigned sequentially: log_001 through log_{n}
  Ground truth logs placed at PREDETERMINED positions
  (seeded deterministic placement, not random)
  Ground truth log IDs are STABLE across resets.

VERIFICATION PROTOCOL:
  Generate logs 100 times for each task.
  SHA-256 hash all generated log datasets.
  ALL 100 hashes must be IDENTICAL.
5.8 Log Message Templates by Category
text
NORMAL OPERATIONS (Background Noise):
  "User login successful for {user_id} from {ip}"
  "Health check passed: all dependencies healthy"
  "Token refreshed for session {session_id}"
  "Request completed: GET /api/v1/{endpoint} in {ms}ms"
  "Cache hit for key {cache_key}: {hit_rate}% hit rate"
  "Scheduled job {job_name} completed in {ms}ms"
  "Connection pool stats: active={n}/{max}, idle={m}"

EASY TASK — Database Errors:
  "Failed to acquire connection to {db_host}:{port}"
  "java.sql.SQLException: Connection refused"
  "Retry attempt {n}/{max} failed. Circuit breaker OPEN."
  "Database health check failed: {db_host} unreachable"

MEDIUM TASK — Cascading Failure:
  Root cause:
    "Connection pool exhausted: {active}/{max} connections in use.
     All acquire attempts timing out after {timeout}ms"
  Symptoms:
    "Transaction timeout after {ms}ms waiting for database connection"
    "Request queue at capacity ({n}/{max}). Rejecting new requests"
    "Timeout waiting for payment-service response: {ms}ms exceeded"
  Cascading:
    "Message queue depth critical: {depth} pending (threshold: {max})"
    "HTTP 503 Service Unavailable returned to client. Downstream
     dependency {service} not responding"

HARD TASK — Security Attack:
  Reconnaissance:
    "Unusual port scan detected from {ip}: probing ports
     {port_list} on {target_host}"
  Brute force:
    "Failed login attempt for user '{username}' from {ip}
     (attempt {n}, {user_agent})"
  Credential compromise:
    "Successful authentication for '{username}' from {ip}
     after {n} prior failures in {timeframe}"
  Privilege escalation:
    "User '{username}' accessed admin endpoint /api/admin/{path}
     without ADMIN role. Source: {ip}"
  Lateral movement:
    "Authentication token {token_prefix}... used from new service
     context: {service}. Original auth on {orig_service}"
  Data exfiltration:
    "Bulk data export: {n} records from {table} downloaded by
     user '{username}'. Size: {size}MB. Time: {time}"


6. Enhanced Adaptive Reward System
6.1 Design Philosophy
text
PRINCIPLE 1: DENSE SIGNAL
  Every action produces some reward signal.
  Not just binary at end of episode.

PRINCIPLE 2: PARTIAL CREDIT
  Finding 5 of 8 issues scores better than 0.
  Even failed episodes teach something.

PRINCIPLE 3: ANTI-DEGENERATE BEHAVIOR
  Repeated actions, noop loops, and random annotations penalized.

PRINCIPLE 4: EFFICIENCY MATTERS
  Solving in fewer steps rewarded. Simulates real time pressure.

PRINCIPLE 5: REWARD-GRADER ALIGNMENT
  Step rewards must correlate (r > 0.85) with final grader score.
  Trajectory alignment score ensures this.
6.2 ML Models Used for Rewards
text
MODEL                       SIZE    PURPOSE                DETERMINISTIC?
────────────────────────────────────────────────────────────────────────
all-MiniLM-L6-v2           ~80MB   Semantic similarity     Yes (CPU)
                                    for reports, categories
TF-IDF Vectorizer           ~5MB   Search relevance        Yes
(scikit-learn)                      scoring
BM25 (rank-bm25)            ~2MB   Information retrieval   Yes
                                    scoring for searches
────────────────────────────────────────────────────────────────────────
TOTAL: ~87MB RAM, ~3s startup, ~12ms per step
6.3 Per-Action Reward Components
text
ACTION                     BASE        MECHANISM
───────────────────────────────────────────────────────────────

ANNOTATION (correct log)
  Base reward              +0.10       Log ID in ground truth
  × category_similarity   0.0–1.0     MiniLM cosine sim of descriptions
  × information_gain      1.0–3.0     How hard was this to find?
  × temporal_multiplier   0.7–1.3     Early finds worth more
  × difficulty_multiplier 1.0–1.6     Harder tasks pay more
  × coverage_multiplier   0.8–1.2     Coverage-aware, not flat diminishing
  × informed_bonus        0.5–1.1     Did agent VIEW this log first?
  CAPPED at: +0.20 per annotation

ANNOTATION (wrong log)
  Base penalty             −0.05      Log ID not in ground truth
  × precision_scaling     1.0–3.0     Doubles/triples at low precision
  CAPPED at: −0.15 per wrong annotation

CORRELATION (correct pair)
  Base reward              +0.15      Directed pair in ground truth
  × temporal_multiplier   0.7–1.3    Timing bonus
  × difficulty_multiplier 1.0–1.6    Task difficulty
  + chain_bonus           +0.05–0.20 If extending a connected chain
  CAPPED at: +0.25 per correlation

CORRELATION (wrong pair)
  Base penalty             −0.05      Pair not in ground truth
  CAPPED at: −0.08

SEARCH
  BM25 + TF-IDF scoring:
    relevant_hits = results ∩ undiscovered_ground_truth
    precision_at_k = relevant_hits / total_results
    recall_gain = relevant_hits / remaining_undiscovered
    search_reward = 0.02 × (1 + 3 × precision) × (1 + 2 × recall_gain)
  CAPPED at: +0.08 useful, −0.02 useless (<span class="ml-2" /><span class="inline-block w-3 h-3 rounded-full bg-neutral-a12 align-middle mb-[0.1rem]" />
Ask anything
6.4 Information Gain Multiplier
text
For each ground truth log, precomputed at environment initialization:

FACTORS:
  Severity visibility:
    ERROR/FATAL log = easy to find    → multiplier 1.0
    WARN log = moderate               → multiplier 1.5
    INFO log with subtle anomaly      → multiplier 2.5

  Position in dataset:
    First 3 pages = easy to find      → multiplier 1.0
    Pages 4-10                        → multiplier 1.3
    Pages 11+ (buried deep)           → multiplier 1.8

  Semantic distinctiveness:
    Unique/obvious error message      → multiplier 1.0
    Message blends with noise         → multiplier 2.0

  Cross-service requirement:
    Same service as goal mentions     → multiplier 1.0
    Different service (lateral find)  → multiplier 1.5

COMBINED: geometric mean of all factors, CAPPED at 3.0

EXAMPLES:
  Easy task — obvious ERROR on page 1:
    1.0 × 1.0 × 1.0 × 1.0 = 1.0 → annotation worth +0.10

  Hard task — WARN on page 15, blends with noise, different service:
    1.5 × 1.8 × 2.0 × 1.5 = 3.0 (capped) → annotation worth up to +0.30
6.5 Coverage-Aware Scaling
text
Replaces flat diminishing returns.

coverage_so_far = correct_annotations / total_ground_truth

If coverage < 0.5:
  multiplier = 1.2  (encourage finding more)
If coverage 0.5–0.8:
  multiplier = 1.0  (standard)
If coverage > 0.8:
  multiplier = 0.8  (slight diminishing)

COVERAGE MILESTONE BONUSES:
  25% coverage reached → +0.05 bonus
  50% coverage reached → +0.08 bonus
  75% coverage reached → +0.10 bonus
  100% coverage reached → +0.15 bonus
6.6 Correlation Chain Detection
text
After each correlate() action, check agent's correlation graph:

  chain_length = longest connected path in agent's correlations

  Chain bonuses (on top of individual correlation rewards):
    3 connected nodes → +0.05
    4 connected nodes → +0.10
    5 connected nodes → +0.15
    6+ connected nodes → +0.20 (full chain)

EXAMPLE:
  Agent correlates: A→B, B→C, C→D (chain of 4)
    Individual: 3 × 0.15 = 0.45
    Chain bonus: +0.10
    Total: 0.55

  Agent correlates: A→B, C→D, E→F (3 disconnected pairs)
    Individual: 3 × 0.15 = 0.45
    Chain bonus: +0.00 (no chain detected)
    Total: 0.45
6.7 Investigation Strategy Scoring (Rule-Based)
text
REPLACES Isolation Forest (avoids circular training dependency).

Simple rule-based strategy bonuses:

  Rule 1: Searched or filtered BEFORE first annotation?
    → +0.02 (agent explored before deciding)

  Rule 2: Explored multiple services? (medium/hard tasks)
    → +0.02 per additional service explored (max +0.06)

  Rule 3: Submitted or drafted report as one of last 3 actions?
    → +0.01 (proper investigation closure)

  Rule 4: Exploration ratio in healthy range?
    exploration_actions / total_actions
    Optimal range 0.25–0.50 → +0.02
    Outside range → +0.00 (no penalty, just no bonus)

  MAXIMUM STRATEGY BONUS: +0.11 per episode
6.8 Precision-Aware Penalty Scaling
text
running_precision = correct_annotations / total_annotations_submitted

If running_precision >= 0.3:
  wrong_annotation_penalty = −0.05 (standard)

If running_precision < 0.3:
  wrong_annotation_penalty = −0.10 (doubled)

If running_precision < 0.1:
  wrong_annotation_penalty = −0.15 (tripled)
  AND precision_warning added to observation:
    "Low annotation accuracy detected. Focus on high-confidence findings."
6.9 Trajectory Alignment Score
text
At each step, compute projected final score:

  projected = (
    annotations_found / total_gt_annotations × 0.35 +
    correlations_found / total_gt_correlations × 0.25 +
    has_classified_severity × 0.15 +
    has_drafted_report × 0.10 +
    phase_progress × 0.15
  )

  alignment_bonus = max(0, projected - previous_projected) × 0.05
  CAPPED at: +0.01 per step

  This ensures step rewards point toward final grader goals.
6.10 Informed Annotation Tracking
text
Track which log pages the agent has viewed.

When agent annotates log_id:
  If agent previously viewed the page containing log_id:
    → annotation reward × 1.1 (10% informed bonus)

  If agent NEVER viewed that page:
    → annotation reward × 0.5 (50% penalty — blind guessing)

  "Viewed" means the log appeared in visible_logs at some point.
6.11 Reward Magnitude Caps
text
PER-ACTION CAPS:
  search:            [−0.02, +0.08]
  navigate/filter:   [−0.00, +0.03]
  inspect:           [+0.00, +0.00]  (no direct reward)
  annotate:          [−0.15, +0.20]
  correlate:         [−0.08, +0.25]
  classify:          [−0.12, +0.25]
  draft_report:      [−0.00, +0.05]
  submit_report:     [−0.00, +0.35]
  noop:              [−0.03, +0.00]

PER-STEP CAPS (including bonuses):
  Maximum per step:  +0.40
  Minimum per step:  −0.15

PER-EPISODE CAPS:
  Maximum cumulative: +3.00
  Minimum cumulative: −1.00
6.12 Reward Trajectory Examples
text
SMART AGENT — Hard Task
───────────────────────
Step  Action                           Base   Modifiers      Final  Cumul
  1   search("failed login")           +0.02  BM25 prec=0.4  +0.09  0.09
  2   filter_service("auth-service")   +0.00  density +0.01  +0.01  0.10
  3   inspect(log_078)                 +0.00                 +0.00  0.10
  4   annotate(log_078,"brute_force")  +0.10  ×1.0×2.0×1.3  +0.20  0.30
      [informed, exact category, hard-to-find, early]
  5   annotate(log_091,"brute_force")  +0.10  ×1.0×1.5×1.3  +0.16  0.46
  6   search("admin API")             +0.02  BM25 prec=0.6  +0.13  0.59
  7   annotate(log_201,"priv_esc")    +0.10  ×1.0×2.5×1.0  +0.20  0.79
  8   correlate(log_091,log_145)      +0.15  chain=2        +0.15  0.94
  9   correlate(log_145,log_201)      +0.15  chain=3,+0.05  +0.20  1.14
  ...
  35  submit_report("...")            +0.30  sem_sim=0.83   +0.28  2.41
      + efficiency bonus              +0.10                 +0.10  2.51
                                                     GRADER: 0.62

BAD AGENT — Hard Task
─────────────────────
Step  Action                           Final  Cumul
  1   noop()                           −0.02  −0.02
  2   noop()                           −0.02  −0.04
  3   annotate(log_001,"error")        −0.05  −0.09  ← wrong log, blind guess
  4   annotate(log_002,"error")        −0.05  −0.14  ← wrong again
  5   annotate(log_003,"error")        −0.10  −0.24  ← precision < 0.3, doubled
  ...
  40  (max steps, no report)
                                                     GRADER: 0.03



7. Grader System (10 Dimensions)
7.1 Overview
text
The grader runs ONCE at episode end.
It is COMPLETELY INDEPENDENT from step rewards.
It produces a single score 0.0–1.0.
It is 100% DETERMINISTIC.

Same inputs → same score. Always.
7.2 What Gets Submitted to the Grader
text
FROM THE AGENT:
  agent_annotations: Dict[str, str]    — {log_id: category}
  agent_correlations: List[List[str]]  — [[source, target], ...]
  agent_severity: str                  — "HIGH"
  agent_report: str                    — free text
  agent_behavior: Dict                 — steps, actions, services explored

FROM THE TASK:
  gt_annotations: Dict[str, str]       — ground truth annotations
  gt_correlations: List[List[str]]     — ground truth directed pairs
  gt_severity: str                     — expected severity
  gt_key_findings: List[str]           — short finding phrases
  gt_finding_descriptions: List[str]   — longer descriptions for ML matching
  task_config: Dict                    — max_steps, difficulty, log_count
7.3 The 10 Grading Components
text
COMPONENT                    WEIGHT*   TYPE            SCORE RANGE
═════════════════════════════════════════════════════════════════════
A. Annotation Precision       12%      Rule-based      0.0 – 1.0
B. Annotation Recall          12%      Rule-based      0.0 – 1.0
C. Annotation Quality          8%      ML-enhanced     0.0 – 1.0
D. Correlation Precision      10%      Rule-based      0.0 – 1.0
E. Correlation Recall         10%      Rule-based      0.0 – 1.0
F. Chain Reconstruction        8%      Graph analysis  0.0 – 1.0
G. Severity Classification   10%      Rule + partial  0.0 – 1.0
H. Report Completeness       12%      ML-enhanced     0.0 – 1.0
I. Report Coherence            8%      Heuristic       0.0 – 1.0
J. Investigation Efficiency   10%      Rule-based      0.0 – 1.0
═════════════════════════════════════════════════════════════════════
TOTAL                        100%                      0.0 – 1.0

*Weights shown are for MEDIUM task. See Section 7.6 for per-task weights.
7.4 Component Details
A. Annotation Precision (Rule-Based)

text
QUESTION: "Of everything the agent flagged, how much was real?"

FORMULA:
  correct = |agent_annotations ∩ gt_annotations| (by log_id)
  total = |agent_annotations|

  if total > 0:     precision = correct / total
  if total == 0 AND gt is empty:     precision = 1.0
  if total == 0 AND gt is non-empty: precision = 0.0
B. Annotation Recall (Rule-Based)

text
QUESTION: "Of all real issues, how many did the agent find?"

FORMULA:
  found = |agent_annotations ∩ gt_annotations| (by log_id)
  total_gt = |gt_annotations|

  if total_gt > 0:  recall = found / total_gt
  if total_gt == 0: recall = 1.0
C. Annotation Quality (ML-Enhanced)

text
QUESTION: "Did the agent categorize findings correctly?"

For each correctly identified log (log_id matches):
  Compare agent category vs GT category using precomputed
  similarity matrix (Section 3.3).

  Exact match:              quality = 1.0
  Similarity > 0.80:        quality = 0.85
  Similarity 0.60–0.80:     quality = 0.60
  Similarity 0.40–0.60:     quality = 0.30
  Similarity < 0.40:        quality = 0.0

  annotation_quality = mean(quality scores for all correct annotations)
  If no correct annotations: 0.0
D. Correlation Precision (Rule-Based)

text
QUESTION: "Of correlations the agent made, how many were real?"

FORMULA:
  agent_pairs = {(source, target) for [source, target] in agent_correlations}
  gt_pairs = {(source, target) for [source, target] in gt_correlations}

  correct = |agent_pairs ∩ gt_pairs|
  total = |agent_pairs|

  if total > 0: precision = correct / total
  if total == 0 AND gt_pairs empty: precision = 1.0
  if total == 0 AND gt_pairs non-empty: precision = 0.0

  SPECIAL CASE — Easy task (no GT correlations):
    Agent makes no correlations → 1.0
    Agent makes correlations → 0.5 (neutral, not penalized)
E. Correlation Recall (Rule-Based)

text
QUESTION: "Of real correlations, how many did the agent find?"

FORMULA:
  found_direct = exact pair matches (1.0 credit each)
  found_transitive = transitive matches (0.5 credit each)
    (e.g., GT has A→B, B→C; agent submits A→C)

  recall = min(1.0, (found_direct + found_transitive) / total_gt_pairs)
  if total_gt_pairs == 0: recall = 1.0
F. Chain Reconstruction (Graph Analysis)

text
QUESTION: "Did the agent reconstruct the causal chain?"

Treats GT and agent correlations as Directed Acyclic Graphs (DAGs).

1. LONGEST PATH MATCH (weight 40%):
   gt_longest = longest directed path in GT graph
   agent_matching = longest agent path that overlaps GT nodes
   path_score = agent_matching / gt_longest

2. COMPONENT COVERAGE (weight 40%):
   gt_nodes_in_agent = GT nodes present in agent's graph
   coverage = gt_nodes_in_agent / total_gt_nodes

3. EDGE DIRECTION (weight 20%):
   For each agent edge that matches a GT edge:
     Correct direction: 1.0
     Reversed direction: 0.5
   direction_score = mean(direction scores)

chain_score = 0.4 × path_score + 0.4 × coverage + 0.2 × direction_score

TASK-SPECIFIC:
  Easy task (no chains): chain_score = 1.0 (automatic full credit)
  Medium task: chain matters moderately
  Hard task: chain is critical differentiator
G. Severity Classification (Rule + Partial)

text
QUESTION: "Did the agent correctly assess overall severity?"

Severity scale: LOW=1, MEDIUM=2, HIGH=3, CRITICAL=4

  distance = |agent_level - gt_level|

  distance 0 → score = 1.0  (exact match)
  distance 1 → score = 0.5  (off by one)
  distance 2 → score = 0.15 (significantly wrong)
  distance 3 → score = 0.0  (completely wrong)

  If agent never classified → score = 0.0

INPUT NORMALIZATION:
  Strip whitespace, convert to uppercase, strip punctuation.
  "critical!" → "CRITICAL" → match
  "VERY HIGH" → no match in {LOW, MEDIUM, HIGH, CRITICAL} → 0.0
H. Report Completeness (ML-Enhanced)

text
QUESTION: "Does the report cover all key findings?"

For each key finding description in ground truth:
  1. Embed the finding description (precomputed, MiniLM)
  2. Split agent report into sentences
  3. Embed each sentence
  4. best_similarity = max cosine similarity to any sentence

  best_similarity > 0.80 → covered = 1.0
  best_similarity > 0.65 → covered = 0.7  (paraphrased)
  best_similarity > 0.50 → covered = 0.4  (vaguely mentioned)
  best_similarity ≤ 0.50 → covered = 0.0  (not mentioned)

  completeness = mean(covered scores for all findings)

EDGE CASES:
  Empty report → 0.0
  "There was an incident" → very low similarity → ~0.05
  Report copies raw log IDs → moderate at best → ~0.3
  Report synthesizes findings → high similarity → ~0.85
I. Report Coherence (Heuristic, NOT LLM)

text
QUESTION: "Is the report well-structured?"

5 sub-metrics, all rule-based:

1. LENGTH ADEQUACY (weight 25%):
   Task-adjusted expected ranges:
     Easy:   30–300 chars optimal
     Medium: 90–600 chars optimal
     Hard:   150–1500 chars optimal

   too_short      → 0.0
   minimal        → 0.3
   adequate       → 0.8
   detailed       → 1.0
   too_long       → 0.7 (rambling penalty)

2. TEMPORAL ORDERING (weight 25%):
   Count temporal sequence words:
     "first", "then", "subsequently", "followed by",
     "after that", "finally", "initially", "began with",
     "leading to", "next", "eventually"
   score = min(temporal_words / expected_markers, 1.0)
     Easy expected:   1 marker
     Medium expected:  2 markers
     Hard expected:    3 markers

3. CAUSAL LANGUAGE (weight 20%):
   Count causal indicators:
     "caused", "resulted in", "led to", "because",
     "due to", "triggered", "propagated", "as a result",
     "consequently", "root cause"
   causal_density = sentences_with_causal / total_sentences
   score = min(causal_density / 0.3, 1.0)

4. STRUCTURAL MARKERS (weight 15%):
   Check for organizational elements:
     Headers, bullet points, numbered lists,
     "Root cause:", "Impact:", "Timeline:", "Affected services:",
     "Summary:", "Recommendation:"
   score = min(markers_found / 3, 1.0)

5. SPECIFICITY (weight 15%):
   Count specific references:
     Service names, log IDs, IP addresses,
     timestamps, error codes, user IDs, port numbers
   expected_details:
     Easy: 3, Medium: 6, Hard: 10
   score = min(specific_references / expected_details, 1.0)

coherence = 0.25×length + 0.25×temporal + 0.20×causal
            + 0.15×structural + 0.15×specificity
J. Investigation Efficiency (Rule-Based)

text
QUESTION: "How efficiently did the agent work?"

NOT just "fewer steps = better."
Efficiency is QUALITY-ADJUSTED.


7. Grader System (10 Dimensions) — Continued
J. Investigation Efficiency (Rule-Based) — Continued

text
quality_score = weighted average of components A through I
step_ratio = steps_taken / max_steps

IF quality_score >= 0.7 (good results):
  step_ratio ≤ 0.4  → efficiency = 1.0   (excellent)
  step_ratio ≤ 0.6  → efficiency = 0.85
  step_ratio ≤ 0.8  → efficiency = 0.65
  step_ratio ≤ 1.0  → efficiency = 0.50

IF quality_score 0.4–0.7 (moderate results):
  step_ratio ≤ 0.5  → efficiency = 0.70
  step_ratio ≤ 0.7  → efficiency = 0.60
  step_ratio ≤ 1.0  → efficiency = 0.50

IF quality_score < 0.4 (poor results):
  efficiency = 0.2 × (1 - step_ratio)

MEANING:
  Fast + accurate  = high efficiency
  Fast + inaccurate = medium (didn't waste time, but poor work)
  Slow + accurate  = medium (good work, but slow)
  Slow + inaccurate = low (wasted time AND got it wrong)
7.5 Draft Report Grading Fallback
text
SCENARIO: Agent drafts a good report but runs out of steps
without formally submitting.

FALLBACK HIERARCHY:
  1. If agent called submit_report() → grade that report
  2. If agent never submitted but called draft_report() →
     grade the LAST draft with a 20% penalty:
       report_completeness score × 0.80
       report_coherence score × 0.80
  3. If agent never drafted or submitted → report scores = 0.0

RATIONALE:
  Agent did the work but forgot to submit.
  Giving ZERO is too harsh — they showed understanding.
  The 20% penalty incentivizes proper submission while
  not completely destroying the score.

  Documented in task descriptions and README so agents know:
  "Submit your report or lose 20% on report components."
7.6 Per-Task Weight Adjustments
text
Different tasks demand different SRE skills.
The grader dynamically adjusts component weights.

COMPONENT                  EASY    MEDIUM    HARD
──────────────────────────────────────────────────
A. Annotation Precision     15%      12%      10%
B. Annotation Recall        15%      12%      10%
C. Annotation Quality        5%       8%      10%
D. Correlation Precision     2%      10%      12%
E. Correlation Recall        2%      10%      12%
F. Chain Reconstruction      0%       8%      10%
G. Severity Classification  15%      10%       8%
H. Report Completeness      18%      12%      10%
I. Report Coherence         13%       8%       8%
J. Investigation Efficiency 15%      10%      10%
──────────────────────────────────────────────────
TOTAL                      100%     100%     100%

RATIONALE:
  EASY TASK:
    • No correlations → D, E, F nearly zero weight
    • Report and severity matter more (main deliverables)
    • Efficiency matters more (should be quick)

  MEDIUM TASK:
    • Balanced across all components
    • Correlations now meaningful
    • Chain starts to matter

  HARD TASK:
    • Correlation and chain CRITICAL (attack reconstruction)
    • Annotation quality matters more (precise categorization)
    • Less weight on severity (one lucky guess shouldn't save you)
    • Less weight on efficiency (hard task legitimately takes time)
7.7 Score Calibration Examples
text
═══════════════════════════════════════════════════════════════
EASY TASK — Expected Score Ranges
═══════════════════════════════════════════════════════════════

EXCELLENT AGENT (score: 0.90–0.97):
  Found 3/3 errors (precision 1.0, recall 1.0)
  Correct categories (quality 1.0)
  Classified MEDIUM correctly (severity 1.0)
  Report covers all findings with good structure
  Done in 8/15 steps

GOOD AGENT (score: 0.65–0.85):
  Found 2/3 errors, 1 false positive (precision 0.67, recall 0.67)
  Categories mostly right (quality 0.80)
  Classified MEDIUM correctly (severity 1.0)
  Report covers 2/3 findings
  Done in 12/15 steps

MEDIOCRE AGENT (score: 0.30–0.55):
  Found 1/3 errors, 2 false positives (precision 0.33, recall 0.33)
  Categories wrong (quality 0.30)
  Classified HIGH instead of MEDIUM (severity 0.5)
  Report is vague
  Used all 15 steps

POOR AGENT (score: 0.05–0.20):
  Found 0/3 errors, annotated random logs
  Never classified severity
  Report empty or irrelevant
  Used all steps

RANDOM AGENT (score: 0.00–0.05):
  Random actions, random annotations
  Everything near zero


═══════════════════════════════════════════════════════════════
MEDIUM TASK — Expected Score Ranges
═══════════════════════════════════════════════════════════════

EXCELLENT AGENT (score: 0.75–0.94):
  Found 6/6 annotations, correct categories
  3/3 correlations with correct direction
  Chain fully reconstructed
  Classified HIGH correctly
  Detailed report with causal analysis
  Done in 18/25 steps

GOOD AGENT (score: 0.45–0.70):
  Found 4/6 annotations, 1 false positive
  2/3 correlations correct
  Partial chain reconstruction
  Correct severity
  Report covers most findings
  Done in 22/25 steps

MEDIOCRE AGENT (score: 0.20–0.40):
  Found 2/6 annotations, several false positives
  1/3 correlations
  No chain
  Wrong severity by 1 level
  Vague report

RANDOM AGENT (score: 0.00–0.05):
  Near zero on everything


═══════════════════════════════════════════════════════════════
HARD TASK — Expected Score Ranges
═══════════════════════════════════════════════════════════════

EXCELLENT AGENT (score: 0.60–0.90):
  Found 7/8 attack logs, high precision
  Correct attack stage categories
  5/6 correlations correct with chain
  Classified CRITICAL
  Detailed security incident report
  Done in 30/40 steps

GOOD AGENT (score: 0.35–0.55):
  Found 5/8 attack logs with some false positives
  Categories partially right
  3/6 correlations
  Partial chain
  Classified HIGH (off by 1)
  Report covers 3/6 findings

MEDIOCRE AGENT (score: 0.10–0.30):
  Found 2/8 logs, many false positives
  Categories mostly wrong
  1/6 correlations, no chain
  Wrong severity
  Vague report

RANDOM AGENT (score: 0.00–0.03):
  Near zero on everything
7.8 Grader Edge Cases & Exploit Guards
text
EDGE CASE 1: Agent Submits Empty Everything
───────────────────────────────────────────
  annotations={}, correlations=[], severity="", report=""

  A(precision) = 0.0
  B(recall) = 0.0
  C(quality) = 0.0
  D–F = 0.0 (or 1.0 on easy if no GT correlations)
  G(severity) = 0.0
  H(completeness) = 0.0
  I(coherence) = 0.0
  J(efficiency) = low

  TOTAL: ~0.00–0.02 ✓


EDGE CASE 2: Agent Annotates ALL 500 Logs
──────────────────────────────────────────
  annotations = {all 500 log_ids: "error"}

  A(precision) = 8/500 = 0.016 (terrible)
  B(recall) = 8/8 = 1.0 (found everything)
  C(quality) = low (all "error", not specific categories)

  Weighted: 0.016×12% + 1.0×12% + 0.3×8% = ~0.15 from annotations
  Plus minimal correlation/report scores.

  TOTAL: ~0.15–0.25 ✓ (spam strategy doesn't pay)


EDGE CASE 3: Submit Report Without Investigation
─────────────────────────────────────────────────
  Step 3: submit_report("I think there was a breach")

  A–F: 0.0 (no annotations or correlations)
  G: 0.0 (no severity)
  H: ~0.1 (vaguely mentions breach)
  I: ~0.2 (short but some structure)
  J: quality < 0.4, so efficiency very low

  TOTAL: ~0.03 ✓


EDGE CASE 4: Perfect Annotations But No Report
───────────────────────────────────────────────
  All annotations correct, all correlations correct,
  severity correct, but never submitted a report.

  A–G: excellent (~0.95)
  H: 0.0 (no report)
  I: 0.0 (no report)
  J: moderate (good quality but used all steps)

  With draft fallback (if drafts exist):
    H: score × 0.80 (20% penalty)
    I: score × 0.80

  Without any draft:
    TOTAL: ~0.55–0.65 ✓
    Rewards investigation but penalizes lack of synthesis.


EDGE CASE 5: Duplicate Correlations
────────────────────────────────────
  agent_correlations = [["A","B"], ["A","B"], ["A","B"]]

  After DEDUPLICATION: {("A","B")}
  Counted as 1 correlation, not 3.
  Grader deduplicates BEFORE scoring. ✓


EDGE CASE 6: Invalid Severity String
─────────────────────────────────────
  agent_severity = "VERY HIGH" or "critical!" or "3"

  Normalization:
    Strip whitespace → uppercase → strip punctuation
    Check against {LOW, MEDIUM, HIGH, CRITICAL}

    "VERY HIGH" → no match → 0.0
    "critical!" → strip punct → "CRITICAL" → match ✓
    "3" → no match → 0.0


EDGE CASE 7: Report Contains Log IDs Instead of Descriptions
─────────────────────────────────────────────────────────────
  Report: "log_078 and log_145 are attack indicators"

  Semantic similarity checks DESCRIPTIONS, not log IDs.
  "log_078 and log_145 are attack indicators"
  vs "brute force login attempts" → similarity ~0.25 → low score

  Agent must DESCRIBE what happened, not just list IDs. ✓


EDGE CASE 8: Easy Task Correlation Handling
───────────────────────────────────────────
  Easy task has no GT correlations.

  Agent makes no correlations → D=1.0, E=1.0 ✓
  Agent makes correlations → D=0.5, E=1.0 (neutral) ✓
  Chain: automatic 1.0 (no chain expected) ✓


EDGE CASE 9: Transitive Correlations on Recall
───────────────────────────────────────────────
  GT: A→B, B→C
  Agent submits: A→C (skipped B)

  Direct match: 0 of 2
  Transitive match: A→C covers A→B→C path → 0.5 credit

  recall = (0 + 0.5) / 2 = 0.25

  Agent understood A and C are related but missed intermediate.
  Partial credit is fair. ✓


EDGE CASE 10: Agent Calls step() After done=True
─────────────────────────────────────────────────
  Returns last observation with done=True.
  last_action_message = "Episode already complete."
  No state change, no reward, no grader re-run.
7.9 Grader Output Format
json
{
  "task_id": "task_hard",
  "final_score": 0.5832,
  "components": {
    "annotation_precision": {
      "score": 0.7143,
      "weight": 0.10,
      "weighted": 0.0714,
      "detail": "5 correct of 7 submitted"
    },
    "annotation_recall": {
      "score": 0.6250,
      "weight": 0.10,
      "weighted": 0.0625,
      "detail": "5 of 8 ground truth found"
    },
    "annotation_quality": {
      "score": 0.8200,
      "weight": 0.10,
      "weighted": 0.0820,
      "detail": "avg category similarity 0.82"
    },
    "correlation_precision": {
      "score": 0.7500,
      "weight": 0.12,
      "weighted": 0.0900,
      "detail": "3 correct of 4 submitted"
    },
    "correlation_recall": {
      "score": 0.5000,
      "weight": 0.12,
      "weighted": 0.0600,
      "detail": "3 of 6 ground truth found (incl 0 transitive)"
    },
    "chain_reconstruction": {
      "score": 0.6520,
      "weight": 0.10,
      "weighted": 0.0652,
      "detail": "path 3/5, coverage 4/6, direction 1.0"
    },
    "severity_classification": {
      "score": 0.5000,
      "weight": 0.08,
      "weighted": 0.0400,
      "detail": "predicted HIGH, expected CRITICAL (off by 1)"
    },
    "report_completeness": {
      "score": 0.7167,
      "weight": 0.10,
      "weighted": 0.0717,
      "detail": "4.3/6 findings covered semantically"
    },
    "report_coherence": {
      "score": 0.6800,
      "weight": 0.08,
      "weighted": 0.0544,
      "detail": "length:0.8 temporal:0.7 causal:0.5 struct:0.6 spec:0.8"
    },
    "investigation_efficiency": {
      "score": 0.6500,
      "weight": 0.10,
      "weighted": 0.0650,
      "detail": "quality 0.58, used 32/40 steps"
    }
  },
  "metadata": {
    "steps_taken": 32,
    "max_steps": 40,
    "annotations_submitted": 7,
    "correlations_submitted": 4,
    "report_length_chars": 487,
    "report_source": "submitted",
    "severity_submitted": "HIGH",
    "grader_version": "3.0",
    "determinism_hash": "a3f8c2d1e5..."
  }
}
7.10 Grader Determinism Verification
text
DETERMINISM CONTRACT:
  Every component satisfies:
    grade(inputs_X) == grade(inputs_X)  ALWAYS

POTENTIAL NON-DETERMINISM SOURCES AND MITIGATIONS:

1. ML Embeddings
   RISK: floating point differences across runs
   FIX: precompute ALL reference embeddings at build time.
        Store as numpy arrays with fixed precision (float32).
        Pin exact model version in requirements.txt.
        CPU-only inference (no GPU non-determinism).

2. Set Operations
   RISK: Python set ordering varies across runs
   FIX: convert sets to sorted lists before comparison.
        All counting operations are order-independent.

3. Floating Point Arithmetic
   RISK: accumulation order affects final digits
   FIX: round all intermediate scores to 4 decimal places.
        Round final score to 4 decimal places.

4. Text Processing
   RISK: Unicode normalization differences
   FIX: normalize all text to NFKC form.
        Lowercase before comparison.
        Strip all whitespace consistently.

5. Sentence Splitting
   RISK: different tokenizers split differently
   FIX: use simple rule-based sentence splitter (split on ". ", "! ", "? ")
        not NLTK or spaCy (which may update models).

VERIFICATION PROTOCOL:
  Create 10 fixed agent outputs (ranging from terrible to perfect).
  Run grader on each 100 times.
  Assert ALL 100 runs produce IDENTICAL scores.
  If any score differs by even 0.0001 → BUILD FAILS.
8. Session Management & API Contract
8.1 Session Architecture
text
APPROACH: Session tokens (supports concurrent access)

  Server stores: Dict[str, LogTriageEnv]
  Each session identified by UUID.
  Session cleanup: TTL of 30 minutes, max 10 concurrent sessions.

MEMORY MATH:
  Each session ≈ 5MB (logs + state + ML cache reference)
  10 sessions × 5MB = 50MB
  ML models shared across sessions (loaded once): ~90MB
  Total worst case: ~140MB (well within 8GB) ✓
8.2 API Endpoints
text
POST /reset
  Body:    {"task_id": "task_easy"}
  Returns: {
    "session_id": "a1b2c3d4-...",
    "observation": { ... full observation object ... },
    "info": {"task_name": "Database Connection Failures", "max_steps": 15}
  }
  Creates new session. Destroys previous session with same task_id if exists.


POST /step
  Body:    {
    "session_id": "a1b2c3d4-...",
    "action_type": "search",
    "params": {"pattern": "connection refused"}
  }
  Returns: {
    "observation": { ... },
    "reward": {
      "value": 0.087,
      "components": {"useful_search": 0.087},
      "cumulative": 0.087
    },
    "done": false,
    "info": {}
  }
  When done=true, info includes "grader_result" with full grading breakdown.


GET /state
  Query:   ?session_id=a1b2c3d4-...
  Returns: {
    "task_id": "task_easy",
    "step": 5,
    "done": false,
    "annotations": {"log_008": "error"},
    "correlations": [],
    "severity": null,
    "report": null,
    "filters": {"severity": "ERROR"},
    "session_id": "a1b2c3d4-..."
  }


GET /health
  Returns: {"status": "ok", "version": "3.0", "sessions_active": 2}
  No authentication required.


GET /tasks
  Returns: [
    {"id": "task_easy", "name": "Database Connection Failures",
     "difficulty": "easy", "max_steps": 15, "log_count": 50},
    {"id": "task_medium", "name": "Cascading Service Failure",
     "difficulty": "medium", "max_steps": 25, "log_count": 200},
    {"id": "task_hard", "name": "Security Breach Investigation",
     "difficulty": "hard", "max_steps": 40, "log_count": 500}
  ]
8.3 Session Lifecycle
text
SESSION STATES:
  ACTIVE    → accepting step() calls
  DONE      → episode complete, grading available, no more steps
  EXPIRED   → TTL exceeded, session cleaned up

STATE TRANSITIONS:
  reset()                   → ACTIVE
  step() [not terminal]     → ACTIVE
  step() [terminal action
    or max steps reached]   → DONE
  30 min inactivity         → EXPIRED

SESSION ERROR RESPONSES:
  Invalid session_id     → 404 {"error": "Session not found. Call /reset first."}
  Step on DONE session   → 200 {last observation, done=true,
                                 "info": {"message": "Episode already complete."}}
  Step on EXPIRED        → 410 {"error": "Session expired. Call /reset."}
  Reset on active session → destroys old session, creates new one
  Max sessions reached   → 429 {"error": "Too many active sessions. Try again later."}


9. Error Handling & Edge Cases
9.1 Invalid Action Handling
text
Every invalid action:
  • Does NOT crash the server
  • Sets last_action_success = False
  • Sets descriptive last_action_message
  • Causes NO state change
  • Step counter STILL increments (wasting a step IS the penalty)
  • Returns zero reward

SPECIFIC CASES:

INVALID LOG ID (annotate/correlate/inspect):
  last_action_success = False
  last_action_message = "Log ID 'xyz' not found in current dataset."
  No state change.

SELF-CORRELATION:
  correlate("log_045", "log_045")
  last_action_success = False
  last_action_message = "Cannot correlate a log with itself."
  No state change.

INVALID SEVERITY FILTER:
  filter_severity("SUPER_BAD")
  last_action_success = False
  last_action_message = "Invalid severity level. Use: DEBUG, INFO, WARN, ERROR, FATAL."
  Filters unchanged.

INVALID SERVICE FILTER:
  filter_service("nonexistent-service")
  last_action_success = False
  last_action_message = "Service 'nonexistent-service' not found.
    Available: auth-service, api-gateway, ..."
  Filters unchanged.

EMPTY SEARCH:
  search("")
  last_action_success = True  (technically valid)
  Clears search filter. Returns all logs.
  last_action_message = "Search cleared. Showing all logs."

EMPTY REPORT SUBMISSION:
  submit_report("")
  If step >= 3: accepted, episode ends, will score very low.
    last_action_message = "Empty report submitted."
  If step < 3: auto-converted to draft.
    last_action_message = "Too early to submit (min 3 steps). Saved as draft."

EARLY SUBMIT (before step 3):
  submit_report("any text")
  Automatically converted to draft_report("any text").
  last_action_success = True
  last_action_message = "Too early for final submission (step {n}/3 min).
    Saved as draft. Submit again after step 3."

DUPLICATE ANNOTATION (same log_id):
  annotate("log_008", "root_cause") when log_008 already annotated as "error"
  last_action_success = True  (overwrites previous annotation)
  last_action_message = "Updated annotation for log_008: error → root_cause"
  Previous annotation is replaced.

FILTER PRODUCES ZERO RESULTS:
  filter_severity("FATAL") on a dataset with no FATAL logs
  last_action_success = True
  visible_logs = []
  last_action_message = "No logs match current filters. 0 results."
  Agent can clear_filters() or adjust.

ACTION AFTER DONE:
  Any action when done=True
  Returns last observation with done=True.
  last_action_message = "Episode already complete."
  No state change, no reward, no grader re-run.

UNKNOWN ACTION TYPE:
  API returns 400 Bad Request.
  Body: {"error": "Unknown action type: 'xyz'.
    Valid types: search, filter_severity, filter_service,
    filter_time_range, clear_filters, clear_filter,
    scroll, inspect, annotate, correlate,
    classify_incident, draft_report, submit_report, noop"}

MALFORMED PARAMS:
  API returns 422 Unprocessable Entity (Pydantic validation).
  Body: {"error": "Missing required parameter 'pattern' for action 'search'."}
9.2 Episode Termination Conditions
text
Episode ends when ANY of these occur:

1. Agent calls submit_report() at step >= 3
   → done = True
   → Grader runs on submitted report
   → Grader result included in info

2. Agent reaches max_steps
   → done = True
   → Grader runs with whatever agent has produced
   → Uses draft fallback if no submission (Section 7.5)
   → last_action_message = "Maximum steps reached. Episode complete."

3. Session expires (30 min TTL)
   → Session destroyed
   → No grading (session lost)
   → Agent must call /reset to start over

Episode does NOT end on:
  • draft_report() (explicitly designed to not end episode)
  • Any error or invalid action
  • classify_incident() (agent may still want to adjust)
  • noop()
10. ML Infrastructure, Determinism & Risk Management
10.1 Model Inventory
text
MODEL                       SIZE     PURPOSE                  DETERMINISTIC
────────────────────────────────────────────────────────────────────────────
all-MiniLM-L6-v2           ~80MB    Semantic similarity       Yes (CPU only)
                                     Report evaluation
                                     Category matching
                                     Unknown category handling
TF-IDF Vectorizer           ~5MB    Search relevance scoring  Yes
(scikit-learn)
BM25 (rank-bm25)            ~2MB    Information retrieval     Yes
                                     Search quality scoring
────────────────────────────────────────────────────────────────────────────
TOTAL: ~87MB RAM
10.2 Performance Budget
text
STARTUP (one-time, at container start):
  Load MiniLM-L6-v2:              ~2.5 seconds, ~80MB RAM
  Load TF-IDF vectorizer:         ~0.1 seconds, ~5MB RAM
  Precompute category embeddings: ~0.3 seconds (12 categories)
  TOTAL STARTUP:                  ~2.9 seconds, ~87MB RAM

PER-RESET (once per episode):
  Generate/load task logs:        ~0.1 seconds
  Fit BM25 on task logs:          ~0.2 seconds, ~2MB RAM
  Precompute finding embeddings:  ~0.2 seconds (per task)
  Build TF-IDF index:             ~0.1 seconds
  TOTAL PER-RESET:               ~0.6 seconds

PER-STEP:
  Embed search query (if search):  ~5ms
  BM25 score:                      ~2ms
  Category similarity lookup:      ~1ms (precomputed matrix)
  Dynamic scaling computation:     ~1ms
  Strategy rule checks:            ~1ms
  TOTAL PER STEP:                 ~10ms

END-OF-EPISODE (grading):
  Embed report sentences:          ~50ms
  Cosine similarity matrix:        ~5ms
  Graph analysis (chain):          ~2ms
  Heuristic coherence scoring:     ~3ms
  TOTAL GRADING:                  ~60ms

TOTAL MEMORY FOOTPRINT:
  ML models (shared):              ~90MB
  Per session state:               ~5MB
  FastAPI overhead:                ~50MB
  Python runtime:                  ~100MB
  TOTAL:                          ~250MB (well within 8GB)
10.3 Model Bundling Strategy
text
CRITICAL: Models must be bundled IN the Docker image.
Do NOT download at runtime — HF Spaces may have network issues.

DOCKERFILE APPROACH:
  During docker build:
    1. pip install sentence-transformers
    2. Run a Python script that downloads and caches MiniLM-L6-v2
    3. Model saved to /app/models/minilm-l6-v2/
    4. At runtime, load from local path (no network needed)

  FROM python:3.11-slim
  ...
  RUN python -c "from sentence_transformers import SentenceTransformer; \
      m = SentenceTransformer('all-MiniLM-L6-v2'); \
      m.save('/app/models/minilm-l6-v2')"
  ...

  In code:
    model = SentenceTransformer('/app/models/minilm-l6-v2')
10.4 Determinism Guarantees
text
SOURCE                  RISK                     MITIGATION
────────────────────────────────────────────────────────────────────
MiniLM embeddings       Float variations         CPU-only, pinned version,
                                                 precomputed references
TF-IDF                  None                     Pure math
BM25                    None                     Pure math
Python sets             Hash seed varies         Sort before comparison
Float accumulation      Rounding drift           Round to 4 decimals
Text processing         Unicode edge cases       NFKC normalization
Sentence splitting      Tokenizer differences    Simple regex splitter
Random generation       Seed-dependent           Fixed seeds per task
Timestamp generation    Float precision          Millisecond precision,
                                                 integer arithmetic
10.5 Risk Analysis
text
RISK                              LIKELIHOOD   IMPACT    MITIGATION
────────────────────────────────────────────────────────────────────────

MiniLM gives weird similarity     Low          Medium    Cap at [0, 1].
scores on edge cases                                     Prevalidate matrix.

Model download fails during       Medium       Critical  Bundle in Docker.
Docker build

Embedding drift across library    Low          High      Pin exact versions
version updates                                          in requirements.txt.
                                                         sentence-transformers==2.7.0

Category embeddings not           Low          High      Precompute and verify
semantically meaningful                                  similarity matrix manually.

BM25 returns all zeros on         Medium       Low       Fallback: return 0.0
very short queries                                       search reward (neutral).

Memory spike during batch         Low          Medium    Process in batches of 20.
embedding of long report

Container runs out of memory      Low          Critical  Monitor memory in health
during hard task with 10 sessions                        endpoint. Limit to 10 sessions.

HF Space cold start too slow      Medium       Medium    Pre-load models at startup.
                                                         Health endpoint waits for ready.

Grader score differs across       Low          Critical  100-run determinism test
container restarts                                       in CI pipeline.


11. Observation Size Control & Context Management
11.1 The Problem
text
Without controls, observation at step 30 of hard task:
  20 logs × 400 chars     = 8,000 chars
  Dashboard                = 500 chars
  20+ annotations (full)   = 1,000 chars
  10+ correlations (full)  = 500 chars
  Draft report             = 1,000 chars
  Metadata                 = 300 chars
  ──────────────────────────────────
  TOTAL: ~11,300 chars ≈ ~3,000 tokens

With conversation history, this blows up the LLM context window.
11.2 Solutions Implemented
text
A. LOG TRUNCATION (default)
  Each log message in visible_logs: truncated to 200 chars
  Full entry available via inspect(log_id) action

B. WORK SUMMARY (not full dump)
  Instead of listing all annotations:
    annotations_count: 15
    recent_annotations: [last 5 only]
    annotations_by_category: {"error": 3, "brute_force": 2}

  Instead of listing all correlations:
    correlations_count: 8
    recent_correlations: [last 3 pairs only]

C. OBSERVATION VERBOSITY PARAMETER
  POST /step supports optional "verbosity" field:

    "compact":  Only visible logs + summary stats (~2,000 chars)
    "standard": Logs + work summary + feedback (~3,500 chars)
    "full":     Everything including full annotations list (~5,000+ chars)

  Default: "standard"
  inference.py uses "standard" to stay within token limits.

D. RESULTING SIZE TARGETS:
  compact:   ~2,000 chars (~500 tokens)
  standard:  ~3,500 chars (~900 tokens)
  full:      ~5,000 chars (~1,300 tokens)
12. Architecture & Deployment Design
12.1 System Architecture
text
┌──────────────────────────────────────────────────┐
│            Hugging Face Space                     │
│         (Docker Container / vcpu=2, 8GB)          │
│                                                   │
│  ┌────────────────────────────────────┐           │
│  │         FastAPI Server (app.py)    │           │
│  │                                    │           │
│  │  POST /reset  → create session     │           │
│  │  POST /step   → process action     │           │
│  │  GET  /state  → return state       │           │
│  │  GET  /health → status check       │           │
│  │  GET  /tasks  → list tasks         │           │
│  └──────────┬─────────────────────────┘           │
│             │                                     │
│  ┌──────────▼─────────────────────────┐           │
│  │   Session Manager                  │           │
│  │   Dict[session_id, LogTriageEnv]   │           │
│  │   TTL: 30min, Max: 10 sessions     │           │
│  └──────────┬─────────────────────────┘           │
│             │                                     │
│  ┌──────────▼─────────────────────────┐           │
│  │      LogTriageEnv (per session)    │           │
│  │                                    │           │
│  │  ┌───────────┐  ┌──────────────┐   │           │
│  │  │ Log       │  │ Reward       │   │           │
│  │  │ Generator │  │ Calculator   │   │           │
│  │  └───────────┘  └──────────────┘   │           │
│  │  ┌───────────┐  ┌──────────────┐   │           │
│  │  │ Task      │  │ 10D Grader   │   │           │
│  │  │ Config    │  │              │   │           │
│  │  └───────────┘  └──────────────┘   │           │
│  └────────────────────────────────────┘           │
│                                                   │
│  ┌────────────────────────────────────┐           │
│  │   Shared ML Models (loaded once)   │           │
│  │   MiniLM-L6-v2 | TF-IDF | BM25   │           │
│  │   ~90MB total                      │           │
│  └────────────────────────────────────┘           │
│                                                   │
└───────────────────────────────────────────────────┘
         ▲                      │
         │   HTTP / JSON        │
         │                      ▼
┌───────────────────────────────────────────────────┐
│         inference.py (Baseline Agent)              │
│         Uses OpenAI Client for LLM calls           │
│         HTTP client for environment interaction     │
│         Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN  │
└───────────────────────────────────────────────────┘
12.2 Dockerfile
dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache ML model
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('all-MiniLM-L6-v2'); \
m.save('/app/models/minilm-l6-v2')"

# Copy application code
COPY . .

# Precompute embeddings at build time
RUN python -c "\
from src.ml_models import precompute_all_embeddings; \
precompute_all_embeddings('/app/models/minilm-l6-v2', '/app/data/embeddings')"

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
12.3 Requirements.txt
text
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3
sentence-transformers==2.7.0
scikit-learn==1.4.0
rank-bm25==0.2.2
numpy==1.26.3
openai==1.12.0
requests==2.31.0
PyYAML==6.0.1
12.4 Resource Constraints Verification
text
MUST RUN ON: vcpu=2, memory=8GB

Memory breakdown:
  Python runtime + FastAPI:     ~100MB
  ML models (shared):           ~90MB
  10 concurrent sessions:       ~50MB
  OS overhead:                  ~200MB
  Safety buffer:                ~560MB
  TOTAL:                        ~1,000MB of 8,192MB
  HEADROOM:                     ~7,192MB  ✓

CPU:
  Per-step ML inference: ~10ms (no GPU needed)
  FastAPI request handling: ~2ms
  Log generation: ~100ms per reset
  Grading: ~60ms per episode end
  All well within 2 vCPU capacity ✓

Inference time budget (must complete in < 20 minutes):
  HF Space cold start:         ~30 seconds
  ML model warm-up:            ~5 seconds
  Easy task:  15 steps × 3.6s  = ~54 seconds
  Medium task: 25 steps × 3.6s = ~90 seconds
  Hard task:  40 steps × 3.6s  = ~144 seconds
  Buffer for retries:          ~60 seconds
  TOTAL:                       ~6.5 minutes  ✓  (well under 20 min)


13. Inference Script Design
13.1 Environment Client
text
inference.py must connect to the environment via HTTP.

class LogTriageClient:
    """HTTP client wrapping the FastAPI environment."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id = None

    def reset(self, task_id: str) -> dict:
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id}
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
                "params": params or {}
            }
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        resp = requests.get(
            f"{self.base_url}/state",
            params={"session_id": self.session_id}
        )
        resp.raise_for_status()
        return resp.json()

CONNECTION MODES:
  Local development:  SPACE_URL = "http://localhost:7860"
  Remote HF Space:    SPACE_URL = "https://your-space.hf.space"
  
  Reads from environment variable:
    SPACE_URL = os.getenv("SPACE_URL", "http://localhost:7860")
13.2 LLM Integration
text
REQUIRED ENVIRONMENT VARIABLES:
  API_BASE_URL   — The API endpoint for the LLM
  MODEL_NAME     — The model identifier
  HF_TOKEN       — Hugging Face / API key

LLM CLIENT:
  Uses OpenAI Client (as required by competition):

  from openai import OpenAI

  client = OpenAI(
      base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
      api_key=os.getenv("HF_TOKEN") or os.getenv("API_KEY")
  )

  MODEL_NAME = os.getenv("MODEL_NAME")

LLM CALL PARAMETERS:
  temperature: 0.2 (low for reproducibility)
  max_tokens: 300 (enough for action + reasoning)
  timeout per call: 30 seconds
  max retries: 2
  fallback on failure: noop()
13.3 Agent System Prompt
text
SYSTEM_PROMPT = """
You are an SRE investigating system logs during an incident.
You interact with a log investigation environment.

Reply with a JSON object containing your next action.
Format: {"action_type": "<type>", "params": {<params>}}

Available actions:
  NAVIGATION:
    {"action_type": "search", "params": {"pattern": "error keyword"}}
    {"action_type": "filter_severity", "params": {"level": "ERROR"}}
    {"action_type": "filter_service", "params": {"service": "auth-service"}}
    {"action_type": "clear_filters", "params": {}}
    {"action_type": "scroll", "params": {"direction": "down"}}

  INVESTIGATION:
    {"action_type": "inspect", "params": {"log_id": "log_042"}}
    {"action_type": "annotate", "params": {"log_id": "log_042", "category": "error"}}
    {"action_type": "correlate", "params": {"source_log_id": "log_012",
      "target_log_id": "log_042"}}

  CONCLUSION:
    {"action_type": "classify_incident", "params": {"severity": "HIGH"}}
    {"action_type": "draft_report", "params": {"summary": "..."}}
    {"action_type": "submit_report", "params": {"summary": "..."}}

Categories for annotation:
  Infrastructure: error, root_cause, symptom, cascading_failure, warning
  Security: reconnaissance, brute_force, credential_compromise,
    privilege_escalation, lateral_movement, data_exfiltration, persistence

Investigation strategy:
  1. Start by exploring: search and filter to find relevant logs
  2. Investigate: inspect suspicious logs, annotate anomalies
  3. Correlate: link related events (source caused target)
  4. Conclude: classify severity and submit a detailed report

Your report should include:
  - What happened (root cause)
  - What was affected (services, impact)
  - Timeline of events
  - Severity justification

Reply ONLY with the JSON action object. No explanations.
"""
13.4 Agent Action Parsing
text
def parse_agent_action(response_text: str) -> Tuple[str, dict]:
    """Parse LLM response into action_type and params."""
    
    # Try JSON parsing first
    try:
        action = json.loads(response_text.strip())
        return action["action_type"], action.get("params", {})
    except (json.JSONDecodeError, KeyError):
        pass

    # Try extracting JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', 
                           response_text, re.DOTALL)
    if json_match:
        try:
            action = json.loads(json_match.group(1))
            return action["action_type"], action.get("params", {})
        except (json.JSONDecodeError, KeyError):
            pass

    # Try extracting any JSON object from text
    json_match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', response_text)
    if json_match:
        try:
            action = json.loads(json_match.group(0))
            return action["action_type"], action.get("params", {})
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback
    return "noop", {}
13.5 Agent Main Loop
text
def run_task(client, env, task_id, llm_client, model_name):
    """Run one task and return the grader result."""

    result = env.reset(task_id)
    observation = result["observation"]
    history = []

    for step in range(1, observation["max_steps"] + 1):
        # Build user prompt from observation
        user_prompt = format_observation(observation, history)

        # Call LLM
        try:
            completion = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            response = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"LLM call failed: {e}")
            response = '{"action_type": "noop", "params": {}}'

        # Parse action
        action_type, params = parse_agent_action(response)

        # Execute step
        step_result = env.step(action_type, params)

        observation = step_result["observation"]
        reward = step_result["reward"]
        done = step_result["done"]

        history.append(f"Step {step}: {action_type}({params}) → "
                      f"reward={reward['value']:+.3f}")

        if done:
            grader_result = step_result.get("info", {}).get("grader_result")
            return grader_result

    return step_result.get("info", {}).get("grader_result")


def main():
    """Run all 3 tasks and report scores."""
    
    env = LogTriageClient(
        os.getenv("SPACE_URL", "http://localhost:7860")
    )
    llm_client = OpenAI(
        base_url=os.getenv("API_BASE_URL"),
        api_key=os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    )
    model = os.getenv("MODEL_NAME")

    results = {}
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        print(f"\n{'='*50}")
        print(f"Running: {task_id}")
        print(f"{'='*50}")

        grader_result = run_task(env, task_id, llm_client, model)
        results[task_id] = grader_result

        print(f"Final Score: {grader_result['final_score']:.4f}")
        for comp, data in grader_result["components"].items():
            print(f"  {comp}: {data['score']:.4f} "
                  f"(×{data['weight']:.0%} = {data['weighted']:.4f})")

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for task_id, result in results.items():
        print(f"  {task_id}: {result['final_score']:.4f}")
    avg = sum(r["final_score"] for r in results.values()) / 3
    print(f"  AVERAGE: {avg:.4f}")


14. Documentation Requirements (README)
14.1 Required Structure
markdown
# LogTriage — SRE Log Investigation Environment

## Overview
What this environment does and why it matters.
The SRE incident investigation narrative.

## Quick Start
  docker build -t logtriage .
  docker run -p 7860:7860 logtriage
  python inference.py

## Environment Description
What the agent investigates.
The three incident scenarios.
7 cognitive dimensions tested.

## Action Space
Complete table of all 15 actions with parameters, types, examples.

## Observation Space
Field-by-field documentation of the observation object.
Size targets and verbosity modes.

## Category Taxonomy
All 12 predefined categories with descriptions.
Infrastructure vs Security categories.

## Tasks
| Task | Difficulty | Logs | Services | Steps | Description |
With expected score ranges per agent class.

## Reward Design
High-level explanation of hybrid ML+rules approach.
Dense signal philosophy.
Anti-exploit measures.
DO NOT reveal ground truth details.

## Grading System
10 components with weights per task.
Scoring methodology overview.
Theoretical maximum scores.

## Baseline Scores
| Task       | Score  | Steps Used | Time  |
| task_easy  | 0.XX   | YY/15      | ZZs   |
| task_medium| 0.XX   | YY/25      | ZZs   |
| task_hard  | 0.XX   | YY/40      | ZZs   |
| AVERAGE    | 0.XX   |            |       |

## Setup Instructions
Environment variables needed.
How to run locally vs against HF Space.
Docker instructions.

## API Reference
All endpoints with request/response examples.

## Technical Details
ML models used (MiniLM, BM25, TF-IDF).
Resource requirements (CPU, memory).
Determinism guarantees.

## License
14.2 Content Guidelines
text
DO:
  • Explain the domain motivation clearly
  • Show example agent trajectories
  • Document all action parameters with examples
  • Include actual baseline scores (not estimates)
  • Explain reward philosophy at a high level

DO NOT:
  • Reveal ground truth annotations or log IDs
  • Reveal exact reward formula multipliers
  • Reveal exact grading weights (show ranges)
  • Include API keys or secrets
  • Over-claim capability ("state of the art benchmark")
15. Score Calibration Strategy
15.1 Calibration Protocol
text
PHASE 1: INTERNAL TESTING (before baseline agent)
  Run grader on 10 hand-crafted agent outputs:
    2 × "perfect" outputs (should score 0.90+)
    2 × "good" outputs (should score 0.60–0.80)
    2 × "mediocre" outputs (should score 0.25–0.45)
    2 × "poor" outputs (should score 0.05–0.15)
    2 × "random" outputs (should score 0.00–0.05)
  
  If scores don't match expected ranges → adjust weights.

PHASE 2: BASELINE AGENT TESTING
  Run inference.py on all 3 tasks, 5 times each.
  Record mean and standard deviation.
  
  Expected ranges:
    Easy:   0.60 – 0.85 mean, σ < 0.05
    Medium: 0.30 – 0.55 mean, σ < 0.08
    Hard:   0.10 – 0.35 mean, σ < 0.10

  IF easy < 0.50:
    → Simplify easy task (fewer logs, more obvious errors)
    → OR relax grader weights (increase efficiency weight)
  
  IF hard > 0.50:
    → Add more noise to hard task
    → OR add more distractors
    → OR tighten precision requirements
  
  IF σ > 0.15 on any task:
    → Investigate non-determinism source
    → Check LLM temperature setting
    → Verify environment seed determinism

PHASE 3: REWARD-GRADER ALIGNMENT CHECK
  For all 15 baseline runs:
    Plot cumulative_step_reward vs final_grader_score
    Calculate Pearson correlation coefficient
    
    REQUIRED: r > 0.85
    
    If r < 0.70:
      → Step rewards are misleading the agent
      → Adjust reward component weights
      → Re-run and re-check

PHASE 4: EXPLOIT TESTING
  Run specific exploit strategies:
    1. "Annotate everything" agent
    2. "Submit immediately" agent
    3. "Only search, never annotate" agent
    4. "Repeat same action" agent
    5. "Random actions" agent
    
  ALL exploit strategies must score < 0.20 on every task.
  If any scores higher → strengthen exploit guards.

PHASE 5: DETERMINISM VERIFICATION
  Run exact same agent on exact same task 10 times.
  ALL 10 runs must produce IDENTICAL grader scores.
  ALL 10 runs must produce IDENTICAL step reward sequences.
  
  Note: LLM outputs may vary (temperature > 0) so we test
  determinism with FIXED action sequences, not LLM-driven.
15.2 Adjustment Levers
text
If scores are too LOW overall:
  → Reduce distractor count
  → Make ground truth logs more distinct (higher severity, unique messages)
  → Increase step budget
  → Relax report coherence thresholds
  → Increase weight on recall (reward finding things)

If scores are too HIGH overall:
  → Add more distractors
  → Make ground truth logs more subtle (lower severity, blending messages)
  → Decrease step budget
  → Tighten precision requirements
  → Add more noise to log generation

If score VARIANCE is too high:
  → Reduce LLM temperature to 0.0
  → Simplify action parsing (fewer failure modes)
  → Add more deterministic fallback behavior
  
16. Implementation Plan
16.1 Phase Schedule
text
PHASE 1: FOUNDATION (Day 1)
────────────────────────────
□ Define all Pydantic models
    Observation, Action, Reward, LogEntry, GraderResult
    12 categories with descriptions
□ Create openenv.yaml
□ Build FastAPI skeleton WITH session management
□ Define all API error responses
□ Write Dockerfile (without ML model yet)
□ VERIFY: docker build works, /health returns 200

PHASE 2: LOG GENERATION (Day 1–2)
─────────────────────────────────
□ Design log message templates (Java/Spring Boot format)
□ Build timestamp generator (jitter, monotonic, seeded)
□ Build background noise generator (normal operations)
□ Build signal generator (ground truth incident events)
□ Build distractor generator (false-positive errors)
□ Implement seeded reproducibility (SHA-256 task hash → seed)
□ Generate all 3 task datasets
□ VERIFY: logs look realistic, 100 generations produce identical hashes

PHASE 3: CORE ENVIRONMENT (Day 2)
──────────────────────────────────
□ Implement LogTriageEnv class with full state management
□ Implement all 15 action handlers:
    search, filter_severity, filter_service, filter_time_range,
    clear_filters, clear_filter, scroll, inspect,
    annotate, correlate, classify_incident,
    draft_report, submit_report, noop
□ Implement filter stacking with AND logic
□ Implement all error cases (Section 9)
□ Implement observation builder with truncation and summarization
□ Implement step() / reset() / state()
□ VERIFY: manual integration test through full episode

PHASE 4: ML INTEGRATION (Day 2–3)
──────────────────────────────────
□ Bundle MiniLM-L6-v2 in Docker image
□ Precompute category description embeddings (12 categories)
□ Precompute and store similarity matrix as numpy array
□ Precompute finding description embeddings (per task)
□ Build BM25 index per task (at reset time)
□ Build TF-IDF index per task (at reset time)
□ Implement ML-enhanced reward components
□ VERIFY: all ML outputs deterministic (100-run test)

PHASE 5: REWARD SYSTEM (Day 3)
──────────────────────────────
□ Implement RewardCalculator with all dynamic scaling:
    information gain, temporal, difficulty, coverage-aware,
    informed annotation, chain detection, precision-aware penalty,
    trajectory alignment, strategy rules
□ Implement reward caps (per-action, per-step, per-episode)
□ Unit test: verify reward trajectories for known sequences
□ VERIFY: smart actions get positive signal, bad actions get negative

PHASE 6: GRADER SYSTEM (Day 3)
──────────────────────────────
□ Implement all 10 grading components:
    A–B (rule-based counting)
    C (ML category similarity)
    D–E (rule-based + transitive matching)
    F (graph analysis — chain reconstruction)
    G (severity distance scoring)
    H (ML report completeness — sentence embeddings)
    I (heuristic report coherence — 5 sub-metrics)
    J (quality-adjusted efficiency)
□ Implement per-task weight adjustment
□ Implement draft report fallback logic
□ Implement all edge cases (Section 7.8)
□ Implement grader output format (JSON)
□ Unit test: 10 fixed agent outputs × 100 runs = identical scores
□ VERIFY: score ranges match expectations for each agent quality level

PHASE 6.5: CALIBRATION (Day 3–4)
─────────────────────────────────
□ Run grader on 10 hand-crafted agent outputs
□ Verify score ranges match Section 15.1 Phase 1 expectations
□ Adjust weights if needed
□ Test 5 exploit strategies, verify all score < 0.20
□ Verify reward-grader alignment (Pearson r > 0.85)
□ VERIFY: no exploits, no misalignment, deterministic

PHASE 7: BASELINE AGENT (Day 4)
────────────────────────────────
□ Write LogTriageClient HTTP wrapper
□ Design system prompt (Section 13.3)
□ Implement action parsing with fallbacks (Section 13.4)
□ Implement observation formatting for LLM context
□ Implement main loop with error handling
□ Run against all 3 tasks, 5 times each
□ Record baseline scores
□ Verify: scores in expected ranges, σ < thresholds
□ VERIFY: inference.py completes in < 20 minutes total

PHASE 8: DEPLOYMENT (Day 4–5)
─────────────────────────────
□ Finalize Dockerfile with ML model bundling
□ Deploy to Hugging Face Space
□ Tag with "openenv"
□ Test from external machine (not localhost)
□ Run openenv validate
□ Full end-to-end test: inference.py → remote HF Space → scores
□ VERIFY: Space returns 200, reset/step/state all work remotely

PHASE 9: DOCUMENTATION (Day 5)
──────────────────────────────
□ Write full README (Section 14 structure)
□ Include actual baseline scores table (not estimates)
□ Document all action/observation spaces
□ Document reward design (high-level, no GT leakage)
□ Include example agent trajectories
□ Final review against competition checklist
□ VERIFY: README covers all required sections
16.2 Testing Strategy
text
UNIT TESTS:
  test_log_generator.py
    • Reproducibility (100 runs, same hash)
    • Correct log count per task
    • Ground truth logs at expected positions
    • Timestamps monotonically increasing
    • All severity levels present in expected proportions

  test_environment.py
    • reset() produces valid observation
    • All 15 actions return valid responses
    • Invalid actions handled gracefully
    • Filter stacking works correctly
    • Episode terminates at max_steps
    • submit_report ends episode
    • draft_report does NOT end episode
    • State is clean after reset

  test_reward.py
    • Correct annotations get positive reward
    • Wrong annotations get negative reward
    • Precision penalty scaling works
    • Coverage milestones trigger correctly
    • Chain bonuses trigger at correct lengths
    • Reward caps enforced
    • Trajectory alignment bonus is non-negative
    • Informed vs blind annotation multiplier works

  test_grader.py
    • Each component scores 0.0–1.0
    • Known perfect inputs score near 1.0
    • Known terrible inputs score near 0.0
    • Per-task weights sum to 1.0
    • Draft fallback applies 20% penalty
    • Transitive correlation recall works
    • Chain reconstruction handles DAGs correctly
    • Severity distance scoring is correct
    • Determinism: 100 identical runs

  test_ml_models.py
    • Category similarity matrix is symmetric
    • Diagonal values are 1.0
    • Off-diagonal values are in [0, 1]
    • Similar categories have higher scores
    • Embedding reproducibility across restarts

  test_session.py
    • Concurrent sessions don't interfere
    • Session TTL expiry works
    • Max session limit enforced
    • Invalid session ID returns 404

INTEGRATION TESTS:
  test_full_episode.py
    • Run a scripted agent through all 3 tasks
    • Verify observation format at each step
    • Verify reward accumulation
    • Verify grader output format
    • Verify end-to-end determinism

  test_api.py
    • All endpoints return correct status codes
    • Error responses have correct format
    • Health endpoint always works
    • Tasks endpoint lists all 3 tasks

EXPLOIT TESTS:
  test_exploits.py
    • "Annotate everything" scores < 0.20
    • "Submit immediately" scores < 0.05
    • "Only search" scores < 0.10
    • "Repeat same action" scores < 0.10
    • "Random actions" scores < 0.05
17. Pre-Submission Validation Checklist
17.1 Automated Checks (All Must Pass)
text
□ HF Space deploys and returns 200 on GET /health
□ POST /reset returns valid observation with session_id
□ POST /step returns observation, reward, done, info
□ GET /state returns current state
□ GET /tasks lists 3 tasks
□ openenv.yaml validates against OpenEnv spec
□ docker build completes without errors
□ docker run starts server and responds to /health within 30 seconds
□ inference.py runs and produces scores for all 3 tasks
□ inference.py completes in under 20 minutes
□ All grader scores are in 0.0–1.0 range
□ 3 tasks enumerated in openenv.yaml
□ Typed Pydantic models for Observation, Action, Reward
17.2 Determinism Checks
text
□ Same fixed action sequence × 10 runs = identical rewards (all 10)
□ Same fixed action sequence × 10 runs = identical grader scores (all 10)
□ Log generation × 100 runs = identical hash (all 100)
□ ML embeddings × 100 runs = identical float arrays (all 100)
17.3 Quality Checks
text
□ Easy task baseline: 0.60–0.85
□ Medium task baseline: 0.30–0.55
□ Hard task baseline: 0.10–0.35
□ All exploit strategies score < 0.20
□ Reward-grader correlation > 0.85
□ Logs look realistic to human reviewer
□ README covers all required sections
□ No hardcoded API keys