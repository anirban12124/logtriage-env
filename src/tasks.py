CATEGORY_DESCRIPTIONS = {
    "error": "A system error or failure in normal operation",
    "root_cause": "The original underlying cause that triggered the incident",
    "symptom": "A visible effect or consequence of an underlying problem",
    "cascading_failure": "A failure in one system that propagates to dependent systems",
    "warning": "An anomaly that hasn't caused failure yet",
    "reconnaissance": "Initial probing or scanning to gather information about target",
    "brute_force": "Repeated automated attempts to guess credentials",
    "credential_compromise": "Successful unauthorized access using stolen credentials",
    "privilege_escalation": "Gaining higher access permissions than authorized",
    "lateral_movement": "Moving from one compromised system to another within network",
    "data_exfiltration": "Unauthorized extraction or theft of data from the system",
    "persistence": "Establishing ongoing unauthorized access to maintain foothold",
}

_SIMILARITY_MAP = {
    ("error", "root_cause"): 0.52, ("error", "symptom"): 0.65,
    ("error", "cascading_failure"): 0.58, ("error", "warning"): 0.70,
    ("root_cause", "symptom"): 0.68, ("root_cause", "cascading_failure"): 0.55,
    ("symptom", "cascading_failure"): 0.72, ("symptom", "warning"): 0.58,
    ("reconnaissance", "brute_force"): 0.62, ("reconnaissance", "credential_compromise"): 0.55,
    ("brute_force", "credential_compromise"): 0.72, ("brute_force", "privilege_escalation"): 0.50,
    ("credential_compromise", "privilege_escalation"): 0.65,
    ("credential_compromise", "lateral_movement"): 0.48,
    ("privilege_escalation", "lateral_movement"): 0.55,
    ("privilege_escalation", "data_exfiltration"): 0.52,
    ("lateral_movement", "data_exfiltration"): 0.58,
    ("lateral_movement", "persistence"): 0.50,
    ("data_exfiltration", "persistence"): 0.42,
}


def get_category_similarity(cat_a: str, cat_b: str) -> float:
    if cat_a == cat_b:
        return 1.0
    pair = tuple(sorted([cat_a, cat_b]))
    return _SIMILARITY_MAP.get(pair, 0.15)


TASKS = {
    "task_easy": {
        "id": "task_easy",
        "name": "Database Connection Failures",
        "difficulty": "easy",
        "max_steps": 15,
        "log_count": 50,
        "services": ["auth-service"],
        "goal": (
            "Review the auth-service logs. Identify all database connection "
            "errors, annotate them with the category 'error', classify the "
            "incident severity, and submit an incident report summarizing "
            "the database connectivity issues."
        ),
        "ground_truth": {
            "annotations": {
                "log_008": "error",
                "log_023": "error",
                "log_041": "error",
            },
            "correlations": [],
            "severity": "MEDIUM",
            "key_findings": [
                "database connection refused",
                "auth-service affected",
                "postgresql port 5432",
            ],
        },
        "grader_weights": {
            "annotation_precision": 0.15,
            "annotation_recall": 0.15,
            "annotation_quality": 0.05,
            "correlation_precision": 0.02,
            "correlation_recall": 0.02,
            "chain_reconstruction": 0.00,
            "severity_classification": 0.15,
            "report_completeness": 0.18,
            "report_coherence": 0.13,
            "investigation_efficiency": 0.15,
        },
    },
    "task_medium": {
        "id": "task_medium",
        "name": "Cascading Service Failure",
        "difficulty": "medium",
        "max_steps": 25,
        "log_count": 200,
        "services": ["api-gateway", "order-service", "payment-service"],
        "goal": (
            "Investigate logs from api-gateway, order-service, and payment-service. "
            "A cascading failure started in one service and propagated. Find all "
            "errors, identify the root cause, correlate related events across "
            "services (specify which event caused which), classify severity, "
            "and submit a root cause analysis report."
        ),
        "ground_truth": {
            "annotations": {
                "log_045": "root_cause",
                "log_067": "symptom",
                "log_089": "symptom",
                "log_102": "symptom",
                "log_134": "cascading_failure",
                "log_156": "cascading_failure",
            },
            "correlations": [
                ["log_045", "log_067"],
                ["log_067", "log_134"],
                ["log_089", "log_156"],
            ],
            "severity": "HIGH",
            "key_findings": [
                "payment service database pool exhausted",
                "order service queue backup",
                "api gateway 503 errors",
                "cascading failure from payment to gateway",
            ],
        },
        "grader_weights": {
            "annotation_precision": 0.12,
            "annotation_recall": 0.12,
            "annotation_quality": 0.08,
            "correlation_precision": 0.10,
            "correlation_recall": 0.10,
            "chain_reconstruction": 0.08,
            "severity_classification": 0.10,
            "report_completeness": 0.12,
            "report_coherence": 0.08,
            "investigation_efficiency": 0.10,
        },
    },
    "task_hard": {
        "id": "task_hard",
        "name": "Security Breach Investigation",
        "difficulty": "hard",
        "max_steps": 40,
        "log_count": 500,
        "services": [
            "auth-service", "api-gateway", "user-service",
            "file-service", "audit-log",
        ],
        "goal": (
            "Investigate a potential security breach across 5 services. "
            "Logs contain normal errors mixed with subtle attack indicators. "
            "Identify the attack stages (reconnaissance, brute force, credential "
            "compromise, privilege escalation, lateral movement, data exfiltration), "
            "correlate the attack chain (which event caused the next), classify "
            "severity, and submit a detailed security incident report."
        ),
        "ground_truth": {
            "annotations": {
                "log_023": "reconnaissance",
                "log_078": "brute_force",
                "log_091": "brute_force",
                "log_145": "credential_compromise",
                "log_201": "privilege_escalation",
                "log_267": "data_exfiltration",
                "log_312": "lateral_movement",
                "log_389": "data_exfiltration",
            },
            "correlations": [
                ["log_023", "log_078"],
                ["log_091", "log_145"],
                ["log_145", "log_201"],
                ["log_201", "log_267"],
                ["log_201", "log_312"],
                ["log_312", "log_389"],
            ],
            "severity": "CRITICAL",
            "key_findings": [
                "brute force login attempts",
                "privilege escalation via admin api",
                "data exfiltration of customer records",
                "lateral movement to file service",
                "compromised credentials",
                "unauthorized admin access",
            ],
        },
        "grader_weights": {
            "annotation_precision": 0.10,
            "annotation_recall": 0.10,
            "annotation_quality": 0.10,
            "correlation_precision": 0.12,
            "correlation_recall": 0.12,
            "chain_reconstruction": 0.10,
            "severity_classification": 0.08,
            "report_completeness": 0.10,
            "report_coherence": 0.08,
            "investigation_efficiency": 0.10,
        },
    },
}