import random
import hashlib
from typing import List, Dict


def _seed(task_id: str) -> int:
    return int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)


def _fmt_ts(seconds_from_midnight: int, ms: int) -> str:
    h = seconds_from_midnight // 3600
    m = (seconds_from_midnight % 3600) // 60
    s = seconds_from_midnight % 60
    return f"2024-01-15T{h:02d}:{m:02d}:{s:02d}.{ms:03d}Z"


# ─── Background templates ───────────────────────────────────────

_BG_INFO = [
    ("User login successful", {"user_id": "u_{uid}", "source_ip": "10.0.{a}.{b}"}),
    ("Health check passed: all dependencies healthy", {"endpoint": "/health"}),
    ("Token refreshed for session", {"session_id": "sess-{sid}"}),
    ("Request completed: GET /api/v1/users in {ms}ms", {"status": "200"}),
    ("Cache hit for key user_profile_{uid}", {"hit_rate": "{hr}%"}),
    ("Scheduled cleanup job completed", {"rows_deleted": "{rd}"}),
    ("Connection pool stats: active={pa}/{pm} idle={pi}", {}),
    ("Outbound request to downstream service completed", {"latency_ms": "{ms}"}),
    ("Configuration reloaded successfully", {"properties": "{pc}"}),
    ("Audit event recorded", {"action": "read", "resource": "users"}),
]

_BG_DEBUG = [
    ("Entering method processRequest", {}),
    ("SQL query executed in {ms}ms", {"query": "SELECT * FROM sessions"}),
    ("Cache lookup for key config_v{v}", {}),
    ("Thread pool utilization: {tp}%", {}),
]

_BG_WARN = [
    ("Slow query detected: {ms}ms for SELECT on users table", {"threshold": "1000ms"}),
    ("Connection pool utilization high: {pct}%", {"pool": "HikariPool-1"}),
    ("Response time elevated: {ms}ms (p99 threshold: 500ms)", {}),
    ("Retry attempt {n}/3 for downstream call", {"target": "cache-service"}),
]

_DISTRACTORS = [
    ("WARN", "DNS resolution timeout for metrics.internal (retried OK)", {"resolver": "10.0.0.2"}),
    ("WARN", "Disk usage at {pct}% on /var/log partition", {"host": "prod-{h}"}),
    ("WARN", "TLS certificate expiring in 14 days for api.example.com", {}),
    ("WARN", "JVM garbage collection pause: {ms}ms", {"gc_type": "G1 Young"}),
    ("ERROR", "Connection reset by peer: 10.0.5.22:3306", {"retry": "true"}),
    ("WARN", "Slow query detected: {ms}ms for INSERT on audit_log", {}),
    ("ERROR", "Health check timeout for cache-service (recovered)", {"downtime_ms": "{ms}"}),
    ("WARN", "Thread pool near capacity: {pct}% utilization", {}),
    ("ERROR", "SMTP connection timeout for alert-mailer", {"retry_in": "30s"}),
    ("WARN", "Rate limit threshold reached: {n}/1000 requests", {}),
    ("ERROR", "Backup job completed with warnings", {"warnings": "2"}),
    ("WARN", "Memory usage elevated: {pct}% of heap", {"heap_max": "4096MB"}),
    ("ERROR", "Log rotation failed: permission denied on /var/log/app.log", {}),
    ("WARN", "Stale NFS handle detected for shared mount", {}),
    ("ERROR", "Config reload partial failure: 1 property skipped", {}),
]

# ─── Ground truth templates per task ────────────────────────────

_GT_EASY = {
    8: {
        "severity": "ERROR",
        "service": "auth-service",
        "message": "Failed to acquire connection to db-primary:5432. "
                   "java.sql.SQLException: Connection refused. "
                   "Retry attempt 1/3 failed.",
        "metadata": {"db_host": "db-primary.internal", "port": "5432",
                      "pool_active": "48", "pool_max": "50"},
    },
    23: {
        "severity": "ERROR",
        "service": "auth-service",
        "message": "Failed to acquire connection to db-primary:5432. "
                   "java.sql.SQLException: Connection refused. "
                   "Circuit breaker OPEN after 3 failures.",
        "metadata": {"db_host": "db-primary.internal", "port": "5432",
                      "pool_active": "50", "pool_max": "50"},
    },
    41: {
        "severity": "ERROR",
        "service": "auth-service",
        "message": "Database connection pool exhausted. "
                   "java.sql.SQLException: Cannot acquire connection from pool. "
                   "All 50 connections in use, 12 requests queued.",
        "metadata": {"db_host": "db-primary.internal", "port": "5432",
                      "pool_active": "50", "pool_max": "50", "queue_depth": "12"},
    },
}

_GT_MEDIUM = {
    45: {
        "severity": "ERROR",
        "service": "payment-service",
        "message": "Connection pool exhausted: 50/50 connections in use. "
                   "All acquire attempts timing out after 30000ms. "
                   "Transactions failing.",
        "metadata": {"pool_active": "50", "pool_max": "50",
                      "timeout_ms": "30000", "pending_tx": "23"},
    },
    67: {
        "severity": "ERROR",
        "service": "payment-service",
        "message": "Transaction timeout after 30200ms waiting for database "
                   "connection. Payment processing failed for order ord-8812.",
        "metadata": {"order_id": "ord-8812", "timeout_ms": "30200"},
    },
    89: {
        "severity": "ERROR",
        "service": "payment-service",
        "message": "Request queue at capacity: 500/500. "
                   "Rejecting new payment requests. Service degraded.",
        "metadata": {"queue_depth": "500", "queue_max": "500",
                      "rejected_count": "45"},
    },
    102: {
        "severity": "ERROR",
        "service": "order-service",
        "message": "Timeout waiting for payment-service response. "
                   "POST /api/v1/payments timed out after 60000ms. "
                   "Order ord-9923 stuck in PENDING.",
        "metadata": {"order_id": "ord-9923", "timeout_ms": "60000",
                      "target": "payment-service"},
    },
    134: {
        "severity": "FATAL",
        "service": "order-service",
        "message": "Message queue depth critical: 15000 messages pending "
                   "(threshold: 5000). Consumer lag increasing. "
                   "Order processing stalled.",
        "metadata": {"queue_depth": "15000", "threshold": "5000",
                      "consumer_lag_ms": "45000"},
    },
    156: {
        "severity": "ERROR",
        "service": "api-gateway",
        "message": "HTTP 503 Service Unavailable returned to client. "
                   "Downstream dependency order-service not responding. "
                   "Circuit breaker OPEN.",
        "metadata": {"downstream": "order-service", "status": "503",
                      "circuit_breaker": "OPEN", "client_ip": "203.0.113.45"},
    },
}

_GT_HARD = {
    23: {
        "severity": "WARN",
        "service": "auth-service",
        "message": "Unusual connection pattern detected from 198.51.100.23. "
                   "Sequential probing of ports 22, 80, 443, 8080, 8443, 5432 "
                   "within 30 seconds.",
        "metadata": {"source_ip": "198.51.100.23", "ports_probed": "22,80,443,8080,8443,5432",
                      "geo": "RO", "user_agent": "python-requests/2.28.0"},
    },
        78: {
        "severity": "WARN",
        "service": "auth-service",
        "message": "Failed login attempt for user 'admin' from 198.51.100.23. "
                   "Attempt count: 47. Automated pattern detected. "
                   "User-Agent: python-requests/2.28.0",
        "metadata": {"source_ip": "198.51.100.23", "username": "admin",
                      "attempt_count": "47", "geo": "RO",
                      "user_agent": "python-requests/2.28.0"},
    },
    91: {
        "severity": "WARN",
        "service": "auth-service",
        "message": "Failed login attempt for user 'svc-deploy' from 198.51.100.23. "
                   "Attempt count: 23. Known service account targeted. "
                   "User-Agent: python-requests/2.28.0",
        "metadata": {"source_ip": "198.51.100.23", "username": "svc-deploy",
                      "attempt_count": "23", "geo": "RO",
                      "user_agent": "python-requests/2.28.0"},
    },
    145: {
        "severity": "INFO",
        "service": "auth-service",
        "message": "Successful authentication for user 'svc-deploy' from "
                   "198.51.100.23. Note: 91 prior failed attempts from this "
                   "IP in last 2 hours. Session token issued.",
        "metadata": {"source_ip": "198.51.100.23", "username": "svc-deploy",
                      "prior_failures": "91", "geo": "RO",
                      "session_id": "sess-x7k2m9"},
    },
    201: {
        "severity": "WARN",
        "service": "api-gateway",
        "message": "User 'svc-deploy' accessed admin endpoint "
                   "GET /api/admin/users without ADMIN role. "
                   "Request allowed by legacy policy. Review required.",
        "metadata": {"source_ip": "198.51.100.23", "username": "svc-deploy",
                      "endpoint": "/api/admin/users", "role": "SERVICE",
                      "policy": "legacy-allow"},
    },
    267: {
        "severity": "INFO",
        "service": "user-service",
        "message": "Bulk data export: 15,247 user records downloaded by "
                   "user 'svc-deploy'. Export size: 48.3MB. "
                   "Includes PII fields: email, phone, address.",
        "metadata": {"username": "svc-deploy", "records": "15247",
                      "size_mb": "48.3", "pii_fields": "email,phone,address",
                      "source_ip": "198.51.100.23"},
    },
    312: {
        "severity": "INFO",
        "service": "file-service",
        "message": "Authentication token sess-x7k2m9 used from new service "
                   "context: file-service. Original auth on auth-service. "
                   "Cross-service token reuse detected.",
        "metadata": {"session_id": "sess-x7k2m9", "username": "svc-deploy",
                      "origin_service": "auth-service",
                      "current_service": "file-service",
                      "source_ip": "198.51.100.23"},
    },
    389: {
        "severity": "INFO",
        "service": "file-service",
        "message": "Bulk file download: 342 files from /confidential/reports/ "
                   "by user 'svc-deploy'. Total size: 2.1GB. "
                   "Download completed at 03:47 UTC.",
        "metadata": {"username": "svc-deploy", "file_count": "342",
                      "directory": "/confidential/reports/",
                      "size_gb": "2.1", "source_ip": "198.51.100.23",
                      "time": "03:47"},
    },
}


def _fill_template(text: str, rng: random.Random) -> str:
    replacements = {
        "{uid}": str(rng.randint(10000, 99999)),
        "{sid}": f"{rng.randint(1000, 9999):04x}",
        "{ms}": str(rng.randint(50, 3500)),
        "{hr}": str(rng.randint(60, 99)),
        "{rd}": str(rng.randint(0, 500)),
        "{pa}": str(rng.randint(5, 45)),
        "{pm}": "50",
        "{pi}": str(rng.randint(5, 30)),
        "{pc}": str(rng.randint(1, 8)),
        "{v}": str(rng.randint(1, 20)),
        "{tp}": str(rng.randint(30, 95)),
        "{pct}": str(rng.randint(70, 95)),
        "{n}": str(rng.randint(1, 3)),
        "{h}": f"node-{rng.randint(1,12):02d}",
        "{a}": str(rng.randint(0, 255)),
        "{b}": str(rng.randint(1, 254)),
    }
    result = text
    for key, val in replacements.items():
        result = result.replace(key, val)
    return result


def _fill_meta(meta: dict, rng: random.Random) -> dict:
    filled = {}
    for k, v in meta.items():
        filled[k] = _fill_template(v, rng)
    return filled


def _pick_severity(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.10:
        return "DEBUG"
    elif roll < 0.75:
        return "INFO"
    elif roll < 0.92:
        return "WARN"
    else:
        return "ERROR"


def generate_logs(task_id: str) -> list:
    from src.tasks import TASKS
    task = TASKS[task_id]
    rng = random.Random(_seed(task_id))
    total = task["log_count"]
    services = task["services"]

    if task_id == "task_easy":
        gt_templates = _GT_EASY
        distractor_count = 0
    elif task_id == "task_medium":
        gt_templates = _GT_MEDIUM
        distractor_count = 5
    else:
        gt_templates = _GT_HARD
        distractor_count = 15

    gt_positions = set(gt_templates.keys())
    logs = []

    # Pick distractor positions (not overlapping with GT)
    available_positions = [i for i in range(1, total + 1) if i not in gt_positions]
    distractor_positions = set(rng.sample(available_positions, min(distractor_count, len(available_positions))))

    # Generate each log
    current_time = 32400  # 09:00 in seconds from midnight
    if task_id == "task_hard":
        current_time = 7200  # 02:00 for security scenario

    for i in range(1, total + 1):
        log_id = f"log_{i:03d}"
        ms = rng.randint(0, 999)

        # Advance timestamp
        gap = rng.randint(2, 30)
        current_time += gap

        if i in gt_positions:
            # Ground truth entry
            gt = gt_templates[i]
            entry = {
                "id": log_id,
                "timestamp": _fmt_ts(current_time, ms),
                "service": gt["service"],
                "severity": gt["severity"],
                "message": gt["message"],
                "metadata": gt.get("metadata", {}),
            }
        elif i in distractor_positions:
            # Distractor entry
            dist = rng.choice(_DISTRACTORS)
            sev, msg_tpl, meta_tpl = dist
            entry = {
                "id": log_id,
                "timestamp": _fmt_ts(current_time, ms),
                "service": rng.choice(services),
                "severity": sev,
                "message": _fill_template(msg_tpl, rng),
                "metadata": _fill_meta(meta_tpl, rng),
            }
        else:
            # Background noise
            severity = _pick_severity(rng)
            service = rng.choice(services)

            if severity == "DEBUG":
                tpl = rng.choice(_BG_DEBUG)
            elif severity == "WARN":
                tpl = rng.choice(_BG_WARN)
            else:
                tpl = rng.choice(_BG_INFO)

            msg_tpl, meta_tpl = tpl
            entry = {
                "id": log_id,
                "timestamp": _fmt_ts(current_time, ms),
                "service": service,
                "severity": severity,
                "message": _fill_template(msg_tpl, rng),
                "metadata": _fill_meta(meta_tpl, rng),
            }

        logs.append(entry)

    return logs