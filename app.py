from src.environment import LogTriageEnv
from src.tasks import TASKS
from src.session import SessionManager

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import math

app = FastAPI(title="LogTriage Environment", version="3.0")

# Use SessionManager instead of inline dict
session_mgr = SessionManager(max_sessions=10, ttl_seconds=1800)


# ─── Score Clamping Safety Net ──────────────────────────────

SCORE_KEYS = frozenset({
    "score", "final_score", "task_score", "value", "cumulative",
    "annotation_precision", "annotation_recall", "annotation_quality",
    "correlation_precision", "correlation_recall", "chain_reconstruction",
    "severity_classification", "report_completeness", "report_coherence",
    "investigation_efficiency",
})


def _safe_clamp(v, eps=0.001):
    """Clamp a single numeric value to strictly (0, 1)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.5
    if math.isnan(v) or math.isinf(v):
        return 0.5
    return max(eps, min(1.0 - eps, v))


def _deep_clamp_scores(obj):
    """Recursively find and clamp any score-like numeric fields in a dict/list."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(v, (int, float)) and k in SCORE_KEYS:
                result[k] = _safe_clamp(v)
            elif isinstance(v, dict):
                result[k] = _deep_clamp_scores(v)
            elif isinstance(v, list):
                result[k] = _deep_clamp_scores(v)
            else:
                result[k] = v
        return result
    elif isinstance(obj, list):
        return [_deep_clamp_scores(item) for item in obj]
    return obj


# ─── Request Models ─────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_easy"


class StepRequest(BaseModel):
    session_id: str
    action_type: str
    params: dict = {}


# ─── Endpoints ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0",
        "sessions_active": session_mgr.active_count(),
    }


@app.get("/tasks")
def list_tasks():
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "difficulty": t["difficulty"],
            "max_steps": t["max_steps"],
            "log_count": t["log_count"],
            "services": t["services"],
        }
        for t in TASKS.values()
    ]


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()

    if req.task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: '{req.task_id}'. "
                   f"Available: {list(TASKS.keys())}",
        )

    try:
        session_id = session_mgr.create_session(req.task_id)
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))

    entry = session_mgr.get_session(session_id)
    result = entry.env.reset(req.task_id)

    return {
        "session_id": session_id,
        "observation": result["observation"],
        "info": result["info"],
    }


@app.post("/step")
def step(req: StepRequest):
    entry = session_mgr.get_session(req.session_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /reset first.",
        )

    result = entry.env.step(req.action_type, req.params)

    # ── Clamp ALL scores at the API boundary — final safety net ──
    raw_reward = result.get("reward", {})
    if isinstance(raw_reward, dict):
        reward = _deep_clamp_scores(raw_reward)
    else:
        reward = _safe_clamp(raw_reward)

    info = _deep_clamp_scores(result.get("info", {}))
    done = result.get("done", False)

    # Build the response
    response = {
        "observation": result["observation"],
        "reward": reward,
        "done": done,
        "info": info,
    }

    # If done, also put score at the top level of the response
    # so the validator can find it no matter where it looks
    if done:
        top_score = None

        # Try to extract from info
        if isinstance(info, dict):
            top_score = info.get("score") or info.get("task_score") or info.get("final_score")
            # Try nested grader_result
            if top_score is None and "grader_result" in info:
                gr = info["grader_result"]
                if isinstance(gr, dict):
                    top_score = gr.get("score") or gr.get("final_score")

        # Try to extract from reward
        if top_score is None and isinstance(reward, dict):
            top_score = reward.get("score") or reward.get("value")

        # Final fallback
        if top_score is None:
            top_score = 0.5

        top_score = _safe_clamp(top_score)
        response["score"] = top_score
        response["task_score"] = top_score
        response["final_score"] = top_score

    return response


@app.get("/state")
def get_state(session_id: str):
    entry = session_mgr.get_session(session_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /reset first.",
        )

    state = entry.env.state()
    state["session_id"] = session_id
    return state


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()