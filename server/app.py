from src.environment import LogTriageEnv
from src.tasks import TASKS
from src.session import SessionManager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="LogTriage Environment", version="3.0")

# Use SessionManager instead of inline dict
session_mgr = SessionManager(max_sessions=10, ttl_seconds=1800)


class ResetRequest(BaseModel):
    task_id: str


class StepRequest(BaseModel):
    session_id: str
    action_type: str
    params: dict = {}


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
def reset(req: ResetRequest):
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

    return {
        "observation": result["observation"],
        "reward": result["reward"],
        "done": result["done"],
        "info": result["info"],
    }


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