from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ActionType(str, Enum):
    SEARCH = "search"
    FILTER_SEVERITY = "filter_severity"
    FILTER_SERVICE = "filter_service"
    FILTER_TIME_RANGE = "filter_time_range"
    CLEAR_FILTERS = "clear_filters"
    SCROLL = "scroll"
    INSPECT = "inspect"
    ANNOTATE = "annotate"
    CORRELATE = "correlate"
    CLASSIFY_INCIDENT = "classify_incident"
    DRAFT_REPORT = "draft_report"
    SUBMIT_REPORT = "submit_report"
    NOOP = "noop"


class Action(BaseModel):
    action_type: ActionType
    params: Dict[str, str] = {}


class LogEntry(BaseModel):
    id: str
    timestamp: str
    service: str
    severity: str
    message: str
    metadata: Dict[str, str] = {}


class Observation(BaseModel):
    task_id: str
    goal: str
    step_number: int
    max_steps: int
    visible_logs: List[LogEntry]
    total_log_count: int
    current_page: int
    total_pages: int
    severity_counts: Dict[str, int]
    available_services: List[str]
    current_filters: Dict[str, str]
    annotations_count: int
    recent_annotations: List[Dict[str, str]]
    annotations_by_category: Dict[str, int]
    correlations_count: int
    recent_correlations: List[List[str]]
    severity_classified: Optional[str] = None
    current_report_draft: Optional[str] = None
    inspected_log: Optional[LogEntry] = None
    last_action_success: bool = True
    last_action_message: str = "OK"
    draft_feedback: Optional[str] = None


class Reward(BaseModel):
    value: float = 0.0
    components: Dict[str, float] = {}
    cumulative: float = 0.0


class GraderComponent(BaseModel):
    score: float
    weight: float
    weighted: float
    detail: str


class GraderResult(BaseModel):
    task_id: str
    final_score: float
    components: Dict[str, GraderComponent]


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}


class ResetResult(BaseModel):
    observation: Observation
    info: Dict[str, Any] = {}