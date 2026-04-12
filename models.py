"""
SREBench – typed Pydantic models for the Incident Response OpenEnv environment.

Action   → what the agent can do each step
Observation → what the agent sees after each action
State    → internal episode metadata
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models used inside Observation
# ---------------------------------------------------------------------------

class Alert(BaseModel):
    alert_id: str
    severity: Literal["critical", "warning", "info"]
    service: str
    message: str
    timestamp: str
    acknowledged: bool = False


class LogEntry(BaseModel):
    timestamp: str
    service: str
    level: Literal["ERROR", "WARN", "INFO", "DEBUG"]
    message: str


class ServiceMetric(BaseModel):
    service: str
    cpu_pct: float
    memory_pct: float
    error_rate: float          # errors per second
    p99_latency_ms: float
    requests_per_sec: float


# ---------------------------------------------------------------------------
# Core OpenEnv models
# ---------------------------------------------------------------------------

class IRAction(BaseModel):
    """
    Action the agent takes each step.

    action_type options
    -------------------
    query_logs        – search log stream for a service/keyword
    check_metrics     – get current metrics for a service
    get_metric_trends – get time-series trend data for a metric (proactive mode)
    run_diagnostic    – run a diagnostic tool (ping, heap-dump, explain_query …)
    apply_fix         – apply a remediation to a service
    acknowledge_alert – ack a PagerDuty-style alert
    escalate          – page another team
    add_note          – append text to incident timeline
    lookup_runbook    – search the runbook database by symptom or service
    predict_incident  – file a formal incident prediction with confidence score
    set_proactive_alert – register a proactive threshold alert
    mark_resolved     – close the incident (terminal action)
    """

    action_type: Literal[
        "query_logs",
        "check_metrics",
        "get_metric_trends",
        "run_diagnostic",
        "apply_fix",
        "acknowledge_alert",
        "escalate",
        "add_note",
        "lookup_runbook",
        "predict_incident",
        "set_proactive_alert",
        "mark_resolved",
    ]
    parameters: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class IRObservation(BaseModel):
    """
    What the agent perceives after each action.
    Extends OpenEnv base Observation fields (done, reward, metadata).
    """

    # OpenEnv required fields
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # SREBench-specific fields
    task_id: str = ""
    step_count: int = 0
    max_steps: int = 20
    active_alerts: List[Alert] = Field(default_factory=list)
    log_results: List[LogEntry] = Field(default_factory=list)
    metrics: List[ServiceMetric] = Field(default_factory=list)
    incident_timeline: List[str] = Field(default_factory=list)
    last_action_result: str = ""
    available_action_types: List[str] = Field(
        default_factory=lambda: [
            "query_logs",
            "check_metrics",
            "get_metric_trends",
            "run_diagnostic",
            "apply_fix",
            "acknowledge_alert",
            "escalate",
            "add_note",
            "lookup_runbook",
            "predict_incident",
            "set_proactive_alert",
            "mark_resolved",
        ]
    )
    trend_data: Dict[str, Any] = Field(default_factory=dict)
    runbook_results: List[Dict[str, Any]] = Field(default_factory=list)
    prediction_filed: Optional[str] = None

    class Config:
        extra = "forbid"


class IRState(BaseModel):
    """
    Internal episode state (returned by state() endpoint).
    """

    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: str = ""
    done: bool = False
    partial_score: float = 0.0

    # Grader progress flags – which scoring criteria have been met
    service_identified: bool = False
    root_cause_identified: bool = False
    fix_applied: bool = False
    fix_correct: bool = False
    escalated_when_needed: bool = False
    resolution_note_written: bool = False
    # Proactive fields
    prediction_filed: bool = False
    prediction_correct: bool = False
    acted_before_failure: bool = False
    proactive_step: Optional[int] = None  # step at which agent filed prediction

    class Config:
        extra = "allow"


class IRReward(BaseModel):
    """Decomposed reward breakdown (stored in metadata)."""

    service_identification: float = 0.0
    root_cause_diagnosis: float = 0.0
    fix_applied: float = 0.0
    resolution_speed: float = 0.0
    escalation: float = 0.0
    documentation: float = 0.0
    penalty: float = 0.0

    @property
    def total(self) -> float:
        return max(
            0.0,
            min(
                1.0,
                self.service_identification
                + self.root_cause_diagnosis
                + self.fix_applied
                + self.resolution_speed
                + self.escalation
                + self.documentation
                + self.penalty,
            ),
        )
