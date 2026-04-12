"""
IncidentResponseEnv – the core OpenEnv environment for SREBench.

Implements the OpenEnv interface:
  reset(seed, episode_id, **kwargs) → IRObservation
  step(action, **kwargs)            → IRObservation
  state                             → IRState (property)
  close()                           → None
"""

from __future__ import annotations

import sys
import os
import uuid
from typing import Any, Dict, List, Optional

# Ensure project root is on path when running from server/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    Alert,
    IRAction,
    IRObservation,
    IRState,
    LogEntry,
    ServiceMetric,
)
from server.simulator import InfraSimulator
from server.graders import grade_episode, GradeResult

VALID_TASKS = ["task1_memory_leak", "task2_db_cascade", "task3_ddos_exfil", "task4_predictive"]
DEFAULT_TASK = "task1_memory_leak"


class IncidentResponseEnv:
    """
    OpenEnv-compliant environment simulating SRE on-call incident response.

    Each episode:
      1. reset(task_id=...) loads a scenario and returns the initial observation
      2. The agent calls step(action) repeatedly
      3. The episode ends when the agent calls mark_resolved or max_steps is reached
      4. The final score (0.0–1.0) is computed by a deterministic grader
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        self._sim = InfraSimulator()
        self._state = IRState()
        self._timeline: List[str] = []
        self._last_grade: Optional[GradeResult] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = DEFAULT_TASK,
        **kwargs,
    ) -> IRObservation:
        """Start a new episode for the given task."""
        if task_id not in VALID_TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid tasks: {VALID_TASKS}"
            )

        ep_id = episode_id or str(uuid.uuid4())
        self._sim.reset(task_id)
        self._timeline = []
        self._last_grade = None

        self._state = IRState(
            episode_id=ep_id,
            step_count=0,
            task_id=task_id,
            done=False,
            partial_score=0.0,
        )

        self._timeline.append(f"[EPISODE START] Task: {task_id}")

        return self._build_observation(
            last_action_result="New incident opened. Review active alerts and begin investigation.",
            done=False,
            reward=None,
        )

    def step(self, action: IRAction, timeout_s: float = 30.0, **kwargs) -> IRObservation:
        """Execute one action and return the resulting observation."""
        if self._state.done:
            return self._build_observation(
                last_action_result="Episode already finished. Call reset() to start a new episode.",
                done=True,
                reward=self._last_grade.total if self._last_grade else 0.0,
            )

        self._state.step_count += 1
        self._timeline.append(
            f"[STEP {self._state.step_count}] action={action.action_type} params={action.parameters}"
        )

        result_text, reward_delta, done_from_action, flags = self._sim.step(action)

        self._timeline.append(f"  → {result_text[:200]}")

        # Update state flags from action result
        if flags.get("fix_applied"):
            self._state.fix_applied = True
            fix_key = flags["fix_applied"]
            criteria = self._sim._grader_criteria
            if fix_key == criteria.get("correct_fix") or fix_key == criteria.get("acceptable_fix", ""):
                self._state.fix_correct = True
        if flags.get("escalated_correct_team"):
            self._state.escalated_when_needed = True
        if flags.get("resolved"):
            self._state.resolution_note_written = bool(flags.get("has_summary"))

        # Check terminal conditions
        max_steps_reached = self._state.step_count >= self._sim.max_steps
        done = done_from_action or max_steps_reached

        final_reward = None
        if done:
            self._last_grade = grade_episode(self._sim, self._state.step_count)
            final_reward = self._last_grade.total
            self._state.partial_score = final_reward
            self._state.done = True
            if max_steps_reached and not done_from_action:
                self._timeline.append(
                    f"[TRUNCATED] Max steps ({self._sim.max_steps}) reached without resolution."
                )
        else:
            # Partial score update for informational purposes
            current_grade = grade_episode(self._sim, self._state.step_count)
            self._state.partial_score = current_grade.total

        return self._build_observation(
            last_action_result=result_text,
            done=done,
            reward=final_reward,
        )

    @property
    def state(self) -> IRState:
        """Current episode metadata."""
        return self._state

    def close(self) -> None:
        """No resources to clean up."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        last_action_result: str,
        done: bool,
        reward: Optional[float],
    ) -> IRObservation:
        meta: Dict[str, Any] = {
            "task_id": self._state.task_id,
            "step_count": self._state.step_count,
            "max_steps": self._sim.max_steps,
            "partial_score": self._state.partial_score,
        }
        if done and self._last_grade is not None:
            meta["grade"] = self._last_grade.as_dict()

        # Proactive mode: surface prediction state
        if self._sim.is_proactive and self._sim.prediction_filed:
            meta["prediction_filed"] = self._sim.prediction_filed
            meta["prediction_confidence"] = self._sim.prediction_confidence

        return IRObservation(
            done=done,
            reward=reward,
            metadata=meta,
            task_id=self._state.task_id,
            step_count=self._state.step_count,
            max_steps=self._sim.max_steps,
            active_alerts=list(self._sim.alerts),
            log_results=[],
            metrics=list(self._sim.metrics),
            incident_timeline=list(self._timeline[-15:]),
            last_action_result=last_action_result,
        )

    def get_task_description(self, task_id: str) -> str:
        """Return human-readable task description."""
        descriptions = {
            "task1_memory_leak": (
                "[EASY] Memory Leak OOM – Payment Service\n"
                "The payment-service is crashing repeatedly (OOMKilled). "
                "Investigate, identify the root cause, apply the correct fix, and document the resolution."
            ),
            "task2_db_cascade": (
                "[MEDIUM] Database Cascade Failure – Slow Query Avalanche\n"
                "Multiple services are degraded. A CPU spike is a red herring. "
                "Trace the dependency chain back to the true root cause in the database layer."
            ),
            "task3_ddos_exfil": (
                "[HARD] DDoS Attack with Suspected Data Exfiltration\n"
                "Auth endpoints are under a DDoS attack AND a background process is exfiltrating data "
                "to Tor exit nodes. The DDoS may be a distraction. Mitigate both threats, "
                "escalate to security, and preserve forensic evidence."
            ),
            "task4_predictive": (
                "[MEDIUM] Proactive Incident Prevention – No alerts have fired yet.\n"
                "Metric trends show auth-service memory climbing ~2.2%/min and session-cache filling up. "
                "If unchecked, a P1 OOM incident will fire in ~13 minutes. "
                "Use get_metric_trends to read the signals, lookup_runbook for guidance, "
                "file a prediction with predict_incident, fix the root cause, "
                "and resolve BEFORE the failure threshold is breached for maximum score."
            ),
        }
        return descriptions.get(task_id, f"Unknown task: {task_id}")
