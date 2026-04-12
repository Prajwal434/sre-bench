"""
Infrastructure simulator for SREBench.

Maintains the mutable state of a simulated microservice environment for one
episode.  Each task JSON is loaded once at reset(); actions mutate the state
and produce deterministic results so episodes are fully reproducible given the
same seed.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from models import Alert, IRAction, LogEntry, ServiceMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TASK_FILES = {
    "task1_memory_leak": "task1_memory_leak.json",
    "task2_db_cascade": "task2_db_cascade.json",
    "task3_ddos_exfil": "task3_ddos_exfil.json",
    "task4_predictive": "task4_predictive.json",
}


def _load_runbooks() -> List[Dict]:
    path = os.path.join(DATA_DIR, "runbooks.json")
    with open(path) as f:
        return json.load(f)["runbooks"]


_RUNBOOKS: Optional[List[Dict]] = None


def get_runbooks() -> List[Dict]:
    global _RUNBOOKS
    if _RUNBOOKS is None:
        _RUNBOOKS = _load_runbooks()
    return _RUNBOOKS


def _load_task(task_id: str) -> Dict[str, Any]:
    path = os.path.join(DATA_DIR, TASK_FILES[task_id])
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class InfraSimulator:
    """
    Stateful simulation of the microservice environment for a single episode.

    Public API
    ----------
    reset(task_id)          → initial observation dict
    step(action)            → (result_text, reward_delta, done, flags_dict)
    alerts                  → current alert list
    metrics                 → current metrics list
    applied_fixes           → set of fix keys applied this episode
    acknowledged_alerts     → set of acknowledged alert_ids
    """

    def __init__(self) -> None:
        self._task: Dict[str, Any] = {}
        self.task_id: str = ""
        self.alerts: List[Alert] = []
        self.metrics: List[ServiceMetric] = []
        self.applied_fixes: Dict[str, Any] = {}   # fix_key → fix data
        self.acknowledged_alerts: set = set()
        self.escalated_teams: set = set()
        self.notes: List[str] = []
        self.resolution_summary: str = ""
        self.wrong_fix_count: int = 0
        self.resolved: bool = False
        self._log_db: Dict[str, List[Dict]] = {}
        self._diag_db: Dict[str, str] = {}
        self._valid_fixes: Dict[str, Any] = {}
        self._grader_criteria: Dict[str, Any] = {}
        self.max_steps: int = 20
        # Proactive / predictive fields
        self._metric_trends: Dict[str, Any] = {}
        self.prediction_filed: Optional[str] = None  # predicted_issue text
        self.prediction_confidence: float = 0.0
        self.prediction_step: Optional[int] = None   # step when filed
        self.proactive_alerts_set: List[Dict] = []
        self.is_proactive: bool = False
        self.failure_at_step: int = 999  # step at which simulated failure fires

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> None:
        self._task = _load_task(task_id)
        self.task_id = task_id
        self.max_steps = self._task.get("max_steps", 20)

        self.alerts = [Alert(**a) for a in self._task["initial_alerts"]]
        self.metrics = [ServiceMetric(**m) for m in self._task["initial_metrics"]]
        self._log_db = self._task["log_database"]
        self._diag_db = self._task.get("diagnostic_results", {})
        self._valid_fixes = self._task["valid_fixes"]
        self._grader_criteria = self._task["grader_criteria"]

        self.applied_fixes = {}
        self.acknowledged_alerts = set()
        self.escalated_teams = set()
        self.notes = []
        self.resolution_summary = ""
        self.wrong_fix_count = 0
        self.resolved = False
        self._metric_trends = self._task.get("metric_trends", {})
        self.is_proactive = self._task.get("proactive_mode", False)
        self.failure_at_step = self._task.get("failure_at_step", 999)
        self.prediction_filed = None
        self.prediction_confidence = 0.0
        self.prediction_step = None
        self.proactive_alerts_set = []

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def step(self, action: IRAction) -> Tuple[str, float, bool, Dict]:
        """
        Returns (result_text, immediate_reward_delta, done, progress_flags).
        """
        t = action.action_type
        p = action.parameters

        if self.resolved:
            return "Incident already resolved.", 0.0, True, {}

        dispatch = {
            "query_logs": self._query_logs,
            "check_metrics": self._check_metrics,
            "get_metric_trends": self._get_metric_trends,
            "run_diagnostic": self._run_diagnostic,
            "apply_fix": self._apply_fix,
            "acknowledge_alert": self._acknowledge_alert,
            "escalate": self._escalate,
            "add_note": self._add_note,
            "lookup_runbook": self._lookup_runbook,
            "predict_incident": self._predict_incident,
            "set_proactive_alert": self._set_proactive_alert,
            "mark_resolved": self._mark_resolved,
        }

        handler = dispatch.get(t)
        if handler is None:
            return f"Unknown action type: {t}", -0.05, False, {}

        return handler(p)

    # ------------------------------------------------------------------
    # Individual handlers
    # ------------------------------------------------------------------

    def _query_logs(self, p: Dict) -> Tuple[str, float, bool, Dict]:
        service = p.get("service", "")
        keyword = p.get("filter", "").lower()

        if service not in self._log_db:
            available = list(self._log_db.keys())
            return (
                f"No logs found for service '{service}'. Available services: {available}",
                -0.02,
                False,
                {},
            )

        entries = self._log_db[service]
        if keyword:
            entries = [e for e in entries if keyword in e["message"].lower()]

        if not entries:
            return f"No log entries matching filter '{keyword}' in {service}.", 0.0, False, {}

        lines = "\n".join(
            f"[{e['timestamp']}] [{e['level']}] {e['message']}" for e in entries
        )
        return f"Logs for {service}:\n{lines}", 0.0, False, {"log_results": entries}

    def _check_metrics(self, p: Dict) -> Tuple[str, float, bool, Dict]:
        service = p.get("service", "")
        metric_filter = p.get("metric_name", "")

        metrics_map = {m.service: m for m in self.metrics}
        if service and service not in metrics_map:
            available = list(metrics_map.keys())
            return (
                f"No metrics for service '{service}'. Available: {available}",
                -0.02,
                False,
                {},
            )

        targets = [metrics_map[service]] if service and service in metrics_map else self.metrics

        lines = []
        for m in targets:
            lines.append(
                f"{m.service}: CPU={m.cpu_pct:.1f}% MEM={m.memory_pct:.1f}% "
                f"ERR={m.error_rate:.1f}/s P99={m.p99_latency_ms:.0f}ms RPS={m.requests_per_sec:.0f}"
            )

        return "\n".join(lines), 0.0, False, {}

    def _run_diagnostic(self, p: Dict) -> Tuple[str, float, bool, Dict]:
        tool = p.get("tool", "")
        target = p.get("target", "")
        key = f"{tool} {target}".strip()

        # Try exact match then prefix match
        result = self._diag_db.get(key)
        if result is None:
            for k, v in self._diag_db.items():
                if tool.lower() in k.lower() and (not target or target.lower() in k.lower()):
                    result = v
                    break

        if result is None:
            return (
                f"Diagnostic '{tool}' on '{target}' returned no data. "
                f"Available diagnostics: {list(self._diag_db.keys())}",
                0.0,
                False,
                {},
            )

        return f"Diagnostic [{tool}] on [{target}]:\n{result}", 0.0, False, {}

    def _apply_fix(self, p: Dict) -> Tuple[str, float, bool, Dict]:
        fix_type = p.get("fix_type", "")
        target = p.get("target", "")

        if fix_type in self.applied_fixes:
            return f"Fix '{fix_type}' already applied.", 0.0, False, {}

        fix_data = self._valid_fixes.get(fix_type)
        if fix_data is None:
            # Unknown fix – penalise
            self.wrong_fix_count += 1
            return (
                f"Unknown fix type '{fix_type}'. This action had no effect and may have made things worse.",
                -0.1,
                False,
                {"wrong_fix": True},
            )

        self.applied_fixes[fix_type] = fix_data
        partial = fix_data.get("partial_credit", 0.0)
        note = fix_data.get("note", "Fix applied.")

        # Update metrics to reflect partial recovery if correct fix
        criteria = self._grader_criteria
        correct = fix_type == criteria.get("correct_fix")
        acceptable = fix_type == criteria.get("acceptable_fix", "")

        if correct or acceptable:
            self._apply_metric_recovery(full=correct)

        reward = partial * 0.4  # fix credit feeds into grader; give small immediate signal
        return f"Fix applied: {fix_type} on {target}.\n{note}", reward, False, {"fix_applied": fix_type}

    def _acknowledge_alert(self, p: Dict) -> Tuple[str, float, bool, Dict]:
        alert_id = p.get("alert_id", "")
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                if alert.acknowledged:
                    return f"Alert {alert_id} already acknowledged.", 0.0, False, {}
                alert.acknowledged = True
                self.acknowledged_alerts.add(alert_id)
                return f"Alert {alert_id} acknowledged.", 0.01, False, {}
        return f"Alert {alert_id} not found. Active alerts: {[a.alert_id for a in self.alerts]}", -0.01, False, {}

    def _escalate(self, p: Dict) -> Tuple[str, float, bool, Dict]:
        team = p.get("team", "unknown")
        message = p.get("message", "")
        criteria = self._grader_criteria

        if team in self.escalated_teams:
            return f"Already escalated to {team}.", 0.0, False, {}

        self.escalated_teams.add(team)

        if criteria.get("needs_escalation") and team == criteria.get("escalation_team", ""):
            return (
                f"Escalated to {team} team: '{message}'\n"
                f"On-call for {team} team acknowledged. They are joining the incident.",
                0.1,
                False,
                {"escalated_correct_team": True},
            )

        if not criteria.get("needs_escalation"):
            return (
                f"Escalated to {team} team. Note: escalation was not required for this incident.",
                0.0,
                False,
                {},
            )

        return (
            f"Escalated to {team} team. Note: the correct team to escalate to may be different.",
            0.02,
            False,
            {},
        )

    def _add_note(self, p: Dict) -> Tuple[str, float, bool, Dict]:
        text = p.get("text", "").strip()
        if not text:
            return "Note was empty.", -0.01, False, {}
        self.notes.append(text)
        return f"Note added to incident timeline: '{text}'", 0.01, False, {}

    def _mark_resolved(self, p: Dict) -> Tuple[str, float, bool, Dict]:
        root_cause = p.get("root_cause", "").lower()
        summary = p.get("resolution_summary", "").strip()

        self.resolution_summary = summary
        criteria = self._grader_criteria
        root_cause_keywords = [kw.lower() for kw in criteria.get("root_cause_keywords", [])]

        correct_rc = any(kw in root_cause for kw in root_cause_keywords)
        has_summary = len(summary) > 20

        if not self.applied_fixes:
            return (
                "Cannot mark resolved – no fix has been applied yet.",
                -0.05,
                False,
                {},
            )

        self.resolved = True
        flags = {
            "resolved": True,
            "correct_root_cause": correct_rc,
            "has_summary": has_summary,
        }

        if correct_rc and has_summary:
            msg = "Incident marked resolved with correct root cause and documentation. Well done."
        elif correct_rc:
            msg = "Incident marked resolved. Root cause correctly identified, but resolution note is sparse."
        else:
            msg = "Incident marked resolved. Root cause description does not match the actual cause."

        return msg, 0.0, True, flags

    def _get_metric_trends(self, p: Dict) -> Tuple[str, float, bool, Dict]:
        service = p.get("service", "")
        metric = p.get("metric", "")
        window = p.get("window_minutes", 30)

        if not self._metric_trends:
            return (
                "No trend data available for this task. Use check_metrics for current snapshot.",
                0.0, False, {},
            )

        if service and service not in self._metric_trends:
            available = list(self._metric_trends.keys())
            return (
                f"No trend data for '{service}'. Available: {available}",
                0.0, False, {},
            )

        targets = {service: self._metric_trends[service]} if service else self._metric_trends
        lines = [f"Metric trend data (last {window} minutes):"]

        trend_out: Dict[str, Any] = {}
        for svc, metrics_data in targets.items():
            lines.append(f"\n{svc}:")
            svc_data: Dict[str, Any] = {}
            for metric_name, data in metrics_data.items():
                if metric and metric.lower() not in metric_name.lower():
                    continue
                values = data["values"]
                timestamps = data.get("timestamps_ago_minutes", [])
                trend = data.get("trend", "UNKNOWN")
                rate = data.get("rate_per_minute", 0)
                note = data.get("note", "")
                projected = data.get("projected_critical_minutes", None)

                lines.append(f"  {metric_name}: {trend}")
                lines.append(f"    Values (oldest→latest): {values}")
                lines.append(f"    Rate: {rate:+.2f}/min | Current: {values[-1]}")
                if projected:
                    lines.append(f"    ⚠ PROJECTED TIME TO CRITICAL: {projected} minutes")
                if note:
                    lines.append(f"    Note: {note}")

                svc_data[metric_name] = {
                    "trend": trend, "rate": rate, "current": values[-1],
                    "projected_critical_minutes": projected,
                }
            trend_out[svc] = svc_data

        return "\n".join(lines), 0.0, False, {"trend_data": trend_out}

    def _lookup_runbook(self, p: Dict) -> Tuple[str, float, bool, Dict]:
        symptom = p.get("symptom", "").lower()
        service = p.get("service", "").lower()

        runbooks = get_runbooks()
        matches = []

        for rb in runbooks:
            score = 0
            if symptom:
                for s in rb.get("symptoms", []):
                    if symptom in s.lower() or s.lower() in symptom:
                        score += 2
            if service:
                for svc in rb.get("applicable_services", []):
                    if service in svc.lower() or svc.lower() in service:
                        score += 1
            if score > 0:
                matches.append((score, rb))

        if not matches:
            return (
                f"No runbook found matching symptom='{symptom}' service='{service}'. "
                f"Try different keywords or check check_metrics / query_logs first.",
                0.0, False, {},
            )

        matches.sort(key=lambda x: -x[0])
        best_score, rb = matches[0]
        steps_text = "\n".join(rb["steps"])
        do_not_text = "\n".join(f"  ⛔ {d}" for d in rb.get("do_not", []))
        causes_text = "\n".join(f"  • {c}" for c in rb.get("common_root_causes", []))

        result = (
            f"RUNBOOK MATCH: [{rb['runbook_id']}] {rb['title']} (v{rb['version']})\n"
            f"P-Level: {rb['p_level']} | Escalate to: {rb['escalate_to']}\n\n"
            f"Steps:\n{steps_text}\n\n"
            f"Common root causes:\n{causes_text}\n\n"
            f"Do NOT:\n{do_not_text}"
        )

        return result, 0.02, False, {"runbook_used": rb["runbook_id"]}

    def _predict_incident(self, p: Dict) -> Tuple[str, float, bool, Dict]:
        service = p.get("service", "")
        predicted_issue = p.get("predicted_issue", "").strip()
        confidence = float(p.get("confidence", 0.5))

        if not predicted_issue:
            return "predict_incident requires 'predicted_issue' and 'confidence' parameters.", -0.01, False, {}

        if self.prediction_filed:
            return (
                f"Prediction already filed: '{self.prediction_filed}'. "
                "You can only file one prediction per episode.",
                0.0, False, {},
            )

        self.prediction_filed = predicted_issue
        self.prediction_confidence = confidence

        # Check if prediction matches root cause keywords
        criteria = self._grader_criteria
        rc_keywords = [k.lower() for k in criteria.get("root_cause_keywords", [])]
        correct = any(kw in predicted_issue.lower() for kw in rc_keywords)

        msg = (
            f"Incident prediction filed:\n"
            f"  Service: {service}\n"
            f"  Predicted issue: {predicted_issue}\n"
            f"  Confidence: {confidence*100:.0f}%\n"
            f"Prediction logged to incident management system. "
            f"If correct and filed early, this improves your proactive score."
        )

        return msg, 0.05 if correct else 0.01, False, {
            "prediction_filed": True,
            "prediction_correct": correct,
        }

    def _set_proactive_alert(self, p: Dict) -> Tuple[str, float, bool, Dict]:
        service = p.get("service", "")
        metric = p.get("metric", "")
        threshold = p.get("threshold", None)
        condition = p.get("condition", "above")

        if not service or not metric:
            return "set_proactive_alert requires 'service', 'metric', and 'threshold' parameters.", -0.01, False, {}

        alert_def = {
            "service": service,
            "metric": metric,
            "threshold": threshold,
            "condition": condition,
        }
        self.proactive_alerts_set.append(alert_def)

        return (
            f"Proactive alert set: [{service}] {metric} {condition} {threshold}\n"
            f"Alert will fire before incident becomes critical. "
            f"Total proactive alerts configured: {len(self.proactive_alerts_set)}",
            0.03, False, {"proactive_alert_set": True},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_metric_recovery(self, full: bool) -> None:
        """Simulate metric improvement after a correct/acceptable fix."""
        recovery = 0.8 if full else 0.4
        for m in self.metrics:
            m.error_rate = round(m.error_rate * (1 - recovery), 2)
            m.p99_latency_ms = round(m.p99_latency_ms * (1 - recovery), 0)
            m.cpu_pct = round(max(5.0, m.cpu_pct * (1 - recovery * 0.5)), 1)

    def get_current_log_results(self) -> List[LogEntry]:
        """Return flat list of all available logs (for initial observation)."""
        results = []
        for entries in self._log_db.values():
            results.extend([LogEntry(**e) for e in entries[:2]])  # 2 per service as preview
        return results
