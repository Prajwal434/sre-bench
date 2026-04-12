"""
Per-task graders for SREBench.

Each grader receives the completed episode state and produces a score in [0, 1]
with a breakdown of what was achieved.  Graders are deterministic – same state
always produces same score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from models import IRReward


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GradeResult:
    total: float                          # 0.0 – 1.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    feedback: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
            "feedback": self.feedback,
        }


# ---------------------------------------------------------------------------
# Base grader
# ---------------------------------------------------------------------------

class BaseGrader:
    """
    Scoring weights (sum = 1.0):

    service_identification  0.10  – identified which service is the root cause
    root_cause_diagnosis    0.25  – correctly named the root cause
    fix_applied             0.30  – applied the correct (or acceptable) fix
    resolution_speed        0.15  – resolved within the step budget
    escalation              0.10  – escalated to the right team (when required)
    documentation           0.10  – wrote a resolution note referencing root cause
    penalty                 –     – subtracted for wrong fixes / wasted actions
    """

    W_SERVICE   = 0.10
    W_ROOTCAUSE = 0.25
    W_FIX       = 0.30
    W_SPEED     = 0.15
    W_ESCALATE  = 0.10
    W_DOCS      = 0.10

    def grade(
        self,
        sim,       # InfraSimulator instance (post-episode)
        step_count: int,
        max_steps: int,
        criteria: Dict[str, Any],
    ) -> GradeResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 1 grader – Memory Leak (easy)
# ---------------------------------------------------------------------------

class Task1Grader(BaseGrader):
    """
    Grades: single-service OOM caused by unbounded cache.
    Correct fix: patch_cache_config
    Acceptable partial credit: restart_service (temporary), increase_memory_limit (workaround)
    """

    def grade(self, sim, step_count: int, max_steps: int, criteria: Dict) -> GradeResult:
        bd: Dict[str, float] = {}
        fb: List[str] = []

        # --- Service identification (0.10) ---
        target = criteria["target_service"]
        queried_target = any(
            target in note or target in (result or "")
            for note in sim.notes
            for result in [sim.resolution_summary]
        )
        # Also check if they queried logs for the right service
        log_queries_on_target = target in str(sim.applied_fixes) or any(
            target in str(f) for f in sim.applied_fixes.values()
        )
        service_score = self.W_SERVICE if (queried_target or log_queries_on_target or sim.applied_fixes) else 0.0
        # Generous: if any fix was applied to payment-service, they found it
        if any(f.get("service") == target for f in sim.applied_fixes.values()):
            service_score = self.W_SERVICE
        bd["service_identification"] = service_score

        # --- Root cause (0.25) ---
        rc_keywords = [k.lower() for k in criteria["root_cause_keywords"]]
        rc_text = (sim.resolution_summary + " ".join(sim.notes)).lower()
        rc_matches = sum(1 for kw in rc_keywords if kw in rc_text)
        rc_score = min(self.W_ROOTCAUSE, self.W_ROOTCAUSE * (rc_matches / max(1, len(rc_keywords) * 0.4)))
        bd["root_cause_diagnosis"] = round(rc_score, 4)
        if rc_matches == 0:
            fb.append("Root cause (unbounded TransactionCache / OOM) not mentioned in resolution.")

        # --- Fix applied (0.30) ---
        correct_fix = criteria["correct_fix"]
        fix_score = 0.0
        if correct_fix in sim.applied_fixes:
            fix_score = self.W_FIX
            fb.append("Correct fix applied: patch_cache_config.")
        elif "restart_service" in sim.applied_fixes:
            fix_score = self.W_FIX * 0.2
            fb.append("Applied restart (temporary). Did not fix root cause.")
        elif "increase_memory_limit" in sim.applied_fixes:
            fix_score = self.W_FIX * 0.1
            fb.append("Applied memory increase (workaround). Root cause unaddressed.")
        bd["fix_applied"] = round(fix_score, 4)

        # --- Speed (0.15) ---
        speed_score = 0.0
        if sim.resolved:
            ratio = step_count / max(1, max_steps)
            speed_score = self.W_SPEED * max(0.0, 1.0 - ratio)
        bd["resolution_speed"] = round(speed_score, 4)

        # --- Escalation (0.10) – not required for task 1, full marks for not escalating unnecessarily ---
        if not criteria.get("needs_escalation"):
            bd["escalation"] = self.W_ESCALATE  # full marks: escalation not needed
        else:
            bd["escalation"] = 0.0

        # --- Documentation (0.10) ---
        doc_score = 0.0
        if sim.resolution_summary and len(sim.resolution_summary) > 20:
            doc_score = self.W_DOCS * 0.5
            if any(kw in sim.resolution_summary.lower() for kw in rc_keywords):
                doc_score = self.W_DOCS
        bd["documentation"] = round(doc_score, 4)

        # --- Penalty ---
        penalty = -0.05 * sim.wrong_fix_count
        bd["penalty"] = round(penalty, 4)

        total = max(0.0, min(1.0, sum(bd.values())))
        return GradeResult(total=round(total, 4), breakdown=bd, feedback=fb)


# ---------------------------------------------------------------------------
# Task 2 grader – DB Cascade (medium)
# ---------------------------------------------------------------------------

class Task2Grader(BaseGrader):
    """
    Grades: invalid Postgres index causing cascade failure.
    Must identify postgres-primary as root cause, not order-service CPU.
    Correct fix: reindex_invalid_index. Acceptable: drop_invalid_index.
    """

    def grade(self, sim, step_count: int, max_steps: int, criteria: Dict) -> GradeResult:
        bd: Dict[str, float] = {}
        fb: List[str] = []

        # --- Service identification (0.10) ---
        service_score = 0.0
        if any(f.get("service") == "postgres-primary" for f in sim.applied_fixes.values()):
            service_score = self.W_SERVICE
        elif any("postgres" in str(f).lower() for f in sim.applied_fixes.values()):
            service_score = self.W_SERVICE * 0.7
        bd["service_identification"] = round(service_score, 4)
        if service_score == 0:
            fb.append("Did not identify postgres-primary as the root service (targeted order-service instead).")

        # --- Root cause (0.25) ---
        rc_keywords = [k.lower() for k in criteria["root_cause_keywords"]]
        rc_text = (sim.resolution_summary + " ".join(sim.notes)).lower()
        rc_matches = sum(1 for kw in rc_keywords if kw in rc_text)
        rc_score = min(self.W_ROOTCAUSE, self.W_ROOTCAUSE * (rc_matches / max(1, len(rc_keywords) * 0.35)))
        bd["root_cause_diagnosis"] = round(rc_score, 4)
        if "invalid" not in rc_text and "index" not in rc_text:
            fb.append("Did not identify the INVALID index as root cause. Red herring: CPU spike on order-service.")

        # --- Fix applied (0.30) ---
        correct = criteria["correct_fix"]
        acceptable = criteria.get("acceptable_fix", "")
        fix_score = 0.0
        if correct in sim.applied_fixes:
            fix_score = self.W_FIX
            fb.append("Optimal fix applied: REINDEX CONCURRENTLY.")
        elif acceptable and acceptable in sim.applied_fixes:
            fix_score = self.W_FIX * 0.7
            fb.append("Acceptable fix applied: DROP invalid index (temporary).")
        elif "restart_order_service" in sim.applied_fixes:
            fix_score = 0.0
            fb.append("Restarted order-service – wrong target, root cause in postgres-primary.")
        elif "scale_order_service" in sim.applied_fixes:
            fix_score = 0.0
            fb.append("Scaled order-service – made DB connection exhaustion worse.")
        bd["fix_applied"] = round(fix_score, 4)

        # --- Speed (0.15) ---
        speed_score = 0.0
        if sim.resolved:
            ratio = step_count / max(1, max_steps)
            speed_score = self.W_SPEED * max(0.0, 1.0 - ratio)
        bd["resolution_speed"] = round(speed_score, 4)

        # --- Escalation (not needed) ---
        bd["escalation"] = self.W_ESCALATE

        # --- Documentation (0.10) ---
        doc_score = 0.0
        if sim.resolution_summary and len(sim.resolution_summary) > 20:
            doc_score = self.W_DOCS * 0.5
            if any(kw in sim.resolution_summary.lower() for kw in rc_keywords):
                doc_score = self.W_DOCS
        bd["documentation"] = round(doc_score, 4)

        # --- Penalty ---
        penalty = -0.05 * sim.wrong_fix_count
        bd["penalty"] = round(penalty, 4)

        total = max(0.0, min(1.0, sum(bd.values())))
        return GradeResult(total=round(total, 4), breakdown=bd, feedback=fb)


# ---------------------------------------------------------------------------
# Task 3 grader – DDoS + Exfil (hard)
# ---------------------------------------------------------------------------

class Task3Grader(BaseGrader):
    """
    Grades: DDoS as distraction + active data exfiltration.
    Must address BOTH vectors, escalate to security, preserve forensics.
    Scoring uses required_actions checklist + bonus actions.
    """

    W_SERVICE   = 0.05
    W_ROOTCAUSE = 0.20
    W_FIX       = 0.35   # distributed across required_actions
    W_SPEED     = 0.05
    W_ESCALATE  = 0.20   # higher weight – security escalation is critical
    W_DOCS      = 0.15

    def grade(self, sim, step_count: int, max_steps: int, criteria: Dict) -> GradeResult:
        bd: Dict[str, float] = {}
        fb: List[str] = []

        required = criteria.get("required_actions", [])
        bonus = criteria.get("bonus_actions", [])

        # --- Service identification (0.05) ---
        # Must recognise both attack vectors (DDoS + exfil via data-pipeline)
        rc_text = (sim.resolution_summary + " ".join(sim.notes)).lower()
        vectors_found = sum([
            "ddos" in rc_text or "rate limit" in rc_text,
            "exfil" in rc_text or "data-pipeline" in rc_text or "compromised" in rc_text,
        ])
        bd["service_identification"] = round(self.W_SERVICE * vectors_found / 2, 4)

        # --- Root cause (0.20) ---
        rc_keywords = [k.lower() for k in criteria.get("root_cause_keywords", [])]
        rc_matches = sum(1 for kw in rc_keywords if kw in rc_text)
        rc_score = min(self.W_ROOTCAUSE, self.W_ROOTCAUSE * (rc_matches / max(1, len(rc_keywords) * 0.4)))
        bd["root_cause_diagnosis"] = round(rc_score, 4)
        if "compromised" not in rc_text and "admin" not in rc_text:
            fb.append("Did not identify the compromised admin account as attack entry point.")
        if "distraction" not in rc_text and "exfil" not in rc_text:
            fb.append("Did not recognise DDoS as a distraction for data exfiltration.")

        # --- Fix applied (0.35) – required_actions checklist ---
        applied_required = [r for r in required if r in sim.applied_fixes]
        fix_per_action = self.W_FIX / max(1, len(required))
        fix_score = len(applied_required) * fix_per_action

        # Bonus actions add up to 0.05 extra
        applied_bonus = [b for b in bonus if b in sim.applied_fixes]
        bonus_score = min(0.05, len(applied_bonus) * 0.01)
        fix_score = min(self.W_FIX + 0.05, fix_score + bonus_score)

        bd["fix_applied"] = round(fix_score, 4)
        missing = [r for r in required if r not in sim.applied_fixes]
        if missing:
            fb.append(f"Missing required actions: {missing}")

        # --- Speed (0.05) ---
        speed_score = 0.0
        if sim.resolved:
            ratio = step_count / max(1, max_steps)
            speed_score = self.W_SPEED * max(0.0, 1.0 - ratio)
        bd["resolution_speed"] = round(speed_score, 4)

        # --- Escalation (0.20) – critical for security incident ---
        escalate_score = 0.0
        if "security" in sim.escalated_teams:
            escalate_score = self.W_ESCALATE
            fb.append("Security team correctly escalated.")
        elif sim.escalated_teams:
            escalate_score = self.W_ESCALATE * 0.3
            fb.append(f"Escalated to {sim.escalated_teams} but security team was required.")
        else:
            fb.append("Failed to escalate to security team – critical miss for a data exfil incident.")
        bd["escalation"] = round(escalate_score, 4)

        # --- Documentation (0.15) ---
        doc_score = 0.0
        if sim.resolution_summary and len(sim.resolution_summary) > 30:
            doc_score = self.W_DOCS * 0.5
            if any(kw in sim.resolution_summary.lower() for kw in rc_keywords):
                doc_score = self.W_DOCS
        bd["documentation"] = round(doc_score, 4)

        # --- Penalty ---
        penalty = -0.05 * sim.wrong_fix_count
        bd["penalty"] = round(penalty, 4)

        total = max(0.0, min(1.0, sum(bd.values())))
        return GradeResult(total=round(total, 4), breakdown=bd, feedback=fb)


# ---------------------------------------------------------------------------
# Task 4 grader – Predictive Prevention (medium)
# ---------------------------------------------------------------------------

class Task4Grader(BaseGrader):
    """
    Grades proactive incident prevention.

    Key difference from other tasks: the PRIMARY signal is WHEN the agent acted.
    Acting before the failure_at_step = full speed bonus.
    Acting after = diminishing returns.

    Weights (sum = 1.0):
      trend_analysis      0.15  – called get_metric_trends + correctly read the trend
      runbook_lookup      0.05  – used lookup_runbook (realistic SRE workflow)
      prediction          0.20  – filed a correct predict_incident
      fix_applied         0.25  – applied the correct fix
      proactive_speed     0.20  – acted BEFORE the failure step (bonus for early action)
      documentation       0.15  – wrote a resolution note referencing root cause
      penalty             –     – wrong fixes
    """

    W_TREND    = 0.15
    W_RUNBOOK  = 0.05
    W_PREDICT  = 0.20
    W_FIX      = 0.25
    W_SPEED    = 0.20
    W_DOCS     = 0.15

    def grade(self, sim, step_count: int, max_steps: int, criteria: Dict) -> GradeResult:
        bd: Dict[str, float] = {}
        fb: List[str] = []

        failure_step = criteria.get("proactive_failure_step", 12)
        rc_keywords = [k.lower() for k in criteria.get("root_cause_keywords", [])]
        rc_text = (sim.resolution_summary + " ".join(sim.notes)).lower()

        # --- Trend analysis (0.15) ---
        # Did agent call get_metric_trends? We track via prediction/notes heuristic
        # (Simulator doesn't track which actions called, so we check notes + prediction text)
        trend_evidence = (
            "trend" in rc_text
            or "creep" in rc_text
            or "increasing" in rc_text
            or "2.2%/min" in rc_text
            or (sim.prediction_filed is not None)
        )
        trend_score = self.W_TREND if trend_evidence else 0.0
        bd["trend_analysis"] = round(trend_score, 4)
        if not trend_evidence:
            fb.append("Did not demonstrate trend analysis (call get_metric_trends and reference findings).")

        # --- Runbook lookup (0.05) ---
        # Reward if agent used lookup_runbook (tracked via note or we give benefit of doubt)
        runbook_score = self.W_RUNBOOK if sim.notes else 0.0  # generous: if they added notes, partial credit
        bd["runbook_lookup"] = round(runbook_score, 4)

        # --- Prediction (0.20) ---
        predict_score = 0.0
        if sim.prediction_filed:
            pred_text = sim.prediction_filed.lower()
            correct = any(kw in pred_text for kw in rc_keywords)
            if correct and sim.prediction_confidence >= 0.7:
                predict_score = self.W_PREDICT
                fb.append(f"Correct prediction filed with high confidence ({sim.prediction_confidence*100:.0f}%).")
            elif correct:
                predict_score = self.W_PREDICT * 0.7
                fb.append(f"Correct prediction filed but confidence was low ({sim.prediction_confidence*100:.0f}%).")
            else:
                predict_score = self.W_PREDICT * 0.1
                fb.append(f"Prediction filed but did not match root cause keywords.")
        else:
            fb.append("No incident prediction filed. Use predict_incident to log your forecast.")
        bd["incident_prediction"] = round(predict_score, 4)

        # --- Fix applied (0.25) ---
        correct_fix = criteria.get("correct_fix", "")
        acceptable_fix = criteria.get("acceptable_fix", "")
        fix_score = 0.0
        if correct_fix in sim.applied_fixes:
            fix_score = self.W_FIX
            fb.append("Correct fix applied: patch_session_ttl.")
        elif acceptable_fix and acceptable_fix in sim.applied_fixes:
            fix_score = self.W_FIX * 0.65
            fb.append("Acceptable partial fix applied.")
        elif "flush_expired_sessions" in sim.applied_fixes:
            fix_score = self.W_FIX * 0.4
            fb.append("Emergency flush applied (temporary). Root cause (ttl=-1) not fixed.")
        elif "restart_auth_service" in sim.applied_fixes:
            fix_score = self.W_FIX * 0.1
            fb.append("Restart applied (temporary workaround only).")
        bd["fix_applied"] = round(fix_score, 4)

        # --- Proactive speed (0.20) – the unique reward signal ---
        # Full marks if fix was applied BEFORE failure_step.
        # Diminishing returns after.
        speed_score = 0.0
        if sim.applied_fixes:
            if step_count < failure_step:
                # Acted before failure: reward inversely proportional to step
                # Step 1 = 1.0, step failure_step-1 = 0.1
                ratio = (failure_step - step_count) / max(1, failure_step - 1)
                speed_score = self.W_SPEED * ratio
                fb.append(f"Proactive! Fixed at step {step_count}, before failure at step {failure_step}. Speed bonus: {speed_score:.2f}")
            else:
                # Acted after failure: still some credit but heavily penalised
                speed_score = self.W_SPEED * 0.1
                fb.append(f"Reactive: fixed at step {step_count} after failure would have fired at step {failure_step}.")
        bd["proactive_speed"] = round(speed_score, 4)

        # --- Documentation (0.15) ---
        doc_score = 0.0
        if sim.resolution_summary and len(sim.resolution_summary) > 20:
            doc_score = self.W_DOCS * 0.5
            if any(kw in sim.resolution_summary.lower() for kw in rc_keywords):
                doc_score = self.W_DOCS
        bd["documentation"] = round(doc_score, 4)

        # --- Penalty ---
        penalty = -0.05 * sim.wrong_fix_count
        bd["penalty"] = round(penalty, 4)

        total = max(0.0, min(1.0, sum(bd.values())))
        return GradeResult(total=round(total, 4), breakdown=bd, feedback=fb)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

GRADERS: Dict[str, BaseGrader] = {
    "task1_memory_leak": Task1Grader(),
    "task2_db_cascade": Task2Grader(),
    "task3_ddos_exfil": Task3Grader(),
    "task4_predictive": Task4Grader(),
}


def grade_episode(sim, step_count: int) -> GradeResult:
    """Grade a completed (or truncated) episode."""
    grader = GRADERS.get(sim.task_id)
    if grader is None:
        return GradeResult(total=0.0, feedback=[f"No grader for task {sim.task_id}"])
    return grader.grade(sim, step_count, sim.max_steps, sim._grader_criteria)
