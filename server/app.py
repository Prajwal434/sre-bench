"""
FastAPI + Gradio server for SREBench.

Exposes:
  POST /reset   – start a new episode
  POST /step    – take an action
  GET  /state   – get current episode state
  GET  /health  – health check
  GET  /        – Gradio web UI (for HF Spaces demo)

Compatible with the OpenEnv HTTP interface spec.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import smtplib
import threading
import time
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import IRAction, IRObservation, IRState
from server.environment import IncidentResponseEnv, VALID_TASKS

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SREBench – Incident Response OpenEnv",
    description=(
        "An OpenEnv-compatible environment simulating SRE on-call incident response. "
        "Agents must diagnose and remediate production incidents across 3 difficulty levels."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server (single-session mode)
_env_lock = threading.Lock()
_env = IncidentResponseEnv()


# ------------------------------------------------------------------
# Request / response schemas
# ------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1_memory_leak"
    episode_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = {}


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "env": "sre-bench"}


@app.get("/metadata")
def metadata():
    return {
        "name": "sre-bench",
        "description": (
            "An OpenEnv environment simulating SRE on-call incident response. "
            "Agents diagnose production incidents (OOM crashes, database cascades, "
            "DDoS + data exfiltration) across 3 difficulty levels and receive "
            "reward signals for correct diagnosis, effective remediation, "
            "appropriate escalation, and documentation quality."
        ),
        "version": "1.0.0",
        "tasks": VALID_TASKS,
    }


@app.get("/schema")
def schema():
    return {
        "action": IRAction.model_json_schema(),
        "observation": IRObservation.model_json_schema(),
        "state": IRState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp_endpoint(request: dict):
    """Minimal JSON-RPC 2.0 endpoint for MCP compatibility."""
    method = request.get("method", "")
    req_id = request.get("id", 1)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "sre-bench", "version": "1.0.0"},
            },
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {"name": t, "description": f"SREBench action: {t}", "inputSchema": {"type": "object"}}
                    for t in ["query_logs", "check_metrics", "run_diagnostic", "apply_fix",
                              "acknowledge_alert", "escalate", "add_note", "mark_resolved",
                              "get_metric_trends", "lookup_runbook", "predict_incident", "set_proactive_alert"]
                ]
            },
        }
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


@app.post("/reset", response_model=IRObservation)
def reset(req: Optional[ResetRequest] = None):
    with _env_lock:
        try:
            if req is None:
                req = ResetRequest()
            obs = _env.reset(
                task_id=req.task_id,
                episode_id=req.episode_id,
                seed=req.seed,
            )
            return obs
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=IRObservation)
def step(req: StepRequest):
    with _env_lock:
        try:
            action = IRAction(
                action_type=req.action_type,
                parameters=req.parameters,
            )
            obs = _env.step(action)
            return obs
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=IRState)
def state():
    with _env_lock:
        return _env.state


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id": t,
                "description": _env.get_task_description(t),
            }
            for t in VALID_TASKS
        ]
    }


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------

def _fmt_obs(obs_dict: dict) -> str:
    """Format observation dict for display."""
    lines = []
    if obs_dict.get("last_action_result"):
        lines.append(f"**Last Action Result:**\n```\n{obs_dict['last_action_result']}\n```")

    alerts = obs_dict.get("active_alerts", [])
    if alerts:
        lines.append("\n**Active Alerts:**")
        for a in alerts:
            ack = " ✓" if a.get("acknowledged") else ""
            lines.append(f"- [{a['severity'].upper()}{ack}] {a['service']}: {a['message']}")

    metrics = obs_dict.get("metrics", [])
    if metrics:
        lines.append("\n**Service Metrics:**")
        for m in metrics:
            lines.append(
                f"- `{m['service']}`: CPU={m['cpu_pct']}% MEM={m['memory_pct']}% "
                f"ERR={m['error_rate']}/s P99={m['p99_latency_ms']}ms"
            )

    timeline = obs_dict.get("incident_timeline", [])
    if timeline:
        lines.append("\n**Timeline (last 10):**")
        for t in timeline[-10:]:
            lines.append(f"  {t}")

    step = obs_dict.get("step_count", 0)
    max_s = obs_dict.get("max_steps", 20)
    done = obs_dict.get("done", False)
    reward = obs_dict.get("reward")
    lines.append(f"\n---\nStep {step}/{max_s} | Done: {done}")
    if reward is not None:
        grade = obs_dict.get("metadata", {}).get("grade", {})
        lines.append(f"**Final Score: {reward:.3f}**")
        if grade:
            lines.append(f"Breakdown: {json.dumps(grade.get('breakdown', {}), indent=2)}")
            for fb in grade.get("feedback", []):
                lines.append(f"- {fb}")

    return "\n".join(lines)


_gradio_state: Dict[str, Any] = {"obs": None, "task_id": "task1_memory_leak"}

# ------------------------------------------------------------------
# Email alert
# ------------------------------------------------------------------

ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")


def send_escalation_email(team: str, message: str, task_id: str):
    if not all([ALERT_EMAIL, EMAIL_USER, EMAIL_PASSWORD]):
        return
    try:
        body = f"""
SREБENCH ESCALATION ALERT
==========================
Task:    {task_id}
Team:    {team}
Message: {message}

Human intervention required. Please review the incident immediately.
        """.strip()
        msg = MIMEText(body)
        msg["Subject"] = f"[SREBench] ESCALATION → {team.upper()} team required"
        msg["From"] = EMAIL_USER
        msg["To"] = ALERT_EMAIL
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(EMAIL_USER, EMAIL_PASSWORD)
            s.sendmail(EMAIL_USER, ALERT_EMAIL, msg.as_string())
    except Exception:
        pass


# ------------------------------------------------------------------
# Mock agent scripts (same as inference.py)
# ------------------------------------------------------------------

MOCK_SCRIPTS: Dict[str, list] = {
    "task1_memory_leak": [
        ("check_metrics", {}),
        ("query_logs", {"service": "payment-service", "filter": "ERROR"}),
        ("lookup_runbook", {"symptom": "OOMKilled memory crash", "service": "payment-service"}),
        ("run_diagnostic", {"tool": "heap_dump", "target": "payment-service"}),
        ("acknowledge_alert", {"alert_id": "ALT-001"}),
        ("apply_fix", {"fix_type": "patch_cache_config", "target": "payment-service"}),
        ("mark_resolved", {"root_cause": "TransactionCache unbounded maxSize=-1 caused heap exhaustion",
                           "resolution_summary": "Patched cache maxSize=50000 with TTL eviction. Payment service memory stabilised."}),
    ],
    "task2_db_cascade": [
        ("check_metrics", {}),
        ("query_logs", {"service": "order-service", "filter": "ERROR"}),
        ("lookup_runbook", {"symptom": "high latency slow queries", "service": "postgres-primary"}),
        ("run_diagnostic", {"tool": "explain_query", "target": "postgres-primary"}),
        ("acknowledge_alert", {"alert_id": "ALT-001"}),
        ("add_note", {"text": "orders_customer_id_idx is INVALID — all queries doing full seq scan of 2.8M rows"}),
        ("apply_fix", {"fix_type": "reindex_invalid_index", "target": "postgres-primary"}),
        ("query_logs", {"service": "order-service", "filter": "latency"}),
        ("mark_resolved", {"root_cause": "Invalid index orders_customer_id_idx forced full seq scans (25s/query)",
                           "resolution_summary": "REINDEX CONCURRENTLY rebuilt index. Query time 25s→8ms. All services recovered."}),
    ],
    "task3_ddos_exfil": [
        ("check_metrics", {}),
        ("acknowledge_alert", {"alert_id": "ALT-001"}),
        ("query_logs", {"service": "auth-service", "filter": "ERROR"}),
        ("run_diagnostic", {"tool": "threat_intel", "target": "auth-service"}),
        ("acknowledge_alert", {"alert_id": "ALT-002"}),
        ("query_logs", {"service": "data-pipeline", "filter": "export"}),
        ("run_diagnostic", {"tool": "netstat", "target": "data-pipeline"}),
        ("apply_fix", {"fix_type": "rate_limit_auth", "target": "auth-service"}),
        ("apply_fix", {"fix_type": "reset_admin_password", "target": "auth-service"}),
        ("escalate", {"team": "security", "message": "Compromised admin account exfiltrating user PII to Tor exit nodes via data-pipeline. DDoS on auth is a distraction."}),
        ("apply_fix", {"fix_type": "kill_rogue_export", "target": "data-pipeline"}),
        ("mark_resolved", {"root_cause": "Compromised admin account used to exfiltrate PII; DDoS was distraction",
                           "resolution_summary": "Rate-limited auth, reset admin password, killed rogue export job, escalated to security team."}),
    ],
    "task4_predictive": [
        ("get_metric_trends", {"service": "auth-service", "metric": "memory_pct", "window_minutes": 30}),
        ("lookup_runbook", {"symptom": "memory increasing gradually", "service": "auth-service"}),
        ("query_logs", {"service": "auth-service", "filter": "session"}),
        ("predict_incident", {"service": "auth-service", "predicted_issue": "OOM crash due to session accumulation with ttl=-1", "confidence": 0.92}),
        ("set_proactive_alert", {"service": "auth-service", "metric": "memory_pct", "threshold": 85, "condition": "above"}),
        ("apply_fix", {"fix_type": "patch_session_ttl", "target": "auth-service"}),
        ("mark_resolved", {"root_cause": "session.ttl=-1 causing unbounded session accumulation",
                           "resolution_summary": "Patched session.ttl=3600. Memory growth stopped. Incident prevented before OOM."}),
    ],
}


def gradio_reset(task_id: str) -> str:
    import requests
    resp = requests.post(
        "http://localhost:7860/reset",
        json={"task_id": task_id},
    )
    obs = resp.json()
    _gradio_state["obs"] = obs
    _gradio_state["task_id"] = task_id
    return _fmt_obs(obs)


def gradio_step(action_type: str, params_json: str) -> str:
    import requests
    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as e:
        return f"Invalid JSON in parameters: {e}"

    resp = requests.post(
        "http://localhost:7860/step",
        json={"action_type": action_type, "parameters": params},
    )
    obs = resp.json()
    _gradio_state["obs"] = obs

    # Fire email if agent escalated
    if action_type == "escalate":
        send_escalation_email(
            team=params.get("team", "unknown"),
            message=params.get("message", ""),
            task_id=_gradio_state.get("task_id", "unknown"),
        )

    return _fmt_obs(obs)


def gradio_auto_run(task_id: str):
    """Run the mock agent automatically, yielding UI updates after each step."""
    import requests

    # Reset first
    resp = requests.post("http://localhost:7860/reset", json={"task_id": task_id})
    obs = resp.json()
    _gradio_state["obs"] = obs
    _gradio_state["task_id"] = task_id
    yield _fmt_obs(obs) + "\n\n---\n_Agent starting..._"
    time.sleep(1.5)

    script = MOCK_SCRIPTS.get(task_id, [])
    for action_type, params in script:
        resp = requests.post(
            "http://localhost:7860/step",
            json={"action_type": action_type, "parameters": params},
        )
        obs = resp.json()
        _gradio_state["obs"] = obs

        if action_type == "escalate":
            send_escalation_email(
                team=params.get("team", "unknown"),
                message=params.get("message", ""),
                task_id=task_id,
            )

        status = f"_Agent action: **{action_type}**_"
        yield _fmt_obs(obs) + f"\n\n---\n{status}"
        time.sleep(2)

        if obs.get("done"):
            break


ACTION_EXAMPLES = {
    "query_logs": '{"service": "auth-service", "filter": "WARN"}',
    "check_metrics": '{"service": "auth-service"}',
    "get_metric_trends": '{"service": "auth-service", "metric": "memory_pct", "window_minutes": 30}',
    "run_diagnostic": '{"tool": "heap_dump", "target": "auth-service"}',
    "apply_fix": '{"fix_type": "patch_session_ttl", "target": "auth-service"}',
    "acknowledge_alert": '{"alert_id": "ALT-001"}',
    "escalate": '{"team": "security", "message": "Possible data exfiltration in progress"}',
    "add_note": '{"text": "Memory trend: +2.2%/min for 30min. Sessions never expire (ttl=-1). Will OOM in ~13min."}',
    "lookup_runbook": '{"symptom": "memory increasing", "service": "auth-service"}',
    "predict_incident": '{"service": "auth-service", "predicted_issue": "OOM due to session accumulation with ttl=-1", "confidence": 0.9}',
    "set_proactive_alert": '{"service": "auth-service", "metric": "memory_pct", "threshold": 85, "condition": "above"}',
    "mark_resolved": '{"root_cause": "session.ttl=-1 causing unbounded session accumulation", "resolution_summary": "Patched session.ttl to 3600s. Redis TTL applied to all keys."}',
}


def fill_example(action_type: str) -> str:
    return ACTION_EXAMPLES.get(action_type, "{}")


with gr.Blocks(title="SREBench – Incident Response OpenEnv") as demo:
    gr.Markdown(
        """
# SREBench — Production Incident Response Environment
**OpenEnv-compatible | 4 Tasks: Easy → Medium → Hard + Predictive**

AI agent automatically diagnoses production incidents, applies fixes, escalates to humans when needed, and scores 0.0–1.0.
        """
    )

    with gr.Row():
        task_selector = gr.Dropdown(
            choices=VALID_TASKS,
            value="task3_ddos_exfil",
            label="Select Task",
        )
        auto_btn = gr.Button("Run AI Agent", variant="primary", scale=2)
        reset_btn = gr.Button("Reset", scale=1)

    observation_box = gr.Markdown(
        value="_Select a task and click **Run AI Agent** to watch the agent work autonomously._",
        label="Live Incident Feed",
    )

    gr.Markdown("### Manual Control")
    with gr.Row():
        action_type = gr.Dropdown(
            choices=list(ACTION_EXAMPLES.keys()),
            value="check_metrics",
            label="Action Type",
        )
        params_input = gr.Textbox(
            value='{}',
            label='Parameters (JSON)',
            lines=2,
        )

    with gr.Row():
        autofill_btn = gr.Button("Autofill Example", size="sm")
        step_btn = gr.Button("Step →")

    def live_update():
        obs = _gradio_state.get("obs")
        if obs is None:
            return "_Agent initializing..._"
        return _fmt_obs(obs)

    timer = gr.Timer(2)
    timer.tick(live_update, outputs=observation_box)

    action_type.change(fill_example, inputs=action_type, outputs=params_input)
    autofill_btn.click(fill_example, inputs=action_type, outputs=params_input)
    reset_btn.click(gradio_reset, inputs=task_selector, outputs=observation_box)
    step_btn.click(gradio_step, inputs=[action_type, params_input], outputs=observation_box)
    auto_btn.click(gradio_auto_run, inputs=task_selector, outputs=observation_box)

    gr.Markdown(
        """
---
**Task 3 (DDoS + Exfil):** Agent detects a DDoS is a distraction, finds hidden data exfiltration, applies fixes, and **emails the security team** automatically.

**Task 4 (Predictive):** No alerts firing — agent spots a memory trend and prevents the crash before it happens.
        """
    )


# ------------------------------------------------------------------
# Mount Gradio into FastAPI
# ------------------------------------------------------------------

app = gr.mount_gradio_app(app, demo, path="/")


def _background_agent_loop():
    """Runs all tasks in an infinite loop — auto-starts with the server."""
    import requests
    time.sleep(4)  # wait for server to be fully ready
    while True:
        for task_id in VALID_TASKS:
            try:
                requests.post("http://localhost:7860/reset", json={"task_id": task_id}, timeout=5)
                time.sleep(2)
                script = MOCK_SCRIPTS.get(task_id, [])
                for action_type, params in script:
                    requests.post(
                        "http://localhost:7860/step",
                        json={"action_type": action_type, "parameters": params},
                        timeout=5,
                    )
                    if action_type == "escalate":
                        send_escalation_email(
                            team=params.get("team", "unknown"),
                            message=params.get("message", ""),
                            task_id=task_id,
                        )
                    time.sleep(2.5)
            except Exception:
                time.sleep(2)
        time.sleep(5)  # short pause between full cycles


# Start the agent loop in background immediately at import time
_agent_thread = threading.Thread(target=_background_agent_loop, daemon=True)
_agent_thread.start()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
