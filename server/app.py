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
import threading
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
    return {"status": "ok", "env": "sre-bench"}


@app.post("/reset", response_model=IRObservation)
def reset(req: ResetRequest):
    with _env_lock:
        try:
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
    return _fmt_obs(obs)


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
**OpenEnv-compatible | 3 Tasks: Easy → Medium → Hard**

Simulate an SRE engineer responding to production incidents. Diagnose root causes,
apply remediations, escalate when needed, and document resolutions. Score: 0.0–1.0.
        """
    )

    with gr.Row():
        task_selector = gr.Dropdown(
            choices=VALID_TASKS,
            value="task1_memory_leak",
            label="Select Task",
        )
        reset_btn = gr.Button("Reset / New Episode", variant="primary")

    observation_box = gr.Markdown(
        value="_Click **Reset** to start an episode._",
        label="Observation",
    )

    gr.Markdown("### Take an Action")
    with gr.Row():
        action_type = gr.Dropdown(
            choices=list(ACTION_EXAMPLES.keys()),
            value="get_metric_trends",
            label="Action Type",
        )
        params_input = gr.Textbox(
            value='{"service": "payment-service", "filter": "ERROR"}',
            label='Parameters (JSON)',
            lines=2,
        )

    with gr.Row():
        autofill_btn = gr.Button("Autofill Example", size="sm")
        step_btn = gr.Button("Step →", variant="primary")

    action_type.change(fill_example, inputs=action_type, outputs=params_input)
    autofill_btn.click(fill_example, inputs=action_type, outputs=params_input)
    reset_btn.click(gradio_reset, inputs=task_selector, outputs=observation_box)
    step_btn.click(gradio_step, inputs=[action_type, params_input], outputs=observation_box)

    gr.Markdown(
        """
---
### API Usage
```bash
# Reset
curl -X POST http://localhost:7860/reset -H 'Content-Type: application/json' \\
  -d '{"task_id": "task1_memory_leak"}'

# Step
curl -X POST http://localhost:7860/step -H 'Content-Type: application/json' \\
  -d '{"action_type": "query_logs", "parameters": {"service": "payment-service"}}'

# State
curl http://localhost:7860/state
```
        """
    )


# ------------------------------------------------------------------
# Mount Gradio into FastAPI
# ------------------------------------------------------------------

app = gr.mount_gradio_app(app, demo, path="/")


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
