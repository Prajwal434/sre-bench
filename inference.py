"""
SREBench – Inference Script
============================
MANDATORY env vars:
  API_BASE_URL      The API endpoint for the LLM  (default: https://router.huggingface.co/v1)
  MODEL_NAME        The model identifier           (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN          Your HuggingFace / API key
  LOCAL_IMAGE_NAME  Local docker image name (unused – SREBench runs as HTTP server)
  SREBENCH_URL      SREBench server URL            (default: http://localhost:7860)

STDOUT FORMAT (mandatory – do not change field names or order):
  [START] task=<task_name> env=sre-bench model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<null|msg>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

When API_BASE_URL / MODEL_NAME / HF_TOKEN are not set, falls back to deterministic MOCK
mode (no API calls, always reproducible, completes in < 30 seconds).
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN:     Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")  # unused – we use HTTP
SREBENCH_URL: str = os.getenv("SREBENCH_URL", "http://localhost:7860").rstrip("/")

BENCHMARK   = "sre-bench"
MAX_STEPS   = 20          # safety cap (each task has its own lower budget)
TEMPERATURE = 0.0
MAX_TOKENS  = 350

VALID_TASKS = [
    "task1_memory_leak",
    "task2_db_cascade",
    "task3_ddos_exfil",
    "task4_predictive",
]

# ---------------------------------------------------------------------------
# Mandatory stdout log helpers  (exact field names / order required)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Truncate action to keep line readable
    action_str = action.replace("\n", " ")[:120]
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# SREBench HTTP helpers
# ---------------------------------------------------------------------------

def _reset(session: requests.Session, task_id: str) -> Dict[str, Any]:
    r = session.post(f"{SREBENCH_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def _step(session: requests.Session, action_type: str, parameters: dict) -> Dict[str, Any]:
    r = session.post(
        f"{SREBENCH_URL}/step",
        json={"action_type": action_type, "parameters": parameters},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _action_str(action_type: str, parameters: dict) -> str:
    """Compact action string for [STEP] log line."""
    if parameters:
        params_compact = json.dumps(parameters, separators=(",", ":"))
        return f"{action_type}({params_compact})"
    return f"{action_type}()"


def _extract_score(obs: Dict[str, Any]) -> float:
    return float(obs.get("reward") or 0.0)


# ---------------------------------------------------------------------------
# System prompt for LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an experienced Site Reliability Engineer (SRE) responding to a production incident.

    Available actions:
    - query_logs(service, filter="")
    - check_metrics(service="")
    - get_metric_trends(service, metric="", window_minutes=30)
    - run_diagnostic(tool, target)   [tools: heap_dump, explain_query, netstat, threat_intel,
                                       check_migrations, check_api_keys, redis_info, config_check,
                                       check_job_history, waf_rules, network_capture]
    - apply_fix(fix_type, target)
    - acknowledge_alert(alert_id)
    - escalate(team, message)
    - add_note(text)
    - lookup_runbook(symptom, service="")
    - predict_incident(service, predicted_issue, confidence)
    - set_proactive_alert(service, metric, threshold, condition)
    - mark_resolved(root_cause, resolution_summary)   ← TERMINAL

    Strategy:
    1. Start with check_metrics (or get_metric_trends for proactive tasks with no alerts).
    2. Query logs and run diagnostics before applying any fix.
    3. For proactive tasks (no active alerts): use get_metric_trends → lookup_runbook → predict_incident → fix.
    4. Always end with mark_resolved containing a clear root_cause and resolution_summary.
    5. Be efficient – you have a limited step budget.

    Reply with EXACTLY ONE raw JSON object, no markdown, no explanation:
    {"action_type": "...", "parameters": {...}}
""").strip()


def _fmt_obs(obs: Dict[str, Any]) -> str:
    alerts = "; ".join(
        f"[{a['severity'].upper()}] {a['service']}: {a['message'][:80]}"
        for a in obs.get("active_alerts", [])
    ) or "(none – check metric trends for proactive task)"

    metrics = " | ".join(
        f"{m['service']} CPU={m['cpu_pct']}% MEM={m['memory_pct']}% ERR={m['error_rate']}/s P99={m['p99_latency_ms']}ms"
        for m in obs.get("metrics", [])
    ) or "(none)"

    timeline = "\n".join(obs.get("incident_timeline", [])[-5:]) or "(empty)"
    result = obs.get("last_action_result", "")[:300]
    step = obs.get("step_count", 0)
    max_steps = obs.get("max_steps", 20)

    return (
        f"=== STEP {step}/{max_steps} ===\n"
        f"ALERTS: {alerts}\n"
        f"METRICS: {metrics}\n"
        f"LAST RESULT:\n{result}\n"
        f"TIMELINE:\n{timeline}\n"
    )


def _parse_action(raw: str) -> Optional[Dict[str, Any]]:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# LLM agent  (OpenAI client → API_BASE_URL)
# ---------------------------------------------------------------------------

def run_llm_episode(task_id: str, client: OpenAI) -> Dict[str, Any]:
    session   = requests.Session()
    rewards:  List[float] = []
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    parse_errors = 0
    step = 0
    score = 0.0
    success = False
    error_msg: Optional[str] = None

    log_start(task_id, BENCHMARK, MODEL_NAME)

    try:
        obs = _reset(session, task_id)
        max_steps = obs.get("max_steps", MAX_STEPS)

        while not obs.get("done") and step < max_steps:
            step += 1
            messages.append({"role": "user", "content": _fmt_obs(obs)})

            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stream=False,
                )
                raw = (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                error_msg = str(exc)[:80]
                log_step(step, "llm_error", 0.0, False, error_msg)
                rewards.append(0.0)
                break

            messages.append({"role": "assistant", "content": raw})
            action = _parse_action(raw)

            if action is None:
                parse_errors += 1
                error_msg = f"parse_error({parse_errors})"
                log_step(step, raw[:60], 0.0, False, error_msg)
                rewards.append(0.0)
                if parse_errors >= 3:
                    break
                messages.append({"role": "user", "content": "ERROR: Reply with one JSON object only, e.g. {\"action_type\": \"check_metrics\", \"parameters\": {}}"})
                step -= 1
                continue

            parse_errors = 0
            at     = action.get("action_type", "mark_resolved")
            params = action.get("parameters", {})
            obs    = _step(session, at, params)

            step_reward = _extract_score(obs) if obs.get("done") else 0.0
            done        = obs.get("done", False)
            rewards.append(step_reward)

            log_step(step, _action_str(at, params), step_reward, done, None)

        # Force close if still running
        if not obs.get("done"):
            obs = _step(session, "mark_resolved", {
                "root_cause": "truncated",
                "resolution_summary": "Episode truncated at step budget.",
            })
            step_reward = _extract_score(obs)
            rewards.append(step_reward)
            log_step(step + 1, "mark_resolved(truncated)", step_reward, True, "truncated")

        score   = _extract_score(obs)
        success = score > 0.0

    except Exception as exc:
        error_msg = str(exc)[:100]
        score, success = 0.0, False

    finally:
        log_end(success, step, score, rewards)

    return {"task_id": task_id, "score": score, "steps": step, "success": success}


# ---------------------------------------------------------------------------
# MOCK agent  (deterministic, no API calls)
# ---------------------------------------------------------------------------

MOCK_SCRIPTS: Dict[str, List[Dict[str, Any]]] = {
    "task1_memory_leak": [
        {"action_type": "check_metrics",       "parameters": {}},
        {"action_type": "query_logs",           "parameters": {"service": "payment-service", "filter": "ERROR"}},
        {"action_type": "run_diagnostic",       "parameters": {"tool": "heap_dump", "target": "payment-service"}},
        {"action_type": "acknowledge_alert",    "parameters": {"alert_id": "ALT-001"}},
        {"action_type": "apply_fix",            "parameters": {"fix_type": "patch_cache_config", "target": "payment-service"}},
        {"action_type": "add_note",             "parameters": {"text": "Root cause: TransactionCache maxSize=-1 (unbounded). Patched to 50000 with TTL."}},
        {"action_type": "mark_resolved",        "parameters": {
            "root_cause": "Unbounded TransactionCache grew to 2.8M entries causing Java heap OOM",
            "resolution_summary": "Patched TransactionCache: maxSize=50000, ttl=3600s. Memory stable at 34%."}},
    ],
    "task2_db_cascade": [
        {"action_type": "check_metrics",        "parameters": {}},
        {"action_type": "query_logs",           "parameters": {"service": "order-service"}},
        {"action_type": "query_logs",           "parameters": {"service": "postgres-primary"}},
        {"action_type": "run_diagnostic",       "parameters": {"tool": "explain_query", "target": "postgres-primary"}},
        {"action_type": "run_diagnostic",       "parameters": {"tool": "check_migrations", "target": "postgres-primary"}},
        {"action_type": "acknowledge_alert",    "parameters": {"alert_id": "ALT-101"}},
        {"action_type": "apply_fix",            "parameters": {"fix_type": "reindex_invalid_index", "target": "postgres-primary"}},
        {"action_type": "add_note",             "parameters": {"text": "Root cause: INVALID index orders_customer_id_idx (failed migration). Rebuilt via REINDEX CONCURRENTLY."}},
        {"action_type": "mark_resolved",        "parameters": {
            "root_cause": "INVALID Postgres index causing full sequential scans on 2.8M row orders table",
            "resolution_summary": "Rebuilt index via REINDEX CONCURRENTLY. Query time 24.8s → 8ms. All services recovered."}},
    ],
    "task3_ddos_exfil": [
        {"action_type": "check_metrics",        "parameters": {}},
        {"action_type": "query_logs",           "parameters": {"service": "data-pipeline"}},
        {"action_type": "query_logs",           "parameters": {"service": "auth-service-audit"}},
        {"action_type": "run_diagnostic",       "parameters": {"tool": "netstat", "target": "data-pipeline"}},
        {"action_type": "run_diagnostic",       "parameters": {"tool": "threat_intel", "target": "185.220.101.0/24"}},
        {"action_type": "run_diagnostic",       "parameters": {"tool": "check_api_keys", "target": "data-pipeline"}},
        {"action_type": "apply_fix",            "parameters": {"fix_type": "revoke_api_key", "target": "data-pipeline"}},
        {"action_type": "apply_fix",            "parameters": {"fix_type": "rate_limit_auth_endpoint", "target": "api-gateway"}},
        {"action_type": "apply_fix",            "parameters": {"fix_type": "block_tor_subnet", "target": "api-gateway"}},
        {"action_type": "apply_fix",            "parameters": {"fix_type": "preserve_forensics", "target": "data-pipeline"}},
        {"action_type": "escalate",             "parameters": {"team": "security", "message": "CRITICAL: Compromised admin account exfiltrating 2.4GB PII to Tor nodes via data-pipeline. DDoS is a distraction."}},
        {"action_type": "apply_fix",            "parameters": {"fix_type": "reset_admin_password", "target": "auth-service"}},
        {"action_type": "mark_resolved",        "parameters": {
            "root_cause": "Compromised admin account created API key for data exfiltration; DDoS as distraction",
            "resolution_summary": "Revoked API key, rate limited auth, blocked Tor subnet, escalated to security, reset admin password."}},
    ],
    "task4_predictive": [
        {"action_type": "get_metric_trends",    "parameters": {"service": "auth-service", "window_minutes": 30}},
        {"action_type": "get_metric_trends",    "parameters": {"service": "session-cache", "window_minutes": 30}},
        {"action_type": "lookup_runbook",       "parameters": {"symptom": "memory increasing", "service": "auth-service"}},
        {"action_type": "query_logs",           "parameters": {"service": "session-cache"}},
        {"action_type": "run_diagnostic",       "parameters": {"tool": "config_check", "target": "auth-service"}},
        {"action_type": "run_diagnostic",       "parameters": {"tool": "redis_info", "target": "session-cache"}},
        {"action_type": "predict_incident",     "parameters": {
            "service": "auth-service",
            "predicted_issue": "OOM due to session accumulation with ttl=-1 causing unbounded memory growth",
            "confidence": 0.92}},
        {"action_type": "set_proactive_alert",  "parameters": {
            "service": "auth-service", "metric": "memory_pct", "threshold": 85, "condition": "above"}},
        {"action_type": "apply_fix",            "parameters": {"fix_type": "patch_session_ttl", "target": "auth-service"}},
        {"action_type": "add_note",             "parameters": {"text": "Proactive prevention: memory trend +2.2%/min for 30min. session.ttl=-1. Fixed before OOM at step 9/12."}},
        {"action_type": "mark_resolved",        "parameters": {
            "root_cause": "session.ttl=-1 in auth-service config caused sessions to accumulate indefinitely",
            "resolution_summary": "Patched session.ttl=3600s. Redis TTL applied. Prevented P1 OOM incident proactively."}},
    ],
}


def run_mock_episode(task_id: str) -> Dict[str, Any]:
    """Deterministic scripted agent — no API calls, always reproducible."""
    session  = requests.Session()
    rewards: List[float] = []
    step    = 0
    score   = 0.0
    success = False

    log_start(task_id, BENCHMARK, "mock-deterministic-v1")

    try:
        obs    = _reset(session, task_id)
        script = MOCK_SCRIPTS[task_id]

        for action in script:
            if obs.get("done"):
                break
            step += 1
            at     = action["action_type"]
            params = action["parameters"]
            obs    = _step(session, at, params)

            step_reward = _extract_score(obs) if obs.get("done") else 0.0
            done        = obs.get("done", False)
            rewards.append(step_reward)

            log_step(step, _action_str(at, params), step_reward, done, None)

        score   = _extract_score(obs)
        success = score > 0.0

    except Exception as exc:
        log_step(step + 1, "error", 0.0, True, str(exc)[:80])
        score, success = 0.0, False

    finally:
        log_end(success, step, score, rewards)

    return {"task_id": task_id, "score": score, "steps": step, "success": success}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="SREBench inference script")
    parser.add_argument("--task", choices=VALID_TASKS, help="Run one task (default: all)")
    parser.add_argument("--mock", action="store_true",
                        help="Force mock mode even if env vars are present")
    args = parser.parse_args()

    # Mode selection: LLM if all three vars are set AND --mock not forced
    use_llm = bool(API_BASE_URL and MODEL_NAME and HF_TOKEN) and not args.mock
    # Even if api_base_url has the default value but HF_TOKEN is missing, fall back to mock
    if not HF_TOKEN:
        use_llm = False

    # Verify server is reachable
    try:
        r = requests.get(f"{SREBENCH_URL}/health", timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: SREBench server unreachable at {SREBENCH_URL}: {e}", file=sys.stderr)
        print("Start with: uvicorn server.app:app --host 0.0.0.0 --port 7860", file=sys.stderr)
        return 1

    if use_llm:
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    else:
        client = None  # mock mode

    tasks   = [args.task] if args.task else VALID_TASKS
    results = []

    for task_id in tasks:
        try:
            if use_llm:
                result = run_llm_episode(task_id, client)
            else:
                result = run_mock_episode(task_id)
            results.append(result)
        except Exception as exc:
            print(f"ERROR: {task_id} failed: {exc}", file=sys.stderr, flush=True)
            results.append({"task_id": task_id, "score": 0.0, "steps": 0, "success": False})
        time.sleep(0.2)

    # Print summary to stderr so it doesn't pollute the validator's stdout parse
    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\n# Results: avg={avg:.4f}", file=sys.stderr, flush=True)
    for r in results:
        print(f"#   {r['task_id']}: {r['score']:.4f} ({r['steps']} steps)", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
