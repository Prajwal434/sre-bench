"""
SREBench Baseline Inference Script
====================================
Runs an agent against all 3 SREBench tasks and reports reproducible scores.

Three backend modes (no OPENAI_API_KEY required for the first two):

  --mock      Deterministic scripted agent – zero API calls, always reproducible.
              Best for CI / judging demos.

  --ollama    Local Ollama model (free). Requires `ollama serve` running.
              Default model: llama3.2  Override with --model mistral etc.

  --openai    OpenAI API. Requires OPENAI_API_KEY env var.
              Default model: gpt-4o-mini

Usage:
  python baseline.py --mock
  python baseline.py --ollama
  python baseline.py --ollama --model mistral
  python baseline.py --openai --model gpt-4o-mini
  python baseline.py --task task2_db_cascade --mock
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterator, List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TASKS = [
    "task1_memory_leak",
    "task2_db_cascade",
    "task3_ddos_exfil",
    "task4_predictive",
]

EXPECTED_BASELINE_SCORES = {
    "task1_memory_leak": 0.74,
    "task2_db_cascade":  0.48,
    "task3_ddos_exfil":  0.22,
    "task4_predictive":  0.55,
}

# ---------------------------------------------------------------------------
# System prompt (shared by Ollama + OpenAI modes)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an experienced Site Reliability Engineer (SRE) responding to a production incident.

You have access to these actions:
- query_logs(service, filter="")                   – search log stream
- check_metrics(service="")                        – view CPU/mem/error/latency metrics
- get_metric_trends(service, metric="", window_minutes=30) – time-series trend data (use for proactive tasks)
- run_diagnostic(tool, target)                     – tools: heap_dump, explain_query, netstat, threat_intel,
                                                     check_migrations, check_api_keys, check_job_history,
                                                     waf_rules, network_capture, thread_dump, ping,
                                                     redis_info, config_check
- apply_fix(fix_type, target)                      – apply a named remediation
- acknowledge_alert(alert_id)                      – acknowledge an alert
- escalate(team, message)                          – page a team (security, database, network)
- add_note(text)                                   – append to incident timeline
- lookup_runbook(symptom, service="")              – search runbook database for guidance
- predict_incident(service, predicted_issue, confidence) – file a proactive incident prediction (0.0-1.0)
- set_proactive_alert(service, metric, threshold, condition) – register a proactive threshold alert
- mark_resolved(root_cause, resolution_summary)    – close incident (TERMINAL)

For proactive tasks (no active alerts): use get_metric_trends first to detect trends, then lookup_runbook, then predict_incident before fixing.

Rules:
1. Always start with check_metrics and query_logs to gather evidence.
2. Run diagnostics before applying any fix.
3. Always end with mark_resolved once fixed.
4. Respond with EXACTLY ONE raw JSON object — no markdown, no explanation.

Example: {"action_type": "query_logs", "parameters": {"service": "payment-service", "filter": "ERROR"}}
"""

OBSERVATION_TEMPLATE = """\
=== INCIDENT OBSERVATION (Step {step}/{max_steps}) ===

ACTIVE ALERTS:
{alerts}

SERVICE METRICS:
{metrics}

LAST ACTION RESULT:
{last_result}

TIMELINE (recent):
{timeline}
"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def fmt_observation(obs: Dict[str, Any]) -> str:
    alerts = obs.get("active_alerts", [])
    alert_lines = [
        f"  [{a['severity'].upper()}{'✓' if a.get('acknowledged') else ' '}]"
        f" {a['service']}: {a['message']}"
        for a in alerts
    ] or ["  (none)"]

    metrics = obs.get("metrics", [])
    metric_lines = [
        f"  {m['service']}: CPU={m['cpu_pct']}%"
        f" MEM={m['memory_pct']}%"
        f" ERR={m['error_rate']}/s"
        f" P99={m['p99_latency_ms']}ms"
        for m in metrics
    ] or ["  (none)"]

    timeline = obs.get("incident_timeline", [])[-6:]
    timeline_str = "\n".join(f"  {t}" for t in timeline) or "  (empty)"

    return OBSERVATION_TEMPLATE.format(
        alerts="\n".join(alert_lines),
        metrics="\n".join(metric_lines),
        timeline=timeline_str,
        last_result=obs.get("last_action_result", "(none)"),
        step=obs.get("step_count", 0),
        max_steps=obs.get("max_steps", 20),
    )


def _parse_action(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from model output."""
    raw = raw.strip()
    # Strip markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Find first {...}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _env_reset(session: requests.Session, env_url: str, task_id: str) -> Dict:
    r = session.post(f"{env_url}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def _env_step(session: requests.Session, env_url: str, action_type: str, parameters: dict) -> Dict:
    r = session.post(
        f"{env_url}/step",
        json={"action_type": action_type, "parameters": parameters},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _extract_grade(obs: Dict) -> tuple[float, dict, list]:
    score = obs.get("reward") or 0.0
    grade = obs.get("metadata", {}).get("grade", {})
    return score, grade.get("breakdown", {}), grade.get("feedback", [])


def _print_result(task_id: str, score: float, breakdown: dict, feedback: list, steps: int) -> None:
    print(f"\n  Score : {score:.3f}  (steps used: {steps})")
    if breakdown:
        print("  Grade breakdown:")
        for k, v in breakdown.items():
            bar = "█" * int(v * 20)
            print(f"    {k:<28} {v:.3f}  {bar}")
    if feedback:
        print("  Feedback:")
        for fb in feedback:
            print(f"    ! {fb}")


# ---------------------------------------------------------------------------
# MOCK AGENT
# ---------------------------------------------------------------------------

# Pre-scripted optimal action sequences per task
MOCK_SCRIPTS: Dict[str, List[Dict[str, Any]]] = {
    "task1_memory_leak": [
        {"action_type": "check_metrics", "parameters": {}},
        {"action_type": "query_logs", "parameters": {"service": "payment-service", "filter": "ERROR"}},
        {"action_type": "run_diagnostic", "parameters": {"tool": "heap_dump", "target": "payment-service"}},
        {"action_type": "acknowledge_alert", "parameters": {"alert_id": "ALT-001"}},
        {"action_type": "apply_fix", "parameters": {"fix_type": "patch_cache_config", "target": "payment-service"}},
        {"action_type": "add_note", "parameters": {"text": "Root cause: TransactionCache initialized with maxSize=-1 (unbounded). Fixed by patching maxSize=50000 with TTL eviction."}},
        {"action_type": "mark_resolved", "parameters": {
            "root_cause": "Unbounded TransactionCache grew to 2.8M entries causing Java heap OOM",
            "resolution_summary": "Patched TransactionCache: maxSize=50000, ttl=3600s. Service redeployed. Memory stable at 34%.",
        }},
    ],
    "task2_db_cascade": [
        {"action_type": "check_metrics", "parameters": {}},
        {"action_type": "query_logs", "parameters": {"service": "order-service"}},
        {"action_type": "query_logs", "parameters": {"service": "postgres-primary"}},
        {"action_type": "run_diagnostic", "parameters": {"tool": "explain_query", "target": "postgres-primary"}},
        {"action_type": "run_diagnostic", "parameters": {"tool": "check_migrations", "target": "postgres-primary"}},
        {"action_type": "acknowledge_alert", "parameters": {"alert_id": "ALT-101"}},
        {"action_type": "apply_fix", "parameters": {"fix_type": "reindex_invalid_index", "target": "postgres-primary"}},
        {"action_type": "add_note", "parameters": {"text": "Root cause: INVALID index orders_customer_id_idx (failed migration). All queries falling back to seq scan of 2.8M rows. Rebuilt via REINDEX CONCURRENTLY."}},
        {"action_type": "mark_resolved", "parameters": {
            "root_cause": "INVALID Postgres index orders_customer_id_idx causing full sequential scans on 2.8M row orders table",
            "resolution_summary": "Rebuilt orders_customer_id_idx via REINDEX CONCURRENTLY. Query time: 24.8s → 8ms. All downstream services recovered. CPU spike on order-service was a red herring (thread contention from DB waits).",
        }},
    ],
    "task4_predictive": [
        {"action_type": "get_metric_trends", "parameters": {"service": "auth-service", "window_minutes": 30}},
        {"action_type": "get_metric_trends", "parameters": {"service": "session-cache", "window_minutes": 30}},
        {"action_type": "lookup_runbook", "parameters": {"symptom": "memory increasing", "service": "auth-service"}},
        {"action_type": "query_logs", "parameters": {"service": "auth-service", "filter": "WARN"}},
        {"action_type": "run_diagnostic", "parameters": {"tool": "config_check", "target": "auth-service"}},
        {"action_type": "run_diagnostic", "parameters": {"tool": "redis_info", "target": "session-cache"}},
        {"action_type": "predict_incident", "parameters": {
            "service": "auth-service",
            "predicted_issue": "OOM due to session accumulation – session.ttl=-1 means sessions never expire, causing unbounded memory growth",
            "confidence": 0.92
        }},
        {"action_type": "set_proactive_alert", "parameters": {
            "service": "auth-service",
            "metric": "memory_pct",
            "threshold": 85,
            "condition": "above"
        }},
        {"action_type": "apply_fix", "parameters": {"fix_type": "patch_session_ttl", "target": "auth-service"}},
        {"action_type": "add_note", "parameters": {"text": "Proactive prevention: memory trend +2.2%/min for 30min. Root cause: session.ttl=-1. Fixed before OOM at step 9/12. Redis TTL applied to all existing keys."}},
        {"action_type": "mark_resolved", "parameters": {
            "root_cause": "session.ttl=-1 in auth-service config caused sessions to accumulate indefinitely in Redis and in-process SessionStore. Memory growing 2.2%/min, projected OOM in 13 minutes.",
            "resolution_summary": "Patched session.ttl to 3600s. Redis allkeys TTL applied. auth-service redeployed. Memory stabilising. Prevented P1 OOM incident proactively.",
        }},
    ],
    "task3_ddos_exfil": [
        {"action_type": "check_metrics", "parameters": {}},
        {"action_type": "query_logs", "parameters": {"service": "data-pipeline"}},
        {"action_type": "query_logs", "parameters": {"service": "auth-service-audit"}},
        {"action_type": "run_diagnostic", "parameters": {"tool": "netstat", "target": "data-pipeline"}},
        {"action_type": "run_diagnostic", "parameters": {"tool": "threat_intel", "target": "185.220.101.0/24"}},
        {"action_type": "run_diagnostic", "parameters": {"tool": "check_api_keys", "target": "data-pipeline"}},
        {"action_type": "apply_fix", "parameters": {"fix_type": "revoke_api_key", "target": "data-pipeline"}},
        {"action_type": "apply_fix", "parameters": {"fix_type": "rate_limit_auth_endpoint", "target": "api-gateway"}},
        {"action_type": "apply_fix", "parameters": {"fix_type": "block_tor_subnet", "target": "api-gateway"}},
        {"action_type": "apply_fix", "parameters": {"fix_type": "preserve_forensics", "target": "data-pipeline"}},
        {"action_type": "escalate", "parameters": {"team": "security", "message": "CRITICAL: Compromised admin account used to create API key and trigger export_users_full_v2, exfiltrating 2.4GB PII to Tor exit nodes (185.220.101.0/24). DDoS on /api/auth/login was a distraction. Exfil stopped, forensics preserved."}},
        {"action_type": "apply_fix", "parameters": {"fix_type": "reset_admin_password", "target": "auth-service"}},
        {"action_type": "mark_resolved", "parameters": {
            "root_cause": "Compromised admin account (admin@company.com) logged in from Tor exit node, created API key with data-pipeline access, triggered rogue export job exfiltrating 2.4GB PII. DDoS on auth endpoints was a distraction to overwhelm SRE.",
            "resolution_summary": "1) Revoked compromised API key dpk_f8a2c. 2) Rate limited /api/auth/login. 3) Blocked 185.220.101.0/24 subnet. 4) Preserved forensic evidence. 5) Escalated to security team. 6) Forced admin password reset. Exfiltration stopped at 2.4GB.",
        }},
    ],
}


def run_mock_agent(env_url: str, task_id: str, verbose: bool = True) -> Dict[str, Any]:
    """Run a pre-scripted deterministic agent. No API calls required."""
    session = requests.Session()

    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK  : {task_id}")
        print(f"MODE  : MOCK (deterministic scripted agent)")
        print(f"{'='*60}")

    obs = _env_reset(session, env_url, task_id)
    script = MOCK_SCRIPTS[task_id]
    actions_taken = []
    step = 0

    for action in script:
        if obs.get("done"):
            break
        step += 1
        at = action["action_type"]
        params = action["parameters"]

        if verbose:
            print(f"  [STEP {step}] {at}({json.dumps(params)[:70]})")

        obs = _env_step(session, env_url, at, params)
        actions_taken.append({"step": step, "action_type": at, "parameters": params})

        if verbose and obs.get("last_action_result"):
            print(f"         → {obs['last_action_result'][:100].replace(chr(10), ' ')}")

    final_score, breakdown, feedback = _extract_grade(obs)

    if verbose:
        _print_result(task_id, final_score, breakdown, feedback, step)

    return {
        "task_id": task_id,
        "mode": "mock",
        "score": final_score,
        "steps": step,
        "actions": actions_taken,
        "grade_breakdown": breakdown,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# OLLAMA AGENT
# ---------------------------------------------------------------------------

def _ollama_chat(model: str, messages: List[Dict], ollama_url: str) -> str:
    """Call local Ollama /api/chat endpoint."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    r = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def run_ollama_agent(
    env_url: str,
    task_id: str,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    max_steps: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run an LLM agent via local Ollama. Free, no API key needed."""
    session = requests.Session()

    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK  : {task_id}")
        print(f"MODE  : OLLAMA  model={model}  url={ollama_url}")
        print(f"{'='*60}")

    obs = _env_reset(session, env_url, task_id)
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    actions_taken = []
    final_score = 0.0
    breakdown: dict = {}
    feedback: list = []

    step = 0
    effective_max = max_steps or obs.get("max_steps", 20)
    parse_errors = 0

    while not obs.get("done") and step < effective_max:
        step += 1
        obs_text = fmt_observation(obs)
        messages.append({"role": "user", "content": obs_text})

        try:
            raw = _ollama_chat(model, messages, ollama_url)
        except Exception as e:
            print(f"  [ERROR] Ollama call failed: {e}")
            break

        messages.append({"role": "assistant", "content": raw})
        action = _parse_action(raw)

        if action is None:
            parse_errors += 1
            if verbose:
                print(f"  [STEP {step}] Parse error. Raw: {raw[:80]}")
            if parse_errors >= 3:
                print("  [ABORT] Too many parse errors.")
                break
            messages.append({
                "role": "user",
                "content": "ERROR: Invalid JSON. Reply with EXACTLY one JSON object, e.g.: {\"action_type\": \"check_metrics\", \"parameters\": {}}",
            })
            step -= 1  # don't count parse error as an env step
            continue

        parse_errors = 0
        at = action.get("action_type", "")
        params = action.get("parameters", {})

        if verbose:
            print(f"  [STEP {step}] {at}({json.dumps(params)[:70]})")

        obs = _env_step(session, env_url, at, params)
        actions_taken.append({"step": step, "action_type": at, "parameters": params})

        if verbose and obs.get("last_action_result"):
            print(f"         → {obs['last_action_result'][:100].replace(chr(10), ' ')}")

    # Finalise if not done yet
    if not obs.get("done"):
        obs = _env_step(session, env_url, "mark_resolved", {
            "root_cause": "episode truncated",
            "resolution_summary": "Truncated at step budget without full resolution.",
        })

    final_score, breakdown, feedback = _extract_grade(obs)

    if verbose:
        _print_result(task_id, final_score, breakdown, feedback, step)

    return {
        "task_id": task_id,
        "mode": f"ollama/{model}",
        "score": final_score,
        "steps": step,
        "actions": actions_taken,
        "grade_breakdown": breakdown,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# OPENAI AGENT
# ---------------------------------------------------------------------------

def run_openai_agent(
    env_url: str,
    task_id: str,
    model: str = "gpt-4o-mini",
    max_steps: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run an LLM agent via OpenAI API. Requires OPENAI_API_KEY."""
    try:
        from openai import OpenAI as _OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    client = _OpenAI(api_key=api_key)
    session = requests.Session()

    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK  : {task_id}")
        print(f"MODE  : OPENAI  model={model}")
        print(f"{'='*60}")

    obs = _env_reset(session, env_url, task_id)
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    actions_taken = []
    parse_errors = 0

    step = 0
    effective_max = max_steps or obs.get("max_steps", 20)

    while not obs.get("done") and step < effective_max:
        step += 1
        messages.append({"role": "user", "content": fmt_observation(obs)})

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=300,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [ERROR] OpenAI call failed: {e}")
            break

        messages.append({"role": "assistant", "content": raw})
        action = _parse_action(raw)

        if action is None:
            parse_errors += 1
            if verbose:
                print(f"  [STEP {step}] Parse error. Raw: {raw[:80]}")
            if parse_errors >= 3:
                break
            messages.append({
                "role": "user",
                "content": "ERROR: Invalid JSON. Reply with one JSON object only.",
            })
            step -= 1
            continue

        parse_errors = 0
        at = action.get("action_type", "")
        params = action.get("parameters", {})

        if verbose:
            print(f"  [STEP {step}] {at}({json.dumps(params)[:70]})")

        obs = _env_step(session, env_url, at, params)
        actions_taken.append({"step": step, "action_type": at, "parameters": params})

        if verbose and obs.get("last_action_result"):
            print(f"         → {obs['last_action_result'][:100].replace(chr(10), ' ')}")

    if not obs.get("done"):
        obs = _env_step(session, env_url, "mark_resolved", {
            "root_cause": "episode truncated",
            "resolution_summary": "Truncated at step budget.",
        })

    final_score, breakdown, feedback = _extract_grade(obs)

    if verbose:
        _print_result(task_id, final_score, breakdown, feedback, step)

    return {
        "task_id": task_id,
        "mode": f"openai/{model}",
        "score": final_score,
        "steps": step,
        "actions": actions_taken,
        "grade_breakdown": breakdown,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SREBench Baseline – runs agents against all 3 tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python baseline.py --mock
  python baseline.py --ollama
  python baseline.py --ollama --model mistral
  python baseline.py --openai --model gpt-4o-mini
  python baseline.py --mock --task task1_memory_leak
        """,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--mock",   action="store_true", help="Deterministic scripted agent (no API key needed)")
    mode_group.add_argument("--ollama", action="store_true", help="Local Ollama LLM (free, no API key)")
    mode_group.add_argument("--openai", action="store_true", help="OpenAI API (requires OPENAI_API_KEY)")

    parser.add_argument("--model",     default=None,    help="Model name (ollama: llama3.2, openai: gpt-4o-mini)")
    parser.add_argument("--task",      choices=VALID_TASKS, help="Run one task only")
    parser.add_argument("--max-steps", type=int, default=None, help="Override step budget")
    parser.add_argument("--url",       default=None,    help="SREBench server URL")
    parser.add_argument("--ollama-url",default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--quiet",     action="store_true", help="Suppress per-step output")
    parser.add_argument("--output",    default="baseline_results.json", help="Output JSON file")
    args = parser.parse_args()

    # Default mode: mock
    if not args.mock and not args.ollama and not args.openai:
        args.mock = True

    env_url = (args.url or os.environ.get("SREBENCH_URL", "http://localhost:7860")).rstrip("/")

    # Verify server
    try:
        r = requests.get(f"{env_url}/health", timeout=10)
        r.raise_for_status()
        print(f"Connected to SREBench at {env_url}")
    except Exception as e:
        print(f"ERROR: Cannot reach SREBench at {env_url}: {e}")
        print("Start with:  uvicorn server.app:app --host 0.0.0.0 --port 7860")
        return 1

    # Verify Ollama is running (if selected)
    if args.ollama:
        ollama_url = args.ollama_url
        try:
            r = requests.get(f"{ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
            available = [m["name"] for m in r.json().get("models", [])]
            print(f"Ollama running. Available models: {available or '(none pulled yet)'}")
        except Exception as e:
            print(f"ERROR: Cannot reach Ollama at {ollama_url}: {e}")
            print("Start with:  ollama serve")
            print("Pull model:  ollama pull llama3.2")
            return 1

    tasks = [args.task] if args.task else VALID_TASKS
    results = []

    for task_id in tasks:
        if args.mock:
            result = run_mock_agent(env_url, task_id, verbose=not args.quiet)
        elif args.ollama:
            model = args.model or "llama3.2"
            result = run_ollama_agent(
                env_url, task_id, model=model,
                ollama_url=args.ollama_url,
                max_steps=args.max_steps,
                verbose=not args.quiet,
            )
        else:  # openai
            model = args.model or "gpt-4o-mini"
            result = run_openai_agent(
                env_url, task_id, model=model,
                max_steps=args.max_steps,
                verbose=not args.quiet,
            )
        results.append(result)
        time.sleep(0.5)

    # Summary
    mode_label = (
        "mock" if args.mock
        else f"ollama/{args.model or 'llama3.2'}" if args.ollama
        else f"openai/{args.model or 'gpt-4o-mini'}"
    )
    print(f"\n{'='*62}")
    print(f"BASELINE RESULTS  [{mode_label}]")
    print(f"{'='*62}")
    print(f"{'Task':<30} {'Score':>7}  {'Expected':>9}  {'Steps':>6}")
    print(f"{'-'*62}")

    total = 0.0
    for r in results:
        expected = EXPECTED_BASELINE_SCORES.get(r["task_id"], 0.0)
        print(f"{r['task_id']:<30} {r['score']:>7.3f}  {expected:>9.3f}  {r['steps']:>6}")
        total += r["score"]

    avg = total / len(results) if results else 0.0
    print(f"{'-'*62}")
    print(f"{'AVERAGE':<30} {avg:>7.3f}")
    print(f"{'='*62}")

    with open(args.output, "w") as f:
        json.dump({
            "mode": mode_label,
            "env_url": env_url,
            "results": results,
            "summary": {"average_score": round(avg, 4), "tasks": len(results)},
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
