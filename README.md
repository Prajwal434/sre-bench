---
title: SREBench
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - sre
  - incident-response
  - agent-evaluation
---

# SREBench — Production Incident Response OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/prajwal434/sre-bench)
[![GitHub](https://img.shields.io/badge/GitHub-Prajwal434%2Fsre--bench-black)](https://github.com/Prajwal434/sre-bench)

> **Train and evaluate AI agents on real SRE on-call workflows.**  
> Not a game. Not a toy. Production incidents, deterministic graders, partial reward signals.

---

## What is SREBench?

SREBench simulates a **Site Reliability Engineer responding to production incidents**. The agent receives live alerts, logs, and metrics from a simulated microservice environment, then must diagnose root causes and apply remediations — exactly what engineers do every day at scale.

This is a domain where:
- Billions of dollars in engineering time are spent annually
- Mistakes have real consequences (data loss, prolonged outages, security breaches)
- Multi-step reasoning, tool use, and causal thinking are all required
- Partial progress is meaningful (identifying the right service vs. correct root cause vs. correct fix)

---

## Novel Features

### 1. Runbook / Playbook Lookup System

Real SRE teams maintain runbooks — step-by-step guides for known failure patterns. SREBench models this with a `lookup_runbook` action that searches a library of 4 runbooks by symptom and service:

| Runbook | Covers |
|---------|--------|
| RB-001 | OOM / memory leak diagnosis |
| RB-002 | Database performance degradation |
| RB-003 | DDoS attack response |
| RB-004 | Proactive memory trend prevention |

Agents that consult runbooks before acting perform better — mirroring real on-call behavior.

### 2. Predictive Incident Prevention (Task 4)

**A new class of SRE task:** instead of reacting to an active incident, the agent must detect a developing problem from metric trends and intervene *before* it causes an outage.

- The agent observes memory growing at +2.2%/min on `auth-service` (sessions never expire: `ttl=-1`)
- No alerts are firing yet — the agent must notice the trend and act proactively
- A simulated failure occurs at step 12 if the agent does nothing
- **Proactive speed bonus:** the earlier the correct fix is applied, the higher the score

This tests a fundamentally different capability: **prediction over reaction**.

---

## Environment Description

The agent acts as an SRE on-call engineer with access to:

| Tool | What it does |
|------|-------------|
| `query_logs` | Search log streams for any service |
| `check_metrics` | View CPU, memory, error rate, latency for services |
| `get_metric_trends` | Get historical metric trends over a time window |
| `run_diagnostic` | Run tools: `heap_dump`, `explain_query`, `netstat`, `threat_intel`, etc. |
| `apply_fix` | Apply a named remediation to a service |
| `acknowledge_alert` | Acknowledge a PagerDuty-style alert |
| `escalate` | Page another team (security, DBA, network) |
| `add_note` | Add a note to the incident timeline |
| `lookup_runbook` | Search runbook library by symptom and service |
| `predict_incident` | File a prediction of an upcoming failure |
| `set_proactive_alert` | Set a metric threshold alert before failure occurs |
| `mark_resolved` | Close the incident with root cause + summary (terminal) |

### Action Space

```python
class IRAction(BaseModel):
    action_type: Literal[
        "query_logs", "check_metrics", "get_metric_trends", "run_diagnostic",
        "apply_fix", "acknowledge_alert", "escalate", "add_note",
        "lookup_runbook", "predict_incident", "set_proactive_alert", "mark_resolved"
    ]
    parameters: Dict[str, Any]
```

### Observation Space

```python
class IRObservation(BaseModel):
    done: bool
    reward: Optional[float]          # None during episode, 0.0–1.0 at termination
    active_alerts: List[Alert]        # PagerDuty-style alerts with severity
    log_results: List[LogEntry]       # Results of query_logs actions
    metrics: List[ServiceMetric]      # Per-service health metrics
    incident_timeline: List[str]      # Recent action history
    last_action_result: str           # Result of the last action taken
    step_count: int
    max_steps: int
    trend_data: Optional[Dict]        # Metric trend data (Task 4)
    runbook_results: Optional[List]   # Runbook search results
    metadata: Dict[str, Any]          # Includes partial_score and final grade
```

---

## Tasks

### Task 1 — Memory Leak OOM (Easy)

**Scenario:** The `payment-service` is crashing with `OOMKilled` errors. It is restarting 4 times in 10 minutes. The `api-gateway` is returning 502 errors.

**Root cause:** The `TransactionCache` was initialized with `maxSize=-1` (unbounded). It has grown to 2.8M entries with no TTL eviction, consuming all available heap.

**Correct fix:** Patch the cache config to set `maxSize=50000` and enable TTL eviction.

**Red herring:** None — this is a straightforward OOM investigation.

**Step budget:** 15 | **Expected agent score (GPT-4o-mini):** ~0.74

---

### Task 2 — Database Cascade Failure (Medium)

**Scenario:** Frontend is returning 503 errors. `order-service` has a CPU spike to 89% and extreme latency. `inventory-service` is timing out. Multiple services degraded simultaneously.

**Root cause:** A database migration left the `orders_customer_id_idx` index in an `INVALID` state. All queries are falling back to full sequential scans of a 2.8M row table, taking 25 seconds each.

**Correct fix:** `REINDEX CONCURRENTLY` to rebuild the invalid index.

**Red herring:** The CPU spike on `order-service` looks like a compute problem, but it is caused by 200 threads blocking on slow database queries.

**Step budget:** 20 | **Expected agent score (GPT-4o-mini):** ~0.48

---

### Task 3 — DDoS Attack with Data Exfiltration (Hard)

**Scenario:** Auth endpoints are being hammered at 48,000 req/s (normal: 200 req/s). Simultaneously, `data-pipeline` is transferring 2.4GB to Tor exit nodes. An `info` alert is easy to miss.

**Root cause (multi-vector):**
1. A compromised admin account logged in from a Tor exit node
2. The attacker created a new API key with data-pipeline access
3. A rogue export job (`export_users_full_v2`) is exfiltrating the entire users table (PII)
4. The DDoS on auth endpoints is a **distraction** to overwhelm the SRE team

**Required actions:** Rate-limit auth endpoint + revoke compromised API key + escalate to security team.

**Red herring:** Focusing exclusively on the DDoS without investigating the exfiltration alert.

**Step budget:** 25 | **Expected agent score (GPT-4o-mini):** ~0.22

---

### Task 4 — Proactive Incident Prevention (Medium / Novel)

**Scenario:** No active alerts. Metric trends show `auth-service` memory growing steadily at +2.2%/min. Sessions are accumulating because `session.ttl=-1` (never expire). If uncorrected, OOM crash occurs at step 12.

**Goal:** Detect the trend, predict the failure, and apply the fix *before* the crash happens.

**Correct fix:** `patch_session_ttl` on `auth-service`.

**Unique mechanic:** Proactive speed bonus — score increases the earlier the agent acts. Acting at step 3 scores higher than acting at step 10.

**Step budget:** 15

---

## Reward Function

Rewards are computed at episode end by a deterministic grader. Partial credit is given throughout:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| `service_identification` | 0.10 | Did the agent identify the correct root service? |
| `root_cause_diagnosis` | 0.25 | Did the agent correctly name the root cause? |
| `fix_applied` | 0.30 | Was the correct (or acceptable) fix applied? |
| `resolution_speed` | 0.15 | How efficiently was the incident resolved? |
| `escalation` | 0.10 | Was the right team escalated to (when required)? |
| `documentation` | 0.10 | Did the resolution note reference the root cause? |
| `penalty` | -0.05× | Applied per wrong fix / destructive action |

**Task 4 replaces `resolution_speed` with `proactive_speed`:** `(failure_step - act_step) / (failure_step - 1)`

**Final score is always in [0.0, 1.0].**

---

## Setup & Usage

### Option 1: HuggingFace Space (live now)

```python
from client import SREBenchClient

env = SREBenchClient(base_url="https://prajwal434-sre-bench.hf.space")
obs = env.reset(task_id="task1_memory_leak")
```

### Option 2: Docker

```bash
git clone https://github.com/Prajwal434/sre-bench
cd sre-bench

docker build -t sre-bench .
docker run -p 7860:7860 sre-bench
```

Server starts at `http://localhost:7860`. Gradio UI available at `/`.

### Option 3: Local Python

```bash
cd sre-bench
pip install -r requirements.txt

uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## Running the Baseline

```bash
# Mock mode — no API key needed, deterministic optimal agent
python inference.py

# With a real LLM
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_...
export SREBENCH_URL=http://localhost:7860

python inference.py
```

### Reproducible Baseline Scores (mock-deterministic-v1)

```
Task                           Score    Steps
-----------------------------------------------
task1_memory_leak              0.930      7
task2_db_cascade               0.870     10
task3_ddos_exfil               0.820     12
task4_predictive               0.940      5
-----------------------------------------------
AVERAGE                        0.890
```

---

## API Reference

```bash
# Health check
GET /health

# Environment metadata
GET /metadata

# Action / observation / state schemas
GET /schema

# List tasks
GET /tasks

# Reset (start new episode)
POST /reset
{"task_id": "task1_memory_leak", "episode_id": null, "seed": null}

# Step (take an action)
POST /step
{"action_type": "query_logs", "parameters": {"service": "payment-service", "filter": "ERROR"}}

# State
GET /state

# MCP (JSON-RPC 2.0)
POST /mcp
{"jsonrpc": "2.0", "method": "tools/list", "id": 1}
```

### Example Agent Loop (Python)

```python
from client import SREBenchClient

env = SREBenchClient(base_url="http://localhost:7860")

obs = env.reset(task_id="task2_db_cascade")
print(obs.last_action_result)

obs = env.step("check_metrics")
obs = env.step("query_logs", service="order-service")
obs = env.step("run_diagnostic", tool="explain_query", target="postgres-primary")
obs = env.step("apply_fix", fix_type="reindex_invalid_index", target="postgres-primary")
obs = env.step("mark_resolved",
    root_cause="invalid index on orders table causing full seq scans",
    resolution_summary="Rebuilt orders_customer_id_idx via REINDEX CONCURRENTLY. "
                       "Query times dropped from 25s to 8ms. All services recovered.")

print(f"Final score: {obs.reward:.3f}")
```

---

## Project Structure

```
sre-bench/
├── openenv.yaml              ← OpenEnv manifest
├── Dockerfile                ← Container definition
├── pyproject.toml            ← Package metadata + uv support
├── requirements.txt
├── README.md
├── models.py                 ← Pydantic models (IRAction, IRObservation, IRState)
├── client.py                 ← Python client (SREBenchClient)
├── inference.py              ← Hackathon inference script ([START]/[STEP]/[END] format)
├── baseline.py               ← Extended baseline with mock + Ollama modes
├── data/
│   ├── task1_memory_leak.json
│   ├── task2_db_cascade.json
│   ├── task3_ddos_exfil.json
│   ├── task4_predictive.json ← Novel proactive task
│   └── runbooks.json         ← SRE runbook library (4 runbooks)
└── server/
    ├── environment.py        ← IncidentResponseEnv (OpenEnv interface)
    ├── simulator.py          ← Infrastructure state machine
    ├── graders.py            ← Deterministic per-task graders
    └── app.py                ← FastAPI + Gradio server
```

---

## Design Notes

**Why SRE incident response?**  
Every tech company has an on-call rotation. Incidents happen at 3am. The cost of slow diagnosis is measured in dollars per minute. Training agents to do this well has immediate commercial value — yet no RL environment models it.

**Why partial rewards?**  
Binary pass/fail rewards collapse the signal. An agent that identifies the right service but applies a wrong fix has learned something valuable. Grading each step of the reasoning chain (identify service → diagnose root cause → apply fix → document) gives dense training signal across the full trajectory.

**Why red herrings?**  
Real incidents always have noise. Task 2's CPU spike is a symptom, not the cause. Task 3's DDoS is a distraction. An environment without red herrings does not prepare agents for the real world.

**Why runbooks?**  
Senior SREs consult runbooks. An agent that can search institutional knowledge before acting is more useful than one that reasons from scratch every time. The runbook system rewards this behavior.

**Why predictive tasks?**  
Reactive incident response is table stakes. The real value is catching problems before they page anyone. Task 4 tests this capability with a proactive speed bonus — earlier action means higher reward.

---

## License

MIT
