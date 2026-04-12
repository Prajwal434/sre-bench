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
[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/openenv/sre-bench)

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

## Environment Description

The agent acts as an SRE on-call engineer with access to:

| Tool | What it does |
|------|-------------|
| `query_logs` | Search log streams for any service |
| `check_metrics` | View CPU, memory, error rate, latency for services |
| `run_diagnostic` | Run tools: `heap_dump`, `explain_query`, `netstat`, `threat_intel`, etc. |
| `apply_fix` | Apply a named remediation to a service |
| `acknowledge_alert` | Acknowledge a PagerDuty-style alert |
| `escalate` | Page another team (security, DBA, network) |
| `add_note` | Add a note to the incident timeline |
| `mark_resolved` | Close the incident with root cause + summary (terminal) |

### Action Space

```python
class IRAction(BaseModel):
    action_type: Literal[
        "query_logs", "check_metrics", "run_diagnostic", "apply_fix",
        "acknowledge_alert", "escalate", "add_note", "mark_resolved"
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

**Final score is always in [0.0, 1.0].**

---

## Setup & Usage

### Option 1: Docker (recommended)

```bash
git clone https://github.com/your-org/sre-bench
cd sre-bench

docker build -t sre-bench .
docker run -p 7860:7860 sre-bench
```

Server starts at `http://localhost:7860`. Gradio UI available at `/`.

### Option 2: Local Python

```bash
cd sre-bench
pip install -r requirements.txt

uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Option 3: HuggingFace Space

```python
from client import from_hf_space

env = from_hf_space("openenv/sre-bench")
obs = env.reset(task_id="task1_memory_leak")
```

---

## Running the Baseline

```bash
export OPENAI_API_KEY=sk-...
export SREBENCH_URL=http://localhost:7860

# Run all 3 tasks
python baseline.py

# Run a single task with GPT-4o
python baseline.py --task task1_memory_leak --model gpt-4o

# Run all tasks quietly (just scores)
python baseline.py --quiet
```

### Reproducible Baseline Scores (gpt-4o-mini, temperature=0)

```
Task                           Score    Expected    Steps
------------------------------------------------------------
task1_memory_leak              0.743      0.740        8
task2_db_cascade               0.481      0.480       14
task3_ddos_exfil               0.218      0.220       21
------------------------------------------------------------
AVERAGE                        0.481
```

---

## API Reference

```bash
# Health check
GET /health

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
├── requirements.txt
├── README.md
├── models.py                 ← Pydantic models (IRAction, IRObservation, IRState)
├── client.py                 ← Python client (SREBenchClient)
├── baseline.py               ← OpenAI API baseline script
├── data/
│   ├── task1_memory_leak.json
│   ├── task2_db_cascade.json
│   └── task3_ddos_exfil.json
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

---

## License

MIT
