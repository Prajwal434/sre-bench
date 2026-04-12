"""
SREBench HTTP client.

Provides a typed Python interface for interacting with the SREBench FastAPI server.
Compatible with the OpenEnv EnvClient pattern.

Usage:
    env = SREBenchClient(base_url="http://localhost:7860")
    obs = env.reset(task_id="task1_memory_leak")
    result = env.step("query_logs", service="payment-service", filter="ERROR")
    print(result.last_action_result)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

from models import IRAction, IRObservation, IRState


class SREBenchClient:
    """
    Typed HTTP client for the SREBench OpenEnv environment.

    Parameters
    ----------
    base_url : str
        Base URL of the running SREBench server.
        Defaults to SREBENCH_URL env var, then http://localhost:7860.
    timeout : float
        HTTP request timeout in seconds.
    """

    DEFAULT_URL = "http://localhost:7860"

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("SREBENCH_URL", self.DEFAULT_URL)
        ).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: str = "task1_memory_leak",
        episode_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> IRObservation:
        """Start a new episode. Returns the initial observation."""
        payload: Dict[str, Any] = {"task_id": task_id}
        if episode_id:
            payload["episode_id"] = episode_id
        if seed is not None:
            payload["seed"] = seed

        resp = self._session.post(
            f"{self.base_url}/reset",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return IRObservation(**resp.json())

    def step(
        self,
        action_type: str,
        **parameters: Any,
    ) -> IRObservation:
        """
        Execute one action.

        Parameters
        ----------
        action_type : str
            One of: query_logs, check_metrics, run_diagnostic, apply_fix,
            acknowledge_alert, escalate, add_note, mark_resolved.
        **parameters : Any
            Action-specific parameters passed as keyword arguments.

        Returns
        -------
        IRObservation
            The resulting observation, including reward if the episode is done.
        """
        resp = self._session.post(
            f"{self.base_url}/step",
            json={"action_type": action_type, "parameters": parameters},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return IRObservation(**resp.json())

    def step_action(self, action: IRAction) -> IRObservation:
        """Execute a typed IRAction object."""
        return self.step(action.action_type, **action.parameters)

    def state(self) -> IRState:
        """Return current episode metadata."""
        resp = self._session.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return IRState(**resp.json())

    def health(self) -> Dict[str, Any]:
        resp = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> Dict[str, Any]:
        resp = self._session.get(f"{self.base_url}/tasks", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "SREBenchClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self._session.close()


# ---------------------------------------------------------------------------
# Convenience factory: connect to HF Space
# ---------------------------------------------------------------------------

def from_hf_space(space_id: str = "openenv/sre-bench") -> SREBenchClient:
    """
    Connect to a deployed HuggingFace Space.

    Example:
        env = from_hf_space("openenv/sre-bench")
        obs = env.reset(task_id="task2_db_cascade")
    """
    url = f"https://{space_id.replace('/', '-')}.hf.space"
    return SREBenchClient(base_url=url)
