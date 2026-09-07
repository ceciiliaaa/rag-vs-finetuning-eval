"""Driving a fine-tuning job at the provider: upload, start, poll, record.

Everything here costs money and takes time at the provider, so nothing runs implicitly. The
class holds no state beyond the client and the identifiers it is given, which keeps a resumed
session as simple as passing the job id back in.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})


@dataclass(frozen=True)
class JobRecord:
    """What is worth keeping about a fine-tuning run once it has finished."""

    job_id: str
    base_model: str
    training_file_id: str
    status: str
    fine_tuned_model: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target


class FineTuningJob:
    """Thin wrapper over the provider's fine-tuning endpoints."""

    def __init__(
        self,
        base_model: str = "gpt-3.5-turbo",
        suffix: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - needs the optional extra
            raise RuntimeError(
                "FineTuningJob needs the 'openai' extra: uv sync --extra openai"
            ) from exc
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("set OPENAI_API_KEY, or pass api_key")
        self.base_model = base_model
        self.suffix = suffix
        self._client = OpenAI(api_key=key)

    def upload(self, jsonl_path: str | Path) -> str:
        """Upload a training file and return its id."""
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            response = self._client.files.create(file=handle, purpose="fine-tune")
        return response.id

    def start(self, training_file_id: str) -> str:
        """Create the job and return its id. Hyper-parameters are left at provider defaults."""
        kwargs: dict[str, object] = {
            "training_file": training_file_id,
            "model": self.base_model,
        }
        if self.suffix:
            kwargs["suffix"] = self.suffix
        return self._client.fine_tuning.jobs.create(**kwargs).id

    def status(self, job_id: str) -> JobRecord:
        """Fetch the current state of a job."""
        response = self._client.fine_tuning.jobs.retrieve(job_id)
        return JobRecord(
            job_id=job_id,
            base_model=self.base_model,
            training_file_id=str(getattr(response, "training_file", "")),
            status=str(response.status),
            fine_tuned_model=getattr(response, "fine_tuned_model", None),
        )


def poll_until_finished(
    job: FineTuningJob,
    job_id: str,
    interval_seconds: float = 60.0,
    timeout_seconds: float = 7200.0,
) -> JobRecord:
    """Poll a job until it reaches a terminal state, or raise on timeout.

    Training a small set takes tens of minutes, so the default interval is a minute and the
    default ceiling two hours.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        record = job.status(job_id)
        if record.status in TERMINAL_STATES:
            return record
        if time.monotonic() >= deadline:
            raise TimeoutError(f"job {job_id} still {record.status} after {timeout_seconds:.0f}s")
        time.sleep(interval_seconds)
