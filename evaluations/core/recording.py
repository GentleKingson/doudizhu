from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

from evaluations.core.types import EvaluationJob, JobResult, QueueEvent


class RunEventRecorder:
    def __init__(
        self,
        *,
        logdir: Path,
        summary_writer: object,
    ) -> None:
        self._logdir = Path(logdir).resolve()
        self._summary_writer = summary_writer
        self._results: dict[str, JobResult] = {}
        self._logdir.mkdir(parents = True, exist_ok = True)
        self._events_handle = (self._logdir / "events.jsonl").open("w", encoding = "utf-8")

    def record(self, event: QueueEvent) -> None:
        if self._events_handle is None:
            raise RuntimeError("RunEventRecorder must be initialized before record().")

        payload = {
            "timestamp": time.time(),
            "kind": str(event.kind),
            "job_id": str(event.job_id),
            "body": dict(event.body),
        }
        self._events_handle.write(json.dumps(payload, sort_keys = True) + "\n")
        self._events_handle.flush()

        record_event = getattr(self._summary_writer, "record_event", None)
        if callable(record_event):
            record_event(event)

        if event.kind == "result":
            result = JobResult(
                job_id = str(event.job_id),
                ok = bool(event.body.get("ok", False)),
                payload = dict(event.body.get("payload", {})),
                artifacts = {str(k): str(v) for k, v in dict(event.body.get("artifacts", {})).items()},
            )
            self._results[result.job_id] = result
            _ = self._summary_writer.write_job(result)

        rendered = self._summary_writer.format_event(event)
        if rendered:
            print(rendered, flush = True)

    def finalize(self, jobs: Sequence[EvaluationJob]) -> list[JobResult]:
        logical_job_order = {
            str(job.logical_job_id): int(index)
            for index, job in enumerate(jobs)
        }
        ordered = sorted(
            self._results.values(),
            key = lambda result: (
                int(
                    logical_job_order.get(
                        str(result.payload.get("logical_job_id", result.job_id)),
                        len(logical_job_order),
                    )
                ),
                int(result.payload.get("worker_index", 0)),
                str(result.job_id),
            ),
        )
        _ = self._summary_writer.write_run(ordered)
        render_report = getattr(self._summary_writer, "render_report", None)
        if callable(render_report):
            report = render_report(ordered)
            if report:
                print(report, end = "" if report.endswith("\n") else "\n")
        return ordered

    def close(self) -> None:
        if self._events_handle is not None:
            self._events_handle.close()
            self._events_handle = None


__all__ = ["RunEventRecorder"]
