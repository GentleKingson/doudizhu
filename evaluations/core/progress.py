from __future__ import annotations

from collections.abc import Mapping, Sequence


def build_progress_metric(
    *,
    key: str,
    label: str,
    value: float,
    ci: float | None = None,
    precision: int = 3,
) -> dict[str, object]:
    if not str(key).strip():
        raise ValueError("Progress metric key must be non-empty.")
    if not str(label).strip():
        raise ValueError("Progress metric label must be non-empty.")
    if int(precision) < 0:
        raise ValueError("Progress metric precision must be non-negative.")
    payload = {
        "key": str(key),
        "label": str(label),
        "value": float(value),
        "precision": int(precision),
    }
    if ci is not None:
        payload["ci"] = float(ci)
    return payload


def build_normalized_progress(
    *,
    scope_kind: str,
    scope_label: str,
    completed: int,
    total: int,
    unit: str,
    metrics: Sequence[Mapping[str, object]] = (),
    throughput_label: str | None = None,
    throughput_value: float | None = None,
    throughput_precision: int = 2,
    job_id: str | None = None,
    logical_job_id: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not str(scope_kind).strip():
        raise ValueError("Progress scope_kind must be non-empty.")
    if not str(scope_label).strip():
        raise ValueError("Progress scope_label must be non-empty.")
    if int(total) <= 0:
        raise ValueError("Progress total must be positive.")
    if int(completed) < 0 or int(completed) > int(total):
        raise ValueError("Progress completed must be within [0, total].")
    if not str(unit).strip():
        raise ValueError("Progress unit must be non-empty.")
    if int(throughput_precision) < 0:
        raise ValueError("Progress throughput_precision must be non-negative.")

    payload = {
        "schema_version": 1,
        "progress_schema": "core.v1",
        "scope_kind": str(scope_kind),
        "scope_label": str(scope_label),
        "completed": int(completed),
        "total": int(total),
        "unit": str(unit),
        "metrics": [dict(item) for item in metrics],
    }
    if job_id is not None:
        payload["job_id"] = str(job_id)
    if logical_job_id is not None:
        payload["logical_job_id"] = str(logical_job_id)
    if throughput_label is not None and throughput_value is not None:
        if not str(throughput_label).strip():
            raise ValueError("Progress throughput_label must be non-empty.")
        payload["throughput"] = {
            "label": str(throughput_label),
            "value": float(throughput_value),
            "precision": int(throughput_precision),
        }
    elif throughput_label is not None or throughput_value is not None:
        raise ValueError("Progress throughput_label and throughput_value must be provided together.")

    if extra is not None:
        overlap = set(payload).intersection(extra)
        if overlap:
            raise ValueError(f"Progress extra fields overlap reserved keys: {sorted(overlap)}.")
        payload.update({str(key): value for key, value in extra.items()})
    return payload


def is_normalized_progress(body: Mapping[str, object]) -> bool:
    return str(body.get("progress_schema", "")).strip() == "core.v1"


def format_normalized_progress(body: Mapping[str, object]) -> str | None:
    if not is_normalized_progress(body):
        return None

    scope_label = str(body.get("scope_label", "")).strip()
    unit = str(body.get("unit", "")).strip()
    if not scope_label or not unit:
        return None

    completed = int(body["completed"])
    total = int(body["total"])
    fields = [f"{completed} / {total} {unit}"]

    for raw_metric in body.get("metrics", ()):
        metric = dict(raw_metric)
        label = str(metric.get("label", "")).strip()
        if not label:
            continue
        precision = int(metric.get("precision", 3))
        value = float(metric["value"])
        piece = f"{label}: {value:.{precision}f}"
        if "ci" in metric:
            ci = float(metric["ci"])
            piece += f" ± {ci:.{precision}f}"
        fields.append(piece)

    throughput = body.get("throughput")
    if throughput is not None:
        throughput_payload = dict(throughput)
        label = str(throughput_payload.get("label", "")).strip()
        if label:
            precision = int(throughput_payload.get("precision", 2))
            value = float(throughput_payload["value"])
            fields.append(f"{label}: {value:.{precision}f}")

    return f"{scope_label} | " + " | ".join(fields)


__all__ = [
    "build_normalized_progress",
    "build_progress_metric",
    "format_normalized_progress",
    "is_normalized_progress",
]
