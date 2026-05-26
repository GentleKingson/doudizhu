from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Mapping, Sequence


class EvaluationWandbRun:
    """Thin W&B run wrapper for evaluation workflows.

    Runs may be finished externally (e.g. by a subsequent
    ``wandb.init(reinit=True)``).  Methods that write to the run call
    :meth:`_ensure_active` first, which transparently resumes a
    finished run via ``wandb.init(resume="allow", id=…)``.
    """

    def __init__(
        self,
        *,
        module: Any,
        run: Any,
        name: str,
        run_id: str,
        project: str,
        entity: str | None,
    ) -> None:
        self._module = module
        self._run = run
        self._name = str(name)
        self._run_id = str(run_id)
        self._project = str(project)
        self._entity = None if entity is None else str(entity)
        self._closed = False

    @staticmethod
    def _import_required() -> Any:
        try:
            import wandb  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "wandb is required for evaluation logging. Install wandb or disable W&B explicitly."
            ) from exc
        return wandb

    @classmethod
    def open(
        cls,
        *,
        project: str,
        entity: str | None,
        name: str,
        config: Mapping[str, object] | None = None,
    ) -> EvaluationWandbRun:
        module = cls._import_required()
        try:
            run = module.init(
                project = str(project),
                entity = None if entity is None else str(entity),
                name = str(name),
                config = dict(config or {}),
                reinit = True,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"W&B initialization failed for evaluation run {name!r}.") from exc

        resolved_run = run if run is not None else getattr(module, "run", None)
        if resolved_run is None:
            raise RuntimeError(f"W&B initialization did not return a run object for evaluation run {name!r}.")
        return cls(
            module = module,
            run = resolved_run,
            name = name,
            run_id = str(resolved_run.id),
            project = str(project),
            entity = None if entity is None else str(entity),
        )

    def define_metric(self, name: str, *, step_metric: str) -> None:
        self._ensure_active()
        define = getattr(self._module, "define_metric", None)
        if callable(define):
            define(name, step_metric = step_metric)

    def _ensure_active(self) -> None:
        """Resume the run if it was finished by a later ``wandb.init()``."""
        if self._closed:
            return
        is_finished = getattr(self._run, "_is_finished", False)
        if not is_finished:
            return
        self._run = self._module.init(
            project = self._project,
            entity = self._entity,
            id = self._run_id,
            resume = "allow",
            reinit = True,
        )

    def log(self, payload: Mapping[str, object], *, step: int | None = None) -> None:
        self._ensure_active()
        try:
            if step is None:
                self._run.log(dict(payload))
            else:
                self._run.log(dict(payload), step = int(step))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"W&B metric logging failed for evaluation run {self._name!r}.") from exc

    def update_summary(self, payload: Mapping[str, object]) -> None:
        self._ensure_active()
        try:
            summary = getattr(self._run, "summary")
            for key, value in payload.items():
                summary[str(key)] = value
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"W&B summary update failed for evaluation run {self._name!r}.") from exc

    def log_table(
        self,
        key: str,
        *,
        columns: Sequence[str],
        rows: Sequence[Sequence[object]],
    ) -> None:
        try:
            table = self._module.Table(
                columns = [str(column) for column in columns],
                data = [list(row) for row in rows],
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"W&B table creation failed for evaluation run {self._name!r}.") from exc
        self.log({str(key): table})

    def save_file(self, path: Path) -> None:
        self._ensure_active()
        target = Path(path).resolve()
        glob_path = glob.escape(str(target))
        base_path = glob.escape(str(target.parent))
        if hasattr(self._run, "save"):
            save = self._run.save
        elif hasattr(self._module, "save"):
            save = self._module.save
        else:
            raise RuntimeError(f"W&B artifact save is not available for evaluation run {self._name!r}.")
        try:
            saved = save(glob_path, base_path = base_path, policy = "now")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"W&B artifact save failed for evaluation run {self._name!r} and file {target}."
            ) from exc
        self._validate_file_staged(target = target, saved = saved)

    @staticmethod
    def _validate_file_staged(*, target: Path, saved: object) -> None:
        if saved:
            return
        raise RuntimeError(f"W&B save did not stage any file for literal path {target}.")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._run.finish()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"W&B finish failed for evaluation run {self._name!r}.") from exc


__all__ = ["EvaluationWandbRun"]
