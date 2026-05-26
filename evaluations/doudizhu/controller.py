from __future__ import annotations

from collections.abc import Callable, Sequence

from evaluations.core.controller_pool import WorkerLocalControllerPool
from evaluations.core.managed_controller import ManagedSocketPolicyController
from evaluations.core.socket_worker import PolicyWorkerRequest
from evaluations.core.types import DecisionRequest
from evaluations.doudizhu.runtime import DEFAULT_NETWORK_CLIENT, REPO_ROOT
from evaluations.doudizhu.specs import AgentSpec, DoudizhuAgentCommandBuilder


def _encode_agent_requests(requests: Sequence[DecisionRequest]) -> list[PolicyWorkerRequest]:
    payloads: list[PolicyWorkerRequest] = []
    for request in requests:
        payload = request.payload
        if not isinstance(payload, PolicyWorkerRequest):
            raise RuntimeError(
                "Dou Dizhu controller expected PolicyWorkerRequest payload, "
                f"got {type(payload).__name__}."
            )
        payloads.append(payload)
    return payloads


class DoudizhuSocketController:
    def __init__(
        self,
        *,
        pool: WorkerLocalControllerPool,
        logdir: Path,
        worker_id: int,
        job_id: str,
        controller_id: str,
        agent: AgentSpec,
        gpu_token: str | None,
        matchup_name: str,
        role_key: str,
        role_label: str,
        scope_label: str,
    ) -> None:
        self._controller_id = str(controller_id)
        self._pool_controller_id = self._agent_pool_id(agent)
        self._agent = agent
        command_builder = DoudizhuAgentCommandBuilder()
        status_context: dict[str, object] = {
            "matchup_name": str(matchup_name),
            "role_key": str(role_key),
            "role_label": str(role_label),
            "scope_label": str(scope_label),
        }

        self._managed = ManagedSocketPolicyController(
            pool = pool,
            logdir = logdir,
            job_id = job_id,
            worker_id = int(worker_id),
            controller_id = self._controller_id,
            pool_controller_id = self._pool_controller_id,
            resource_id = self._agent.canonical_spec,
            gpu_token = gpu_token,
            repo_root = REPO_ROOT,
            command_factory = lambda socket_path, device: command_builder.build(
                self._agent,
                socket_path = socket_path,
                device = device,
            ),
            socket_name = f"{self._pool_controller_id}_{self._agent.kind}",
            connect_timeout_s = float(DEFAULT_NETWORK_CLIENT.connect_timeout_s),
            client_cfg = DEFAULT_NETWORK_CLIENT,
            runtime_label = f"Dou Dizhu controller {self._controller_id} ({self._agent.display_name()})",
            encode_requests = _encode_agent_requests,
            format_failure = lambda context: (
                "Dou Dizhu agent socket request failed. "
                f"controller_id={self._controller_id} agent={self._agent.display_name()}\n"
                f"{context}"
            ),
            status_context = status_context,
            ready_body_factory = lambda log_path, device: {
                "agent": self._agent.display_name(),
                "log_path": str(log_path),
            },
        )

    def bind_queue_event_emitter(self, emitter: Callable) -> None:  # noqa: ANN401
        self._managed.bind_queue_event_emitter(emitter)

    def open(self) -> None:
        self._managed.open()

    def act_batch(self, requests: Sequence[DecisionRequest]) -> Sequence[object]:
        return self._managed.act_batch(requests)

    def close(self) -> None:
        self._managed.close()

    @staticmethod
    def _agent_pool_id(agent: AgentSpec) -> str:
        return str(agent.canonical_spec)


__all__ = ["DoudizhuSocketController"]
