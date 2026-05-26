"""Core orchestration contracts for evaluation workflows."""

import sys as _sys

# The full public API requires Python >= 3.9 (Literal, Protocol, PEP 604
# unions, etc.).  Python 3.7 sub-processes — such as the PerfectDou agent —
# only need ``evaluations.core.socket_worker``; skip the heavy imports so
# the package can be loaded without 3.9+ features.
if _sys.version_info >= (3, 10):
    from evaluations.core.coordinator import EvaluationCoordinator
    from evaluations.core.controller_dispatch import (
        CoalescingDispatchPolicy,
        ControllerDispatchPolicy,
        DirectDispatchPolicy,
    )
    from evaluations.core.dispatch import LogicalJobDispatcher, SharedUnitAllocator, WorkerAssignment
    from evaluations.core.controller_pool import (
        ControllerPoolKey,
        PooledPolicyController,
        WorkerLocalControllerPool,
    )
    from evaluations.core.managed_controller import (
        ControllerReuseScope,
        ManagedPolicyController,
        ManagedSocketPolicyController,
    )
    from evaluations.core.socket_worker import (
        PROTOCOL_VERSION,
        PolicyWorkerAction,
        PolicyWorkerClient,
        PolicyWorkerClientConfigProtocol,
        PolicyWorkerRequest,
        PolicyWorkerServerConfigProtocol,
        SocketSubprocessControllerRuntime,
        SubprocessSocketClientRuntime,
        build_controller_subprocess_env,
        format_policy_worker_actions,
        parse_policy_worker_requests,
        recv_json,
        run_policy_worker_server,
        send_json,
        worker_scoped_resource_dir,
    )
    from evaluations.core.executor import JobCancelledError, JobExecutor
    from evaluations.core.plan import EvaluationRunPlan
    from evaluations.core.progress import (
        build_normalized_progress,
        build_progress_metric,
        format_normalized_progress,
        is_normalized_progress,
    )
    from evaluations.core.protocols import (
        EvaluationScenario,
        HostSession,
        JobRuntimeFactory,
        JobRuntimeSession,
        PolicyController,
        QueueEventEmitterAware,
        ResultAggregator,
        SummaryWriter,
    )
    from evaluations.core.recording import RunEventRecorder
    from evaluations.core.resources import (
        WorkerLayout,
        WorkerLayoutPlanner,
        build_worker_layout,
        build_worker_layout_for_workers_per_gpu,
        discover_usable_gpu_tokens,
        discover_visible_gpu_tokens,
        probe_gpu_token,
        resolve_worker_slot_device,
    )
    from evaluations.core.runtime import CallbackJobRuntimeFactory, HostedJobRuntimeSession, RunScopedControllerRuntimeFactory
    from evaluations.core.stats import MetricStats, RunningStats, metric_from_mean_ci, metric_from_running_stats
    from evaluations.core.types import (
        DecisionBatch,
        DecisionRequest,
        EpisodeUpdate,
        EvaluationJob,
        Finished,
        JobWorkload,
        JobResult,
        ProgressUpdate,
        QueueEvent,
        StatusUpdate,
        WorkerSlot,
    )
    from evaluations.core.wandb import EvaluationWandbRun

__all__ = [
    "CallbackJobRuntimeFactory",
    "CoalescingDispatchPolicy",
    "ControllerDispatchPolicy",
    "ControllerPoolKey",
    "ControllerReuseScope",
    "DecisionBatch",
    "DecisionRequest",
    "DirectDispatchPolicy",
    "EpisodeUpdate",
    "EvaluationCoordinator",
    "EvaluationJob",
    "EvaluationRunPlan",
    "EvaluationScenario",
    "EvaluationWandbRun",
    "Finished",
    "format_normalized_progress",
    "HostSession",
    "HostedJobRuntimeSession",
    "is_normalized_progress",
    "JobCancelledError",
    "JobExecutor",
    "JobResult",
    "JobRuntimeFactory",
    "JobRuntimeSession",
    "JobWorkload",
    "LogicalJobDispatcher",
    "ManagedPolicyController",
    "ManagedSocketPolicyController",
    "MetricStats",
    "PROTOCOL_VERSION",
    "PolicyWorkerAction",
    "PolicyWorkerClient",
    "PolicyWorkerClientConfigProtocol",
    "PooledPolicyController",
    "PolicyWorkerRequest",
    "PolicyWorkerServerConfigProtocol",
    "PolicyController",
    "ProgressUpdate",
    "QueueEvent",
    "QueueEventEmitterAware",
    "ResultAggregator",
    "RunEventRecorder",
    "RunScopedControllerRuntimeFactory",
    "RunningStats",
    "SocketSubprocessControllerRuntime",
    "StatusUpdate",
    "SubprocessSocketClientRuntime",
    "SummaryWriter",
    "SharedUnitAllocator",
    "WorkerAssignment",
    "WorkerLayout",
    "WorkerLayoutPlanner",
    "WorkerLocalControllerPool",
    "WorkerSlot",
    "build_controller_subprocess_env",
    "build_normalized_progress",
    "build_progress_metric",
    "build_worker_layout",
    "build_worker_layout_for_workers_per_gpu",
    "discover_usable_gpu_tokens",
    "discover_visible_gpu_tokens",
    "format_policy_worker_actions",
    "metric_from_mean_ci",
    "metric_from_running_stats",
    "parse_policy_worker_requests",
    "probe_gpu_token",
    "recv_json",
    "resolve_worker_slot_device",
    "run_policy_worker_server",
    "send_json",
    "worker_scoped_resource_dir",
]
