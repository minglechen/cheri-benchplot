from dataclasses import dataclass

from ..core.config import InstancePlatform, ProfileConfig
from ..core.task import ExecutionTask, LocalFileTarget
from pathlib import Path


class QEMUTracingSetupTask(ExecutionTask):
    """
    This is the dependency task that modifies the current benchmark machine
    configuration to enable QEMU tracing.
    The task_config for this should be a ProfileConfig

    XXX need to split public platform options from private platform options that
    are populated here via the ProfileConfig
    """

    public = False
    task_namespace = "qemu.tracing"
    task_config_class = ProfileConfig

    def get_qemu_profile_target(self) -> LocalFileTarget:
        """The QEMU trace output file path"""
        return LocalFileTarget.from_benchmark(
            self.benchmark, f"qemu-perfetto-{self.benchmark.uuid}.pb"
        )

    def get_qemu_interceptor_target(self) -> LocalFileTarget:
        """The QEMU interceptor trace output file path"""
        path = (
            Path("qemu-trace-dir")
            / f"qemu-perfetto-interceptor-{self.benchmark.uuid}.trace.gz"
        )
        return LocalFileTarget.from_benchmark(self.benchmark, path)

    def run(self):
        if self.benchmark.config.instance.platform != InstancePlatform.QEMU:
            raise ValueError("QEMU tracing is only allowed when running on QEMU")
        opts = self.benchmark.config.instance.platform_options
        if self.config.qemu_trace == "perfetto":
            opts.qemu_trace = "perfetto"
            opts.qemu_trace_file = self.get_qemu_profile_target().path
        elif self.config.qemu_trace == "perfetto-dynamorio":
            path = self.get_qemu_interceptor_target().path
            # Ensure that we have the paths
            path.parent.mkdir(exist_ok=True)
            opts.qemu_trace = "perfetto-dynamorio"
            opts.qemu_trace_file = self.get_qemu_profile_target().path
            opts.qemu_interceptor_trace_file = path
        elif self.config.qemu_trace is not None:
            raise ValueError("Unknown configuration for ProfileConfig.qemu_trace")
        opts.qemu_trace_categories = self.config.qemu_trace_categories

    def outputs(self):
        yield "perfetto-trace", self.get_qemu_profile_target()
        if self.config.qemu_trace == "perfetto-dynamorio":
            yield "interceptor-trace", self.get_qemu_interceptor_target()
