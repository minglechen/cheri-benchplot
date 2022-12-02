import typing
from ..core.task import ExecutionTask, LocalFileTarget
from ..qemu.task import QEMUTracingSetupTask
from pathlib import Path
from ..core.config import ConfigPath, TemplateConfig, ProfileConfig
from dataclasses import dataclass, field


@dataclass
class SpecRunConfig(TemplateConfig):

    #: Path to the Spec2006 suite in the guest
    spec_path: ConfigPath = Path("/opt/spec2006")

    #: Path to cd into when running the spec benchmark, defaults to workload input
    spec_rundir: typing.Optional[ConfigPath] = None

    #: Spec workload to run (test/train/ref)
    spec_workload: str = "test"

    #: Spec benchmark name to run
    spec_benchmark: str = "471.omnetpp"

    #: Extra options to the benchmark
    spec_benchmark_options: typing.List[str] = field(default_factory=list)

    #: Benchmark profiling configuration
    profile: ProfileConfig = field(default_factory=ProfileConfig)

    def __post_init__(self):
        assert self.spec_path.is_absolute(), "Remote SPEC suite path must be absolute"
        if self.spec_rundir is None:
            self.spec_rundir = (
                self.spec_path
                / self.spec_benchmark
                / "data"
                / self.spec_workload
                / "input"
            )


class SpecExecTask(ExecutionTask):

    task_namespace = "spec"
    public = True
    task_config_class = SpecRunConfig

    def __init__(
        self,
        benchmark: "Benchmark",
        script: "ScriptBuilder",
        task_config=None,
    ):
        super().__init__(benchmark, script, task_config)
        # Remote path to spec benchmark
        remote_benchmark_dir = self.config.spec_path / self.config.spec_benchmark
        self.spec_benchmark_bin = remote_benchmark_dir / self.config.spec_benchmark

    def get_spec_output_path(self) -> LocalFileTarget:
        path = (
            self.benchmark.get_benchmark_data_path() / f"spec-{self.benchmark.uuid}.txt"
        )
        return LocalFileTarget(self.benchmark, path)

    def dependencies(self) -> typing.Iterable["Task"]:
        trace_dir: Path = (
            self.benchmark.get_benchmark_data_path() / "qemu-trace" / "qemu-trace-dir"
        )
        trace_dir.mkdir(parents=True, exist_ok=True)
        profile = ProfileConfig(
            qemu_trace="perfetto-dynamorio", qemu_trace_categories=["instructions"]
        )
        yield QEMUTracingSetupTask(self.benchmark, self.script, profile)

    def run(self):
        # Run the benchmark
        for i in range(self.benchmark.config.iterations):
            s = self.script.benchmark_sections[i]["benchmark"]
            s.add_cmd("cd", [self.config.spec_rundir])
            s.add_cmd(
                "qtrace",
                ["-u", "exec", self.spec_benchmark_bin, "--"]
                + self.config.spec_benchmark_options.copy(),
            )

    def outputs(self):
        yield "spec-output", self.get_spec_output_path()
