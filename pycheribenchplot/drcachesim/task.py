import typing
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from ..core.config import ConfigPath, ProfileConfig, TemplateConfig
from ..core.task import Task
from subprocess import run, PIPE, CompletedProcess
from ..addr2line.task import Addr2LineTask


@dataclass
class DrCacheSimRunConfig(TemplateConfig):
    #: Path to the drcachesim executable
    drrun_path: ConfigPath = Path("bin64/drrun")
    #: Simulator to run
    simulator: str = "cache"
    #: Output file path for analysis results
    output_path: ConfigPath = None
    #: Cache size to run
    cache_size: str = field(default_factory=str)
    #: Cache level to run
    cache_level: str = field(default_factory=str)
    #: Indir for drcachesim
    indir: ConfigPath = Path("traces")
    #: Record instr misses
    record_instr_misses: bool = False
    #: Working set reset interval
    working_set_reset_interval: int = field(default_factory=int)
    #: Path to addr2line file
    addr2line_file: str = field(default_factory=str)
    #: Output directory for instr count analysis
    output_dir: str = field(default_factory=str)
    #: Rerun drcachesim even if output file exists
    rerun_sim: bool = False
    #: Optional addr2Line config
    addr2line_config: TemplateConfig = None


class DrCacheSimRunTask(Task):
    """Run a single drcachesim analysis"""

    public = True
    task_name = "drcachesim-run"
    task_namespace = "drcachesim-run"
    task_config_class = DrCacheSimRunConfig

    def __init__(self, session, task_config=None):
        super().__init__(task_config)
        self.uuid = uuid4()
        self._session = session

    @property
    def session(self):
        return self._session

    @property
    def task_id(self):
        #: make sure that the task_id is unique
        return f"{self.task_namespace}.{self.task_name}-{self.uuid}"

    def dependencies(self) -> typing.Iterable["Task"]:
        if self.config.addr2line_config:
            yield Addr2LineTask(self.session, self.config.addr2line_config)

    def run(self):
        cache_level = self.config.cache_level
        out_path = self.config.output_path
        size = self.config.cache_size
        indir = self.config.indir
        simulator = self.config.simulator
        assert (
            simulator == "cache"
            or simulator == "working_set"
            or simulator == "instr_count"
        )
        if out_path and out_path.is_file() and not self.config.rerun_sim:
            self.logger.info(out_path, "already exists, skipping")
            return

        if (
            simulator == "instr_count"
            and (self.config.output_dir / "instr_counts.csv").is_file()
            and not self.config.rerun_sim
        ):
            self.logger.info(
                f"{self.config.output_dir / 'instr_counts.csv'} already exists, skipping"
            )
            return
        cmd = [
            self.config.drrun_path,
            "-t",
            "drcachesim",
            "-simulator_type",
            simulator,
            "-indir",
            indir,
        ]
        if cache_level and size:
            cmd.extend([f"-{cache_level}_size", size])

        if self.config.addr2line_file:
            cmd.extend(["-addr2line_file", self.config.addr2line_file])

        if self.config.output_dir:
            cmd.extend(["-output_dir", self.config.output_dir])

        if self.config.record_instr_misses:
            cmd.append("-record_instr_misses")

        if self.config.working_set_reset_interval:
            cmd.extend(
                [
                    "-working_set_reset_interval",
                    str(self.config.working_set_reset_interval),
                ]
            )
        self.logger.info(f"Running drcachesim on {self.config.indir}")
        p: CompletedProcess = run(
            cmd,
            stderr=PIPE,
        )
        result = p.stderr.decode("utf-8")
        if out_path:
            with open(out_path, "w") as f:
                f.write(result)

        self.logger.info(f"Finished running drcachesim on {self.config.indir}")
