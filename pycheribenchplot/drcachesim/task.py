import typing
from dataclasses import dataclass, field
from pathlib import Path

from marshmallow.validate import OneOf

from ..core.config import ConfigPath, ProfileConfig, TemplateConfig
from ..core.task import Task
from ..qemu.task import QEMUTracingSetupTask
from subprocess import run, PIPE, CompletedProcess


@dataclass
class DrCacheSimRunConfig(TemplateConfig):
    #: Path to the drcachesim executable
    drrun_path: ConfigPath = Path("bin64/drrun")
    #: Output file path for analysis results
    output_path: ConfigPath = Path("LL_size/8M.txt")
    #: Cache size to run
    cache_size: str = "8M"
    #: Cache level to run
    cache_level: str = "LL"
    #: Indir for drcachesim
    indir: ConfigPath = Path("traces")
    #: Rerun drcachesim even if output file exists
    rerun_sim: bool = False


class DrCaheSimRunTask(Task):
    """Run a single drcachesim analysis"""

    public = True
    task_name = "drcachesim-run"
    task_namespace = "drcachesim-run"
    task_config_class = DrCacheSimRunConfig

    @property
    def task_id(self):
        #: This is a bit of a hack to make sure that the task_id is unique
        return f"{self.task_namespace}.{self.task_name}-{self.config.cache_level}-{self.config.cache_size}"

    def _run_drcachesim(self):
        level_arg = self.config.cache_level + "_size"
        out_path = self.config.output_path
        size = self.config.cache_size
        indir = self.config.indir
        if out_path.is_file() and not self.config.rerun_sim:
            print(out_path, "already exists, skipping")
            return
        p: CompletedProcess = run(
            [
                self.config.drrun_path,
                "-t",
                "drcachesim",
                "-indir",
                indir,
                "-" + level_arg,
                size,
            ],
            stderr=PIPE,
        )
        result = p.stderr.decode("utf-8")
        with open(out_path, "w") as f:
            f.write(result)

    def run(self):
        self.logger.info(f"Running drcachesim on {self.config.indir}")
        self._run_drcachesim()
        self.logger.info(f"Finished running drcachesim on {self.config.indir}")
