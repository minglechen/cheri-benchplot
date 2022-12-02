import typing
from dataclasses import dataclass, field
from pathlib import Path

from marshmallow.validate import OneOf

from ..core.config import ConfigPath, ProfileConfig, TemplateConfig
from ..core.task import DataFileTarget, AnalysisTask
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


class DrCaheSimRunTask(AnalysisTask):
    """Run a single drcachesim analysis"""

    public = True
    task_name = "drcachesim-run"
    task_namespace = "drcachesim-run"
    task_config_class = DrCacheSimRunConfig

    def __init__(self, benchmark, config):
        super().__init__(benchmark, config)

    def _run_drcachesim(self):
        level_arg = self.config.cache_level + "_size"
        out_path: Path = self.config.output_path
        size = self.config.cache_size
        indir = self.config.indir
        if out_path.is_file() and not self.config.rerun_sim:
            return
        p: CompletedProcess = run(
            self.config.drrun_path,
            "-t",
            "drcachesim",
            "-indir",
            indir,
            "-" + level_arg,
            size,
            stderr=PIPE,
        )
        p.stderr.decode("utf-8")

    def run(self):
        self._run_drcachesim()
