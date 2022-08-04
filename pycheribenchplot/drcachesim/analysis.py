from pathlib import Path
import typing
from ..core.analysis import BenchmarkAnalysis
from ..core.dataset import DatasetName
import asyncio as aio
import subprocess, os, shutil

from dataclasses import dataclass, field
from ..core.config import TemplateConfig, ConfigPath

@dataclass
class DrCacheSimConfig(TemplateConfig):
    drrun_path: ConfigPath = Path("dynamorio/bin64/drrun")
    remove_saved_results: bool = False
    LL_cache_sizes: typing.List[str] = field(default_factory=list)
    rerun_sim: bool = False

class DrCacheSimRun(BenchmarkAnalysis):
    require = {DatasetName.QEMU_DYNAMORIO}
    name: str = "drcachesim"
    description: str = "Run drcachesim"
    analysis_options_class = DrCacheSimConfig
    def __init__(self, benchmark, config):
        super().__init__(benchmark, config)
        self.processes_dict = {}
        self.args = self.config.handlers[0].options
        self.out_paths = {}
    async def process_datasets(self):
        dset = self.get_dataset(DatasetName.QEMU_DYNAMORIO)
        trace_file = dset.output_file()
        indir = trace_file.parent
        base = indir.parent / "drcachesim_results" 
        if self.args['remove_saved_results']:
            shutil.rmtree(str(base), ignore_errors=True)
        if not base.exists():
            base.mkdir(parents=True)

        for s in self.args['LL_cache_sizes']:
            out_path = base / ("LL_size_" + s + ".txt")
            if os.path.isfile(out_path) and not self.args['rerun_sim']:
                continue
            self.out_paths[s] = out_path
            self.logger.info(f"Running drcachesim on {indir}")
            p = await aio.create_subprocess_exec(self.args['drrun_path'], '-t', 'drcachesim', '-indir', indir, '-LL_size', s, stderr=aio.subprocess.PIPE)
            self.processes_dict[p] = s

        for kvp in self.processes_dict.items():
            p = kvp[0]
            size = kvp[1]
            err = (await p.communicate())[1];
            with open(self.out_paths[size], "w") as f: 
                f.write(err.decode())
                

 