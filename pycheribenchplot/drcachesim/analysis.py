import shutil
import typing
from dataclasses import dataclass, field
from pathlib import Path

from ..core.analysis import AnalysisTask, AnalysisConfig
from ..core.config import ConfigPath, TemplateConfig, Config
from .task import DrCacheSimRunTask, DrCacheSimRunConfig
from ..addr2line.task import Addr2LineTask, Addr2LineConfig


@dataclass
class DrCacheSimConfig(TemplateConfig):
    drrun_path: ConfigPath = Path("dynamorio/bin64/drrun")
    remove_saved_results: bool = False
    LL_cache_sizes: typing.List[str] = field(default_factory=list)
    L1D_cache_sizes: typing.List[str] = field(default_factory=list)
    L1I_cache_sizes: typing.List[str] = field(default_factory=list)
    run_cache_levels: typing.List[str] = field(default_factory=list)
    rerun_sim: bool = False


@dataclass
class InstrCountRunConfig(TemplateConfig):
    drrun_path: ConfigPath = Path("dynamorio/bin64/drrun")
    remove_saved_results: bool = False


class DrCacheSimAnalyseTask(AnalysisTask):
    """Utility task to run drcachesim cache simulator on different configurations"""

    public = True
    task_name = "drcachesim"
    task_namespace = "drcachesim-run"
    task_config_class: typing.Type[Config] = DrCacheSimConfig

    def dependencies(self) -> typing.Iterable["Task"]:
        for _, row in self.session.benchmark_matrix.iterrows():
            for benchmark in row:
                data_path = benchmark.get_benchmark_data_path()
                base = data_path / "drcachesim-results"
                indir = data_path / "qemu-trace" / "qemu-trace-dir"
                if self.config.remove_saved_results:
                    shutil.rmtree(str(base), ignore_errors=True)
                if not base.exists():
                    base.mkdir(parents=True)

                for level in self.config.run_cache_levels:
                    if level == "LL":
                        sizes = self.config.LL_cache_sizes
                        out_path = base / "LL_size"
                        out_path.mkdir(exist_ok=True)
                        for s in sizes:
                            config = DrCacheSimRunConfig(
                                drrun_path=self.config.drrun_path,
                                indir=indir,
                                cache_level="LL",
                                cache_size=s,
                                output_path=out_path / f"{s}.txt",
                            )
                            yield DrCacheSimRunTask(
                                session=self.session,
                                task_config=config,
                            )
                    elif level == "L1D":
                        sizes = self.config.L1D_cache_sizes
                        out_path = base / "L1D_size"
                        out_path.mkdir(exist_ok=True)
                        for s in sizes:
                            config = DrCacheSimRunConfig(
                                drrun_path=self.config.drrun_path,
                                indir=indir,
                                cache_level="L1D",
                                cache_size=s,
                                output_path=out_path / f"{s}.txt",
                            )
                            yield DrCacheSimRunTask(
                                session=self.session,
                                task_config=config,
                            )
                    elif level == "L1I":
                        sizes = self.config.L1I_cache_sizes
                        out_path = base / "L1I_size"
                        out_path.mkdir(exist_ok=True)
                        for s in sizes:
                            config = DrCacheSimRunConfig(
                                drrun_path=self.config.drrun_path,
                                indir=indir,
                                cache_level="L1I",
                                cache_size=s,
                                output_path=out_path / f"{s}.txt",
                            )
                            yield DrCacheSimRunTask(
                                session=self.session,
                                task_config=config,
                            )
                    else:
                        self.logger.error(f"Unknown cache level {level}")

    def run(self):
        self.logger.info(f"Finished drcachesim analysis")


class InstrCountAnalyseTask(AnalysisTask):
    """Utility task to run drcachesim instr count on spec benchmarks"""

    public = True
    task_name = "instr_count"
    task_namespace = "drcachesim-run"
    task_config_class: typing.Type[Config] = InstrCountRunConfig

    def dependencies(self) -> typing.Iterable["Task"]:
        for multi_ind in self.session.benchmark_matrix.index.values:
            for benchmark in self.session.benchmark_matrix.loc[multi_ind]:
                data_path = benchmark.get_benchmark_data_path()
                variant, spec_variant = multi_ind
                base = data_path / "drcachesim-results"
                indir = data_path / "qemu-trace" / "qemu-trace-dir"
                out_dir = base / "instr_count"
                addr2line_dir = base / "addr2line"
                if self.config.remove_saved_results:
                    shutil.rmtree(str(out_dir), ignore_errors=True)
                if not out_dir.exists():
                    out_dir.mkdir(parents=True)

                out_dir.mkdir(exist_ok=True)
                addr2line_config = Addr2LineConfig(
                    obj_path=self.session.user_config.cheribsd_extra_files_path
                    / "root"
                    / "spec_static"
                    / spec_variant
                    / variant
                    / variant,
                    output_dir=addr2line_dir,
                    raw_output_path=self.session.get_plot_root_path()
                    / variant
                    / f"{spec_variant}_objdump.txt",
                )
                cachesim_config = DrCacheSimRunConfig(
                    drrun_path=self.config.drrun_path,
                    simulator="instr_count",
                    indir=indir,
                    addr2line_file=addr2line_dir / "addr2line.csv",
                    output_dir=out_dir,
                    addr2line_config=addr2line_config,
                )
                yield DrCacheSimRunTask(
                    session=self.session,
                    task_config=cachesim_config,
                )

    def run(self):
        self.logger.info(f"Finished instrcount analysis")
