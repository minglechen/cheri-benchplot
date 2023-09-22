import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from ..compile_db import (BuildConfig, Builder, CheriBSDBuild, CheriBSDBuildHelper)
from ..core.analysis import AnalysisTask
from ..core.artefact import (AnalysisFileTarget, DataFrameTarget, LocalFileTarget)
from ..core.config import InstanceKernelABI
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import DataGenTask, dependency, output
from ..core.util import SubprocessHelper
from ..ext import pychericap, pydwarf
from .model import (ImpreciseSubobjectInfoModel, ImpreciseSubobjectInfoModelRecord, ImpreciseSubobjectLayoutModel,
                    SetboundsKind, SubobjectBoundsModel, SubobjectBoundsUnionModel)


class CheriBSDSubobjectStats(DataGenTask):
    """
    Extract sub-object bounds from the kernel build.

    When this generator is used, a cheribsd kernel is built for every
    platform instance configuration in the pipeline.
    The task will collect clang-generated sub-object bounds statistics.
    Note that this relies on the cheribsd kernel configuration to instruct
    the kernel build to emit sub-object bounds information.
    This can be easily accomplished by adding the following line to the desired
    kernel configuration:

    .. code-block::

        makeoptions     CHERI_SUBOBJECT_BOUNDS_STATS


    Example configuration:

    .. code-block:: json

        {
            "instance_config": {
                "instances": [{
                    "kernel": "KERNEL-CONFIG-WITH-SUBOBJ-STATS",
                    "cheri_target": "riscv64-purecap",
                    "kernelabi": "purecap"
                }, {
                    "kernel": "KERNEL-CONFIG-WITH-SUBOBJ-STATS",
                    "cheri_target": "morello-purecap",
                    "kernelabi": "purecap"
                }]
            },
            "benchmark_config": [{
                "name": "example",
                "generators": [{ "handler": "kernel-static.cheribsd-subobject-stats" }]
            }]
        }

    """
    public = True
    task_namespace = "kernel-static"
    task_name = "cheribsd-subobject-stats"
    task_config_class = BuildConfig

    def __init__(self, benchmark, script, task_config):
        if not task_config.target:
            task_config.target = "cheribsd"
            task_config.builder = Builder.CheriBuild
        assert task_config.target == "cheribsd"
        super().__init__(benchmark, script, task_config)

    def _extract_kernel_stats(self, build_root) -> pd.DataFrame:
        """
        Fetch kernel subobject bounds statistics from the kernel build.

        Load the statistics into a temporary dataframe to be merged with the module data.
        """
        build_path = self.cheribsd.kernel_build_path(build_root)
        path = build_path / "kernel-subobject-bounds-stats.csv"
        if not path.exists():
            self.logger.error("Missing csetbounds stats for %s", self.benchmark)
            raise RuntimeError("Missing kernel csetbounds stats")

        df = pd.read_csv(path)
        df["src_module"] = "kernel"
        return df

    def _extract_module_stats(self, build_root) -> list[pd.DataFrame]:
        build_path = self.cheribsd.kernel_build_path(build_root)
        df_set = []
        for path in build_path.glob("modules/**/kernel-subobject-bounds-stats.csv"):
            df = pd.read_csv(path)
            df["src_module"] = path.parent.name
            df_set.append(df)
        return df_set

    @dependency
    def cheribsd(self):
        return CheriBSDBuildHelper(self.benchmark, self.script, self.config)

    @output
    def subobject_stats(self):
        return LocalFileTarget(self, ext="csv", model=SubobjectBoundsModel)

    def run(self):
        instance_config = self.benchmark.config.instance
        if instance_config.kernelabi != InstanceKernelABI.PURECAP:
            # It makes no sense to build this because no subobject stats will
            # ever be emitted, just make an empty file
            self.subobject_stats.path.touch()
            return

        # Extract the files from the build dir
        df_set = [self._extract_kernel_stats(self.cheribsd.build_root)]
        df_set.extend(self._extract_module_stats(self.cheribsd.build_root))
        df = pd.concat(df_set)

        # Patch the alignment_bits and size for unknown values
        unknown_align = df["alignment_bits"].map(str).str.startswith("<unknown")
        unknown_size = df["size"].map(str).str.startswith("<unknown")
        df["alignment_bits"] = df["alignment_bits"].mask(unknown_align, None)
        df["size"] = df["size"].mask(unknown_size, None)

        df.to_csv(self.subobject_stats.path)


class CheriBSDSubobjectStatsUnion(AnalysisTask):
    """
    Merge all statistics about subobject bounds from differet kernel configurations.

    Note that in order to overlap, a bounds record must match in both location, size and alignment.
    """
    task_namespace = "kernel-static"
    task_name = "subobject-bounds-stats-union"

    @dependency
    def raw_subobject_stats(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(CheriBSDSubobjectStats)
            yield task.subobject_stats.get_loader()

    def run(self):
        df = pd.concat([loader.df.get() for loader in self.raw_subobject_stats])
        df = df.groupby(["source_loc", "compiler_pass", "details"]).first()
        self.subobject_stats.assign(df)

    @output
    def subobject_stats(self):
        return DataFrameTarget(self, SubobjectBoundsUnionModel)


class CheriBSDSubobjectSizeBySize(AnalysisTask):
    """
    Generate a file that shows cases where subobject bounds are larger than 4K.

    Additional outputs are generated to separate large stack bounds and unknown cases.
    """
    public = True
    task_namespace = "kernel-static"
    task_name = "subobject-size-large"

    @dependency
    def stats(self):
        return CheriBSDSubobjectStatsUnion(self.session, self.analysis_config)

    def run(self):
        df = self.stats.subobject_stats.get()
        out_cols = ["src_module", "size", "compiler_pass", "details", "source_loc"]

        # Filter by kind=subobject and deterimne the largest
        out_df = df.loc[df["kind"] == SetboundsKind.SUBOBJECT.value]
        out_df = out_df.loc[out_df["size"] > 2**12]
        out_df = out_df.sort_values(by="size", ascending=False).reset_index()
        out_df[out_cols].to_csv(self.subobject_large.path)

    @output
    def subobject_large(self):
        return AnalysisFileTarget(self, ext="csv")


class CheriBSDSubobjectSizeDistribution(PlotTask):
    """
    Generate plots showing the distribution of subobject bounds sizes.
    """
    public = True
    task_namespace = "kernel-static"
    task_name = "subobject-size-distribution-plot"

    @dependency
    def stats(self):
        return CheriBSDSubobjectStatsUnion(self.session, self.analysis_config)

    def _plot_size_distribution(self, df: pd.DataFrame, target: PlotTarget):
        """
        Helper to plot the distribution of subobject bounds sizes
        """
        # Determine buckets we are going to use
        min_size = max(df["size"].min(), 1)
        max_size = max(df["size"].max(), 1)
        log_buckets = range(int(np.log2(min_size)), int(np.log2(max_size)) + 1)
        buckets = [2**i for i in log_buckets]

        with new_figure(target.paths()) as fig:
            ax = fig.subplots()
            sns.histplot(df, x="size", stat="count", bins=buckets, ax=ax)
            ax.set_yscale("log", base=10)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("size (bytes)")
            ax.set_ylabel("# of csetbounds")

    def run_plot(self):
        df = self.stats.subobject_stats.get()

        # Filter only setbounds that are marked kind=subobject
        data_df = df.loc[df["kind"] == SetboundsKind.SUBOBJECT.value]

        sns.set_theme()

        show_df = data_df.loc[data_df["src_module"] == "kernel"]
        self._plot_size_distribution(show_df, self.size_distribution_kernel)

        show_df = data_df.loc[data_df["src_module"] != "kernel"]
        self._plot_size_distribution(show_df, self.size_distribution_modules)

        self._plot_size_distribution(data_df, self.size_distribution_all)

    @output
    def size_distribution_kernel(self):
        return PlotTarget(self, prefix="size-distribution-kern")

    @output
    def size_distribution_modules(self):
        return PlotTarget(self, prefix="size-distribution-mods")

    @output
    def size_distribution_all(self):
        return PlotTarget(self, prefix="size-distribution-all")
