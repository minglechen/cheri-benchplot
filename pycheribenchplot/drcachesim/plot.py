import os
import typing
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pycheribenchplot.core.config import TemplateConfig
from pycheribenchplot.core.analysis import PlotTask
from .analysis import (
    DrCacheSimAnalyseTask,
    InstrCountAnalyseTask,
    InstrCountRunConfig,
    Addr2LineTask,
    Addr2LineConfig,
)


@dataclass
class CachePlotConfig(TemplateConfig):
    plot_cache_levels: typing.List[str] = field(default_factory=list)


class CacheSizePlot(PlotTask):
    task_name = "cache-plot"
    task_config = CachePlotConfig
    public = True

    def convert_prefix(self, prefix):
        if prefix == "K":
            return 1024
        elif prefix == "M":
            return 1024**2
        elif prefix == "G":
            return 1024**3
        else:
            return 1

    def plot_result_internal(self, cache_level, outdir, run_name):
        if cache_level == "LL":
            file_path = outdir / "LL_size"
            key_str = "Local miss rate:"
            start_str = "LL stats:"
        elif cache_level == "L1D":
            file_path = outdir / "L1D_size"
            key_str = "Miss rate:"
            # for now assume there is only one core
            start_str = "L1D stats:"
        elif cache_level == "L1I":
            file_path = outdir / "L1I_size"
            key_str = "Miss rate:"
            start_str = "L1I stats:"
        if not file_path.exists():
            raise Exception(f"No cache size file found for {file_path}")
        files = os.listdir(file_path)
        cache_sizes = np.array([])
        miss_rates = np.array([])
        for file in files:
            if file.endswith(".txt"):
                cache_size = file.split(".")[0]
                with open(file_path / file, "r") as f:
                    buf = f.read()
                    start_ind = buf.find(start_str)
                    if start_ind == -1:
                        continue
                    else:
                        start_ind += len(start_str)
                    ind = buf.find(key_str, start_ind)
                    if ind == -1:
                        continue
                    else:
                        ind += len(key_str)
                        miss_rate = float(buf[ind:].split()[0].strip("%")) / 100
                        cache_sizes = np.append(cache_sizes, cache_size)
                        miss_rates = np.append(miss_rates, miss_rate)

        cache_sizes_bytes = np.array(
            [int(x[:-1]) * self.convert_prefix(x[-1]) for x in cache_sizes]
        )
        ind = np.argsort(cache_sizes_bytes)
        cache_sizes_bytes = cache_sizes_bytes[ind]
        miss_rates = miss_rates[ind]
        cache_sizes = cache_sizes[ind]

        plt.plot(cache_sizes_bytes, miss_rates, "o-", label=run_name)
        plt.xticks(cache_sizes_bytes, cache_sizes)

    def plot_result(self, cache_level, outdirs, variant_name):
        plt.xscale("log")
        if cache_level == "LL":
            plt.xlabel("LL cache size")
        elif cache_level == "L1D":
            plt.xlabel("L1D cache size")
        elif cache_level == "L1I":
            plt.xlabel("L1I cache size")
        for outdir, spec_variant in outdirs.items():
            outdir = Path(outdir)
            try:
                self.plot_result_internal(cache_level, outdir, spec_variant)
            except Exception as e:
                self.logger.warning(
                    f"Failed to plot cache sizes for {variant_name}: {e}"
                )
                return
        plt.ylabel("Local miss rate")
        # plt.ylim(0, 1)
        plt.title(f"Local miss rate vs {cache_level} cache size: {variant_name}")
        plt.legend(loc="upper right")
        file_path = (
            self.get_plot_root_path()
            / "cache_plots"
            / (cache_level + "_cache_sizes")
            / (variant_name + ".png")
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path, format="png")
        plt.close()

    def dependencies(self) -> typing.Iterable["Task"]:
        yield DrCacheSimAnalyseTask(self.session, self.analysis_config)

    def run(self):
        trace_info = self.session.benchmark_matrix
        groups = trace_info.groupby("variant").groups
        for variant, group in groups.items():
            for level in self.config.plot_cache_levels:
                self.logger.info(f"Plotting cache sizes for variant {variant}")
                self.plot_result(
                    level,
                    {
                        k: v
                        for k, v in zip(
                            trace_info.iloc[group]["cachesim_dir"],
                            trace_info.iloc[group]["spec_variant"],
                        )
                    },
                    variant,
                )


@dataclass
class InstrCountPlotConfig(TemplateConfig):
    drrun_path: Path = field(default_factory=Path)
    benchmark_param_name: str = "variant"
    variant_param_name: str = "spec_variant"
    #: The group level (e.g. "line" or "symbol")
    group_level: str = "line"


class InstrCountPlot(PlotTask):
    """Creates csv files with the instruction count for comparison for each benchmark."""

    task_name = "instr-count-plot"
    task_config = InstrCountPlotConfig
    public = True

    def dependencies(self) -> typing.Iterable["Task"]:
        config = InstrCountRunConfig(drrun_path=self.config["drrun_path"])
        yield InstrCountAnalyseTask(
            self.session, self.analysis_config, task_config=config
        )

    def run(self):
        matrix: pd.DataFrame = self.session.benchmark_matrix
        group = ["path", "line"]
        if self.config["group_level"] == "line":
            group = ["path", "line"]
        elif self.config["group_level"] == "symbol":
            group = ["path", "symbol"]
        else:
            self.logger.error(f"Unknown group level {self.config['group_level']}")

        for bench_name in matrix.index.unique(
            level=self.config["benchmark_param_name"]
        ):
            df_purecap = pd.DataFrame()
            df_hybrid = pd.DataFrame()
            for variant in matrix.index.unique(level=self.config["variant_param_name"]):
                ind = (bench_name, variant)
                # We only need one instance if there are multiple
                benchmark = matrix.loc[ind][0]
                data_path = benchmark.get_benchmark_data_path()
                base = data_path / "drcachesim-results"
                out_path = base / "instr_count" / "instr_counts.csv"
                assert out_path.is_file(), f"Missing {out_path}"

                df = pd.read_csv(out_path)
                if "purecap" in variant:
                    df["path"] = df["path"].map(
                        lambda x: str(x).split("riscv64-purecap-build")[-1]
                    )
                    df_purecap = (
                        df.groupby(group)["count"].sum().rename("purecap_count")
                    )
                elif "hybrid" in variant:
                    df["path"] = df["path"].map(
                        lambda x: str(x).split("riscv64-build")[-1]
                    )
                    df_hybrid = df.groupby(group)["count"].sum().rename("hybrid_count")
                else:
                    raise Exception(f"Unknown variant {variant}")
                print()
                print(bench_name, variant)
            df = pd.concat([df_purecap, df_hybrid], axis=1, join="inner")
            df["ratio"] = df["purecap_count"] / df["hybrid_count"]
            df = df.sort_values(by=["ratio"], ascending=False)
            print(df.head(10).to_string(justify="right"))
            group_level = self.config["group_level"]
            plot_path = (
                self.session.get_plot_root_path()
                / bench_name
                / f"instr_count_{group_level}_cmp.csv"
            )
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(plot_path)


@dataclass
class StaticInstrCountPlotConfig(TemplateConfig):
    benchmark_param_name: str = "variant"
    variant_param_name: str = "spec_variant"
    #: The group level (e.g. "line" or "symbol")
    group_level: str = "line"


class StaticInstrCountPlot(PlotTask):
    task_name = "static-instr-count-plot"
    task_config = StaticInstrCountPlotConfig
    public = True

    def dependencies(self) -> typing.Iterable["Task"]:
        for multi_ind in self.session.benchmark_matrix.index.values:
            for benchmark in self.session.benchmark_matrix.loc[multi_ind]:
                variant, spec_variant = multi_ind
                data_path = benchmark.get_benchmark_data_path()
                base = data_path / "drcachesim-results"
                out_path = base / "instr_count"
                addr2line_path = base / "addr2line.csv"
                plot_path_base = self.session.get_plot_root_path() / variant
                raw_path = plot_path_base / f"{spec_variant}_objdump.txt"
                if not out_path.exists():
                    out_path.mkdir(parents=True)
                config = Addr2LineConfig(
                    obj_path=self.session.user_config.cheribsd_extra_files_path
                    / "root"
                    / "spec_static"
                    / spec_variant
                    / variant
                    / variant,
                    output_path=addr2line_path,
                    raw_output_path=raw_path,
                )
                yield Addr2LineTask(self.session, config)

    def run(self):
        matrix: pd.DataFrame = self.session.benchmark_matrix
        group = ["path", "line"]
        if self.config["group_level"] == "line":
            group = ["path", "line"]
        elif self.config["group_level"] == "symbol":
            group = ["path", "symbol"]
        else:
            self.logger.error(f"Unknown group level {self.config['group_level']}")

        for bench_name in matrix.index.unique(
            level=self.config["benchmark_param_name"]
        ):
            df_purecap = pd.DataFrame()
            df_hybrid = pd.DataFrame()
            for variant in matrix.index.unique(level=self.config["variant_param_name"]):
                ind = (bench_name, variant)
                # We only need one instance if there are multiple
                benchmark = matrix.loc[ind][0]
                data_path = benchmark.get_benchmark_data_path()
                base = data_path / "drcachesim-results"
                out_path = base / "addr2line.csv"
                plot_path_base = self.session.get_plot_root_path() / bench_name
                raw_path = plot_path_base / "objdump.txt"
                assert out_path.is_file(), f"Missing {out_path}"
                df = pd.read_csv(out_path)
                if "purecap" in variant:
                    df["path"] = df["path"].map(
                        lambda x: str(x).split("riscv64-purecap-build")[-1]
                    )
                    df_purecap = df.groupby(group).size().rename("purecap_count")
                elif "hybrid" in variant:
                    df["path"] = df["path"].map(
                        lambda x: str(x).split("riscv64-build")[-1]
                    )
                    df_hybrid = df.groupby(group).size().rename("hybrid_count")
                else:
                    raise Exception(f"Unknown variant {variant}")
                print()
                print("static", bench_name, variant)
            df = pd.concat([df_purecap, df_hybrid], axis=1, join="inner")
            df["ratio"] = df["purecap_count"] / df["hybrid_count"]
            df = df.sort_values(by=["ratio"], ascending=False)
            print(df.head(10).to_string(justify="right"))
            group_level = self.config["group_level"]
            plot_path = plot_path_base / f"static_instr_count_{group_level}_cmp.csv"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(plot_path)
