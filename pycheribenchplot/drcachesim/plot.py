import os
import typing
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from pycheribenchplot.core.config import AnalysisConfig, Config, TemplateConfig, ConfigPath
from pycheribenchplot.core.analysis import PlotTask
from pycheribenchplot.core.task import Target
from .analysis import (
    DrCacheSimAnalyseTask,
    DrCacheSimConfig,
    InstrCountAnalyseTask,
    InstrCountRunConfig,
    Addr2LineTask,
    Addr2LineConfig,
)


@dataclass
class CachePlotConfig(TemplateConfig):
    plot_cache_levels: typing.List[str] = field(default_factory=list)
    benchmark_param_name: str = "variant"
    variant_param_name: str = "spec_variant"
    drcachesim_config: DrCacheSimConfig = None


class CacheSizePlot(PlotTask):
    task_name = "cache-plot"
    task_config_class = CachePlotConfig
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
            self.session.get_plot_root_path()
            / "cache_plots"
            / (cache_level + "_cache_sizes")
            / (variant_name + ".pdf")
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

    def dependencies(self) -> typing.Iterable["Task"]:
        if self.config.drcachesim_config:
            yield DrCacheSimAnalyseTask(self.session, self.config.drcachesim_config)

    def run(self):
        matrix = self.session.benchmark_matrix
        for bench_name in matrix.index.unique(level=self.config.benchmark_param_name):
            dir_map = {}
            for variant in matrix.index.unique(level=self.config.variant_param_name):
                ind = (bench_name, variant)
                # We only need one instance if there are multiple
                benchmark = matrix.loc[ind][0]
                data_path = benchmark.get_benchmark_data_path()
                base = data_path / "drcachesim-results"
                dir_map[base] = variant
                
            for level in self.config.plot_cache_levels:
                self.logger.info(f"Plotting cache sizes for variant {bench_name}")
                self.plot_result(
                    level,
                    dir_map,
                    bench_name,
                )


@dataclass
class InstrCountPlotConfig(TemplateConfig):
    drrun_path: ConfigPath = field(default_factory=Path)
    benchmark_param_name: str = "variant"
    variant_param_name: str = "spec_variant"
    #: The group level (e.g. "line" or "symbol")
    group_level: str = "line"
    #: sort by ("ratio" or "diff")
    sort_by: str = "ratio"
    remove_drcachesim_results: bool = False
    plot_overhead: bool = False


class InstrCountPlot(PlotTask):
    """Creates csv files with the instruction count for comparison for each benchmark."""

    task_name = "instr-count-plot"
    task_config_class = InstrCountPlotConfig
    public = True

    def __init__(self, session: "Session", analysis_config, task_config = None):
        super().__init__(session, analysis_config, task_config=task_config)
        self.output_data = {}

    def dependencies(self) -> typing.Iterable["Task"]:
        config = InstrCountRunConfig(
            drrun_path=self.config.drrun_path,
            remove_saved_results=self.config.remove_drcachesim_results,
        )
        yield InstrCountAnalyseTask(
            self.session, self.analysis_config, task_config=config
        )
    
    def analysis_output(self) -> typing.Dict[str, pd.DataFrame]:
        return self.output_data
    
    def plot_overhead(self, df: pd.DataFrame, dir: Path, bench_name:str, report_top = 5, max_width = 20):

        print("total overhead:", df[df["diff"] > 0]["diff"].sum())

        df = df.sort_values(by=["diff"], ascending=False)
        df1 = df[df["diff"] > 0][["diff"]]
        df_top = df1.iloc[:report_top]
        df_rest = df1.iloc[report_top:]
        df_top = pd.concat(
            [df_top,
            pd.DataFrame(
                {"diff": [df_rest["diff"].sum()]}, index=["other"]
            )]
        )
        labels = list(map(lambda x: x[:max_width] + "..." if len(x) > max_width else x,df_top.index.values.flatten()))
        fig, ax = plt.subplots()
        ax.pie(df_top.values.flatten(), labels=labels, autopct='%1.1f%%')
        ax.axis('equal')
        # ax.legend(loc="upper right")
        plt.title(f"Overhead by {self.config.group_level} for {bench_name} in purecap mode")
        plt.tight_layout()
        plt.savefig(dir / f"{bench_name}.pdf")
        plt.close()

        fig, ax2 = plt.subplots()
        df2 = df[["purecap_count", "hybrid_count", "diff", "ratio"]]
        df2 = df2.sort_values(by=["diff"], ascending=False)
        df2 = df2.iloc[:report_top]
        print (df2)
        max_width = 10
        df2.index = df2.index.map(lambda x: x[:max_width] + "..." if len(x) > max_width else x)
        ax2 = df2.plot.bar(y=["purecap_count", "hybrid_count"], rot=0, use_index=True)
        ax2.set_ylabel("Instruction count")
        ax2.set_xlabel(self.config.group_level)

        plt.title(f"Instruction count for {bench_name}")
        plt.tight_layout()
        plt.savefig(dir / f"{bench_name}_instr.pdf")
        plt.close()




    def run(self):
        matrix: pd.DataFrame = self.session.benchmark_matrix
        group = ["path", "line"]
        if self.config.group_level == "line":
            group = ["path", "line"]
        elif self.config.group_level == "symbol":
            group = ["symbol"]
        else:
            self.logger.error(f"Unknown group level {self.config.group_level}")

        for bench_name in matrix.index.unique(level=self.config.benchmark_param_name):
            df_purecap = pd.DataFrame()
            df_hybrid = pd.DataFrame()
            for variant in matrix.index.unique(level=self.config.variant_param_name):
                ind = (bench_name, variant)
                # We only need one instance if there are multiple
                benchmark = matrix.loc[ind][0]
                data_path = benchmark.get_benchmark_data_path()
                base = data_path / "drcachesim-results"
                out_path = base / "instr_count" / "instr_counts.csv"
                assert out_path.is_file(), f"Missing {out_path}"

                if self.config.group_level == "symbol":
                    symbol_path = base / "addr2line" / "symbol.csv"

                df = pd.read_csv(out_path)
                print(variant, df["count"].sum())
                if "purecap" in variant:
                    df["path"] = df["path"].map(
                        lambda x: str(x).split("riscv64-purecap-build")[-1]
                    )
                    df_purecap = (
                        (df.groupby(group)["count"].sum().rename("purecap_count"))
                        .to_frame()
                        .reset_index()
                    )
                    if self.config.group_level == "symbol":
                        df_symbol = pd.read_csv(symbol_path)
                        purecap_calls = (
                            pd.merge(
                                df[["addr", "count"]],
                                df_symbol,
                                left_on="addr",
                                right_on="addr",
                                how="inner",
                            )[["symbol", "count"]]
                            .rename(columns={"count": "purecap_calls"})
                            .groupby("symbol")
                            .sum()
                        )
                        df_purecap = pd.merge(
                            df_purecap, purecap_calls, on="symbol"
                        ).set_index(group)
                    elif self.config.group_level == "line":
                        df_purecap = df_purecap.set_index(group)

                elif "hybrid" in variant:
                    df["path"] = df["path"].map(
                        lambda x: str(x).split("riscv64-build")[-1]
                    )
                    df_hybrid = (
                        df.groupby(group)["count"]
                        .sum()
                        .rename("hybrid_count")
                        .to_frame()
                        .reset_index()
                    )
                    if self.config.group_level == "symbol":
                        df_symbol = pd.read_csv(symbol_path)
                        hybrid_calls = (
                            pd.merge(
                                df[["addr", "count"]],
                                df_symbol,
                                left_on="addr",
                                right_on="addr",
                                how="inner",
                            )[["symbol", "count"]]
                            .rename(columns={"count": "hybrid_calls"})
                            .groupby("symbol")
                            .sum()
                        )
                        df_hybrid = pd.merge(
                            df_hybrid, hybrid_calls, on="symbol"
                        ).set_index(group)
                    elif self.config.group_level == "line":
                        df_hybrid = df_hybrid.set_index(group)
                else:
                    raise Exception(f"Unknown variant {variant}")

            df = pd.concat([df_purecap, df_hybrid], axis=1, join="inner")
            df["ratio"] = df["purecap_count"] / df["hybrid_count"]
            df["diff"] = df["purecap_count"] - df["hybrid_count"]
            df = df.sort_values(by=[self.config.sort_by], ascending=False)
            self.output_data[bench_name] = df
            print(bench_name)
            print(df.head(10).to_string(justify="right"))
            group_level = self.config.group_level
            plot_path_base = (
                self.session.get_plot_root_path()
                / bench_name
            )
            plot_path = plot_path_base / f"instr_count_{group_level}_{self.config.sort_by}.csv"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            # df.to_csv(plot_path)
            with plot_path.open("w") as f:
                f.write(df.to_csv())
            if self.config.plot_overhead:
                self.plot_overhead(df, plot_path_base, bench_name)


@dataclass
class StaticInstrCountPlotConfig(TemplateConfig):
    benchmark_param_name: str = "variant"
    variant_param_name: str = "spec_variant"
    #: The group level (e.g. "line" or "symbol")
    group_level: str = "line"
    #: sort by ("ratio" or "diff")
    sort_by: str = "ratio"


class StaticInstrCountPlot(PlotTask):
    task_name = "static-instr-count-plot"
    task_config_class = StaticInstrCountPlotConfig
    public = True

    def dependencies(self) -> typing.Iterable["Task"]:
        for multi_ind in self.session.benchmark_matrix.index.values:
            for benchmark in self.session.benchmark_matrix.loc[multi_ind]:
                variant, spec_variant = multi_ind
                data_path = benchmark.get_benchmark_data_path()
                base = data_path / "drcachesim-results"
                out_path = base / "instr_count"
                addr2line_dir = base / "addr2line"
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
                    output_dir=addr2line_dir,
                    raw_output_path=raw_path,
                )
                yield Addr2LineTask(self.session, config)

    def run(self):
        matrix: pd.DataFrame = self.session.benchmark_matrix
        group = ["path", "line"]
        if self.config.group_level == "line":
            group = ["path", "line"]
        elif self.config.group_level == "symbol":
            group = ["symbol"]
        else:
            self.logger.error(f"Unknown group level {self.config.group_level}")

        for bench_name in matrix.index.unique(level=self.config.benchmark_param_name):
            df_purecap = pd.DataFrame()
            df_hybrid = pd.DataFrame()
            for variant in matrix.index.unique(level=self.config.variant_param_name):
                ind = (bench_name, variant)
                # We only need one instance if there are multiple
                benchmark = matrix.loc[ind][0]
                data_path = benchmark.get_benchmark_data_path()
                base = data_path / "drcachesim-results"
                out_path = base / "addr2line" / "addr2line.csv"
                plot_path_base = self.session.get_plot_root_path() / bench_name
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
                # print()
                # print("static", bench_name, variant)
            df = pd.concat([df_purecap, df_hybrid], axis=1, join="inner")
            df["ratio"] = df["purecap_count"] / df["hybrid_count"]
            df["diff"] = df["purecap_count"] - df["hybrid_count"]
            df = df.sort_values(by=[self.config.sort_by], ascending=False)
            # print(df.head(10).to_string(justify="right"))
            group_level = self.config.group_level
            plot_path = (
                plot_path_base
                / f"static_instr_count_{group_level}_{self.config.sort_by}.csv"
            )
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            # df.to_csv(plot_path)
            with plot_path.open("w") as f:
                f.write(df.to_csv())


@dataclass
class FunctionAnalysisPlotConfig(TemplateConfig):
    drrun_path: ConfigPath = field(default_factory=Path)
    benchmark_param_name: str = "variant"
    variant_param_name: str = "spec_variant"
    function_name: str = "main"
    remove_drcachesim_results: bool = False


class FunctionAnalysisPlot(PlotTask):

    task_name = "function-analysis-plot"
    task_config_class = FunctionAnalysisPlotConfig
    public = True

    def dependencies(self) -> typing.Iterable["Task"]:
        config = InstrCountRunConfig(
            drrun_path=self.config.drrun_path,
            remove_saved_results=self.config.remove_drcachesim_results,
        )
        yield InstrCountAnalyseTask(
            self.session, self.analysis_config, task_config=config
        )

    def run(self):
        matrix: pd.DataFrame = self.session.benchmark_matrix

        for bench_name in matrix.index.unique(level=self.config.benchmark_param_name):
            df_purecap = pd.DataFrame()
            df_hybrid = pd.DataFrame()
            for variant in matrix.index.unique(level=self.config.variant_param_name):
                ind = (bench_name, variant)
                # We only need one instance if there are multiple
                benchmark = matrix.loc[ind][0]
                data_path = benchmark.get_benchmark_data_path()
                base = data_path / "drcachesim-results"
                out_path = base / "instr_count" / "instr_counts.csv"
                assert out_path.is_file(), f"Missing {out_path}"

                symbol_path = base / "addr2line" / "symbol.csv"

                df_symbol = pd.read_csv(symbol_path)
                df = pd.read_csv(out_path)
                if "purecap" in variant:
                    symbol_range = df_symbol[
                        df_symbol["symbol"] == self.config.function_name
                    ].reset_index(drop=True)
                    start = symbol_range["addr"].iloc[0]
                    end = symbol_range["end_addr"].iloc[0]
                    purecap_df = (
                        df[(df["addr"] >= start) & (df["addr"] <= end)]
                        .sort_values(by=["addr"])
                        .reset_index(drop=True)
                    )
                    # print("purecap")
                    # print(purecap_df)

                elif "hybrid" in variant:
                    symbol_range = df_symbol[
                        df_symbol["symbol"] == self.config.function_name
                    ].reset_index(drop=True)
                    start = symbol_range["addr"].iloc[0]
                    end = symbol_range["end_addr"].iloc[0]
                    hybrid_df = (
                        df[(df["addr"] >= start) & (df["addr"] <= end)]
                        .sort_values(by=["addr"])
                        .reset_index(drop=True)
                    )
                    # print("hybrid")
                    # print(hybrid_df)
                else:
                    raise Exception(f"Unknown variant {variant}")

            plot_path = (
                self.session.get_plot_root_path()
                / bench_name
                / f"function_analysis_{self.config.function_name}.pdf"
            )

            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.locator_params(axis="x", nbins=10)
            fig, (ax1, ax2) = plt.subplots(2)
            fig.suptitle(
                f"Function Analysis of {self.config.function_name}, {bench_name}"
            )

            ax1.set_title("Purecap")
            ax2.set_title("Hybrid")
            ax1.set_ylabel("Instruction Hits")
            ax2.set_ylabel("Instruction Hits")
            ax1.set_xlabel("Instruction index")
            ax2.set_xlabel("Instruction index")

            ax1.plot(purecap_df["addr"].index, purecap_df["count"])
            ax2.plot(hybrid_df["addr"].index, hybrid_df["count"])
            fig.tight_layout()
            # ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
            # ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
            # ax1.xaxis.set_major_locator(plt.maxNLocator(5))
            # ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%x"))
            # ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
            # ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%x"))
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)

@dataclass
class CrossComparePlotConfig(TemplateConfig):
    optimized_file: ConfigPath = field(default_factory=Path)
    baseline_file: ConfigPath = field(default_factory=Path)
    output_file: ConfigPath = field(default_factory=Path)
    output_plot: ConfigPath = field(default_factory=Path)
    bench_name: str = "471.omnetpp"
    plot_symbols: list[str] = field(default_factory=list)
    additional_labels: list[str] = field(default_factory=list)
    additional_label_names: list[str] = field(default_factory=list)
    instr_count_config: InstrCountPlotConfig = None

class CrossComparePlot(PlotTask):
    task_name = "cross-compare-plot"
    task_config_class = CrossComparePlotConfig
    public = True

    def __init__(self, session: "Session", analysis_config: AnalysisConfig, task_config: Config = None):
        super().__init__(session, analysis_config, task_config)
        self.plot_task = None

    def dependencies(self) -> typing.Iterable["Task"]:
        if self.config.instr_count_config:
            self.plot_task = InstrCountPlot(self.session, self.analysis_config, self.config.instr_count_config)
            yield self.plot_task

    def run(self):
        # Read in the data
        df1 = pd.read_csv(self.config.baseline_file)
        if self.plot_task:
            df2 = self.plot_task.analysis_output()[self.config.bench_name]
        else:
            df2 = pd.read_csv(self.config.optimized_file)

        df_base = df1[["symbol", "purecap_count"]].rename(columns={"purecap_count": "purecap_baseline"})
        df_opt = df2[["symbol", "purecap_count"]].rename(columns={"purecap_count": "purecap_optimization"})

        # Merge the data
        df = pd.merge(df_base, df_opt, on="symbol", how="outer")
        
        if(self.config.additional_labels):
            df = pd.merge(df, df1[["symbol"]+self.config.additional_labels], on="symbol", how="outer")

        df = df.rename(columns={orig: new for orig, new in zip(self.config.additional_labels, self.config.additional_label_names)})
        # Compute the difference
        df["purecap_count_diff"] = df["purecap_baseline"] - df["purecap_optimization"]

        # Sort the data
        df = df.sort_values("purecap_count_diff", ascending=False)
        print(df)

        # Save the data
        df.to_csv(self.config.output_file, index=False)
        df.set_index("symbol", inplace=True)

        df = df.loc[self.config.plot_symbols]
        df /= 1000000
        # plot the data
        fig, ax = plt.subplots()
        fig.suptitle(f"Instruction Count of {self.config.bench_name}")
        ax.set_ylabel("Instruction Count (Million)")
        ax.set_xlabel("Symbol")
        df.plot.bar(y=["purecap_baseline", "purecap_optimization"] + self.config.additional_label_names, ax=ax, use_index=True)
        fig.savefig(self.config.output_plot, bbox_inches="tight")
        plt.close(fig)
