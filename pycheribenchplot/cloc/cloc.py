from dataclasses import dataclass, field

import pandas as pd
import seaborn as sns

from ..compile_db import CompilationDB, CompilationDBModel
from ..core.analysis import AnalysisTask
from ..core.artefact import AnalysisFileTarget, DataFrameTarget
from ..core.config import Config
from ..core.model import check_data_model
from ..core.pandas_util import generalized_xs
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import dependency, output
from .cheribsd import ExtractLoCCheriBSD
from .cloc_generic import ExtractLoCMultiRepo
from .model import LoCCountModel, LoCDiffModel


def to_file_type(path: str):
    # Don't expect C++ in the kernel
    if path.endswith(".h") or path.endswith(".hh") or path.endswith(".hpp"):
        return "header"
    if path.endswith(".c") or path.endswith(".cc") or path.endswith(".cpp"):
        return "source"
    if path.endswith(".S") or path.endswith(".s"):
        return "assembly"
    return "other"


@dataclass
class LoCAnalysisConfig(Config):
    """
    Common analysis configuration for LoC data
    """
    pass


class LoadLoCGenericData(AnalysisTask):
    """
    Load line of code changes by file.
    This is the place to configure filtering based on the compilation DB or other
    criteria, so that all dependent tasks operate on the filtered data normally.

    Note that this the central loader for all LoC data. This allows the consumers
    downstream to ignore the source of the data and just plot/process it.
    The only requirement is that the data input must conform to the
    LoCDiffModel and LoCCountModel.
    """
    task_namespace = "cloc"
    task_name = "load-loc-generic-data"
    task_config_class = LoCAnalysisConfig

    @dependency(optional=True)
    def compilation_db(self):
        """
        Load all compilation databases.

        Note that there is a cdb for each instance, as the instance
        already parameterizes both the kernel ABI and the user world ABI.
        """
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(CompilationDB)
            yield from (t.get_loader() for t in task.cdb)

    @dependency(optional=True)
    def cloc_diff(self):
        task = self.session.find_exec_task(ExtractLoCMultiRepo)
        return [tgt.get_loader() for tgt in task.cloc_diff]

    @dependency(optional=True)
    def cloc_baseline(self):
        task = self.session.find_exec_task(ExtractLoCMultiRepo)
        return [tgt.get_loader() for tgt in task.cloc_baseline]

    @dependency(optional=True)
    def cloc_cheribsd_diff(self):
        task = self.session.find_exec_task(ExtractLoCCheriBSD)
        return task.cloc_diff.get_loader()

    @dependency(optional=True)
    def cloc_cheribsd_baseline(self):
        task = self.session.find_exec_task(ExtractLoCCheriBSD)
        return task.cloc_baseline.get_loader()

    @check_data_model
    def _load_compilation_db(self) -> CompilationDBModel:
        """
        Fetch all compilation DBs and merge them
        """
        cdb_set = []
        for loader in self.compilation_db:
            cdb_set.append(loader.df.get())
        return pd.concat(cdb_set).groupby("file").first().reset_index()

    def run(self):
        # Merge the data from inputs
        all_diff = []
        all_baseline = []
        if self.cloc_diff:
            all_diff += [loader.df.get() for loader in self.cloc_diff]
            all_baseline = [loader.df.get() for loader in self.cloc_baseline]
        if self.cloc_cheribsd_diff:
            all_diff += [self.cloc_cheribsd_diff.df.get()]
            all_baseline += [self.cloc_cheribsd_baseline.df.get()]

        if not all_diff or not all_baseline:
            self.logger.error("No data to consume")
            raise RuntimeError("Empty input")
        if len(all_diff) != len(all_baseline):
            self.logger.error("Must have the same number of Diff and Baseline LoC dataframes. " +
                              "Something is very wrong with dependencies.")
            raise RuntimeError("Dependency mismatch")

        diff_df = pd.concat(all_diff)
        baseline_df = pd.concat(all_baseline)

        if self.compilation_db:
            cdb = self._load_compilation_db()
            # Filter by compilation DB files
            filtered_diff_df = diff_df.loc[diff_df.index.isin(cdb["file"], level="file")]
            filtered_baseline_df = baseline_df.loc[baseline_df.index.isin(cdb["file"], level="file")]
            self.cdb_diff_df.assign(filtered_diff_df)
            self.cdb_baseline_df.assign(filtered_baseline_df)
        self.diff_df.assign(diff_df)
        self.baseline_df.assign(baseline_df)

    @output
    def diff_df(self):
        return DataFrameTarget(self, LoCDiffModel, output_id="all-diff")

    @output
    def baseline_df(self):
        return DataFrameTarget(self, LoCCountModel, output_id="all-baseline")

    @output
    def cdb_diff_df(self):
        return DataFrameTarget(self, LoCDiffModel, output_id="cdb-diff")

    @output
    def cdb_baseline_df(self):
        return DataFrameTarget(self, LoCCountModel, output_id="cdb-baseline")


class ReportLoCGeneric(PlotTask):
    """
    Produce a plot of CLoC changed for each repository
    """
    public = True
    task_namespace = "cloc"
    task_name = "plot-deltas"
    task_config_class = LoCAnalysisConfig

    @dependency
    def loc_data(self):
        return LoadLoCGenericData(self.session, self.analysis_config, task_config=self.config)

    def run_plot(self):
        df = self.loc_data.diff_df.get()
        df = generalized_xs(df, how="same", complement=True)
        base_df = self.loc_data.baseline_df.get()

        # Ensure we have the proper theme
        sns.set_theme()
        # Use tab10 colors, added => green, modified => orange, removed => red
        cmap = sns.color_palette(as_cmap=True)
        # ordered as green, orange, red
        colors = [cmap[2], cmap[1], cmap[3]]

        # Aggregate counts by repo
        base_counts_df = base_df["code"].groupby(["repo"]).sum()
        agg_df = df.groupby(["repo", "how"]).sum().join(base_counts_df, on="repo", rsuffix="_baseline")
        agg_df["code_pct"] = 100 * agg_df["code"] / agg_df["code_baseline"]

        agg_df.to_csv(self.raw_data.path)

        with new_figure(self.plot.paths()) as fig:
            ax_l, ax_r = fig.subplots(1, 2, sharey=True)
            # Absolute SLoC on the left
            show_df = agg_df.reset_index().pivot(index="repo", columns="how", values="code")
            show_df.reset_index().plot(x="repo",
                                       y=["added", "modified", "removed"],
                                       stacked=True,
                                       kind="barh",
                                       ax=ax_l,
                                       color=colors,
                                       legend=False)
            ax_l.tick_params(axis="y", labelsize="x-small")
            ax_l.set_xlabel("# of lines")

            # Percent SLoC on the right
            show_df = agg_df.reset_index().pivot(index="repo", columns="how", values="code_pct")
            show_df.reset_index().plot(x="repo",
                                       y=["added", "modified", "removed"],
                                       stacked=True,
                                       kind="barh",
                                       ax=ax_r,
                                       color=colors,
                                       legend=False)
            ax_r.set_xlabel("% of lines")

            # The legend is shared at the top center
            handles, labels = ax_l.get_legend_handles_labels()
            fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))

    @output
    def plot(self):
        return PlotTarget(self)

    @output
    def raw_data(self):
        return AnalysisFileTarget(self, prefix="raw", ext="csv")


class LoCChangesSummary(AnalysisTask):
    """
    Produce a latex-compatible table summary for the SLoC changes.

    The following columns are present:
    - Type: header, source, assembly, other
    - Total SLoC
    - Changed SLoC
    - % Changed SLoC
    - Total Files
    - Changed Files
    - % Changed Files
    """
    public = True
    task_namespace = "cloc"
    task_name = "diff-summary-table"
    task_config_class = LoCAnalysisConfig

    @dependency
    def loc_data(self):
        return LoadLoCGenericData(self.session, self.analysis_config, task_config=self.config)

    def _add_totals(self, df: pd.Series) -> pd.Series:
        total = df.groupby("repo").sum().to_frame()
        total["Type"] = "total"
        total.set_index("Type", append=True, inplace=True)
        result = pd.concat([df.to_frame(), total], axis=0).iloc[:, 0]

        return result

    def _build_stats(self, df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the table for given file groups.
        """
        # drop the rows with how == "same"
        data_df = generalized_xs(df, how="same", complement=True)
        # don't care about the split in how, all of them are changes
        data_df = data_df.groupby(["repo", "file"]).sum()
        # drop rows that do not have code changes
        data_df = data_df.loc[data_df["code"] != 0]

        # Determine file types
        data_df["Type"] = data_df.index.get_level_values("file").map(to_file_type)
        baseline_df["Type"] = baseline_df.index.get_level_values("file").map(to_file_type)
        # Group by target and type and sum the "code" column
        group_keys = ["repo", "Type"]
        data_count = data_df.groupby(group_keys)["code"].sum()
        baseline_count = baseline_df.groupby(group_keys)["code"].sum()

        data_nfiles = data_df.groupby(group_keys).size()
        baseline_nfiles = baseline_df.groupby(group_keys).size()

        # add a row with the grand total
        data_count = self._add_totals(data_count)
        baseline_count = self._add_totals(baseline_count)
        data_nfiles = self._add_totals(data_nfiles)
        baseline_nfiles = self._add_totals(baseline_nfiles)

        # Finally, build the stats frame
        stats = {
            "Total SLoC": baseline_count,
            "Changed SLoC": data_count,
            "% Changed SLoC": 100 * data_count / baseline_count,
            "Total files": baseline_nfiles,
            "Changed files": data_nfiles,
            "% Changed files": 100 * data_nfiles / baseline_nfiles,
        }
        return pd.DataFrame(stats).round(3)

    def run(self):
        # Build stats without CDB
        stats_df = self._build_stats(self.loc_data.diff_df.get(), self.loc_data.baseline_df.get())
        stats_df.fillna(0).to_csv(self.table.path)

        totals_df = stats_df.xs("total", level="Type")
        totals_df.fillna(0).to_csv(self.totals_table.path)

        if self.loc_data.compilation_db is not None:
            stats_df = self._build_stats(self.loc_data.cdb_diff_df.get(), self.loc_data.cdb_baseline_df.get())
            stats_df.fillna(0).to_csv(self.cdb_table.path)

    @output
    def table(self):
        return AnalysisFileTarget(self, prefix="all-summary", ext="csv")

    @output
    def totals_table(self):
        return AnalysisFileTarget(self, prefix="condensed-summary", ext="csv")

    @output
    def cdb_table(self):
        return AnalysisFileTarget(self, prefix="cdb-summary", ext="csv")


class CheriBSDLineChangesByPath(PlotTask):
    """
    Produce a mosaic plot showing the changes nesting by path

    DEPRECATED / UNFIXED
    """
    public = True
    task_namespace = "cloc"
    task_name = "plot-diff-mosaic"
    task_config_class = LoCAnalysisConfig

    @dependency
    def loc_diff(self):
        return LoadLoCGenericData(self.session, self.analysis_config, task_config=self.config)

    def run_plot(self):
        df = self.loc_diff.diff_df.get()
        # Don't care how the lines changed
        data_df = generalized_xs(df, how="same", complement=True)
        data_df = data_df.groupby("file").sum()
        # Drop the rows where code changes are 0, as these probably mark
        # historical changes or just comments
        data_df = data_df.loc[data_df["code"] != 0]

        # XXX this does not agree well with multithreading
        sns.set_palette("crest")

        with new_figure(self.plot.path) as fig:
            ax = fig.subplots()
            filename_components = data_df.index.get_level_values("file").str.split("/")
            # Remove the leading "sys", as it is everywhere, and the filename component
            data_df["components"] = filename_components.map(lambda c: c[1:-1])
            mosaic_treemap(data_df, fig, ax, bbox=(0, 0, 100, 100), values="code", groups="components", maxlevel=2)

    @output
    def plot(self):
        return PlotTarget(self)


class CheriBSDLineChangesByFile(PlotTask):
    """
    Produce a set of histograms sorted by the file with highest diff.

    There is an histogram for each flavor of each repo in the input data.

    The histogram is stacked by the "how" index, showing whether the kind of diff.
    There are two outputs, one using absolute LoC changes, the other using the
    percentage of LoC changed with respect to the total file LoC.
    """
    public = True
    task_namespace = "cloc"
    task_name = "plot-cloc-by-file"
    task_config_class = LoCAnalysisConfig

    plot_flavors = ["all", "cdb"]

    @dependency
    def loc_data(self):
        return LoadLoCGenericData(self.session, self.analysis_config, task_config=self.config)

    def _prepare_bars(self, df, sort_by, values_col):
        """
        Helper to prepare the dataframe for plotting

        Need to prepare the frame by pivoting how into columns and retaining only
        the relevant code column. This is the format expected by df.plot()
        Filter top N files, stacked by the 'how' index
        """
        NTOP = 50 * len(df.index.unique("how"))
        show_df = df.sort_values(by=[sort_by, "file", "how"], ascending=False).iloc[:NTOP].reset_index()
        show_df["file"] = show_df["file"].str.removeprefix("sys/")
        return show_df.pivot(index=[sort_by, "file"], columns="how", values=values_col)

    def _do_plot(self, df: pd.DataFrame, output: PlotTarget, xlabel: str):
        """
        Helper to plot bars for LoC changes.
        This expects a dataframe produced by _prepare_bars().
        """
        # Use tab10 colors, added => green, modified => orange, removed => red
        cmap = sns.color_palette(as_cmap=True)
        # ordered as green, orange, red
        colors = [cmap[2], cmap[1], cmap[3]]

        with new_figure(output.path) as fig:
            ax = fig.subplots()
            df.reset_index().plot(x="file",
                                  y=["added", "modified", "removed"],
                                  stacked=True,
                                  kind="barh",
                                  ax=ax,
                                  color=colors)
            ax.tick_params(axis="y", labelsize="xx-small")
            ax.set_xlabel(xlabel)

    def _plot_repo(self, output_key: str, df: pd.DataFrame, baseline_df: pd.DataFrame):
        data_df = generalized_xs(df, how="same", complement=True)

        # Find out the total and percent_total LoC changed by file
        total_loc = data_df.groupby("file").sum()
        total_loc = total_loc.join(baseline_df, on="file", rsuffix="_baseline")
        total_loc["percent"] = 100 * total_loc["code"] / total_loc["code_baseline"]

        # Backfill the total for each value of the "how" index so we can do the sorting
        # Note: some files are added completely, we ignore them here because the
        # they are not particularly interesting for the relative diff by file.
        # These would always account for 100% of the changes, polluting the histogram
        # with not very useful information.
        aligned, _ = total_loc.align(data_df, level="file")
        data_df["total"] = aligned["code"]
        data_df["total_percent"] = aligned["percent"]
        data_df["baseline"] = aligned["code_baseline"]
        data_df.dropna(inplace=True, subset=["baseline"])

        # Compute relative added/modified/removed counts
        data_df["code_percent"] = 100 * data_df["code"] / data_df["baseline"]

        # Plot absolute LoC changed
        show_df = self._prepare_bars(data_df, "total", "code")
        self._do_plot(show_df, self.output_map[f"{output_key}-abs"], "# of lines")

        # Plot relative LoC, ranked by absolute value
        show_df = self._prepare_bars(data_df, "total", "code_percent")
        self._do_plot(show_df, self.output_map[f"{output_key}-rel-sorted-by-abs"], "% of file lines")

        # Plot relative LoC changed
        show_df = self._prepare_bars(data_df, "total_percent", "code_percent")
        self._do_plot(show_df, self.output_map[f"{output_key}-rel"], "% of file lines")

        # Plot absolute LoC, ranked by percent change
        show_df = self._prepare_bars(data_df, "total_percent", "code")
        self._do_plot(show_df, self.output_map[f"{output_key}-abs-sorted-by-rel"], "# of lines")

        ftype = data_df.index.get_level_values("file").map(to_file_type)
        # Plot absolute LoC in assembly files, ranked by absolute change
        show_df = self._prepare_bars(data_df.loc[ftype == "assembly"], "total", "code")
        self._do_plot(show_df, self.output_map[f"{output_key}-abs-asm"], "# of lines")

        # Plot absolute LoC in header files, ranked by absolute change
        show_df = self._prepare_bars(data_df.loc[ftype == "header"], "total", "code")
        self._do_plot(show_df, self.output_map[f"{output_key}-abs-hdr"], "# of lines")

        # Plot absolute LoC in source files, ranked by absolute change
        show_df = self._prepare_bars(data_df.loc[ftype == "source"], "total", "code")
        self._do_plot(show_df, self.output_map[f"{output_key}-abs-src"], "# of lines")

    def _plot_flavor(self, prefix: str, df: pd.DataFrame, baseline_df: pd.DataFrame):
        """
        Generate a set of plots for a named set of input data.

        The prefix is used to fetch the group of plot targets to use
        """
        for repo in df.index.unique("repo"):
            repo_df = df.xs(repo, level="repo")
            repo_baseline_df = baseline_df.xs(repo, level="repo")
            self._plot_repo(f"{prefix}-{repo}", repo_df, repo_baseline_df)

    def run_plot(self):
        # Ensure we have the proper theme
        sns.set_theme()

        self._plot_flavor("all", self.loc_data.diff_df.get(), self.loc_data.baseline_df.get())
        if self.loc_data.compilation_db is not None:
            self._plot_flavor("cdb", self.loc_data.cdb_diff_df.get(), self.loc_data.cdb_baseline_df.get())

    def outputs(self):
        """
        Directly generate outputs for the output map.

        This is easier than using @output annotations
        """
        yield from super().outputs()
        df = self.loc_data.diff_df.get()
        for flavor in self.plot_flavors:
            for repo in df.index.unique("repo"):
                key = f"{flavor}-{repo}-abs"
                yield key, PlotTarget(self, prefix=key)
                key = f"{flavor}-{repo}-rel"
                yield key, PlotTarget(self, prefix=key)

                key = f"{flavor}-{repo}-rel-sorted-by-abs"
                yield key, PlotTarget(self, prefix=key)
                key = f"{flavor}-{repo}-abs-sorted-by-rel"
                yield key, PlotTarget(self, prefix=key)

                key = f"{flavor}-{repo}-abs-asm"
                yield key, PlotTarget(self, prefix=key)
                key = f"{flavor}-{repo}-abs-hdr"
                yield key, PlotTarget(self, prefix=key)
                key = f"{flavor}-{repo}-abs-src"
                yield key, PlotTarget(self, prefix=key)
