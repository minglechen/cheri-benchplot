from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pycheribenchplot.core.plot import PlotTarget, PlotTask, new_figure
from pycheribenchplot.netperf.analysis import NetperfStatsMergedParams


class NetperfStatsPlot(PlotTask):
    """
    Produce a box plot of the absolute netperf metrics.
    Note that the depencencies are the same as the task computing the merged statistics frame
    across all parameterizations, but we need to access the raw merged frames here as we use
    seaborn to show individual samples distribution.
    """
    public = True
    task_namespace = "netperf"
    task_name = "plot-stats"

    def _draw_box_metric(self, df, metric: str):
        """
        Produce the box plot for a given metric, we keep grouping by g_uuid.
        """
        with new_figure(self._plot_output(metric).path) as fig:
            ax = fig.subplots()
            # There may be multiple parameterization axes, join as strings for now
            # probably not the best.
            if self.session.parameter_keys:
                param_desc = df.index.to_frame()[self.session.parameter_keys].agg("-".join, axis=1)
            else:
                # Should be only one benchmark, get its name
                name = self.session.config.configurations[0].name
                param_desc = pd.Series(name, index=df.index)
            hue = df.index.get_level_values("dataset_gid").map(lambda gid: self.session.machine_configuration_name(gid))
            sns.boxplot(ax=ax, y=metric, x=param_desc, hue=hue, data=df, palette="pastel")

    def dependencies(self):
        self.stats = NetperfStatsMergedParams(self.session, self.analysis_config)
        yield self.stats

    def run(self):
        df = self.stats.output_map["merged_df"].df
        # Determine how many metrics and parameters we have
        metrics = df.columns.unique("metric")
        for m in metrics:
            self._draw_box_metric(df, m)

    def outputs(self):
        df = self.stats.output_map["merged_df"].df
        for m in df.columns.unique("metric"):
            yield f"plot-{m}", self._plot_output(metric)


class NetperfStatsDeltaPlot(PlotTask):
    public = True
    task_namespace = "netperf"
    task_name = "plot-delta-stats"

    def _draw_bar_metric(self, df, metric: str):
        """
        Produce the bar plot for a given metric, we keep grouping by g_uuid.
        """
        with new_figure(self._plot_output(metric).path) as fig:
            ax = fig.subplots()
            chunk = df.xs(metric, level="metric", axis=1)
            if self.session.parameter_keys:
                param_desc = chunk.index.to_frame()[self.session.parameter_keys].agg("-".join, axis=1)
            else:
                # Should be only one benchmark, get its name
                name = self.session.config.configurations[0].name
                param_desc = pd.Series(name, index=chunk.index)
            hue = chunk.index.get_level_values("dataset_gid")
            err_hi = chunk[("q75", "delta")] - chunk[("median", "delta")]
            err_lo = chunk[("median", "delta")] - chunk[("q25", "delta")]
            sns.barplot(ax=ax,
                        x=param_desc,
                        y=("median", "delta"),
                        hue=hue,
                        data=chunk,
                        palette="pastel",
                        errorbar=lambda v: [err_hi, err_lo])

    def dependencies(self):
        self.stats = NetperfStatsMergedParams(self.session, self.analysis_config)
        yield self.stats

    def run(self):
        df = self.stats.output_map["df"].df
        # Determine how many metrics and parameters we have
        metrics = df.columns.unique("metric")
        for m in metrics:
            self._draw_bar_metric(df, m)

    def outputs(self):
        df = self.stats.output_map["df"].df
        for m in df.columns.unique("metric"):
            yield f"plot-{m}", self._plot_output(metric)
