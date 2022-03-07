import pandas as pd

from ..core.dataset import DatasetName
from ..core.plot import (AALineDataView, BarPlotDataView, BenchmarkPlot, BenchmarkSubPlot, CellData, HistPlotDataView,
                         LegendInfo, Scale)
from ..vmstat.plot import VMStatUMAMetricHist


class UMABucketAnomalyFail(VMStatUMAMetricHist):
    """
    Highlight the purecap kernel vm_pgcache UMA anomaly
    """
    def get_zone_names(self):
        """
        Zone names to include in the plot
        """
        all_zones = self.ds.agg_df.index.get_level_values("name").unique()
        zones = [z for z in all_zones if "vm pgcache" in z]
        zones += ["8 Bucket", "64 Bucket", "128 Bucket", "256 Bucket"]
        return zones

    def get_cell_title(self):
        return "UMA vm_pgcache zone anomaly"

    def get_filtered_df(self, df):
        """
        Only select the zones we care about
        """
        zones = self.get_zone_names()
        return df[df.index.get_level_values("name").isin(zones)].copy()


class UMABucketAllocAnomaly(BenchmarkPlot):
    """
    Collect data to explain the UMA bucket allocation anomaly triggered by
    vm_pgcache. This is currently enabled for all benchmarks as it seems to
    be a common issue.
    """
    @classmethod
    def check_required_datasets(cls, dsets: list):
        """Check dataset list against qemu stats dataset names"""
        required = set([DatasetName.VMSTAT_UMA, DatasetName.VMSTAT_UMA_INFO])
        return required.issubset(set(dsets))

    def _get_uma_stats(self):
        return [
            "rsize", "ipers", "bucket_size", "requests", "free", "fail", "fail_import", "bucket_alloc", "bucket_free"
        ]

    def _make_subplots(self):
        subplots = []
        want_stats = self._get_uma_stats()
        uma_stats = self.get_dataset(DatasetName.VMSTAT_UMA)
        uma_info = self.get_dataset(DatasetName.VMSTAT_UMA_INFO)
        for metric in uma_stats.data_columns():
            if metric in want_stats:
                subplots.append(UMABucketAnomalyFail(self, uma_stats, metric))
        for metric in uma_info.data_columns():
            if metric in want_stats:
                subplots.append(UMABucketAnomalyFail(self, uma_info, metric))
        return subplots

    def get_plot_name(self):
        return "UMA Bucket vm_pgcache anomaly"

    def get_plot_file(self):
        return self.benchmark.manager.plot_output_path / "uma-bucket-vm-pgcache-anomaly"


class UMABucketAffinityHist(BenchmarkSubPlot):
    """
    Show histogram of number of zones using each bucket zone. This helps understanding
    the pressure a zone receives from the rest of the system.
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.VMSTAT_UMA_INFO]
        return dsets

    def get_cell_title(self):
        return "Bucket affinity distribution"

    def generate(self, surface, cell):
        df = self.get_dataset(DatasetName.VMSTAT_UMA_INFO).agg_df
        buckets = [2**i for i in range(0, 9)]

        size_col = ("bucket_size", "-", "sample")
        view = HistPlotDataView(df, x=size_col, buckets=buckets, bucket_group="dataset_id")
        view.legend_level = ["dataset_id"]
        view.legend_info = self.build_legend_by_dataset()
        view.bar_align = "right"
        cell.x_config.label = "Bucket size"
        cell.x_config.ticks = buckets
        cell.x_config.tick_labels = buckets
        cell.x_config.scale = Scale("log", base=2)
        cell.yleft_config.label = "# Zones using the bucket zone"
        cell.add_view(view)


class UMABucketAnalysis(BenchmarkPlot):
    """
    Collect bucket-related data for the pgcache anomaly analysis
    """
    subplots = [UMABucketAffinityHist]

    def get_plot_name(self):
        return "UMA Bucket analysis"

    def get_plot_file(self):
        return self.benchmark.manager.plot_output_path / "uma-bucket-vm-pgcache-summary"
