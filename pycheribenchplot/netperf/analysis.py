from pycheribenchplot.core.analysis import (AnalysisTask, BenchmarkDataLoadTask, StatsByParamGroupTask,
                                            StatsForAllParamSetsTask)

from .model import NetperfInputModel, NetperfStatsModel
from .task import NetperfExecTask

# class NetperfSanityCheck(BenchmarkAnalysis):
#     """
#     Sanity check for netperf stats
#     """
#     name = "netperf-sanity-check"
#     require = {DatasetName.NETPERF_DATA}

#     async def process_datasets(self):
#         self.logger.info("Verify integrity of netperf datasets")
#         dset = self.get_dataset(DatasetName.NETPERF_DATA)
#         # Check that all benchmarks report the same number of iterations
#         if "Confidence Iterations Run" in dset.agg_df.columns:
#             if len(dset.agg_df["Confidence Iterations Run"].unique()) > 1:
#                 self.logger.error("Benchmark iteration count does not match across samples")
#         else:
#             self.logger.warning(
#                 "Can not verify netperf iteration count, consider enabling the CONFIDENCE_ITERATION output")
#         # Check that all benchmarks ran a consistent amount of sampling
#         # functions in libstatcounters
#         dset = self.get_dataset(DatasetName.QEMU_STATS_CALL_HIT)
#         if dset:
#             syms_index = dset.ctx_agg_df.index.get_level_values("symbol")
#             cpu_start = syms_index == "cpu_start"
#             cpu_stop = syms_index == "cpu_stop"
#             statcounters_sample = syms_index == "statcounters_sample"
#             check = dset.ctx_agg_df.loc[cpu_start, "call_count"].groupby("dataset_id").sum().unique()
#             if len(check) > 1:
#                 self.logger.error("netperf::cpu_start anomalous #calls %s", check)
#             check = dset.ctx_agg_df.loc[cpu_stop, "call_count"].groupby("dataset_id").sum().unique()
#             if len(check) > 1:
#                 self.logger.error("netperf::cpu_stop anomalous #calls %s", check)
#             check = dset.ctx_agg_df.loc[statcounters_sample, "call_count"].groupby("dataset_id").sum().unique()
#             if len(check) > 1:
#                 self.logger.error("libstatcounters::statcounters_sample anomalous #calls %s", check)


class NetperfStatsLoadTask(BenchmarkDataLoadTask):
    """
    Netperf output data load and pre-processing
    """
    task_namespace = "netperf"
    task_name = "load"
    exec_task = NetperfExecTask
    target_key = "stats"
    model = NetperfInputModel

    def _load_one_csv(self, path, **kwargs):
        kwargs["skiprows"] = 1
        return super()._load_one_csv(path, **kwargs)


class NetperfStatsByParamGroup(StatsByParamGroupTask):
    """
    Generate netperf statistics by parameterization group, along the machine configuration axis.
    """
    task_namespace = "netperf"
    task_name = "stats-by-param-set"
    load_task = NetperfStatsLoadTask
    model = NetperfStatsModel
    extra_group_keys = ["Request Size Bytes", "Response Size Bytes"]


class NetperfStatsMergedParams(StatsForAllParamSetsTask):
    """
    Generate netperf statistics showing the scaling of the netperf benchmark along the parameterization axis.
    """
    task_namespace = "netperf"
    task_name = "merged-stats"
    stats_task = NetperfStatsByParamGroup
    model = NetperfStatsModel


class NetperfStatsPipeline(AnalysisTask):
    public = True
    task_namespace = "netperf"
    task_name = "stats-pipeline"

    def dependencies(self):
        self.stats = NetperfStatsMergedParams(self.session, self.analysis_config)
        yield self.stats

    def run(self):
        print(self.stats.output_map["df"].df)
