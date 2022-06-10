import logging
import re
import typing
from functools import reduce
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from pypika import Order, Query, Table
from sortedcontainers import SortedList

from ..core.dataset import (DatasetArtefact, DatasetName, DatasetProcessingError, Field, align_multi_index_levels,
                            check_multi_index_aligned, rotate_multi_index_level, subset_xs)
from ..core.instance import PlatformOptions
from ..core.perfetto import PerfettoDataSetContainer


class QEMUTraceDataset(PerfettoDataSetContainer):
    """
    Base class for all datasets requiring qemu-perfetto trace output.
    This initializes the qemu instance configuration so that all the datasets will point to
    the same qemu output file. Subclasses should enable their own trace categories.
    """
    def output_file(self):
        return self.benchmark.get_output_path() / f"qemu-perfetto-{self.benchmark.uuid}.pb"

    def configure(self, opts: PlatformOptions) -> PlatformOptions:
        opts = super().configure(opts)
        opts.qemu_trace = True
        opts.qemu_trace_file = self.output_file()
        opts.qemu_trace_categories.add("ctrl")
        return opts


class ContextIntervalStatsBase(QEMUTraceDataset):
    """
    Base class for generating and loading interval-based statistics from the perfetto backend.
    Each interval track is associated to a CHERI context, we attempt to resolve pid and tid
    from the context to a process name and thread name.
    """
    fields = [
        Field.index_field("bucket", dtype=int),
        Field.index_field("process", dtype=str, isderived=True),
        Field.index_field("thread", dtype=str, isderived=True),
        Field.index_field("pid", dtype=int),
        Field.index_field("tid", dtype=int),
        Field.index_field("cid", dtype=int),
        Field.index_field("EL", dtype=int),
        # Field.index_field("AS", dtype=int),
    ]

    def delta_columns(self):
        """Return columns for which to compute derived delta columns"""
        raise NotImplementedError("Must override")

    def extract_intervals(self, tp: "TraceProcessor", ts_start: float, ts_end: float):
        """
        Extract intervals associated to a given track from the intervals table, within a
        timestamp interval.
        """
        t_interval = Table("interval")
        t_track = Table("cheri_context_interval_track")
        t_process = Table("process")
        t_thread = Table("thread")
        t_comp = Table("compartment")
        query = Query.from_(t_track).join(t_interval).on(t_interval.track_id == t_track.id)
        query = query.join(t_process).on_field("upid")
        query = query.join(t_thread).on_field("utid")
        query = query.join(t_comp).on_field("ucid")
        query = query.select(t_process.pid, t_thread.tid, t_comp.cid, t_comp.el, t_interval.ts, t_interval.start,
                             t_interval.end, t_interval.value)
        query = self._query_filter_ts(query, t_interval.ts, ts_start, ts_end)
        df = self._query_to_df(tp, query)
        # Merge overlapping intervals for each context
        grouped = df.groupby(["pid", "tid", "cid", "el"])
        df = grouped.apply(lambda g: self._merge_intervals(g))
        return df[df["value"] != 0].reset_index()

    def _merge_intervals(self, df):
        """
        Merge intervals belonging to the same context.
        Assume that the dataframe contains all the intervals from a context. We accumulate
        the value of the overlapping intervals, in case there are multiple snapshots in the
        trace. This can happen if the internal interval container in QEMU is flushed multiple
        times during the same iteration.
        """
        points = np.concatenate((df["start"], df["end"]))
        limits = sorted(np.unique(points))
        intervals = []
        for i in range(len(limits) - 1):
            start = limits[i]
            end = limits[i + 1]
            # select intervals that fully contain interval #i
            sel = (df["start"] <= start) & (df["end"] >= end)
            value = df[sel]["value"].sum()
            intervals.append((start, end, value))
        return pd.DataFrame.from_records(intervals, columns=["start", "end", "value"])

    def _resolve_pid_tid(self):
        """
        Resolve process and thread names for the current loaded data frame.
        """
        pidmap = self.benchmark.get_dataset_by_artefact(DatasetArtefact.PIDMAP)
        resolver = self.benchmark.sym_resolver
        assert pidmap is not None, "The pidmap dataset is required for qemu stats"

        # Find missing TIDs in pidmap and attempt to match them with the QEMU data
        detected = self.df.groupby(["pid", "tid"]).size().reset_index()[["pid", "tid"]]
        # Suffix is added to avoid clash during join
        pid_df = pidmap.fixup_missing_tid(detected)
        pid_df = pid_df.set_index(["pid", "tid"]).add_suffix("_pidmap")
        join_df = self.df.join(pid_df, how="left", on=["pid", "tid"])
        # There may be some NaN due to PID that were running during the benchmark but have since been terminated
        # We mark these as undetected
        na_cmd = join_df["command_pidmap"].isna()
        na_thr = join_df["thread_name_pidmap"].isna()

        join_df.loc[na_cmd, "command_pidmap"] = join_df.index.get_level_values("pid")[na_cmd].map(
            lambda pid: f"undetected:{pid}")
        join_df.loc[na_thr, "thread_name_pidmap"] = join_df.index.get_level_values("tid")[na_thr].map(
            lambda tid: f"unknown:{tid}")
        self.df["process"] = join_df["command_pidmap"]
        self.df["thread"] = join_df["thread_name_pidmap"]

    def configure(self, opts: PlatformOptions) -> PlatformOptions:
        opts = super().configure(opts)
        opts.qemu_trace_categories.add("marker")
        return opts

    def _extract_iteration(self, tp: "TraceProcessor", i: int, start: int, end: int):
        """
        Extract events from an iteration of the benchmark.
        """
        raise NotImplementedError("Must override")

    def load(self):
        tp = self._get_trace_processor(self.output_file())
        iterations = self.extract_iteration_markers(tp)
        # Verify that the iteration markers agree with the configured number of iterations
        if len(iterations) != self.benchmark.config.iterations:
            self.logger.error("QEMU trace does not have the expected iteration markers: %d configured %d",
                              len(iterations), self.benchmark.config.iterations)
            raise DatasetProcessingError("QEMU trace has invalid iteration markers")
        for i, interval in enumerate(iterations):
            start, end = interval
            self._extract_iteration(tp, i, start, end)

    def pre_merge(self):
        """
        Common pre-merge resolves the file/symbol and valid_symbol.
        This uses a combination of the PID map and the benchmark symbolizer.
        """
        super().pre_merge()
        self._resolve_pid_tid()

    def _pre_aggregate(self):
        self.agg_df = self.merged_df

    def aggregate(self):
        """
        Aggregation occurs in two steps:
        The first step gives the chanche to subclasses to aggregate per-context data for each iteration.
        This happens in the _pre_aggregate() hook.
        The second step aggregates data across iterations.
        """
        super().aggregate()
        self._pre_aggregate()
        agg_levels = self.agg_df.index.names.difference(["iteration"])
        grouped = self.agg_df.groupby(agg_levels)
        self.agg_df = self._compute_aggregations(grouped)

    def post_aggregate(self):
        """
        Make sure the frame is aligned on the correct index levels, so that we can produce a meaningful delta
        even with different sets of function calls/line numbers etc.
        """
        super().post_aggregate()
        align_levels = self._delta_align_levels()
        tmp_df = align_multi_index_levels(self.agg_df, align_levels, fill_value=0)
        tmp_df = self._add_delta_columns(tmp_df)
        self.agg_df = self._compute_delta_by_dataset(tmp_df)


class QEMUStatsBBHitDataset(ContextIntervalStatsBase):
    dataset_source_id = DatasetArtefact.QEMU_STATS_BB_HIT
    dataset_config_name = DatasetName.QEMU_STATS_BB_HIT
    fields = [Field("start", dtype=np.uint), Field("end", dtype=np.uint), Field.data_field("hit_count", dtype=int)]

    def configure(self, opts: PlatformOptions) -> PlatformOptions:
        opts = super().configure(opts)
        opts.qemu_trace_categories.add("bb_hit")
        return opts

    def _extract_iteration(self, tp, i, start, end):
        intervals_df = self.extract_intervals(tp, start, end)
        intervals_df.rename(columns={"value": "hit_count", "el": "EL"}, inplace=True)
        intervals_df["iteration"] = i
        # Append and add ID columns
        self._append_df(intervals_df)

    def pre_merge(self):
        super().pre_merge()
        # Temporary as_key column
        self.df["as_key"] = self.df["process"].map(lambda p: Path(p).name)
        resolved = self.benchmark.dwarf_helper.addr2line_range_to_df(self.df, "start", "end", "as_key")
        # Melt the line info column into file/line/symbol columns and use them as index levels
        print(resolved)
        self.df = self.df.drop(columns=["as_key"])


class QEMUStatsBBICountDataset(ContextIntervalStatsBase):
    dataset_source_id = DatasetArtefact.QEMU_STATS_BB_ICOUNT
    dataset_config_name = DatasetName.QEMU_STATS_BB_ICOUNT
    fields = [
        Field.index_field("file", dtype=str, isderived=True),
        Field.index_field("symbol", dtype=str, isderived=True),
        Field("start", dtype=np.uint),
        Field("end", dtype=np.uint),
        Field.data_field("instr_count", dtype=int),
        Field.derived_field("valid_symbol", dtype=object, isdata=False)
    ]

    def configure(self, opts: PlatformOptions) -> PlatformOptions:
        opts = super().configure(opts)
        opts.qemu_trace_categories.add("bb_icount")
        return opts

    def _extract_iteration(self, tp, i, start, end):
        intervals_df = self.extract_intervals(tp, start, end)
        intervals_df.rename(columns={"value": "instr_count", "el": "EL"}, inplace=True)
        intervals_df["iteration"] = i
        # Append and add ID columns
        self._append_df(intervals_df)

    def pre_merge(self):
        """
        Common pre-merge resolves the file/symbol/line and valid symbol column based on
        the PID and DWARF information
        """
        super().pre_merge()
        # Temporary as_key column
        self.df["as_key"] = self.df["process"].map(lambda p: Path(p).name)
        resolved = self.benchmark.sym_resolver.lookup_fn_to_df(self.df, "start", "as_key")
        self.df = self.df.drop(columns=["as_key"])
        self.df = pd.concat([self.df, resolved], axis=1)

    def _pre_aggregate(self):
        """
        For each iteration, collapse the icount for each file/function in the index.
        Then aggregate the iterations normally.
        """
        icount_agg_levels = self.dataset_id_columns() + ["process", "thread", "EL", "file", "symbol", "iteration"]
        grouped = self.merged_df.groupby(icount_agg_levels)
        self.agg_df = grouped.aggregate({"start": "min", "end": "max", "instr_count": "sum"})

    def _delta_align_levels(self):
        """
        Return the levels that need to be aligned across all datasets before computing deltas.
        """
        return ["process", "thread", "EL", "file", "symbol"]

    def get_icount_per_function(self):
        """
        Return a dataframe similar to the aggregated frame but where the instruction count is accumulated for
        all matching symbols, regardless of the context.
        """
        # Aggregate ignoring the context instead of preserving it
        grouped = self.merged_df.groupby(self.dataset_id_columns() + ["symbol", "iteration"])
        df = grouped.aggregate({"start": "min", "end": "max", "instr_count": "sum"})
        grouped = df.groupby(self.dataset_id_columns() + ["symbol"])
        df = self._compute_aggregations(grouped)
        df = align_multi_index_levels(df, ["dataset_id", "dataset_gid"])
        df = self._add_delta_columns(df)
        df = self._compute_delta_by_dataset(df)
        return df


class QEMUStatsCallHitDataset(ContextIntervalStatsBase):
    dataset_source_id = DatasetArtefact.QEMU_STATS_CALL_HIT
    dataset_config_name = DatasetName.QEMU_STATS_CALL_HIT
    fields = [
        Field.index_field("source_file", dtype=str, isderived=True),
        Field.index_field("source_symbol", dtype=str, isderived=True),
        Field("source", dtype=np.uint),
        Field("target", dtype=np.uint),
        Field("branch_count", dtype=int),
        Field.derived_field("call_count", dtype=int)
    ]

    def configure(self, opts: PlatformOptions) -> PlatformOptions:
        opts = super().configure(opts)
        opts.qemu_trace_categories.add("br_hit")
        return opts

    def _extract_iteration(self, tp, i, start, end):
        intervals_df = self.extract_intervals(tp, start, end)
        intervals_df.rename(columns={"value": "branch_count", "el": "EL"}, inplace=True)
        intervals_df["iteration"] = i
        # Append and add ID columns
        self._append_df(intervals_df)

    def pre_merge(self):
        super().pre_merge()
        self.df["as_key"] = self.df["process"].map(lambda p: Path(p).name)
        src_resolved = self.benchmark.sym_resolver.lookup_fn_to_df(tmp_df, "source", "as_key")
        tgt_resolved = self.benchmark.sym_resolver.lookup_fn_to_df(tmp_df, "target", "as_key")
        self.df = self.df.drop(columns=["as_key"])
        self.df = pd.concat([self.df, tgt_resolved, src_resolved.add_prefix("source_")], axis=1)


#         # Resolve the secondary file/symbol index for the source address
#         resolved = self._resolve_sym_column("source", "process_name")
#         resolved = resolved.add_prefix("source_")
#         self.df = pd.concat((self.df, resolved), axis=1)
#         # Generate now a new column only for entries that exactly match symbols, meaning that
#         # these are function calls is the first basic-block of the function and is considered as an individual call
#         # to that function
#         resolver = self.benchmark.sym_resolver
#         match = self.df.apply(lambda row: resolver.match_fn(row["target"], row["process_name"]), axis=1)
#         is_call = ~match.isna()
#         self.df["call_count"] = self.df["branch_count"].where(is_call, 0).astype(int)


class QEMUGuestCountersDataset(QEMUTraceDataset):
    """
    Dataset collecting global guest-driven qemu counters with the perfetto trace backend.
    This detects and separates counters samples in tracks for each dataset.
    """
    dataset_source_id = DatasetArtefact.QEMU_COUNTERS
    fields = [
        Field.index_field("ts", dtype=float),
        Field.index_field("name", dtype=str),
        Field.index_field("slot", dtype=int),
        Field.data_field("value", dtype=float),
    ]

    def _extract_counters(self, tp, iteration, ts_start, ts_end):
        """
        Extract counters during a single iteration
        """
        cnt_query = Query.from_(self.t_counter_track).join(
            self.t_counter).on(self.t_counter_track.id == self.t_counter.track_id)
        cnt_query = cnt_query.select(self.t_counter_track.name, self.t_counter.ts, self.t_counter.value)
        cnt_query = self._query_filter_ts(cnt_query, self.t_counter.ts, ts_start, ts_end)
        cnt_df = self._query_to_df(tp, cnt_query)

        name_split = cnt_df["name"].str.split(":")
        name = name_split.map(lambda v: v[0])
        slot = name_split.map(lambda v: v[1])
        cnt_df["name"] = name
        cnt_df["slot"] = slot
        cnt_df["dataset_id"] = self.benchmark.uuid
        cnt_df["iteration"] = iteration
        # Make timestamp relative to the beginning of the iteration
        cnt_df["ts"] = cnt_df["ts"] - ts_start
        self._append_df(cnt_df)

    def load(self):
        tp = self._get_trace_processor(self.output_file())
        iterations = self.extract_iteration_markers(tp)
        if len(iterations) != self.benchmark.config.iterations:
            self.logger.error("QEMU trace does not have the expected iteration markers: %d configured %d",
                              len(iterations), self.benchmark.config.iterations)
            raise DatasetProcessingError("QEMU trace has invalid iteration markers")
        for i, (start, end) in enumerate(iterations):
            self._extract_counters(tp, i, start, end)

    def configure(self, opts):
        opts = super().configure(opts)
        opts.qemu_trace_categories.add("counter")
        opts.qemu_trace_categories.add("marker")
        return opts
