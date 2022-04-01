import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.dataset import (DatasetArtefact, DataSetContainer, DatasetName, DatasetRunOrder, Field,
                            align_multi_index_levels)
from ..core.json import JSONDataSetContainer


class UMAZoneInfoDataset(DataSetContainer):
    """
    Extract UMA zone information from the uma SYSCTL nodes
    XXX processing is essentially the same as the other VMStat datasets so may share some code
    """
    dataset_config_name = DatasetName.VMSTAT_UMA_INFO
    dataset_source_id = DatasetArtefact.UMA_ZONE_INFO
    fields = [
        Field.index_field("name", dtype=str),
        Field.data_field("efficiency", dtype=float),
        Field.data_field("ipers", dtype=int),
        Field.data_field("ppera", dtype=int),
        Field.data_field("rsize", dtype=int),
        Field.data_field("bucket_size", dtype=int),
        Field.data_field("bucket_size_max", dtype=int),
        Field.data_field("bucket_refill_efficiency", dtype=float, isderived=True),
    ]

    def __init__(self, benchmark, dset_key, config):
        super().__init__(benchmark, dset_key, config)

    def _output_file_vmstat(self):
        base = super().output_file()
        return base.with_name(base.name + "-vmstat").with_suffix(".json")

    def _output_file_sysctl(self):
        base = super().output_file()
        return base.with_name(base.name + "-sysctl").with_suffix(".csv")

    def _load_vmstat_file(self) -> pd.DataFrame:
        with open(self._output_file_vmstat(), "r") as vmstat_out:
            vmstat_data = json.load(vmstat_out)
        # Create mapping between a convenient ID and the vmstat zone name
        vmstat_df = pd.DataFrame.from_records(vmstat_data["memory-zone-statistics"]["zone"])
        vmstat_df["zone_tmp_id"] = vmstat_df["name"].str.replace("[ .-]", "_", regex=True)
        vmstat_df = vmstat_df.set_index("zone_tmp_id")["name"]
        return vmstat_df

    def _load_sysctl_file(self) -> pd.DataFrame:
        with open(self._output_file_sysctl(), "r") as sysctl_out:
            sysctl_df = pd.read_csv(sysctl_out, sep=":", names=["node", "value"])
        # For each UMA zone extract keg info from sysctl
        base_info = sysctl_df["node"].str.contains(".bucket_size")
        keg_info = sysctl_df["node"].str.contains(".keg")
        sysctl_df = sysctl_df[base_info | keg_info]
        # this should contain ["vm", "uma", "<zone>", ...]
        sysctl_path = sysctl_df["node"].str.split(".")
        sysctl_df["zone_tmp_id"] = sysctl_path.map(lambda l: l[2])
        sysctl_df["metric"] = sysctl_path.map(lambda l: l[-1])
        return pd.pivot(sysctl_df, index="zone_tmp_id", values="value", columns="metric")

    def load(self):
        vmstat_df = self._load_vmstat_file()
        sysctl_df = self._load_sysctl_file()
        merged_df = sysctl_df.join(vmstat_df, how="left", on="zone_tmp_id", rsuffix="_vmstat")
        merged_df["name"] = merged_df["name_vmstat"]
        df = merged_df[self.input_all_columns()].fillna(0).reset_index(drop=True)
        df["dataset_id"] = self.benchmark.uuid
        self._append_df(df)

    def gen_pre_benchmark(self):
        self._script.gen_cmd("vmstat", ["--libxo", "json", "-z"], outfile=self._output_file_vmstat())
        self._script.gen_cmd("sysctl", ["vm.uma"], outfile=self._output_file_sysctl())

    def pre_merge(self):
        super().pre_merge()
        self.df["bucket_refill_efficiency"] = self.df["ipers"] / self.df["bucket_size"]

    def aggregate(self):
        super().aggregate()
        # We just sum if there are repeated index entries, we have no per-iteration data here
        self.agg_df = self.merged_df.groupby(self.merged_df.index.names).sum()
        # We expect no iteration information to be present in the data
        assert (self.agg_df.index.get_level_values("__iteration") == -1).all()
        # Add the aggregate index level
        self.agg_df = self._add_aggregate_columns(self.agg_df)

    def post_aggregate(self):
        super().post_aggregate()
        new_df = align_multi_index_levels(self.agg_df, ["name"], fill_value=0)
        agg_df = self._add_delta_columns(new_df)
        self.agg_df = self._compute_delta_by_dataset(agg_df)


class VMStatDataset(JSONDataSetContainer):
    """
    JSON output generated by the --libxo option of the vmstat command
    Note that we expect two files to exist with the suffix ".pre" and ".post"
    containing vmstat snapshots before and after the benchmark
    """
    def _get_vmstat_records(self, data):
        raise NotImplementedError("Must be defined by subclasses")

    def _vmstat_delta(self, pre_df, post_df):
        raise NotImplementedError("Must be defined by subclasses")

    def load_iteration(self, iteration):
        path = self.iteration_output_file(iteration)
        pre = open(path.with_suffix(".pre"), "r")
        post = open(path.with_suffix(".post"), "r")
        try:
            pre_data = json.load(pre)
            post_data = json.load(post)
            pre_df = pd.DataFrame.from_records(self._get_vmstat_records(pre_data))
            post_df = pd.DataFrame.from_records(self._get_vmstat_records(post_data))
            df = self._vmstat_delta(pre_df, post_df)
            df["dataset_id"] = self.benchmark.uuid
            df["__iteration"] = iteration
            self._append_df(df)
        finally:
            pre.close()
            post.close()

    def aggregate(self):
        super().aggregate()
        # We sum if there are repeated index entries for each iteration
        tmp = self.merged_df.groupby(self.merged_df.index.names).sum()
        # Then aggregate across iterations
        iter_index = list(self.merged_df.index.names)
        iter_index.remove("__iteration")
        grouped = tmp.groupby(iter_index)
        self.agg_df = self._compute_aggregations(grouped)

    def post_aggregate(self):
        super().post_aggregate()
        new_df = align_multi_index_levels(self.agg_df, self._get_align_levels(), fill_value=0)
        agg_df = self._add_delta_columns(new_df)
        self.agg_df = self._compute_delta_by_dataset(agg_df)

    def iteration_output_file(self, iteration):
        """The output file is shared by all vmstat datasets."""
        return self.benchmark.get_iter_output_path(iteration) / f"vmstat-{self.benchmark.uuid}.json"

    def gen_pre_benchmark_iter(self, iteration):
        """Run a vmstat snapshot before the benchmark runs."""
        super().gen_pre_benchmark_iter(iteration)
        pre_output = self.iteration_output_file(iteration).with_suffix(".pre")
        self._script.gen_cmd("vmstat", ["--libxo", "json", "-H", "-i", "-m", "-o", "-P", "-z"], outfile=pre_output)

    def gen_post_benchmark_iter(self, iteration):
        """Run a vmstat snapshot after the benchmark runs."""
        super().gen_post_benchmark_iter(iteration)
        post_output = self.iteration_output_file(iteration).with_suffix(".post")
        self._script.gen_cmd("vmstat", ["--libxo", "json", "-H", "-i", "-m", "-o", "-P", "-z"], outfile=post_output)


class VMStatKMalloc(VMStatDataset):
    dataset_config_name = DatasetName.VMSTAT_MALLOC
    dataset_source_id = DatasetArtefact.VMSTAT
    dataset_run_order = DatasetRunOrder.LAST
    fields = [
        Field.index_field("type", dtype=str),
        Field.data_field("in-use", dtype=int),
        Field.data_field("memory-use", dtype=int),
        Field.data_field("reservation-use", dtype=int),
        Field.data_field("requests", dtype=int),
        Field.data_field("large-malloc", dtype=int),
        Field("size", dtype=object),
    ]

    def _get_align_levels(self):
        return ["type"]

    def _get_vmstat_records(self, data):
        return data["malloc-statistics"]["memory"]

    def _vmstat_delta(self, pre_df, post_df):
        pre_df = pre_df.set_index("type")
        post_df = post_df.set_index("type")
        pre_df["size"] = pre_df["size"].map(lambda v: set(v))
        post_df["size"] = post_df["size"].map(lambda v: set(v))
        return post_df.subtract(pre_df).reset_index()


class VMStatUMA(VMStatDataset):
    """
    XXX TODO should also extract the keg.npages keg.free_items and zone current count.
    """
    dataset_config_name = DatasetName.VMSTAT_UMA
    dataset_source_id = DatasetArtefact.VMSTAT
    dataset_run_order = DatasetRunOrder.LAST
    fields = [
        Field.index_field("name", dtype=str),
        Field.data_field("size", dtype=int),
        Field("limit", dtype=int),
        Field("used", dtype=int),
        Field.data_field("free", dtype=int),
        Field.data_field("requests", dtype=int),
        Field.data_field("fail", dtype=int),
        Field("sleep", dtype=int),
        Field("xdomain", dtype=int),
        Field.data_field("bucket_alloc", dtype=int),
        Field.data_field("bucket_free", dtype=int),
        Field.data_field("fail_import", dtype=int),
        Field.data_field("pressure", dtype=int),
    ]

    def _get_align_levels(self):
        return ["name"]

    def _get_vmstat_records(self, data):
        return data["memory-zone-statistics"]["zone"]

    def _vmstat_delta(self, pre_df, post_df):
        pre_df = pre_df.set_index(["name"])
        post_df = post_df.set_index(["name"])
        # The size field is not supposed to change from pre to post
        # so we use the initial size for each bucket
        delta = post_df.subtract(pre_df)
        delta["size"] = pre_df["size"]
        return delta.reset_index()
