import io
import json
import re
import typing
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DatasetArtefact, DatasetName
from .dataset import DataSetContainer, Field
from .json import JSONDataSetContainer


class PidMap(DataSetContainer):
    """
    Per-benchmark pid mapping container.
    This should be filled by one of the pidmap datasets.

    Note that this looks like a dataset but does not follow the
    normal registration, and is not named.
    In particular, currently the merge and analysis steps are unused.
    """
    fields = [
        Field("pid", dtype=int),
        Field("tid", dtype=int),
        Field.str_field("command"),
        Field.str_field("thread_name"),
        Field("thread_flags", dtype=int),
        Field("proc_flags", dtype=int),
    ]


class HWPMCPidMapDataset(JSONDataSetContainer):
    """
    JSON output generated by the --libxo option of pmcstat
    Currently we only support a subset of the output values

    The pidmap for each benchmark iteration should be produced by the
    benchmark handler itself. Benchmarks that can do that natively can
    use this dataset to grab the destination output path.
    Benchmarks that require help to do this should use pmcstat to run.
    """
    dataset_config_name = DatasetName.PIDMAP
    dataset_source_id = DatasetArtefact.PIDMAP
    fields = [
        Field("pid", dtype=int),
        Field("tid", dtype=int),
        Field.str_field("command"),
        Field.str_field("thread_name"),
        Field("thread_flags", dtype=int),
        Field("proc_flags", dtype=int),
    ]

    def iteration_output_file(self, iteration):
        return super().iteration_output_file(iteration).with_suffix(".json")

    def get_pmcstat_command(self, iteration: int, cmd: list[str]) -> list[str]:
        """
        Helper to produce a pmcstat command that only tracks processes and mappings
        """
        return ["pmcstat", "-O", self.iteration_output_file(iteration), "-S", "DUMMY.DUMMY"] + cmd

    async def after_extract_results(self, script, instance):
        # If we are supposed to get pids from other pmcstat sources skip
        # XXX need to integrate this better
        if self.benchmark.get_dataset(DatasetName.PMC_PROFCLOCK_STACKSAMPLE):
            return
        # Here we just check that we have the output files
        for i in range(self.benchmark.config.iterations):
            check_path = self.iteration_output_file(i)
            if not check_path.exists():
                self.logger.error("Missing %s output, maybe pmcstat did not run?", check_path)

    def load_iteration(self, iteration):
        super().load_iteration(iteration)
        profclock = self.benchmark.get_dataset(DatasetName.PMC_PROFCLOCK_STACKSAMPLE)
        if profclock:
            path = profclock.iteration_output_file(iteration)
        else:
            path = self.iteration_output_file(iteration)
        with open(path, "r") as fd:
            data = json.load(fd, strict=False)
        pmc_data = data["pmcstat"]["pmc-log-entry"]
        json_df = pd.DataFrame.from_records(pmc_data)
        json_df["iteration"] = iteration
        self.load_from_hwpmc(json_df)

    def load_from_hwpmc(self, hwpmc_df):
        """
        Load pid/tid entries from an hwpmc dump file
        """
        procs = hwpmc_df[hwpmc_df["type"].str.strip() == "create"].dropna(axis=1)
        threads = hwpmc_df[hwpmc_df["type"].str.strip() == "thr-create"].dropna(axis=1)
        entries = pd.merge(procs, threads, on="pid", suffixes=("", "_thr"))
        # Rename columns to match the pidmap dataset values
        pid_df = entries.rename(columns={
            "value": "command",
            "tdname": "thread_name",
            "flags_thr": "thread_flags",
            "flags": "proc_flags"
        })
        self._append_df(pid_df)

    def resolve_user_binaries(self, dataset_id) -> pd.DataFrame:
        df = self.df.xs(dataset_id)
        P_KPROC_MASK = 0x4
        user_commands = (df["flags"] & P_KPROC_MASK) == 0

        def resolve_path(cmd):
            target_path = Path(cmd)
            if target_path.is_absolute():
                return self.benchmark.rootfs / target_path.relative_to("/")
            else:
                # attempt to locate file in /bin /usr/bin /usr/sbin
                dollarpath = [Path("bin"), Path("usr/bin"), Path("usr/sbin")]
                for path in dollarpath:
                    candidate = self.benchmark.rootfs / path / target_path
                    if candidate.exists():
                        return candidate
            return None

        result_df = df.copy()
        result_df["command"] = df["command"].map(resolve_path)
        return result_df[~result_df["command"].isna()]
