import io
import json
import re
import shutil
import subprocess
import typing
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pandera as pa
import seaborn as sns

from pycheribenchplot.core.analysis import PlotTask
from pycheribenchplot.core.config import Config, ConfigPath
from pycheribenchplot.core.plot import new_figure
from pycheribenchplot.core.task import (AnalysisTask, DataFrameTarget, LocalFileTarget, SessionDataGenTask)
from pycheribenchplot.core.util import SubprocessHelper, new_logger
from pycheribenchplot.kernel_history.model import CheriBSDChangesModel


class CheriBSDKernelFiles(SessionDataGenTask):
    """
    This task extracts the set of files that are actually used during a build.
    These are considered interesting in terms of changes, especially regarding drivers.
    We only care about files in the cheribsd source tree during the compilation process.
    Currently build both the riscv and morello targets and make an union of the files
    we touch. Both the FPGA/Morello HW and QEMU targets are built. We only build the
    kernels, in a temporary directory to ensure reproducibility.
    """
    task_namespace = "kernel-history"
    task_name = "cheribuild-trace"

    def __init__(self, session, task_config=None):
        super().__init__(session, task_config=task_config)
        #: Path to cheribuild script
        self._cheribuild = self.session.user_config.cheribuild_path / "cheribuild.py"
        #: strace is required for this
        self._strace = Path(shutil.which("strace"))

    def _extract_files(self, path: Path) -> set[str]:
        open_re = re.compile(r"openat\(.*, ?\"([a-zA-Z0-9_/.-]+\.[hcmS])\"")
        file_set = set()
        with open(path, "r") as strace_fd:
            for line in strace_fd:
                m = open_re.search(line)
                if not m:
                    continue
                p = Path(m.group(1))
                try:
                    rp = p.relative_to(self.session.user_config.cheribsd_path)
                    file_set.add(str(rp))
                except ValueError:
                    # not relative to cheribsd_path
                    continue
        return file_set

    def run(self):
        """
        Try to build cheribsd and record the files that are built/used.
        We only care about files in the cheribsd source tree during the compilation process.
        Currently build both the riscv and morello targets and make an union of the files
        we touch. Both the FPGA/Morello HW and QEMU targets are built. We only build the
        kernels, in a temporary directory to ensure reproducibility.
        """
        file_set = set()
        with TemporaryDirectory() as build_dir:
            strace_file = Path(build_dir) / "strace-output.txt"
            strace_opts = ["-o", strace_file, "-qqq", "--signal=!SIGCHLD", "-f", "--trace=open,openat", "-z"]
            cbuild_opts = [
                "--build-root", build_dir, "--skip-update", "--skip-buildworld", "--skip-install",
                "cheribsd-riscv64-purecap", "--cheribsd-riscv64-purecap/build-fpga-kernels", "cheribsd-morello-purecap"
            ]
            cbuild_cmd = [self._cheribuild] + cbuild_opts
            build_cmd = SubprocessHelper(self._strace, strace_opts + cbuild_cmd)
            build_cmd.run()

            file_set = self._extract_files(strace_file)
            with open(self.output_map["strace"].path, "w+") as outfd:
                json.dump(list(file_set), outfd)

    def outputs(self):
        yield "strace", LocalFileTarget.from_task(self, ext="json")


@dataclass
class CheriBSDKernelLineChangesConfig(Config):
    #: The freebsd head baseline tag or commit SHA to use, mandatory
    freebsd_head: str


class CheriBSDKernelLineChanges(SessionDataGenTask):
    """
    This task extracts cheribsd kernel changes in terms of lines of codes.

    Use git diff with respect to a given freebsd-main-xxxx tag as the baseline
    to extract the diff.
    """
    task_namespace = "kernel-history"
    task_name = "cheribsd-diff"


class CheriBSDKernelFileChanges(SessionDataGenTask):
    """
    This task extracts cheribsd kernel changes tags using the tooling in cheribsd.
    For this to work, the cheribsd source tree must be synchronized to the kernel version we are running.
    XXX One day we will have a copy of some of the cherisdk assets in the assets path, maybe.

    Note that we do not need one of these for each benchmark in the matrix but just one per session.
    The way we handle this is by making the task_id alias everywhere, scheduling and borg will take care of the rest.
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-changes"

    def __init__(self, session, task_config=None):
        super().__init__(session, task_config=task_config)
        awkscript = Path("tools") / "tools" / "cheri-changes" / "extract-cheri-changes.awk"
        #: The awk script that generates the cheri-changes json output
        self._cheribsd_changes_awkscript = self.session.user_config.cheribsd_path / awkscript

    def _output_target(self) -> LocalFileTarget:
        return LocalFileTarget.from_task(self, ext="json")

    def _decode_changes(self, raw_data: str) -> list:
        decoder = json.JSONDecoder()
        chunks = []
        while raw_data:
            data, idx = decoder.raw_decode(raw_data)
            chunks.extend(data)
            raw_data = raw_data[idx:].strip()
        return chunks

    def dependencies(self):
        self._kernel_files = CheriBSDKernelFiles(self.session)
        yield self._kernel_files

    def run(self):
        self.logger.debug("Extracting kernel changes")
        if not self._cheribsd_changes_awkscript.exists():
            self.logger.error("Can not extract cheribsd kernel changes, missing awk script %s",
                              self._cheribsd_changes_awkscript)
            raise RuntimeError("Can not extract kernel changes")

        # Load list of kernel files we have touched
        with open(self._kernel_files.output_map["strace"].path, "r") as file_list_fd:
            kernel_files = json.load(file_list_fd)

        # For each of the files, we extract the changes annotations, if any
        # We chunk the list in groups of 20 files to lower the number of calls to awk
        chunks = np.array_split(np.array(kernel_files), len(kernel_files) / 20)
        json_chunks = []
        awk_cmd = ["awk", "-f", self._cheribsd_changes_awkscript]
        for chunk in chunks:
            cmd = awk_cmd + [self.session.user_config.cheribsd_path / p for p in chunk]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                self.logger.error("Failed to extract CheriBSD changes from %s: %s", chunk, result.stderr)
                raise RuntimeError("Failed to run awk")
            json_chunks.extend(self._decode_changes(result.stdout.decode("utf-8")))

        # Build a dataframe that we can verify
        df = pd.DataFrame.from_records(json_chunks)

        df["changes"] = df["changes"].map(lambda v: v if isinstance(v, list) else [])
        df["changes_purecap"] = df["changes_purecap"].map(lambda v: v if isinstance(v, list) else [])
        df.set_index(self.session.parameter_keys + ["file"], inplace=True)
        df["hybrid_specific"] = df["hybrid_specific"].fillna(False)
        df = CheriBSDChangesModel.to_schema(self.session).validate(df)

        output_path = self._output_target().path
        df.to_json(output_path, orient="table", index=True)

    def outputs(self):
        yield "changes", self._output_target()


# @dataclass
# class CheriBSDKernelChangesConfig(Config):
#     line_config: CheriBSDKernelLineChangesConfig
#     file_config: CheriBSDKernelFileChangesConfig

# class CheriBSDKernelChanges(SessionDataGenTask):
#     """
#     Top level task for use in configuration.

#     This triggers all kernel-history data sources.
#     """
#     public = True
#     task_namespace = "kernel-history"
#     task_name = "exec"
#     task_config_class = CheriBSDKernelChangesConfig

#     def dependencies(self):
#         yield CheriBSDKernelFiles(self.session)
#         yield CheriBSDKernelLineChanges(self.session, task_config=self.config.line_config)
#         yield CheriBSDKernelFileChanges(self.session, task_config=self.config.file_config)


class CheriBSDChangesLoad(AnalysisTask):
    """
    Load the kernel changes data
    """
    task_namespace = "kernel-history"
    task_name = "cheribsd-changes-load"

    def run(self):
        gen_task = CheriBSDKernelFileChanges(self.session)
        df = pd.read_json(gen_task.output_map["changes"].path, orient="table")
        self.output_map["df"].assign(df)

    def outputs(self):
        yield "df", DataFrameTarget(CheriBSDChangesModel.to_schema(self.session))


class CheriBSDFilesLoad(AnalysisTask):
    """
    Load the kernel changes data
    """
    task_namespace = "kernel-history"
    task_name = "cheribsd-files-load"

    def run(self):
        gen_task = CheriBSDKernelFiles(self.session)
        df = pd.read_json(gen_task.output_map["strace"].path)
        df.columns = ["files"]
        self.output_map["df"].assign(df)

    def outputs(self):
        schema = pa.DataFrameSchema({"files": pa.Column(str, nullable=False, unique=True)})
        yield "df", DataFrameTarget(schema)


class CheriBSDChangesByType(PlotTask):
    """
    Generate a bar plot with the number files containing changes of a given type.
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-changes-plot"

    def dependencies(self):
        self._kchanges = CheriBSDChangesLoad(self.session, self.analysis_config)
        yield self._kchanges
        self._kfiles = CheriBSDFilesLoad(self.session, self.analysis_config)
        yield self._kfiles

    def run(self):
        df = self._kchanges.output_map["df"].get()
        df["all_changes"] = (df["changes"] + df["changes_purecap"]).map(set)
        changed_files = len(df)
        files_df = self._kfiles.output_map["df"].get()
        total_files = len(files_df)

        data_df = df.explode("all_changes")
        data_df = data_df.groupby("all_changes").size().to_frame(name="Count").reset_index()
        data_df["% of changed files"] = data_df["Count"] * 100 / changed_files
        data_df = data_df.rename({"all_changes": "Type of change"}, axis=1)
        data_df["% of all files"] = data_df["Count"] * 100 / total_files

        with new_figure(self.output_map["changes"].path) as fig:
            ax = fig.subplots()
            sns.barplot(ax=ax, data=data_df, x="Type of change", y="Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

        with new_figure(self.output_map["rel-changes"].path) as fig:
            ax = fig.subplots()
            sns.barplot(ax=ax, data=data_df, x="Type of change", y="% of changed files")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

        with new_figure(self.output_map["rel-total-changes"].path) as fig:
            ax = fig.subplots()
            sns.barplot(ax=ax, data=data_df, x="Type of change", y="% of all files")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

    def outputs(self):
        yield "changes", self._plot_output("changes")
        yield "rel-changes", self._plot_output("rel-changes")
        yield "rel-total-changes", self._plot_output("rel-total-changes")
