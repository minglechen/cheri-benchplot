import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from git import Repo

from ..compile_db import CompilationDB, CompilationDBModel
from ..core.analysis import AnalysisTask
from ..core.artefact import DataFrameTarget, LocalFileTarget
from ..core.config import Config
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import SessionDataGenTask, dependency, output
from ..core.util import resolve_system_command
from .model import CheriBSDAnnotationsModel


class CheriBSDKernelAnnotations(SessionDataGenTask):
    """
    This task extracts cheribsd kernel changes tags using the tooling in cheribsd.
    For this to work, the cheribsd source tree must be synchronized to the kernel version we are running.
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-changes"

    def __init__(self, session, task_config=None):
        super().__init__(session, task_config=task_config)
        awkscript = Path("tools") / "tools" / "cheri-changes" / "extract-cheri-changes.awk"
        #: The awk script that generates the cheri-changes json output
        self._cheribsd_changes_awkscript = self.session.user_config.cheribsd_path / awkscript
        self._awk = resolve_system_command("awk")
        self._repo = Repo(self.session.user_config.cheribsd_path)

    def _decode_changes(self, raw_data: str) -> list:
        decoder = json.JSONDecoder()
        chunks = []
        while raw_data:
            data, idx = decoder.raw_decode(raw_data)
            chunks.extend(data)
            raw_data = raw_data[idx:].strip()
        return chunks

    def run(self):
        self.logger.debug("Extracting kernel changes")
        if not self._cheribsd_changes_awkscript.exists():
            self.logger.error("Can not extract cheribsd kernel changes, missing awk script %s",
                              self._cheribsd_changes_awkscript)
            raise RuntimeError("Can not extract kernel changes")

        # Scan all files for annotations
        json_chunks = []
        awk_cmd = [self._awk, "-f", self._cheribsd_changes_awkscript]
        for path in self._repo.git.grep("CHERI CHANGES", l=True).splitlines():
            cmd = awk_cmd + [self.session.user_config.cheribsd_path / path]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                self.logger.error("Failed to extract CheriBSD changes from %s: %s", path, result.stderr)
                raise RuntimeError("Failed to run awk")
            data = json.loads(result.stdout.decode("utf-8"))
            json_chunks.extend(data)

        # Build a dataframe that we can verify
        df = pd.DataFrame.from_records(json_chunks)
        df["file"] = df["file"].map(lambda path: Path(path).relative_to(self.session.user_config.cheribsd_path)).map(
            str)
        df["changes"] = df.get("changes", None).map(lambda v: v if isinstance(v, list) else [])
        df["changes_purecap"] = df.get("changes_purecap", None).map(lambda v: v if isinstance(v, list) else [])
        df["hybrid_specific"] = df.get("hybrid_specific", False).fillna(False)

        df.to_json(self.changes.path)

    @output
    def changes(self):
        return LocalFileTarget(self, ext="json", model=CheriBSDAnnotationsModel)


class CheriBSDAnnotationsUnion(AnalysisTask):
    task_namespace = "kernel-history"
    task_name = "cheribsd-annotations-union"

    @dependency
    def load_annotations(self):
        exec_task = self.session.find_exec_task(CheriBSDKernelAnnotations)
        return exec_task.changes.get_loader()

    @dependency
    def load_cdb(self):
        for b in self.session.all_benchmarks():
            exec_task = b.find_exec_task(CompilationDB)
            yield exec_task.compilation_db.get_loader()

    def run(self):
        cdb_set = []
        df = self.load_annotations.df.get()

        # Merge the CompilationDB data
        for loader in self.load_cdb:
            cdb_set.append(loader.df.get())
        # Make sure we only grab cheribsd kernel compilation DB data
        cdb_df = pd.concat(cdb_set).xs("cheribsd", level="target", drop_level=False)
        # Merge everything that was built under the same target, i.e. remove duplicates
        cdb_df = cdb_df.groupby(["target", "file"]).first().reset_index("file")

        # Filter annotations by compilation DB
        ann_df = cdb_df.merge(df, left_on="file", right_on="file", how="inner")
        ann_df.set_index("file", inplace=True)

        # Ensure that the changes columns are hashable
        ann_df["changes"] = ann_df["changes"].map(sorted).map(tuple)
        ann_df["changes_purecap"] = ann_df["changes_purecap"].map(sorted).map(tuple)
        # Verify that the groups are consistent, otherwise we have invalid input data
        check = (ann_df.groupby(level="file").nunique() <= 1).all().all()
        assert check, "Invalid file changes data, conflicting changes tags found"

        self.df.assign(ann_df)
        self.compilation_db.assign(cdb_df)

    @output
    def df(self):
        return DataFrameTarget(self, CheriBSDAnnotationsModel)

    @output
    def compilation_db(self):
        return DataFrameTarget(self, CompilationDBModel.as_groupby(["target", "file"]))


class CheriBSDChangesByType(PlotTask):
    """
    Generate a bar plot with the number files containing changes of a given type.
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-changes-plot"

    @dependency
    def merged_annotations(self):
        return CheriBSDAnnotationsUnion(self.session, self.analysis_config)

    def run_plot(self):
        df = self.merged_annotations.df.get()
        cdb_df = self.merged_annotations.compilation_db.get()

        df["all_changes"] = (df["changes"] + df["changes_purecap"]).map(set)
        changed_files = len(df)
        total_files = len(cdb_df)

        count_label = "Number of Files"

        data_df = df.explode("all_changes")
        data_df = data_df.groupby("all_changes").size().to_frame(name=count_label).reset_index()
        data_df["% of changed files"] = data_df[count_label] * 100 / changed_files
        data_df = data_df.rename({"all_changes": "Type of change"}, axis=1)
        data_df["% of all files"] = data_df[count_label] * 100 / total_files

        sns.set_theme()

        with new_figure(self.changes.paths()) as fig:
            ax = fig.subplots()
            sns.barplot(ax=ax, data=data_df, x="Type of change", y=count_label, color="steelblue")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

        with new_figure(self.rel_changes.paths()) as fig:
            ax = fig.subplots()
            sns.barplot(ax=ax, data=data_df, x="Type of change", y="% of changed files", color="steelblue")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

        with new_figure(self.rel_total_changes.paths()) as fig:
            ax = fig.subplots()
            sns.barplot(ax=ax, data=data_df, x="Type of change", y="% of all files", color="steelblue")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

    @output
    def changes(self):
        return PlotTarget(self, prefix="changes")

    @output
    def rel_changes(self):
        return PlotTarget(self, prefix="rel-changes")

    @output
    def rel_total_changes(self):
        return PlotTarget(self, prefix="rel-total-changes")


class CheckMissingAnnotation(AnalysisTask):
    """
    Simple lint task that verifies that the CheriBSD annotations agree
    with the CheriBSD git diff.
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-annotation-check"

    @dependency
    def load_annotations(self):
        exec_task = self.session.find_exec_task(CheriBSDKernelAnnotations)
        return exec_task.changes.get_loader()

    @dependency
    def load_cloc(self):
        for b in self.session.all_benchmarks():
            exec_task = b.find_exec_task(CompilationDB)
            yield exec_task.compilation_db.get_loader()

    def run(self):
        # Sanity check: warn about files in the compilation db that do not
        # annotations
        # XXX TODO
        pass

    @output
    def failures(self):
        return LocalFileTarget(self, ext="csv")
