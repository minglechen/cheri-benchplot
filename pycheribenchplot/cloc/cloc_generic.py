import json
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pandera.typing as pa
import seaborn as sns
from git import Repo
from git.exc import InvalidGitRepositoryError

from ..core.analysis import AnalysisTask
from ..core.artefact import DataFrameTarget, LocalFileTarget
from ..core.config import Config, ConfigPath
from ..core.model import check_data_model
from ..core.pandas_util import generalized_xs, map_index
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import SessionDataGenTask, dependency, output
from ..core.util import SubprocessHelper, resolve_system_command
from .model import LoCCountModel, LoCDiffModel


@dataclass
class ExtractRepoLoCConfig(Config):
    #: Human-readable name of the cloc data, defaults to the repo path
    name: str | None = None
    #: Target repository, relative to UserConfig CHERI repository path
    repo_path: ConfigPath | None = None
    #: Repository baseline ref
    baseline_ref: ConfigPath | str = None
    #: Repository head ref
    head_ref: ConfigPath | str = "HEAD"
    #: Extensions to accept e.g. ".c"
    accept_ext: str | None = None
    #: Filter directories matching any of the regexes
    accept_filter: list[str] = field(default_factory=list)
    #: Filter out directories matching any of the regexes
    reject_filter: list[str] = field(default_factory=list)

    # XXX implement validators here

    def __post_init__(self):
        if self.name is None:
            self.name = self.repo_path


@dataclass
class ExtractLoCConfig(Config):
    #: Configuration for each repository to extract
    repos: list[ExtractRepoLoCConfig] = field(default_factory=list)


@dataclass
class LoCAnalysisConfig(Config):
    pass


class ExtractLoCBase(SessionDataGenTask):
    """
    Base task for extracting LoC information from repositories
    """
    task_namespace = "cloc"
    task_config_class = ExtractRepoLoCConfig

    def __init__(self, session, task_config=None):
        super().__init__(session, task_config=task_config)

        self._cloc = resolve_system_command("cloc", self.logger)

    def _get_common_cloc_args(self):
        # concurrency = self.session.config.concurrent_workers
        args = [
            "--skip-uniqueness",
            # f"--processes={concurrency}",
            "--exclude-content=DO NOT EDIT",
            "--file-encoding=UTF-8",
            "--fullpath",
            "--by-file",
            "--json",
            "--git",
            "--count-and-diff"
        ]
        return args

    def _apply_filters(self, df: pd.DataFrame, config: ExtractLoCConfig) -> pd.DataFrame:
        paths = df.index.get_level_values("file")
        match = paths.str.contains(config.accept_ext)
        accept = [paths.str.contains(expr) for expr in config.accept_filter]
        if accept:
            match = match & reduce(lambda a, m: a | m, accept[1:], accept[0])
        for expr in config.reject_filter:
            match = match & ~paths.str.contains(expr)
        return df.loc[match]

    @check_data_model
    def _load_diff_data(self, config: ExtractRepoLoCConfig, diff_file: Path) -> pa.DataFrame[LoCDiffModel]:
        """
        Load cloc diff output into a normalized LoCDiffModel dataframe

        :param diff_file: Output file from cloc containg the diff information
        :returns: A pandas dataframe
        """
        if not diff_file.exists():
            self.logger.error("cloc diff file %s does not exist", diff_file)
            raise FileNotFoundError(f"Missing file {diff_file}")

        with open(diff_file, "r") as raw_cloc:
            raw_data = json.load(raw_cloc)

        df_set = []
        for key in ["added", "same", "modified", "removed"]:
            chunk = raw_data[key]
            df = pd.DataFrame(chunk).transpose()
            df["how"] = key
            df["repo"] = config.name
            # Drop the nFiles column as it is not meaningful
            df = df.drop("nFiles", axis=1)
            df.index.name = "file"
            df = df.set_index(["how", "repo"], append=True).reorder_levels(["repo", "file", "how"])
            df_set.append(df)
        df = pd.concat(df_set)
        return df

    @check_data_model
    def _load_count_data(self, config: ExtractRepoLoCConfig, count_file: Path) -> pa.DataFrame[LoCCountModel]:
        """
        Load cloc count output into a LoCCountModel dataframe.

        This is usually the baseline line count.
        :param count_file: Output file from cloc containing count by file
        :returns: A pandas dataframe
        """
        if not count_file.exists():
            self.logger.error("cloc count file %s does not exist", count_file)
        with open(count_file, "r") as raw_cloc:
            raw_data = json.load(raw_cloc)

        raw_data.pop("header")
        raw_data.pop("SUM")
        df = pd.DataFrame(raw_data).transpose()
        df.index.name = "file"
        df["repo"] = config.name
        return df.set_index("repo", append=True).reorder_levels(["repo", "file"])

    @check_data_model
    def _extract_loc(self,
                     config: ExtractRepoLoCConfig,
                     extra_cloc_args: list[str] = None,
                     subtree: Path = None) -> tuple[pa.DataFrame[LoCDiffModel], pa.DataFrame[LoCCountModel]]:
        """
        Extract cloc stats from a repository.

        The repository must be configured in the task configuration.

        :param config: Configuration for the extraction, this may be different from the
        task configuration in case we are extracting the diff from a subtree or subrepo.
        :param extra_cloc_args: Additional arguments to cloc
        :param subtree: If extracing LoC data from a subtree, this must be the base path of the
        subtree relative to the repository workdir.
        :returns: Tuple containing the diff dataframe and the baseline dataframe
        """
        baseline_outfile_suffix = str(config.baseline_ref).replace("/", "_")
        head_outfile_suffix = str(config.head_ref).replace("/", "_")
        if config.repo_path.is_absolute():
            cwd_path = config.repo_path
        else:
            cwd_path = self.session.user_config.src_path / config.repo_path

        try:
            repo = Repo(cwd_path)
            self.logger.info("Extract LoC from %s diff %s %s", cwd_path, config.baseline_ref, config.head_ref)
        except InvalidGitRepositoryError:
            self.logger.error("Attempt to extract LoC data from invalid git repository %s", cwd_path)
            raise

        with TemporaryDirectory() as out_dir:
            outfile = Path(out_dir) / "cloc"

            output_baseline_file = outfile.with_suffix("." + baseline_outfile_suffix)
            output_head_file = outfile.with_suffix("." + head_outfile_suffix)
            output_diff_file = outfile.with_suffix(f".diff.{baseline_outfile_suffix}.{head_outfile_suffix}")
            cloc_args = self._get_common_cloc_args()
            if extra_cloc_args:
                cloc_args += extra_cloc_args
            cloc_args += [f"--report-file={outfile}", config.baseline_ref, config.head_ref]

            cloc_cmd = SubprocessHelper(self._cloc, cloc_args)
            cloc_cmd.run(cwd=cwd_path)

            diff_df = self._load_diff_data(config, Path(output_diff_file))
            baseline_df = self._load_count_data(config, Path(output_baseline_file))

            # cloc generates multiple files, should move them to debug outputs?
            # shutil.copy(output_baseline_file, params.raw_cloc_baseline_path)
            # shutil.copy(output_head_file, params.raw_cloc_head_path)
            # shutil.copy(output_diff_file, params.raw_cloc_diff_path)

        # Fixup the paths to always be relative to the main repo
        # this may happen if head_ref or baseline_ref are in a different repository
        # e.g. in the case of a subtree or subrepo.
        def patch_path(pathstr: str):
            path = Path(pathstr)
            if subtree and path.is_absolute():
                if path.is_relative_to(config.baseline_ref):
                    path = path.relative_to(config.baseline_ref)
                elif path.is_relative_to(config.head_ref):
                    path = path.relative_to(config.head_ref)
                return str(subtree / path)
            return pathstr

        diff_df = map_index(diff_df, "file", patch_path)
        baseline_df = map_index(baseline_df, "file", patch_path)

        # Filter directories/files
        diff_df = self._apply_filters(diff_df, config)
        baseline_df = self._apply_filters(baseline_df, config)

        return (diff_df, baseline_df)


class ExtractLoCGenericRepo(ExtractLoCBase):
    """
    Generate the LoC diff for a generic repository specified in configuration.
    """
    task_name = "repo-generic"

    def __init__(self, session, task_config=None):
        # Required for task_id
        assert task_config is not None
        self._repo_key = str(task_config.name).replace("/", "_").replace(" ", "_")
        super().__init__(session, task_config=task_config)

    @property
    def task_id(self):
        return super().task_id + "-" + self._repo_key

    def run(self):
        diff_df, base_df = self._extract_loc(self.config)
        diff_df.to_csv(self.cloc_diff.path)
        base_df.to_csv(self.cloc_baseline.path)

    @output
    def cloc_diff(self):
        return LocalFileTarget(self, prefix="cloc-diff", ext="csv", model=LoCDiffModel)

    @output
    def cloc_baseline(self):
        return LocalFileTarget(self, prefix="cloc-baseline", ext="csv", model=LoCCountModel)


class ExtractLoCMultiRepo(SessionDataGenTask):
    """
    Generate the LoC diff for a set of repositories specified in configuration
    """
    public = True
    task_namespace = "cloc"
    task_name = "generic"
    task_config_class = ExtractLoCConfig

    @dependency
    def repos(self):
        for repo in self.config.repos:
            yield ExtractLoCGenericRepo(self.session, task_config=repo)

    def run(self):
        pass

    @output
    def cloc_diff(self):
        return [r.cloc_diff for r in self.repos]

    @output
    def cloc_baseline(self):
        return [r.cloc_baseline for r in self.repos]
