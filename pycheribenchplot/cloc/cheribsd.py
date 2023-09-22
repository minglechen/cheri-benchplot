from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import pandas as pd
import seaborn as sns
from git import Repo
from git.exc import BadName, InvalidGitRepositoryError

from ..core.artefact import LocalFileTarget
from ..core.config import Config
from ..core.pandas_util import generalized_xs
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import dependency, output
from .cloc_generic import ExtractLoCBase, ExtractRepoLoCConfig
from .model import LoCCountModel, LoCDiffModel


def abspath_or(path: Path | str, default_base: Path) -> Path:
    """
    Utility function to get either an absolute path or a default value
    """
    path = Path(path)
    if path.is_absolute():
        return path
    else:
        return default_base / path


@dataclass
class ExtractCheriBSDLoCConfig(ExtractRepoLoCConfig):
    drm: ExtractRepoLoCConfig | None = None
    zfs: ExtractRepoLoCConfig | None = None


class ExtractLoCCheriBSD(ExtractLoCBase):
    """
    Task that generates LoC information for the CheriBSD kernel changes.

    This is a convenience task to avoid to have to manually specify complex
    path filtering rules in the configuration file.

    Cheri kernel sources include a number of subrepos that must be cloned
    separately in order to be used as baseline:
     - drm: https://github.com/evadot/drm-subtree
     - zfs: https://github.com/CTSRD-CHERI/zfs.git
    """
    public = True
    task_name = "cheribsd-kernel"
    task_config_class = ExtractCheriBSDLoCConfig

    def __init__(self, session, task_config=None):
        super().__init__(session, task_config=task_config)

        # Determine defaults for the main kernel repo
        if self.config.repo_path is None:
            repo_fullpath = self.session.user_config.cheribsd_path
            self.config.repo_path = repo_fullpath.relative_to(self.session.user_config.src_path)
        else:
            repo_fullpath = self.session.user_config.src_path / self.config.repo_path
        self._repo = Repo(repo_fullpath)

        # Load ref to last freebsd merged commit
        with open(repo_fullpath / ".last_merge", "r") as last_merge:
            default_baseline = last_merge.read().strip()

        # Cleanup the freebsd refs, this can not be done in config because
        # we can't access the user_config there...
        self.config.baseline_ref = self._resolve_git_ref(self._repo, self.config.baseline_ref, default_baseline)
        self.config.head_ref = self._resolve_git_ref(self._repo, self.config.head_ref, self._repo.head.commit.hexsha)

        if self.config.baseline_ref is None:
            self.config.baseline_ref = last_merge

        # Initialize the subrepo configurations if there are no overrides
        if self.config.drm is None:
            self.config.drm = ExtractRepoLoCConfig(
                repo_path=Path("drm-subtree"),
                # Note this is in the drm-subtree repo
                baseline_ref="master",
                # Note this is in the cheribsd repo
                head_ref=str(self.config.head_ref) + ":sys/dev/drm",
                accept_ext=r"\.[chSsm]$")
        if self.config.zfs is None:
            self.config.zfs = ExtractRepoLoCConfig(
                repo_path=Path("zfs"),
                # Note this is in the zfs repo
                baseline_ref="cheri-hybrid",
                # Note this is in the cheribsd repo
                head_ref=str(self.config.head_ref) + ":sys/contrib/subrepo-openzfs",
                accept_ext=r"\.[chSsm]$")

        # Ensure that drm and zfs are there
        drm_repo = abspath_or(self.config.drm.repo_path, self.session.user_config.src_path)
        zfs_repo = abspath_or(self.config.zfs.repo_path, self.session.user_config.src_path)
        try:
            self._drm_repo = Repo(drm_repo)
            self.logger.debug("DRM subrepo baseline at %s", drm_repo)
        except InvalidGitRepositoryError:
            self.logger.error("Invalid DRM subrepo clone at %s", drm_repo)
            raise
        try:
            self._zfs_repo = Repo(zfs_repo)
            self.logger.debug("ZFS subrepo baseline at %s", zfs_repo)
        except InvalidGitRepositoryError:
            self.logger.error("Invalid ZFS subrepo clone at %s", zfs_repo)
            raise

        # Add standard kernel file filters
        self.config.accept_ext = r"\.[chSsmy]$"
        self.config.accept_filter += [r"^sys/"]
        self.config.reject_filter += [r"tests?", r"^sys/contrib/subrepo-openzfs", r"^sys/dev/drm"]

        self.logger.debug("Constructed LoC CheriBSD configuration: %s", asdict(self.config))

    def _resolve_git_ref(self, repo, ref, default):
        if ref is None:
            try:
                resolved = repo.commit(default).hexsha
            except BadName:
                self.logger.exception("Invalid default commit ref %s", default)
                raise
        else:
            try:
                resolved = repo.commit(ref).hexsha
            except BadName:
                self.logger.exception("Invalid cheribsd ref %s", ref)
                raise
        return resolved

    def run(self):
        # Extract base kernel LoC data
        diff_df, baseline_df = self._extract_loc(self.config, ["--match-d=sys/"])
        # Extract drm and zfs LoC data

        # Ensure that the subrepos are checked out ot the requested refs
        self._drm_repo.git.checkout(self.config.drm.baseline_ref)
        self._zfs_repo.git.checkout(self.config.zfs.baseline_ref)

        # Need to do this because cloc does not understand adding a remote to drm
        # directly in the cheribsd repo.
        subrepo_worktree = Path(self._drm_repo.working_tree_dir)
        subrepo_config = replace(self.config.drm, repo_path=self.config.repo_path, baseline_ref=subrepo_worktree)
        diff_drm_df, baseline_drm_df = self._extract_loc(subrepo_config, subtree="sys/dev/drm")

        subrepo_worktree = Path(self._zfs_repo.working_tree_dir)
        subrepo_config = replace(self.config.zfs, repo_path=self.config.repo_path, baseline_ref=subrepo_worktree)
        diff_zfs_df, baseline_zfs_df = self._extract_loc(subrepo_config, subtree="sys/contrib/subrepo-openzfs")

        all_diff = pd.concat([diff_df, diff_drm_df, diff_zfs_df])
        all_diff.to_csv(self.cloc_diff.path)
        all_baseline = pd.concat([baseline_df, baseline_drm_df, baseline_zfs_df])
        all_baseline.to_csv(self.cloc_baseline.path)

    @output
    def cloc_diff(self):
        return LocalFileTarget(self, prefix="cloc-diff", ext="csv", model=LoCDiffModel)

    @output
    def cloc_baseline(self):
        return LocalFileTarget(self, prefix="cloc-baseline", ext="csv", model=LoCCountModel)


class CheriBSDLoCDiffByComponent(PlotTask):
    """
    Produce an histogram sorted by the component with highest diff.

    Components are matched by path. This kind of plot is useful as it shows the
    distribution of kernel changes with intermediate granularity.
    File granularity is too confusing, while directory hierarchy may miss the
    relationship between different kernel components.
    This uses the same reporting histogram as cloc-by-file.
    There are two outputs, one using absolute diff numbers, the other using
    the changed LoC percentage with respect with total component lines.
    """
    public = True
    task_namespace = "cloc"
    task_name = "plot-cheribsd-by-component"

    components = {
        "platform": r"sys/(riscv|arm)",
        "vm": r"sys/vm/(?!uma)",
        "net": r"sys/net",
        "alloc": r"(sys/(vm/uma|kern/.*vmem)|sys/sys/vmem)",
        "dev": r"sys/dev(?!/drm)",
        "kern": r"(sys/kern/(?!.*vmem|vfs)|sys/sys/(?!vmem))",
        "compat": r"sys/compat(?!/freebsd64)",
        "compat64": r"sys/compat/freebsd64",
        "vfs": r"sys/kern/vfs",
        "fs": r"sys/fs",
        "cheri": r"sys/cheri",
        "drm": r"sys/dev/drm",
        "zfs": r"sys/contrib/subrepo-openzfs"
    }

    # Extra filters that are useful to ensure the sanity of the results
    # but should not go in the main plot
    extra_components = {
        "all_platforms": r"sys/(riscv|arm|amd|x86|i386|power)",
    }

    @dependency
    def loc_data(self):
        # Prevent circular import
        from .cloc import LoadLoCGenericData
        return LoadLoCGenericData(self.session, self.analysis_config, task_config=self.config)

    def _filter_component(self, name, filter_, data_df, base_df) -> pd.DataFrame:
        return changed_sloc

    def _do_plot(self, df: pd.DataFrame, base_df: pd.DataFrame, component_map: dict, target: PlotTarget):
        """
        Produce the plot given a set of component filters and a target where to emit the plot
        """
        # Filter only the cheribsd data
        # XXX assume the repository is "cheribsd", should grab this from the config
        df = df.xs("cheribsd", level="repo")
        base_df = base_df.xs("cheribsd", level="repo")
        # Drop the "same" data as we don't care
        df = generalized_xs(df, how="same", complement=True)

        # Ensure we have the proper theme
        sns.set_theme()
        # Use tab10 colors, added => green, modified => orange, removed => red
        cmap = sns.color_palette(as_cmap=True)
        # ordered as green, orange, red
        colors = [cmap[2], cmap[1], cmap[3]]

        # Produce the dataframe containing counts for each component
        index = pd.MultiIndex.from_product([self.components.keys(), ["added", "modified", "removed"]],
                                           names=["component", "how"])
        data_set = []
        extra_set = []
        all_matches = None
        all_base_matches = None
        for name, filter_ in component_map.items():
            matches = df.index.get_level_values("file").str.match(filter_)
            base_matches = base_df.index.get_level_values("file").str.match(filter_)
            self.logger.debug("Component %s ('%s') matched %d diff entries and %d baseline entries", name, filter_,
                              matches.sum(), base_matches.sum())
            if all_matches is None:
                all_matches = matches
                all_base_matches = base_matches
            else:
                all_matches = (all_matches | matches)
                all_base_matches = (all_base_matches | base_matches)
            changed_sloc = df.loc[matches].groupby("how").sum()
            changed_sloc["component"] = name
            changed_sloc["baseline"] = base_df.loc[base_matches]["code"].sum()
            data_set.append(changed_sloc)
        # Catch-all set
        other = df.loc[~all_matches].groupby("how").sum()
        other["component"] = "other"
        other["baseline"] = base_df.loc[~all_base_matches]["code"].sum()
        data_set.append(other)
        # Finally merge everything
        data_df = pd.concat(data_set)
        data_df["percent"] = 100 * data_df["code"] / data_df["baseline"]

        with new_figure(target.paths()) as fig:
            ax_l, ax_r = fig.subplots(1, 2, sharey=True)
            # Absolute SLoC on the left
            show_df = data_df.reset_index().pivot(index="component", columns="how", values="code")
            show_df.reset_index().plot(x="component",
                                       y=["added", "modified", "removed"],
                                       stacked=True,
                                       kind="barh",
                                       ax=ax_l,
                                       color=colors,
                                       legend=False)
            ax_l.tick_params(axis="y", labelsize="x-small")
            ax_l.set_xlabel("# of lines")

            # Percent SLoC on the right
            show_df = data_df.reset_index().pivot(index="component", columns="how", values="percent")
            show_df.reset_index().plot(x="component",
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

    def run_plot(self):
        """
        Produce plot variants by splitting the LoC changes by components.

        We produce two plots, one with all the changes and one only considering the
        changes that are built in the compilation DB.
        """
        self._do_plot(self.loc_data.diff_df.get(), self.loc_data.baseline_df.get(), self.components, self.plot)
        self._do_plot(self.loc_data.cdb_diff_df.get(), self.loc_data.cdb_baseline_df.get(), self.components,
                      self.cdb_plot)

        debug_components = dict(self.components)
        debug_components.update(self.extra_components)
        self._do_plot(self.loc_data.diff_df.get(), self.loc_data.baseline_df.get(), debug_components, self.debug_plot)
        self._do_plot(self.loc_data.cdb_diff_df.get(), self.loc_data.cdb_baseline_df.get(), debug_components,
                      self.cdb_debug_plot)

    @output
    def plot(self):
        return PlotTarget(self, prefix="all")

    @output
    def debug_plot(self):
        return PlotTarget(self, prefix="all-debug")

    @output
    def cdb_plot(self):
        return PlotTarget(self, prefix="cdb")

    @output
    def cdb_debug_plot(self):
        return PlotTarget(self, prefix="cdb-debug")
