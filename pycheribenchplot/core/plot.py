"""
General purpose matplotlib helpers
"""
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import matplotlib.pyplot as plt
from matplotlib import rc_context

from .analysis import AnalysisTask
from .artefact import AnalysisFileTarget
from .config import AnalysisConfig, Config, InstanceConfig
from .task import Task


@contextmanager
def new_figure(dest: Path | list[Path], **kwargs):
    kwargs.setdefault("constrained_layout", True)
    fig = plt.figure(**kwargs)
    yield fig
    if isinstance(dest, Path):
        dest = [dest]
    for path in dest:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


class PlotTarget(AnalysisFileTarget):
    """
    Target pointing to a plot path.

    The plot target assumes that the associated task is an analysis task.
    """
    def __init__(self, task: Task, prefix: str = "", ext: str | None = None):
        super().__init__(task, prefix=prefix, ext=ext)
        if ext:
            self._plot_ext = [ext]
        else:
            self._plot_ext = self._task.analysis_config.plot.plot_output_format

    def paths(self):
        files = []
        for ext in self._plot_ext:
            if not ext.startswith("."):
                ext = f".{ext}"
            files.append(self._file_name.with_suffix(ext))
        return [self._task.session.get_plot_root_path() / fname for fname in files]


class PlotTask(AnalysisTask):
    """
    Base class for plotting tasks.
    Plot tasks generate one or more plots from some analysis task data.
    These are generally the public-facing tasks that are selected in the analysis
    configuration.
    Each plot task is responsible for setting up a figure and axes.
    Note that the ID of the task is generated assuming that there is only one plot per session.
    """
    task_namespace = "analysis.plot"

    def __init__(self, session: "Session", analysis_config: AnalysisConfig, task_config: Config = None):
        super().__init__(session, analysis_config, task_config=task_config)

    def _plot_output(self, suffix: str = None) -> PlotTarget:
        if suffix:
            name = f"{self.task_id}-{suffix}.pdf"
        else:
            name = f"{self.task_id}.pdf"
        return PlotTarget(self.session.get_plot_root_path() / name)

    def get_instance_config(self, g_uuid: UUID) -> InstanceConfig:
        """
        Helper to retreive an instance configuration for the given g_uuid.
        """
        gid_column = self.session.benchmark_matrix[g_uuid]
        return gid_column[0].config.instance

    def g_uuid_to_label(self, g_uuid: UUID | str) -> str:
        """
        Helper that maps group UUIDs to a human-readable label that describes the instance
        """
        if isinstance(g_uuid, str):
            g_uuid = UUID(g_uuid)
        instance_config = self.get_instance_config(g_uuid)
        return instance_config.name

    def run(self):
        with rc_context():
            self.run_plot()

    def run_plot(self):
        """
        Plot task body.

        This runs within a matplotlib RC parameter context, so that local
        RC params are not propagated.
        """
        raise NotImplementedError("Must override")
