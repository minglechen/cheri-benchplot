import re
from pathlib import Path
from typing import ForwardRef, Type

import pandas as pd
import pandera as pa
from jinja2 import (Environment, PackageLoader, TemplateNotFound, select_autoescape)
from typing_extensions import Self

from .borg import Borg
from .model import BaseDataModel, DerivedSchemaBuilder
from .task import BenchmarkTask, SessionTask, Task, output


class Target(Borg):
    """
    Base class that represents an output artefact of a task.

    Targets are borg objects that share the internal state, as there is only
    one output for each unique output identifier of a task.
    The output identifier is derived from the task_id and other parameters.
    """
    def __init__(self, task: Task, output_id: str):
        """
        A target is associated to a task and output_id, which can be used to
        produce the unique target identifier.

        :param task: Task that produces this artefact
        :param output_id: Unique identifier of the artefact within the task.
        """
        self._task = task
        self._output_id = output_id

        # Borg state initialization occurs here
        super().__init__()

    @property
    def borg_state_id(self):
        """
        Note that since the unique output ID depends on the task_id, intentional
        collision of task_id values for session-scoped tasks will result in
        session-scoped outputs
        """
        output_suffix = f"-{self._output_id}" if self._output_id else ""
        return f"{self._task.task_id}-output{output_suffix}"

    def is_file(self) -> bool:
        """
        :return: True if the target is a subclass of :class:`FileTarget`
        """
        return False


class DataFrameTarget(Target):
    """
    Target wrapping an output dataframe from a task.
    """
    def __init__(self, task: Task, model: BaseDataModel | DerivedSchemaBuilder | None, output_id: str | None = None):
        if model:
            self.schema = model.to_schema(task.session)
        else:
            self.schema = None
        #: The dataframe currently assigned
        self._df = None
        # Original model for checking
        self._model = model

        if output_id is None and model is not None:
            output_id = model.get_name()
        # Borg state initialization occurs here
        super().__init__(task, output_id)

        # Sanity check to verify that the model is consistent
        if model and self._model != model:
            task.logger.error("DataFrameTarget ID collision detected for %s, output_id=%s", self.borg_state_id,
                              output_id)
            raise RuntimeError("DataFrameTarget Target ID collision")

    def assign(self, df: pd.DataFrame):
        if self.schema:
            df = self.schema.validate(df)
        self._df = df

    def get(self) -> pd.DataFrame:
        return self._df.copy()


class FileTarget(Target):
    """
    Base class for a target output file.

    The factory methods should be used to generate paths for the file targets.
    """
    def __init__(self,
                 task: Task,
                 prefix: str = "",
                 ext: str | None = None,
                 use_iterations: bool = False,
                 model: BaseDataModel | None = None,
                 file_path: Path | str | None = None,
                 loader: Task | None = None):
        self.use_iterations = use_iterations
        self._loader_model = model
        self._loader = loader
        # Prepare paths
        if file_path is None:
            name = re.sub(r"\.", "-", task.task_id)
            if prefix:
                name = f"{prefix}-{name}"
            self._file_name = Path(name)
        else:
            # Normalize to path
            file_path = Path(file_path)
            if prefix:
                file_path = file_path.with_name(f"{prefix}-{file_path.name}")
            self._file_name = file_path
        if ext:
            if not ext.startswith("."):
                ext = "." + ext
            self._file_name = self._file_name.with_suffix(ext)

        # Borg state initialization occurs here
        super().__init__(task, prefix)

    @property
    def path(self) -> Path:
        """
        Shorthand to return the path for targets that do not depend on iterations.
        If the path depends on the iteration index, there may be multiple paths for this target,
        in which case this accessor fails in order to prevent accidental misuse.

        :return: The path for the target.
        """
        if self.use_iterations:
            raise ValueError("Can not use shorthand path property when path depends on iteration index")
        return self.paths()[0]

    def paths(self) -> list[Path]:
        """
        Produce the paths for this target.

        Multiple paths may be produced if the target uses the benchmark iterations count to produce
        a separate file for each iteration.

        :return: A sequence of paths, ordered by iteration number.
        """
        raise NotImplementedError("Must override")

    @property
    def remote_path(self) -> Path:
        """
        Shorthand to return the path for targets that do not depend on iterations.
        If the path depends on the iteration index, this raises an exception.

        :return: The remote path for the target, this can be used to copy the file out
        of the guest.
        """
        if self.use_iterations:
            raise ValueError("Can not use shorthand remote_path property when path depends on iteration index")
        return self.remote_paths()[0]

    def remote_paths(self) -> list[Path]:
        """
        Same as path but all paths are coverted rebased to the guest data output directory.

        :return: A sequence of paths, ordered by iteration number.
        """
        raise NotImplementedError("Must override")

    def is_file(self):
        return True

    def needs_extraction(self):
        raise NotImplementedError("Must override")

    def get_loader(self) -> Task:
        """
        The loader for a target is a task that can be scheduled to load the output data.

        :return: A task that can be depended upon to load the target data.
        """
        if self._loader:
            if self._loader.is_session_task():
                return self._loader(self._task.session, self)
            elif self._loader.is_benchmark_task():
                return self._loader(self._task.benchmark, self)

        if self._task.is_session_task():
            return TargetSessionLoadTask(self._task.session, self, self._loader_model)
        elif self._task.is_benchmark_task():
            return TargetLoadTask(self._task.benchmark, self, self._loader_model)
        raise TypeError("Task %s is not supported by the file loader", self._task)


class DataFileTarget(FileTarget):
    """
    A target output file that is generated on the guest and needs to be extracted.

    Note that guest-generated targets are only valid when associated to a benchmark,
    session-wide files are currently not allowed.
    This restriction may be lifted if required.
    """
    def __init__(self,
                 task: Task,
                 prefix: str = "",
                 ext: str | None = None,
                 use_iterations: bool = False,
                 model: BaseDataModel | None = None):
        if task.is_session_task():
            raise TypeError(f"{self.__class__} does not support session-wide "
                            "scope tasks, only benchmark scope")
        super().__init__(task, prefix=prefix, ext=ext, use_iterations=use_iterations, model=model)

    def paths(self):
        benchmark = self._task.benchmark
        if self.use_iterations:
            base_paths = map(benchmark.get_benchmark_iter_data_path, range(benchmark.config.iterations))
            return [base / self._file_name for base in base_paths]
        else:
            return [benchmark.get_benchmark_data_path() / self._file_name]

    def remote_paths(self):
        benchmark = self._task.benchmark
        benchmark_data_root = benchmark.get_benchmark_data_path()
        guest_data_root = benchmark.config.remote_output_dir
        return [guest_data_root / p.relative_to(benchmark_data_root) for p in self.paths()]

    def needs_extraction(self):
        return True


class LocalFileTarget(FileTarget):
    """
    A target output file that is generated on the host and does not need to be extracted
    """
    def _session_paths(self) -> list[Path]:
        session = self._task.session
        if self.use_iterations:
            raise NotImplementedError("per-iteration session local files are not implemented")
        else:
            return [session.get_data_root_path() / self._file_name]

    def _benchmark_paths(self) -> list[Path]:
        benchmark = self._task.benchmark
        benchmark_data_root = benchmark.get_benchmark_data_path()
        if self.use_iterations:
            base_paths = map(benchmark.get_benchmark_iter_data_path, range(benchmark.config.iterations))
            return [base / self._file_name for base in base_paths]
        else:
            return [benchmark_data_root / self._file_name]

    def paths(self):
        if self._task.is_session_task():
            return self._session_paths()
        elif self._task.is_benchmark_task():
            return self._benchmark_paths()
        raise TypeError("Unrecognised task type")

    def remote_paths(self):
        raise TypeError(f"{self.__class__} does not have remote paths")

    def needs_extraction(self):
        return False


class AnalysisFileTarget(LocalFileTarget):
    """
    A target that identifies the product of an analysis task, which lives in
    the analysis output directory.
    """
    def __init__(self, task: Task, prefix: str = "", ext: str | None = None):
        if not task.is_session_task():
            raise TypeError("AnalysisFileTarget does not support datarun-wide analysis tasks")
        super().__init__(task, prefix=prefix, ext=ext, use_iterations=False)

    def paths(self):
        return [self._task.session.get_plot_root_path() / self._file_name]


class DataRunAnalysisFileTarget(LocalFileTarget):
    """
    A target that identifies the product of an analysis task specific to a benchmark run.

    The file is emitted in the datarun analysis output directory.
    """
    def __init__(self, task: Task, prefix: str = "", ext: str | None = None):
        if not task.is_benchmark_task():
            raise TypeError("DataRunAnalysisFileTarget does not support session-wide analysis tasks")
        super().__init__(task, prefix=prefix, ext=ext, use_iterations=False)

    def paths(self):
        return [self._task.benchmark.get_plot_path() / self._file_name]


class HTMLTemplateTarget(LocalFileTarget):
    """
    A target that identifies an HTML template that renders analysis data.

    Note that the templates are located in pycheribenchplot/templates.
    """
    env = Environment(loader=PackageLoader("pycheribenchplot", "templates"), autoescape=select_autoescape())

    def __init__(self, task: Task, template: str):
        self._template = template
        base_name = template.split(".")[0]
        super().__init__(task, prefix=base_name, ext="html")

    def paths(self):
        if self._task.is_benchmark_task():
            base_path = self._task.benchmark.get_plot_path()
        elif self._task.is_session_task():
            base_path = self._task.session.get_plot_root_path()
        else:
            raise TypeError("HTMLTemplateTarget require a session or benchmark task")
        return [base_path / self._file_name]

    def render(self, **kwargs):
        try:
            tmpl = HTMLTemplateTarget.env.get_template(self._template)
        except TemplateNotFound:
            self._task.logger.error("Can not find file template %s, target setup is wrong", self._template)
            raise RuntimeError("Target error")
        with open(self.path, "w") as fd:
            fd.write(tmpl.render(**kwargs))


class TargetLoadTaskMixin:
    """
    Helper mixing with common file loading operations
    """
    def _load_one_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file into a dataframe.

        :param path: The file path.
        :param **kwargs: Additional arguments to pandas read_csv().
        :return: The resulting dataframe.
        """
        return pd.read_csv(path, **kwargs)

    def _load_one_json(self, path: Path, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file into a dataframe

        :param path: The file path.
        :param **kwargs: Additional arguments to pandas read_json().
        :return: The resulting dataframe.
        """
        return pd.io.json.read_json(path, **kwargs)

    def _load_one(self, path: Path) -> pd.DataFrame:
        """
        Load data from a given path, the format is inferred from the file extension.

        :param path: The target path.
        :return: A dataframe containing the data.
        """
        if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
            df = self._load_one_csv(path)
        elif path.suffix == ".json":
            df = self._load_one_json(path)
        else:
            self.logger.error(
                "Can not determine how to load %s, add extesion or override "
                f"{self.__class__.__name__}._load_one()", path)
            raise RuntimeError("Can not infer file type from extension")
        return df

    def _parameter_index_columns(self) -> list[str]:
        """
        Produce a list of benchmark parameterization keys.

        This is only relevant if this task is loading data for a benchmark, if the
        task is loading session-wide data, do not generate the parameterization keys.

        :return: A list of key names for the index levels
        """
        if self.is_benchmark_task() and self.benchmark.config.parameters:
            return list(self.benchmark.config.parameters.keys())
        return []

    def _prepare_standard_index(self, target: FileTarget, schema: pa.DataFrameSchema, df: pd.DataFrame,
                                iteration: int | None) -> pd.DataFrame:
        """
        If the target gives us a model, automatically create the standard set of index
        levels if they are part of the model.

        :param target: The file target we are loading files from.
        :param df: Input dataframe to transform.
        :return: A new dataframe with the additional index levels.
        """
        if schema is None:
            return df.copy()

        df = df.reset_index()
        names = list(schema.index.names)

        benchmark_only_index = ["dataset_id", "dataset_gid", "iteration"]
        has_benchmark_index = set(names).intersection(benchmark_only_index)
        if has_benchmark_index and not self.is_benchmark_task():
            self.logger.error("The model index %s can only be set for BenchmarkTasks.", has_benchmark_index)
            raise TypeError("Invalid target task type")

        if "dataset_id" in names and "dataset_id" not in df.columns:
            self.logger.debug("No dataset column, using default")
            df["dataset_id"] = self.benchmark.uuid
        if "dataset_gid" in names and "dataset_gid" not in df.columns:
            self.logger.debug("No dataset group, using default")
            df["dataset_gid"] = self.benchmark.g_uuid
        if "iteration" in names and "iteration" not in df.columns:
            self.logger.debug("No iteration index, using default")
            df["iteration"] = iteration if target.use_iterations else -1

        # Generate benchmark parameter index keys if necessary
        for pcol in self._parameter_index_columns():
            if pcol in df.columns:
                continue
            param = self.benchmark.config.parameters[pcol]
            self.logger.debug("No parameter %s column, generate from config %s=%s", pcol, pcol, param)
            df[pcol] = param

        for n in names:
            if n is None:
                self.logger.error("DataFrameSchema does not have name of index level, "
                                  "if this is the only index level, it should have the "
                                  "pandera.Field(check_name=True) descriptor")
                raise ValueError("Invalid DataFrameSchema index")

        return df.set_index(names)

    def _load_from(self, target: FileTarget, schema: pa.DataFrameSchema) -> pd.DataFrame:
        """
        Load all the files defined by a target and merge the resulting dataframes

        :param target: The file target we are loading files from.
        :param schema: The target schema that describes the output dataframe
        :return: The dataframe containing the union of the data.
        """
        df_set = []
        for i, path in enumerate(target.paths()):
            self.logger.debug("Loading data[i=%d] from %s", i, path)
            if not path.exists():
                self.logger.error("Can not load %s, does not exist", path)
                raise FileNotFoundError(f"{path} does not exist")
            df = self._load_one(path)
            df = self._prepare_standard_index(target, schema, df, i)
            df_set.append(df)

        if not df_set:
            # Bail because we can not concatenate and validate an empty frame
            # We could support empty data but there is no use case for it now.
            self.logger.error("No data has been loaded for %s", self)
            raise ValueError("Loader did not find any data")
        if len(df_set) == 1:
            # Skip concatenation. If the frame is large, time saving is significant
            return df_set[0]
        return pd.concat(df_set)


class TargetLoadTask(BenchmarkTask, TargetLoadTaskMixin):
    """
    Internal task to load target data.

    This is the default loader task that loads data from file targets.
    """
    task_namespace = "internal"
    task_name = "target-load"

    def __init__(self, benchmark: "Benchmark", target: FileTarget, model: BaseDataModel | None = None):
        assert target.is_file(), "TargetLoadTask can only operate on file targets"
        self.target = target
        self.model = model

        # Borg state initialization occurs here
        super().__init__(benchmark)

    @property
    def task_id(self):
        return super().task_id + "-for-" + self.target.borg_state_id

    def run(self):
        schema = self.model.to_schema(self.session) if self.model else None
        df = self._load_from(self.target, schema)
        self.df.assign(df)

    @output
    def df(self):
        return DataFrameTarget(self, self.model, output_id=f"loaded-df-for-{super().task_id}")


class TargetSessionLoadTask(SessionTask, TargetLoadTaskMixin):
    """
    Internal task to load target data.

    This is the default loader task that loads data from file targets.
    """
    task_namespace = "internal"
    task_name = "target-session-load"

    def __init__(self, session: "Session", target: FileTarget, model: BaseDataModel | None = None):
        assert target.is_file(), "TargetSessionLoadTask can only operate on file targets"
        self.target = target
        self.model = model

        # Borg state initialization occurs here
        super().__init__(session)

    @property
    def task_id(self):
        return super().task_id + "-for-" + self.target.borg_state_id

    def run(self):
        schema = self.model.to_schema(self.session) if self.model else None
        df = self._load_from(self.target, schema)
        self.df.assign(df)

    @output
    def df(self):
        return DataFrameTarget(self, self.model, output_id=f"loaded-df-for-{super().task_id}")
