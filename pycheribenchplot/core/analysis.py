import typing
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd

from .benchmark import Benchmark
from .config import AnalysisConfig, Config
from .model import DataModel
from .task import AnalysisTask, DataFrameTarget, ExecutionTask


class BenchmarkAnalysisTask(AnalysisTask):
    """
    Base class for analysis tasks that operate on a single benchmark context.
    These generally used to perform per-benchmark operations such as loading
    benchmark output data, pre-processing and preliminary aggregation.
    """
    task_namespace = "analysis.benchmark"

    def __init__(self, benchmark: Benchmark, analysis_config: AnalysisConfig, task_config: Config = None):
        super().__init__(benchmark.session, analysis_config, task_config=task_config)
        #: The associated benchmark context
        self.benchmark = benchmark

    @property
    def uuid(self):
        return self.benchmark.uuid

    @property
    def g_uuid(self):
        return self.benchmark.g_uuid

    @property
    def task_id(self):
        """
        Note that this currently assumes that tasks with the same name are not issued
        more than once for each benchmark run UUID. If this is violated, we need to
        change the task ID generation.
        """
        return f"{self.task_namespace}.{self.task_name}-{self.benchmark.uuid}"


class MachineGroupAnalysisTask(AnalysisTask):
    """
    Base class for analysis tasks that operate on a group of benchmark contexts that
    have the same g_uuid (machine configuration), i.e. columns in the benchmark matrix.
    This is used for operations such as merging multiple data from benchmark parameterizations
    that have run on the same machine
    This is generally used to perform operations such as aggregating along the machine configuration
    axis and compute deltas between different benchmark configurations on the same machine.
    """
    task_namespace = "analysis.mgroup"

    def __init__(self,
                 session: "PipelineSession",
                 analysis_config: AnalysisConfig,
                 g_uuid: UUID,
                 task_config: Config = None):
        """
        :param session: The current session
        :param analysis_config: The analysis configuration for this run.
        :param g_uuid: The machine configuration ID for this group.
        :param task_config: Optional task configuration.
        """
        super().__init__(session, analysis_config, task_config=task_config)
        #: The associated group uuid
        self.g_uuid = g_uuid

    @property
    def task_id(self):
        return f"{self.task_namespace}.{self.task_name}-{self.g_uuid}"


class ParamGroupAnalysisTask(AnalysisTask):
    """
    Base class for analysis tasks that operate on a group of benchmark contexts that
    have the same set of parameterization values, i.e. rows in the benchmark matrix.
    This is used for operations such as merging multiple data from the same benchmark
    setup running different machines.
    This is generally used to perform operations such as aggregating along parameter axes
    and compute deltas between runs on different machine configurations.
    """
    task_namespace = "analysis.pgroup"

    def __init__(self,
                 session: "PipelineSession",
                 analysis_config: AnalysisConfig,
                 parameters: dict[str, any],
                 task_config: Config = None):
        super().__init__(session, analysis_config, task_config=task_config)
        #: The baseline group uuid
        self.baseline = session.baseline_g_uuid
        #: The set of parameters identifying the target benchmark matrix row
        self.parameters = parameters

    @property
    def task_id(self):
        parameter_set = ":".join([f"{key}={value}" for key, value in self.parameters.items()])
        return f"{self.task_namespace}.{self.task_name}-{parameter_set}"


class BenchmarkDataLoadTask(BenchmarkAnalysisTask):
    """
    General-purpose data loading and pre-processing task for benchmarks.

    This task will load some data from a target of a benchmark exec task.
    The load task needs to be pointed to the provider of the target, from which it
    can extract the path information.
    The data is loaded to a dataframe, according to a :class:`DataModel`.
    The input data model must be specified so that the input data is validated and
    the columns of interest are filtered.
    This task generates a DataFrameTarget() that identifies the task result.
    """
    #: The exec task from which to fetch the target
    exec_task: typing.Type[ExecutionTask] = None
    #: The name of the target file to load
    target_key: str = None
    #: Input data model
    model: typing.Type[DataModel] = None

    def __init__(self, benchmark: Benchmark, analysis_config: AnalysisConfig, **kwargs):
        super().__init__(benchmark, analysis_config, **kwargs)
        self._df = []

    def _parameter_index_columns(self):
        if self.benchmark.config.parameters:
            return list(self.benchmark.config.parameters.keys())
        return []

    def _output_df(self) -> pd.DataFrame:
        """
        Produce the output dataframe by joining all the iteration frames
        """
        if len(self._df) == 0:
            # Bail because we can not concatenate and validate an empty frame
            # We could support empty data but there is no use case for it now.
            self.logger.error("No data has been loaded for %s", self)
            raise ValueError("Loader did not find any data")
        schema = self.model.to_schema(self.session)
        df = pd.concat(self._df)
        return schema.validate(df)

    def _append_df(self, df: pd.DataFrame):
        """
        Add a given dataframe to the output dataframe.
        This is used to combine multiple iterations of the same benchmark that
        come from different files.
        Here we also set the index columns based on the benchmark configuration.
        """
        if len(df) == 0:
            self.logger.warning("Appending empty dataframe")
            return

        if "dataset_id" not in df.columns:
            self.logger.debug("No dataset column, using default")
            df["dataset_id"] = self.benchmark.uuid
        if "dataset_gid" not in df.columns:
            self.logger.debug("No dataset group, using default")
            df["dataset_gid"] = self.benchmark.g_uuid
        for pcol in self._parameter_index_columns():
            if pcol in df.columns:
                continue
            param = self.benchmark.config.parameters[pcol]
            self.logger.debug("No parameter %s column, generate from config %s=%s", pcol, pcol, param)
            df[pcol] = param

        # Now set the index based on the data model definition and proceed to validate
        schema = self.model.to_schema(self.session)
        df.set_index(schema.index.names, inplace=True)
        valid_df = schema.validate(df)
        self._df.append(df)

    def _load_one_csv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    def _load_one_json(self, path: Path) -> pd.DataFrame:
        return pd.io.json.read_json(path)

    def _load_one(self, path: Path, iteration: int):
        """
        Load data from the given path. The format is inferred from the extension.
        """
        self.logger.debug("Loading data[i=%d] from %s", iteration, path)
        if path.suffix == ".csv":
            df = self._load_one_csv(path)
        elif path.suffix == ".json":
            df = self._load_one_json(path)
        else:
            self.logger.error(
                "Can not determine how to load %s, add extesion or override BenchmarkDataLoadTask._load_one()", path)
            raise RuntimeError("Can not ifer file type from extension")
        df["iteration"] = iteration
        self._append_df(df)

    def run(self):
        target_task = self.exec_task(self.benchmark, script=None)
        target = target_task.output_map.get(self.target_key)
        if target is None:
            self.logger.error("%s can not load data from task %s, output key %s missing", self, target_task,
                              self.target_key)
            raise KeyError(f"{self.target_key} is not in task output_map")
        if not target.is_file():
            raise NotImplementedError("BenchmarkDataLoadTask only supports loading from files")
        for i, path in enumerate(target.paths):
            if not path.exists():
                self.logger.error("Can not load %s, does not exist", path)
                raise FileNotFoundError(f"{path} does not exist")
            if not target.has_iteration_path:
                i = -1
            self._load_one(path, i)

    def outputs(self):
        """
        Note that the target data will be valid only after the Task.completed
        flag has been set.
        """
        yield "df", DataFrameTarget(self.model, self._output_df())


class BenchmarkStatsByParamGroupTask(ParamGroupAnalysisTask):
    """
    Base task that computes statistics for a group of benchmarks with the same parameter keys.
    This will produce a dataframe with different benchmark runs and machine g_uuids in the index, with the same parameterization values.
    This task depends on the load tasks for all benchmarks with a given set of parameter keys.
    We merge the output of various load tasks into a single dataframe which is used to generate the aggregated output.
    The output dataframe should conform to the DataModel for this task.
    """
    #: The load task to use for loading dependencies
    load_task: typing.Type[BenchmarkDataLoadTask] = None
    #: The output data model
    model: typing.Type[DataModel] = None

    def __init__(self, session, analysis_config, parameters, **kwargs):
        super().__init__(session, analysis_config, parameters, **kwargs)
        self._df = None

    def _make_load_depend(self, benchmark: Benchmark):
        return self.load_task(benchmark, self.analysis_config)

    def _output_df(self) -> pd.DataFrame:
        """
        Produce the output dataframe and validate it with the given model
        """
        if len(self._df) == 0:
            # Bail it is weird to have an empty dataframe
            self.logger.error("Empty stats dataframe")
            raise RuntimeError("Empty stats dataframe")
        schema = self.model.to_schema(self.session)
        return schema.validate(self._df)

    def _transform_merged(self, mdf: pd.DataFrame) -> pd.DataFrame:
        """
        Hook to perform transformations on the merged dataframe before aggregation.
        """
        return mdf

    # Add I/O checks for base models?
    def _transform_aggregate(self, mdf: pd.DataFrame) -> pd.DataFrame:
        """
        Hook to perform aggregation on the merged dataframe.
        This should produce an extra column index level which will become the "aggregate" level.
        """

        # Setup aggregation functions map, note that function names will become names in the
        # 'aggregate' index level.
        def q25(v):
            return np.quantile(v, q=0.25)

        def q75(v):
            return np.quantile(v, q=0.75)

        agg_list = ["mean", "median", "std", q25, q75]
        agg_strategy = {c: agg_list for c in mdf.columns}

        # Group and aggregate
        grouped = mdf.groupby(self.group_keys)
        agg_df = grouped.aggregate(agg_strategy)
        agg_df.columns.set_names("aggregate", level=-1, inplace=True)
        return agg_df

    def _transform_delta_normal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute deltas with uncertainty propagation assuming a normal distribution
        """
        baseline_xs = df.xs(self.baseline, level="dataset_gid")
        # replicate the baseline to align it to each of the other datasets
        _, baseline_align = df.align(baseline_xs, axis=0)
        meanslice = (slice(None), "mean")
        stdslice = (slice(None), "std")
        delta = df.loc[:, meanslice] - baseline_align.loc[:, meanslice]
        std_delta = (df.loc[:, stdslice]**2 + baseline_align.loc[:, stdslice]**2)**0.5
        norm_delta = delta / baseline_align.loc[:, meanslice]
        # Approximated by taylor expansion as standard deviation for f(x,y) = x / y
        # If proof needed see https://www.stat.cmu.edu/~hseltman/files/ratio.pdf
        x = std_delta / delta.values
        y = baseline_align.loc[:, stdslice] / baseline_align.loc[:, meanslice].values
        std_norm_delta = norm_delta.abs().values * ((x**2 + y**2)**0.5)
        out_df = pd.concat([delta, std_delta, norm_delta, std_norm_delta],
                           keys=["delta", "delta", "norm_delta", "norm_delta"],
                           names=["delta"],
                           axis=1)
        return out_df.reorder_levels(["metric", "aggregate", "delta"], axis=1)

    def _transform_delta_skewed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute deltas with uncertainty propagation assuming a skewed distribution.
        Note this is more shaky in theory, we subtract directly the quartile values
        so that we get an estimation of the worst case scenario, where a value in the
        lower quartile gets subtracted a value from the high quartile (and reverse).
        This likely overestimates the uncertainty.
        XXX probably want to replace this with something like this
        https://iopscience.iop.org/article/10.1088/1681-7575/ab2a8d/pdf
        new_err_hi = data_err_hi + bs_err_lo
        new_err_lo = data_err_lo + bs_err_hi
        """
        baseline_xs = df.xs(self.baseline, level="dataset_gid")
        # replicate the baseline to align it to each of the other datasets
        _, baseline_align = df.align(baseline_xs, axis=0)
        medianslice = (slice(None), "median")
        q75slice = (slice(None), "q75")
        q25slice = (slice(None), "q25")

        # Median delta
        delta = df.loc[:, medianslice] - baseline_align.loc[:, medianslice]
        norm_delta = delta / baseline_align.loc[:, medianslice]
        # Estimated quartiles propagation
        delta_q25 = df.loc[:, q25slice] - baseline_align.loc[:, q75slice].values
        delta_q75 = df.loc[:, q75slice] - baseline_align.loc[:, q25slice].values
        norm_delta_q25 = delta_q25 / baseline_align.loc[:, q75slice].values
        norm_delta_q75 = delta_q75 / baseline_align.loc[:, q25slice].values

        out_df = pd.concat([delta, delta_q25, delta_q75, norm_delta, norm_delta_q25, norm_delta_q75],
                           keys=["delta", "delta", "delta", "norm_delta", "norm_delta", "norm_delta"],
                           names=["delta"],
                           axis=1)
        return out_df.reorder_levels(["metric", "aggregate", "delta"], axis=1)

    # Add I/O checks for base models?
    def _transform_delta(self, adf: pd.DataFrame) -> pd.DataFrame:
        """
        Hook to compute statistics delta on the merged dataframe.
        This should produce an extra column index level which will become the "delta" level.
        """
        # We rely on the column index ordering here, so double check it first
        assert adf.columns.names == ["metric", "aggregate"]

        delta_normal = self._transform_delta_normal(adf)
        delta_skewed = self._transform_delta_skewed(adf)
        # Add the delta level also to the absolute values of the stats data
        base_stats_df = pd.concat([adf], keys=["sample"], names=["delta"],
                                  axis=1).reorder_levels(["metric", "aggregate", "delta"], axis=1)
        # Join together all the delta columns with the original data columns
        stats_df = pd.concat([base_stats_df, delta_normal, delta_skewed], axis=1)
        return stats_df

    @property
    def group_keys(self) -> list[str]:
        """
        The list of keys to group by for statistics aggregation.
        By default group by machine configuration ID and parameter keys, so we aggregate along the iterations axis.
        """
        return ["dataset_gid"] + self.session.parameter_keys

    def dependencies(self):
        # Generate dependencies for our parameter set, this corresponds to a benchmark matrix row
        contexts = self.session.benchmark_matrix
        for key, value in self.parameters.items():
            contexts = contexts.xs(value, level=key)
        if len(contexts) != 1:
            self.logger.error(
                "Unexpected set of benchmarks resolved: %s. A single benchmark matrix row should be selected by %s",
                contexts.index, self.parameters)
            raise RuntimeError("Invalid benchmark parameters set")
        for ctx in contexts.to_numpy().ravel():
            yield self._make_load_depend(ctx)

    def run(self):
        """
        This task merges all load dependencies output into a single dataframe.
        Subclasses get a chance to transform the merged data before aggregation via
        :meth:`BenchmarkStatsByParamSetTask._transform_merged`.
        The merged dataframe is then aggregated using the declared set of aggregation
        functions.
        This will produce one additional column index level which is named 'aggregate'.
        The original column index level name is changed to 'metric'.

        The final dataframe will have the following properties:
        The column index has 3 levels: ["metric", "aggregate", "delta"]
        The row index will have a varying set of levels, depending on the aggregation group keys.
        """
        to_merge = []
        for loader in self.resolved_dependencies:
            to_merge.append(loader.output_map["df"].df)
        merged_df = pd.concat(to_merge)
        merged_df = self.load_task.model.to_schema(self.session).validate(merged_df)
        merged_df.columns.name = "metric"
        merged_df = self._transform_merged(merged_df)
        agg_df = self._transform_aggregate(merged_df)
        delta_df = self._transform_delta(agg_df)
        self._df = delta_df

    def outputs(self):
        yield "df", DataFrameTarget(self.model, self._output_df())


class StatsForAllParamSetsTask(AnalysisTask):
    """
    Generate statistics for each set of parameters in the benchmark matrix.
    Merge the statistics in the output dataframe.
    """
    #: The task class to produce the statistics dataframe
    stats_task: typing.Type[BenchmarkStatsByParamGroupTask] = None
    #: Data model for the output dataframe
    model: typing.Type[DataModel] = None

    def __init__(self, session, analysis_config, **kwargs):
        super().__init__(session, analysis_config, **kwargs)
        #: The output dataframe, after the task is completed.
        self._df = None

    def _output_df(self) -> pd.DataFrame:
        """
        Produce the output dataframe
        """
        schema = self.model.to_schema(self.session)
        return schema.validate(self._df)

    def dependencies(self):
        for param_set in self.session.benchmark_matrix.index:
            params = dict(zip(self.session.parameter_keys, param_set))
            yield self.stats_task(self.session, self.analysis_config, params)

    def run(self):
        to_merge = []
        for stats_task in self.resolved_dependencies:
            to_merge.append(stats_task.output_map["df"].df)
        self._df = pd.concat(to_merge)

    def outputs(self):
        yield "df", DataFrameTarget(self.model, self._output_df())
