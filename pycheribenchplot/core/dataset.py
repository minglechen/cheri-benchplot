import typing
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import (is_integer_dtype, is_numeric_dtype, is_object_dtype)

from .util import new_logger


class DatasetProcessingError(Exception):
    pass


class DatasetName(Enum):
    """
    Public name used to identify dataset parsers (and, by association, generators)
    in the configuration file.
    """
    PMC = "pmc"
    NETPERF_DATA = "netperf-data"
    PROCSTAT_NETPERF = "procstat-netperf"
    QEMU_STATS_BB_HIST = "qemu-stats-bb"
    QEMU_STATS_CALL_HIST = "qemu-stats-call"
    QEMU_CTX_CTRL = "qemu-ctx-tracks"
    PIDMAP = "pidmap"
    VMSTAT_UMA_INFO = "vmstat-uma-info"
    VMSTAT_MALLOC = "vmstat-malloc"
    VMSTAT_UMA = "vmstat-uma"
    NETSTAT = "netstat"
    KERNEL_CSETBOUNDS_STATS = "kernel-csetbounds-stats"
    KERNEL_STRUCT_STATS = "kernel-struct-stats"
    KERNEL_STRUCT_MEMBER_STATS = "kernel-struct-member-stats"

    def __str__(self):
        return self.value


class DatasetArtefact(Enum):
    """
    Internal identifier for dataset artifacts that are generated by a dataset object.
    This identifies the artifact that the dataset generates, and is used to avoid
    generating multiple times the same data if multiple datasets reuse the same input
    for processing.
    """
    NETPERF = auto()
    PMC = auto()
    VMSTAT = auto()
    UMA_ZONE_INFO = auto()
    PROCSTAT = auto()
    PIDMAP = auto()
    QEMU_STATS = auto()
    NETSTAT = auto()
    KERNEL_CSETBOUNDS_STATS = auto()
    KERNEL_STRUCT_STATS = auto()

    def __str__(self):
        return self.name


class DatasetRunOrder(IntEnum):
    """
    Run ordering for datasets to extract data.
    This allows to control which dataset should run closer to the benchmark to
    avoid probe effect from other operations.
    """
    FIRST = 0
    ANY = 1
    LAST = 10


@dataclass
class Field:
    """
    Helper class to describe column and associated metadata to aid processing
    XXX-AM: Consider adding some sort of tags to the fields so that we can avoid hardcoding the
    names for some processing steps (e.g. normalized fields that should be shown as percentage,
    or address fields for hex visualization). We should also use the desc field to generate
    legend labels and human-readable version of the fields.
    May also help to split derived fields by the stage in which they are created
    (e.g. pre-merge, post-merge, post-agg). This should move the burden of declaring which fields to process.
    """
    name: str
    desc: str = None
    dtype: typing.Type = float
    isdata: bool = False
    isindex: bool = False
    isderived: bool = False
    importfn: typing.Callable = None

    @classmethod
    def str_field(cls, *args, **kwargs):
        kwargs.setdefault("importfn", str)
        return cls(*args, dtype=str, **kwargs)

    @classmethod
    def data_field(cls, *args, **kwargs):
        """A field representing benchmark measurement data instead of benchmark information."""
        return cls(*args, isdata=True, **kwargs)

    @classmethod
    def index_field(cls, *args, **kwargs):
        """A field representing benchmark setup index over which we can plot."""
        return cls(*args, isindex=True, **kwargs)

    @classmethod
    def derived_field(cls, *args, **kwargs):
        """A field that is generated during processing."""
        kwargs.setdefault("isdata", True)
        return cls(*args, isderived=True, **kwargs)

    @property
    def default_value(self):
        if is_integer_dtype(self.dtype):
            return 0
        elif is_numeric_dtype(self.dtype):
            return np.nan
        elif is_object_dtype(self.dtype):
            return None
        else:
            return ""

    def __post_init__(self):
        if self.desc is None:
            self.desc = self.name


class DatasetRegistry(type):
    dataset_types = defaultdict(list)

    @classmethod
    def resolve_name(cls, ds_name: DatasetName) -> typing.Type["DataSetContainer"]:
        """
        Find the dataset class with the given configuration name.
        It is an error if multiple matches are found.
        """
        resolved = []
        for dset_list in cls.dataset_types.values():
            for dset in dset_list:
                if dset.dataset_config_name == ds_name:
                    resolved.append(dset)
        if len(resolved) == 0:
            raise KeyError("No dataset registered with the name %s", ds_name)
        if len(resolved) > 1:
            raise ValueError("Multiple datasets match %s", ds_name)
        return resolved[0]

    def __init__(self, name, bases, kdict):
        super().__init__(name, bases, kdict)
        if self.dataset_config_name:
            # Only attempt to register datasets that can be named in the configuration file
            assert self.dataset_source_id, "Missing dataset_source_id"
            did = DatasetArtefact(self.dataset_source_id)
            duplicates = [
                dset for dset in DatasetRegistry.dataset_types[did]
                if dset.dataset_config_name == self.dataset_config_name
            ]
            assert len(duplicates) == 0
            DatasetRegistry.dataset_types[did].append(self)
        all_fields = []
        for base in bases:
            if hasattr(base, "fields"):
                all_fields += base.fields
        all_fields += kdict.get("fields", [])
        self._all_fields = all_fields


class DataSetContainer(metaclass=DatasetRegistry):
    """
    Base class to hold collections of fields containing benchmark data
    Each benchmark run is associated with an UUID which is used to cross-reference
    data from different files.

    Each dataset exposes 3 dataframes:
    - df: the input dataframe. There is one input dataframe for each instance of the dataset,
    belonging to each existing benchmark run. This is the dataframe on which we operate until
    we reach the *merge* step.
    There are two mandatory index levels: the dataset_id and __iterations levels.
    The dataset_id is the UUID of the benchmark run for which the data was captured.
    The __iteration index level contains the iteration number, if it is meaningless for the
    data source, then it is set to -1.
    - merged_df: the merged dataframe contains the concatenated data from all the benchmark
    runs of a given benchmark. This dataframe is built only once for each benchmark, in the
    dataset container belonging to the baseline benchmark run. This is by convention, as the
    baseline dataset is used to aggregate all the other runs of the benchmark.
    - agg_df: the aggregate dataset. The aggregation dataset is generated from the merged
    dataset by aggregating across iterations (to compute mean, median, etc...) and any other
    index field relevant to the data source. This is considered to be the final output of the
    dataset processing phase. In the post-aggregation phase, it is expected that the dataset
    will produce delta values between relevant runs of the benchmark.

    Dataframe indexing and fields:
    The index, data and metadata fields that we want to import from the raw dataset should be
    declared as class properties in the DataSetContainer. The registry metaclass will take care
    to collect all the Field properties and make them available via the get_fields()
    method.
    The resulting dataframes use multi-indexes on both rows and columns.
    The row multi-index levels are dataset-dependent and are declared as IndexField(), in addition
    to the implicit dataset_id and __iteration index levels.
    (Note that the __iteration index level should be absent in the agg_df as it would not make sense).
    The column index levels are the following (by convention):
    - The 1st column level contains the name of each non-index field declared as input
    (including derived fields from pre_merge()).
    - The next levels contain the name of aggregates or derived columns that are generated.
    """
    # Unique name of the dataset in the configuration files
    dataset_config_name: DatasetName = None
    # Internal identifier of the dataset, this can be reused if multiple containers use the
    # same source data to produce different datasets
    dataset_source_id: DatasetArtefact = None
    dataset_run_order = DatasetRunOrder.ANY
    # Data class for the dataset-specific run options in the configuration file
    run_options_class = None

    def __init__(self, benchmark: "Benchmarkbase", dset_key: str, config: "BenchmarkDataSetConfig"):
        """
        Arguments:
        benchmark: the benchmark instance this dataset belongs to
        dset_key: the key this dataset is associated to in the BenchmarkRunConfig
        """
        self.name = dset_key
        self.benchmark = benchmark
        self._script = self.benchmark.get_script_builder()
        self.config = config
        self.logger = new_logger(f"{dset_key}", parent=self.benchmark.logger)
        self.df = None
        self.merged_df = None
        self.agg_df = None

    @property
    def bench_config(self):
        # This needs to be dynamic to grab the up-to-date configuration of the benchmark
        return self.benchmark.config

    def input_fields(self) -> typing.Sequence[Field]:
        """
        Return a list of fields that we care about in the input data source.
        This will not contain any derived fields.
        """
        fields = [f for f in self.__class__._all_fields if not f.isderived]
        return fields

    def input_index_fields(self) -> typing.Sequence[Field]:
        """
        Return a list of fields that are the index levels in the input dataset.
        This will not include derived index fields.
        """
        fields = [f for f in self.input_fields() if f.isindex]
        return fields

    def input_non_index_columns(self) -> typing.Sequence[str]:
        """
        Return a list of column names in the input dataset that are not index columns,
        meaning that we return only data and metadata columns.
        """
        return [f.name for f in self.input_fields() if not f.isindex]

    def input_all_columns(self) -> typing.Sequence[str]:
        """
        Return a list of column names that we are interested in the input data source.
        """
        return [f.name for f in self.input_fields()]

    def input_base_index_columns(self) -> typing.Sequence[str]:
        """
        Return a list of column names that represent the index fields present in the
        input data source.
        """
        return [f.name for f in self.input_index_fields()]

    def input_index_columns(self) -> typing.Sequence[str]:
        """
        Return a list of column names that represent the index fields present in the
        input dataframe. Any implicit index column will be reported here as well.
        """
        return self.implicit_index_columns() + [f.name for f in self.input_index_fields()]

    def implicit_index_columns(self):
        return ["dataset_id", "__iteration"]

    def index_columns(self) -> typing.Sequence[str]:
        """
        All column names that are to be used as dataset index in the container dataframe.
        This will contain both input and derived index columns.
        """
        input_cols = self.input_index_columns()
        return input_cols + [f.name for f in self.__class__._all_fields if f.isderived and f.isindex]

    def all_columns(self) -> typing.Sequence[str]:
        """
        All columns (derived or not) in the pre-merge dataframe, including index columns.
        """
        return self.implicit_index_columns() + [f.name for f in self.__class__._all_fields]

    def data_columns(self) -> typing.Sequence[str]:
        """
        All data column names in the container dataframe.
        This, by default, does not include synthetic data columns that are generated after importing the dataframe.
        """
        return [f.name for f in self.__class__._all_fields if f.isdata and not f.isindex]

    def _get_input_columns_dtype(self) -> dict[str, type]:
        """
        Get a dictionary suitable for pandas DataFrame.astype() to normalize the data type
        of input fields.
        This will include both index and non-index fields
        """
        return {f.name: f.dtype for f in self.input_fields()}

    def _get_input_columns_conv(self) -> dict:
        """
        Get a dictionary mapping input columns to the column conversion function, if any
        """
        return {f.name: f.importfn for f in self.input_fields() if f.importfn is not None}

    def _get_all_columns_dtype(self) -> dict[str, type]:
        """
        Get a dictionary suitable for pandas DataFrame.astype() to normalize the data type
        of all dataframe fields.
        This will include both index and non-index fields
        """
        return {f.name: f.dtype for f in self.__class__._all_fields}

    def _append_df(self, df):
        """
        Import the given dataframe for one or more iterations into the main container dataframe.
        This means that:
        - The index columns must be in the given dataframe and must agree with the container dataframe.
        - The columns must be a subset of all_columns().
        - The missing columns that are part of input_all_columns() are added and filled with NaN or None.
        this will not include derived or implicit index columns.
        """
        if "dataset_id" not in df.columns:
            self.logger.debug("No dataset column, using default")
            df["dataset_id"] = self.benchmark.uuid
        if "__iteration" not in df.columns:
            self.logger.debug("No iteration column, using default (-1)")
            df["__iteration"] = -1
        # Normalize columns to always contain at least all input columns
        existing = df.columns.to_list() + list(df.index.names)
        default_columns = []
        for f in self.input_fields():
            if f.name not in existing:
                col = pd.Series(f.default_value, index=df.index, name=f.name)
                default_columns.append(col)
        if default_columns:
            self.logger.debug("Add defaults for fields not found in input dataset.")
            df = pd.concat([df] + default_columns, axis=1)
        # Normalize type for existing columns
        col_dtypes = self._get_input_columns_dtype()
        df = df.astype(col_dtypes)
        df.set_index(self.input_index_columns(), inplace=True)
        # Only select columns from the input that are registered as fields, the ones in the index are
        # already selected
        dataset_columns = set(self.input_non_index_columns())
        avail_columns = set(df.columns)
        column_subset = avail_columns.intersection(dataset_columns)
        if self.df is None:
            self.df = df[column_subset]
        else:
            self.df = pd.concat([self.df, df[column_subset]])
            # Check that we did not accidentally change dtype, this may cause weirdness due to conversions
            dtype_check = self.df.dtypes[column_subset] == df.dtypes[column_subset]
            if not dtype_check.all():
                changed = dtype_check.index[~dtype_check]
                for col in changed:
                    self.logger.error("Unexpected dtype change in %s: %s -> %s", col, df.dtypes[col],
                                      self.df.dtypes[col])
                raise DatasetProcessingError("Unexpected dtype change")

    def _add_delta_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Initialize and add the delta columns index level.
        This is generally used as the third index level to hold the delta for each aggregated
        column with respect to other benchmark runs.
        The original data columns are labeled "sample".
        """
        col_idx = df.columns.to_frame()
        col_idx["delta"] = "sample"
        df = df.copy()
        df.columns = pd.MultiIndex.from_frame(col_idx)
        return df

    def _add_aggregate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Initialize and add the aggregate columns index level.
        This is intended to be used by datasets that do not aggregate on iterations
        but still need to have an empty level for alignment purposes.
        """
        col_idx = df.columns.to_frame()
        col_idx["aggregate"] = "-"
        df = df.copy()
        df.columns = pd.MultiIndex.from_frame(col_idx)
        return df

    def _set_delta_columns_name(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper to rename a new set of columns along the delta column index level.
        Note that we rename every column in the level.
        """
        level_index = df.columns.names.index("delta")
        new_index = df.columns.map(lambda t: t[:level_index] + (name, ) + t[level_index + 1:])
        df.columns = new_index
        return df

    def _compute_delta_by_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        General operation to compute the delta of aggregated data columns between
        benchmark runs (identified by dataset_id).
        This will add the "delta_baseline" column in the delta columns index level.
        It is assumed that the current benchmark instance is the baseline instance,
        this is the case if called from post_aggregate().
        """
        assert check_multi_index_aligned(df, "dataset_id")
        assert "metric" in df.columns.names, "Missing column metric level"
        assert "delta" in df.columns.names, "Missing column delta level"

        datasets = df.index.get_level_values("dataset_id").unique()
        baseline = df.xs(self.benchmark.uuid, level="dataset_id")
        # broadcast the baseline cross-section across the dataset_id
        # and perform the arithmetic operation, we want the right-join
        # result only
        _, aligned_baseline = df.align(baseline)
        metric_cols = df.columns.get_level_values("metric")
        data_cols = set(self.data_columns()).intersection(metric_cols)
        column_idx = metric_cols.isin(data_cols)
        aligned_data_baseline = aligned_baseline.loc[:, column_idx]
        delta = df.subtract(aligned_data_baseline)
        norm_delta = delta.divide(aligned_data_baseline)
        result = pd.concat([
            df,
            self._set_delta_columns_name("delta_baseline", delta),
            self._set_delta_columns_name("norm_delta_baseline", norm_delta)
        ],
                           axis=1)
        return result.sort_index(axis=1)

    def _get_aggregation_strategy(self) -> dict:
        """
        Return the aggregation strategy to use for each data column of interest.
        The return dictionary is suitable to be used in pd.DataFrameGroupBy.aggregate()
        """
        def q25(v):
            return np.quantile(v, q=0.25)

        def q75(v):
            return np.quantile(v, q=0.75)

        def q90(v):
            return np.quantile(v, q=0.90)

        agg_list = ["mean", "median", "std", q25, q75, q90]
        return {c: agg_list for c in self.data_columns()}

    def _compute_aggregations(self, grouped: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        """
        Helper to generate the aggregate dataframe and normalize the column index level names.
        """
        agg = grouped.aggregate(self._get_aggregation_strategy())
        levels = list(agg.columns.names)
        levels[-1] = "aggregate"
        agg.columns = agg.columns.set_names(levels)
        return agg

    def iteration_output_file(self, iteration):
        """
        Generate the output file for this dataset for the current benchmark iteration.
        Any extension suffix should be added in subclasses.
        """
        return self.benchmark.get_iter_output_path(iteration) / f"{self.name}-{self.benchmark.uuid}"

    def output_file(self):
        """
        Generate the iteration-independent output file for this dataset.
        Any extension suffix should be added in subclasses.
        """
        return self.benchmark.get_output_path() / f"{self.name}-{self.benchmark.uuid}"

    def get_addrspace_key(self):
        """
        Return the name of the address-space to use for the benchmark address space in the symbolizer.
        This is only relevant for datasets that are intended to be used as the main benchmark dataset.
        """
        raise NotImplementedError("The address-space key must be specified by subclasses")

    def configure_platform(self, options: "PlatformOptions"):
        """
        Finalize the dataset run_options configuration and add any relevant platform options
        to generate the dataset.
        """
        self.logger.debug("Configure platform")
        if self.run_options_class:
            self.config = self.run_options_class(**self.config.run_options).bind(self.benchmark)
        return options

    def configure(self, options: "PlatformOptions"):
        """
        Finalize the dataset run_options configuration and add any relevant platform options
        to generate the dataset.
        """
        self.logger.debug("Configure dataset")
        if self.run_options_class:
            self.config = self.run_options_class(**self.config.run_options).bind(self.benchmark)
        return options

    def configure_iteration(self, iteration: int):
        """
        Update configuration for the current benchmark iteration, if any depends on it.
        This is called for each iteration, before pre_benchmark_iter()
        (e.g. to update the benchmark output file options)
        """
        self.logger.debug("Configure iteration %d", iteration)

    def gen_pre_benchmark(self):
        self.logger.debug("Gen pre-benchmark")

    def gen_pre_benchmark_iter(self, iteration: int):
        self.logger.debug("Gen pre-benchmark iteration %d", iteration)

    def gen_benchmark(self, iteration: int):
        self.logger.debug("Gen benchmark iteration %d", iteration)

    def gen_post_benchmark_iter(self, iteration: int):
        self.logger.debug("Gen post-benchmark iteration %d", iteration)

    def gen_post_benchmark(self):
        self.logger.debug("Gen post-benchmark")

    async def after_extract_results(self):
        """
        Give a chance to run commands on the live instance after the benchmark has
        completed. Note that this should only be used to extract auxiliary information
        that are not part of a dataset main input file, or to post-process output files.
        """
        self.logger.debug("Run post-extraction hook")

    def load(self):
        """
        Load the dataset from the common output files.
        Note that this is always called after iteration data has been loaded.
        No-op by default
        """
        pass

    def load_iteration(self, iteration: int):
        """
        Load the dataset per-iteration data.
        No-op by default
        """
        pass

    def pre_merge(self):
        """
        Pre-process a dataset from a single benchmark run.
        This can be used as a hook to generate composite metrics before merging the datasets.
        """
        self.logger.debug("Pre-process")

    def init_merge(self):
        """
        Initialize merge state on the baseline instance we are merging into.
        """
        if self.merged_df is None:
            self.merged_df = self.df

    def merge(self, other: "DataSetContainer"):
        """
        Merge datasets from all the runs that we need to compare
        Note that the merged dataset will be associated with the baseline run, so the
        benchmark.uuid on the merge and post-merge operations will refer to the baseline implicitly.
        """
        self.logger.debug("Merge")
        assert self.merged_df is not None, "forgot to call init_merge()?"
        self.merged_df = pd.concat([self.merged_df, other.df])

    def post_merge(self):
        """
        After merging, this can be used to generate composite or relative metrics on the merged dataset.
        """
        self.logger.debug("Post-merge")
        # Setup the name for the first hierarchical column index level
        self.merged_df.columns.name = "metric"

    def aggregate(self):
        """
        Aggregate the metrics in the merged runs.
        """
        self.logger.debug("Aggregate")
        # Do nothing by default
        self.agg_df = self.merged_df

    def post_aggregate(self):
        """
        Generate composite metrics or relative metrics after aggregation.
        """
        self.logger.debug("Post-aggregate")


@contextmanager
def dataframe_debug():
    """Helper context manager to print whole dataframes"""
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        yield


def col2stat(prefix, colnames):
    """
    Map base column namens to the respective statistic column with
    the given prefix
    """
    return list(map(lambda c: "{}_{}".format(prefix, c), colnames))


def check_multi_index_aligned(df: pd.DataFrame, level: str | list[str]):
    """
    Check that the given index level(s) are aligned.
    """
    if len(df) == 0:
        return True
    if not df.index.is_unique:
        return False

    grouped = df.groupby(level)
    # just grab the first group to compare
    first_index = list(grouped.groups.values())[0]
    match = first_index.to_frame().reset_index(drop=True).drop(level, axis=1)
    for _, g in grouped:
        g_match = g.index.to_frame().drop(level, axis=1).reset_index(drop=True)
        if match.shape != g_match.shape:
            # There is no hope of equality
            return False
        if not (match == g_match).all().all():
            return False
    return True


def align_multi_index_levels(df: pd.DataFrame, align_levels: list[str], fill_value=np.nan):
    """
    Align a subset of the levels of a multi-index.
    This will generate the union of the sets of values in the align_levels parameter.
    The union set is then repeated for each other dataframe index level, so that every
    combination of the other levels, have the same set of aligned level combinations.
    If the propagate_columns list is given, the nan values filled during alignment will
    be replaced by the original value of the column for the existing index combination.
    """
    assert df.index.is_unique, "Need unique index"
    # Get an union of the sets of levels to align as the index of the grouped dataframe
    align_sets = df.groupby(align_levels).count()
    # Values of the non-aggregated levels of the dataframe
    other_levels = [lvl for lvl in df.index.names if lvl not in align_levels]
    # Now get the unique combinations of other_levels
    other_sets = df.groupby(other_levels).count()
    # For each one of the other_sets levels, we need to repeat the aligned index union set, so we
    # create repetitions to make room for all the combinations
    align_cols = align_sets.index.to_frame().reset_index(drop=True)
    other_cols = other_sets.index.to_frame().reset_index(drop=True)
    align_cols_rep = align_cols.iloc[align_cols.index.repeat(len(other_sets))].reset_index(drop=True)
    other_cols_rep = other_cols.iloc[np.tile(other_cols.index, len(align_sets))].reset_index(drop=True)
    new_index = pd.concat([other_cols_rep, align_cols_rep], axis=1)
    new_index = pd.MultiIndex.from_frame(new_index)
    return df.reindex(new_index, fill_value=fill_value).sort_index()


def pivot_multi_index_level(df: pd.DataFrame, level: str, rename_map: dict = None) -> pd.DataFrame:
    """
    Pivot a row multi index level into the last level of the columns multi index.
    If a rename_map is given, the resulting column index level values are mapped accordingly to
    transform them into the new column level values.

    Example:
    ID  name  |  value
    A   foo   |    0
    A   bar   |    1
    B   foo   |    2
    B   bar   |    3

    pivots into

    name  | 0:      value
          | ID:   A   |   B
          | ------------------
    foo   |       0   |   2
    bar   |       1   |   3
    """
    keep_index = [lvl for lvl in df.index.names if lvl != level]
    df = df.reset_index().pivot(index=keep_index, columns=[level])
    if rename_map is not None:
        level_index = df.columns.names.index(level)
        mapped_level = df.columns.levels[level_index].map(lambda value: rename_map[value])
        df.columns = df.columns.set_levels(mapped_level, level=level)
    return df


def rotate_multi_index_level(df: pd.DataFrame,
                             level: str,
                             suffixes: dict[str, str] = None,
                             fill_value=None) -> tuple[pd.DataFrame]:
    """
    Given a dataframe with multiple datasets indexed by one level of the multi-index, rotate datasets into
    columns so that the index level is removed and the column values related to each dataset are concatenated
    and renamed with the given suffix map.
    We also emit a dataframe for the level/column mappings as follows.
    XXX deprecate and remove in favor of pivot_multi_index_level

    Example:
    ID  name  |  value
    A   foo   |    0
    A   bar   |    1
    B   foo   |    2
    B   bar   |    3

    is rotated into

    name  |  value_A value_B
    foo   |    0       2
    bar   |    1       3

    with the following column mapping
    ID  |  value
    A   |  value_A
    B   |  value_B
    """

    # XXX-AM: Check that the levels are aligned, otherwise we may get unexpected results due to NaN popping out

    rotate_groups = df.groupby(level)
    if suffixes is None:
        suffixes = rotate_groups.groups.keys()
    colmap = pd.DataFrame(columns=df.columns, index=df.index.get_level_values(level).unique())
    groups = []
    for key, group in rotate_groups.groups.items():
        suffix = suffixes[key]
        colmap.loc[key, :] = colmap.columns.map(lambda c: f"{c}_{suffix}")
        rotated = df.loc[group].reset_index(level, drop=True).add_suffix(f"_{suffix}")
        groups.append(rotated)
    if len(groups):
        new_df = pd.concat(groups, axis=1)
        return new_df, colmap
    else:
        # Return the input dataframe without the index level, but no extra columns as there are
        # no index values to rotate
        return df.reset_index(level=level, drop=True), colmap


def subset_xs(df: pd.DataFrame, selector: pd.Series, complement=False):
    """
    Extract a cross section of the given levels of the dataframe, regarless of frame index ordering,
    where the values match the given set of values.
    """
    if selector.dtype != bool:
        raise TypeError("selector must be a bool series")

    l, _ = selector.align(df)
    l = l.reorder_levels(df.index.names).fillna(False).sort_index()
    assert l.index.equals(df.index), f"{l.index} != {df.index}"
    if complement:
        l = ~l
    return df.loc[l]


def broadcast_xs(df: pd.DataFrame, chunk: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Given a dataframe and a cross-section from it, with some missing index levels, generate
    the complete series or frame with the cross-section aligned to the parent frame.
    This is useful to perform an intermediate operations on a subset (e.g. the baseline frame)
    and then replicate the values for the rest of the datasets.
    """
    _, r = df.align(chunk, axis=0)
    return r.reorder_levels(df.index.names)


def reorder_columns(df: pd.DataFrame, ordered_cols: typing.Sequence[str]):
    """
    Reorder columns as the given column name list. Any remaining column is
    appended at the end.
    """
    extra_cols = list(set(df.columns) - set(ordered_cols))
    result_df = df.reindex(columns=np.append(ordered_cols, extra_cols))
    return result_df


def index_where(df: pd.DataFrame, level: str, cond: pd.Series, alt: pd.Series):
    """
    Operation that mirrors dataframe.where but operates on an index level.
    """
    idx_df = df.index.to_frame()
    idx_df[level] = idx_df[level].where(cond, alt)
    df = df.copy()
    df.index = pd.MultiIndex.from_frame(idx_df)
    return df


def stacked_histogram(df_in: pd.DataFrame, group: str, stack: str, data_col: str, bins: list):
    """
    Helper to compute a dataframe suitable for plotting stacked multi-group
    histograms.
    Currently this only supports a single 'group' and 'stack' levels.
    """
    df = df_in.reset_index()
    g_uniq = df[group].unique()
    s_uniq = df[stack].unique()
    boundaries = np.array(bins)
    b_start = boundaries[:-1]
    b_end = boundaries[1:]
    hidx = pd.MultiIndex.from_product([g_uniq, s_uniq, b_start], names=[group, stack, "bin_start"])
    # preallocate dataframe
    hdf = pd.DataFrame({"count": 0}, index=hidx)

    groups = df.groupby([group, stack])
    for (k_group, k_stack), chunk in groups:
        count, out_bins = np.histogram(chunk[data_col], bins=bins)
        hist_key = (k_group, k_stack, slice(None))
        hdf.loc[hist_key, "bin_end"] = b_end
        hdf.loc[hist_key, "count"] = count
    return hdf.set_index("bin_end", append=True).sort_index()


def quantile_slice(df: pd.DataFrame,
                   columns: list[str | tuple],
                   quantile: float,
                   max_entries: int = None,
                   level: list[str] = None) -> pd.DataFrame:
    """
    Filter a dataset to select the values where the given columns are above/below the given quantile threshold.
    Care is taken to maintain the slice index aligned at the given level (dataset_id by default),
    for this reason if one entry satisfies the threshold for one dataset,
    the values for other datasets will be included as well.
    The max_entries option allows to limit the number of entries that we select for each dataset group.
    Returns the dataframe containing the entries above the given quantile threshold.
    """
    if level is None:
        level = ["dataset_id"]
    if isinstance(level, str):
        level = [level]
    if max_entries is None:
        max_entries = np.inf
    # preliminary checking
    assert check_multi_index_aligned(df, level)

    level_complement = df.index.names.difference(level)
    high_thresh = df[columns].quantile(quantile)

    # We split each level group to determine the top N entries for each group.
    # Then we slice each group at max_entries and realign the values across groups.
    # This will result in potentially more than max_entries per group, but maintains data integrity
    # without dropping interesting values. Note that we select entries based on the global
    # high_thresh, so there may be empty group selections.
    def handle_group(g):
        # Any column may be above
        cond = (g[columns] >= high_thresh).apply(np.any, axis=1)
        sel = pd.Series(False, index=g.index)
        if cond.sum() > max_entries:
            cut = g[cond].sort_values(columns, ascending=False).index[max_entries:]
            cond.loc[cut] = False
        sel[cond] = True
        return sel

    sel = df.groupby(level, group_keys=False).apply(handle_group)
    # Need to propagate True values in sel across `level` containing the
    # complementary key matching the high value, this is necessary to maintain
    # alignment of the frame groups.
    sel = sel.groupby(level_complement, group_keys=True).transform(lambda g: g.any())
    high_df = df[sel]
    # Make sure we are still aligned
    assert check_multi_index_aligned(high_df, level)
    return high_df.copy()


def assign_sorted_coord(df: pd.DataFrame, sort: list[str], group_by=list[str], **sort_kwargs) -> pd.Series:
    """
    Assign coordinates for plotting to dataframe groups, preserving the index mapping between groups.
    This assumes that the dataframe is aligned at the given level.

    df: the dataframe to operate on
    sort: columns to use for sorting
    group_by: grouping levels/columns
    **sort_kwargs: extra sort_values() parameters
    """
    assert check_multi_index_aligned(df, group_by)
    # Do not trash source df
    df = df.copy()
    # We now we find the max for each complementary group. This will be used for cross-group sorting
    index_complement = df.index.names.difference(group_by)
    sort_max_key = df.groupby(index_complement).max()[sort]
    # Generate temporary sort keys
    ngroups = len(df.groupby(group_by))
    tmp_sort_keys = [f"__sort_tmp_{i}" for i in range(len(sort))]
    for tmp_key, col in zip(tmp_sort_keys, sort):
        df[tmp_key] = np.tile(sort_max_key[col].values, ngroups)
    sorted_df = df.sort_values(tmp_sort_keys + index_complement, **sort_kwargs)
    coord_by_group = sorted_df.groupby(group_by).cumcount()
    return coord_by_group.sort_index()


def generalized_xs(df: pd.DataFrame, match: list, levels: list, complement=False):
    """
    Generalized cross section that allows slicing on multiple named levels.
    Example:
    Given a dataframe, generaized_xs(df, [0, 1], levels=["k0", "k1"]) gives:

     k0 | k1 | k2 || V
     0  | 0  | 0  || 1
     0  | 0  | 1  || 2
     0  | 1  | 0  || 3  generalized_xs()   k0 | k1 | k2 || V
     0  | 1  | 1  || 4 ==================> 0  | 1  | 0  || 3
     1  | 0  | 0  || 5                     0  | 1  | 1  || 4
     1  | 0  | 1  || 6
     1  | 1  | 0  || 7
     1  | 1  | 1  || 8
    """
    assert len(match) == len(levels)
    nlevels = len(df.index.names)
    slicer = [slice(None)] * nlevels
    for m, level_name in zip(match, levels):
        level_idx = df.index.names.index(level_name)
        slicer[level_idx] = m
    sel = pd.Series(False, index=df.index)
    sel.loc[tuple(slicer)] = True
    if complement:
        sel = ~sel
    return df[sel]


def filter_aggregate(df: pd.DataFrame, cond: pd.Series, by: list, how="all", complement=False):
    """
    Filter a dataframe with an aggregation function across a set of levels, where the
    aggregation function matches in all groups or in any group, depending on the "how" parameter.

    df: The dataframe to operate on
    cond: condition vector
    by: levels to check the condition on
    how: "all" or "any". If "all", match the rows where `cond` is True across all `by` groups.
    If "any", match the rows where `cond` is true in at least one `by` group.
    complement: If False, return the matched rows, if True return `df` without the matched rows.

    Example:
    Given a dataframe, filter_aggregate(df, df["k1"] == 0, ["k0"]) gives:

     k0 | k1 | k2 || V
     0  | 0  | 0  || 1
     0  | 0  | 1  || 2                     k0 | k1 | k2 || V
     0  | 1  | 0  || 3  filter_aggregate() 0  | 0  | 0  || 1
     0  | 1  | 1  || 4 ==================> 0  | 0  | 1  || 2
     1  | 0  | 0  || 5                     1  | 0  | 0  || 5
     1  | 0  | 1  || 6                     1  | 0  | 1  || 6
     1  | 1  | 0  || 7
     1  | 1  | 1  || 8
    """
    if cond.dtype != bool:
        raise TypeError("cond must be a boolean series")
    if isinstance(by, str):
        by = [by]
    if how == "all":
        agg_fn = lambda g: g.all()
    elif how == "any":
        agg_fn = lambda g: g.any()
    else:
        raise ValueError("how must be 'all' or 'any'")

    # Try to use the index complement first, if not available, dont know
    index_complement = df.index.names.difference(by)
    if len(index_complement) == 0:
        raise ValueError("Can not select across all index levels")
    match = cond.groupby(index_complement).transform(agg_fn)
    if complement:
        match = ~match
    return df[match]
