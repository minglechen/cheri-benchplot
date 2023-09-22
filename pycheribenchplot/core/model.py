import functools
import inspect
import typing
import uuid
from typing import Any, Callable, Type

import hypothesis as hp
import numpy as np
import pandas as pd
import pandera as pa
import pandera.strategies as st
import typing_inspect
import wrapt
from pandera import (Check, Column, DataFrameSchema, DataType, Field, SchemaModel)
from pandera.dtypes import immutable
from pandera.engines import numpy_engine, pandas_engine
from pandera.errors import SchemaError
from pandera.typing import DataFrame, Index, Series
from typing_extensions import Self

from .pandas_util import apply_or

#: Constant name to mark the datarun matrix index as unparameterized
UNPARAMETERIZED_INDEX_NAME = "RESERVED__unparameterized_index"

# Helper type
SchemaTransform = Callable[[DataFrameSchema], DataFrameSchema]


@pandas_engine.Engine.register_dtype
@immutable
class UUIDType(numpy_engine.Object):
    """
    Pandera data type for UUID values
    XXX we may move to having string UUID values instead...
    """
    def check(self, pandera_dtype, data_container):
        return data_container.map(lambda v: isinstance(v, uuid.UUID))

    def coerce(self, series: pd.Series) -> pd.Series:
        return series.map(lambda uid: uid if isinstance(uid, uuid.UUID) else uuid.UUID(uid))


def uuid_strategy(pa_dtype: DataType, strategy: st.SearchStrategy | None, *args):
    if strategy is None:
        return st.pandas_dtype_strategy(pa_dtype, strategy=hp.strategies.just(uuid.uuid4()))
    # This would be useful if we have generators for valid and invalid UUIDs and
    # this would filter them by validity
    raise NotImplementedError("Chaining not supported yet")


@pa.extensions.register_check_method(strategy=uuid_strategy)
def is_valid_uuid(pandas_obj, *args) -> pd.Series:
    return pandas_obj.map(apply_or(lambda v: type(v) == uuid.UUID or uuid.UUID(v), False))


class BaseDataModel(SchemaModel):
    """
    Empty base data model for all schemas.
    This does not assume the presence of any of the dynamic ID fields but provides the
    infrastructure to add dynamic fields based on a session.
    """
    @classmethod
    def dynamic_index_position(cls) -> str:
        """
        Return the name of the index level after which we insert the dynamically
        generated parameter key index fields from the session.
        """
        raise NotImplementedError("Must override")

    @classmethod
    def get_name(cls) -> str:
        # Maybe this should instead return the schema name
        # computed by the pandera schema model
        return cls.__name__.lower()

    @classmethod
    def to_schema(cls, session: "Session") -> DataFrameSchema:
        """
        Generate a dataframe or series schema.
        This takes care to add the dataset index fields, including the dynamic parameters
        fields from the current configuration.

        :param session: the pipeline session to use to fetch dynamic fields
        :return: dataframe or series schema for validation
        """
        assert session is not None, "Need to specify a session for dynamic fields"
        s = super().to_schema()
        index_names = list(s.index.names)
        param_index = session.benchmark_matrix.index
        extra_columns = {}
        if param_index.names[0] != UNPARAMETERIZED_INDEX_NAME:
            for param in param_index.names:
                try:
                    dt = param_index.dtypes[param]
                except AttributeError:
                    dt = param_index.dtype
                extra_columns[param] = Column(dt,
                                              nullable=False,
                                              coerce=True,
                                              required=True,
                                              title=param,
                                              description=f"Benchmark parameter {param}")
        s = s.add_columns(extra_columns).reset_index()
        # Create new index in the following order [<index_up_to_dynamic_position>, <param_index_fields>, <other_index_fields>]
        index_offset = cls.dynamic_index_position()
        if index_offset is not None:
            offset = index_names.index(cls.dynamic_index_position()) + 1
        else:
            offset = 0
        index_names[offset:offset] = extra_columns.keys()
        return s.set_index(index_names)

    @classmethod
    def strategy(cls, session: "Session" = None, **kwargs):
        """
        Similar to the base hypothesis strategy generator.

        If the session is not given, this will return a partial factory function
        that can be used to generate the final strategy with a given session.
        """
        def proxy_strategy(lazy_session: "Session"):
            return cls.to_schema(lazy_session).strategy(**kwargs)

        if session:
            return proxy_strategy(session)
        return proxy_strategy

    @classmethod
    def example(cls, session: "Session" = None, **kwargs) -> Callable[["Session"], pd.DataFrame] | pd.DataFrame:
        """
        Similar to the base example generator.

        If the session is not given, this will return a partial factory function
        that can be used to generate the final example with a given session.
        """
        def proxy_example(lazy_session: "Session"):
            return cls.to_schema(lazy_session).example(**kwargs)

        if session:
            return proxy_example(session)
        return proxy_example

    class Config:
        """
        :noindex:
        Common configuration for pandera schema validation
        """
        # Drop extra columns
        strict = "filter"
        # Coerce data types
        coerce = True


class DerivedSchemaBuilder:
    """
    Proxy schema generator.

    This only supports the to_schema() function from the SchemaModel interface.
    The resulting schema is modified according to a transformation function.
    """
    def __init__(self, id_: str, base_model: Type[BaseDataModel | Self], transform: SchemaTransform):
        #: Identifier used to distinguish the derived schema
        self._id = id_
        #: The underlying model that provides the schema
        self._model = base_model
        #: Schema transformation
        self._transform = transform

    def __eq__(self, other):
        return self._model == other._model and self._id == other._id

    def get_name(self) -> str:
        return self._model.get_name() + f"-{self._id}"

    def to_schema(self, session: "Session") -> DataFrameSchema:
        base_schema = self._model.to_schema(session)
        return self._transform(session, base_schema)


class GlobalModel(BaseDataModel):
    """
    A data model that is not associated to any specific benchmark run or group.
    This is a global set of data without dynamic fields.
    """
    @classmethod
    def dynamic_index_position(cls):
        return None


class DataModel(BaseDataModel):
    """
    Base class for data models for input data.
    Input data is generally loaded and assembled from the benchmark output files.
    We will should always have the full IDs here.
    """
    dataset_id: Index[UUIDType] = Field(nullable=False, title="benchmark run ID", is_valid_uuid=True)
    dataset_gid: Index[UUIDType] = Field(nullable=False, title="platform ID", is_valid_uuid=True)
    # The iteration number may be null if there is no iteration associated to the data, meaning
    # that it is collected for the whole set of iterations of the benchmark.
    iteration: Index[pd.Int64Dtype] = Field(nullable=True, title="Iteration number")

    @classmethod
    def as_groupby(cls, by: list | str) -> DerivedSchemaBuilder:
        """
        Produce a schema for a variation of this model that reflects a groupby operation.

        The groupby operation may elide any of the dataset_id, dataset_gid and iteration
        index levels. Additional groupby indexes may be specified.
        """
        # Normalize to list
        if type(by) == str:
            by = [by]

        assert by, "At least an index level must be specified to create groupby DataModel"
        derived_suffix = "groupby-" + "-".join(by)

        def index_transform(session, schema):
            index_names = list(schema.index.names)
            drop = [n for n in index_names if n not in by]
            return schema.reset_index(drop, drop=True)

        return DerivedSchemaBuilder(derived_suffix, cls, index_transform)

    @classmethod
    def as_raw_model(cls):
        """
        Produce a schema for a variation of this model that does not contain implicit
        index and columns.

        This is useful to validate data before the per-benchmark and per-session
        identifiers are added.
        """
        def index_transform(session, schema):
            index_names = list(schema.index.names)
            param_names = session.benchmark_matrix.index.names
            if param_names[0] == UNPARAMETERIZED_INDEX_NAME:
                param_names = []
            drop = ["dataset_id", "dataset_gid", "iteration"] + param_names
            return schema.reset_index(drop, drop=True)

        return DerivedSchemaBuilder("raw-model-", cls, index_transform)

    @classmethod
    def dynamic_index_position(cls):
        return "iteration"


class ParamGroupDataModel(BaseDataModel):
    """
    Base class for groups of statistics by parameter group.
    These will be a result of aggregation along the iteration or other custom axes.
    """
    # Note: Need the check_name=True because otherwise the single-level index will not
    # have a name associated to it. This does not happen for multi-indexes
    dataset_gid: Index[UUIDType] = Field(nullable=False, title="platform ID", check_name=True)

    @classmethod
    def dynamic_index_position(cls):
        return "dataset_gid"


def _validate_data_model(session: "Session", type_target: Type, value: Any) -> Any:
    if type_target == inspect.Signature.empty:
        return value

    target_origin = typing.get_origin(type_target)
    if typing_inspect.is_union_type(type_target):
        if typing_inspect.is_optional_type(type_target):
            type_target = typing.get_args(type_target)[0]
        else:
            # Dynamically resolve union type?
            return value
    elif typing_inspect.is_tuple_type(type_target):
        tuple_args = typing.get_args(type_target)
        if not isinstance(value, tuple):
            raise TypeError("Invalid return type, expected tuple")
        validated = []
        for type_arg, item in zip(tuple_args, value):
            validated.append(_validate_data_model(session, type_arg, item))
        return tuple(validated)
    elif target_origin and issubclass(target_origin, list):
        item_type = typing.get_args(type_target)[0]
        if not isinstance(value, list):
            raise TypeError("Invalid return type, expected list")
        return [_validate_data_model(session, item_type, item) for item in value]

    if not typing_inspect.is_generic_type(type_target):
        return value

    type_base = typing.get_origin(type_target)
    if issubclass(type_base, DataFrame):
        model = typing.get_args(type_target)[0]
    elif issubclass(type_base, Series):
        model = typing.get_args(type_target)[0]
    else:
        return value

    schema = model.to_schema(session=session)
    try:
        checked = schema.validate(value)
        return checked
    except:
        raise


def check_data_model(fn: Callable = None, warn: bool = False) -> Callable:
    """
    Class method decorator designed to decorate DatasetTask transformations.

    This will automatically insert model checks for the input and output DataModel dataframes from type hints.
    The caveat is that normally models do not contain the dynamic fields generated for each session from config,
    this makes sure that the schema is generated using the correct session from the current DatasetTask instance.
    """
    if fn is None:
        return functools.partial(check_data_model, warn=warn)

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        assert instance is not None, "@check_data_model() must be used on a DatasetTask method"
        # assert isinstance(instance, DatasetTask), "@check_data_model() must be used on a DatasetTask method"

        sig = inspect.signature(wrapped)
        bound_args = sig.bind_partial(*args).arguments
        bound_kwargs = sig.bind_partial(**kwargs).arguments
        wrapped_name = wrapped.__qualname__
        if warn:
            log_function = instance.logger.warning
        else:
            log_function = instance.logger.error

        def validate_arg(name, value):
            try:
                return _validate_data_model(instance.session, sig.parameters[name].annotation, value)
            except SchemaError as ex:
                log_function("%s: argument '%s' failed validation because %s", wrapped_name, name, ex)
                if not warn:
                    raise
            return value

        checked_args = tuple(validate_arg(arg_name, arg_value) for arg_name, arg_value in bound_args.items())
        checked_kwargs = {arg_name: validate_arg(arg_name, arg_value) for arg_name, arg_value in bound_kwargs.items()}
        result = wrapped(*checked_args, **checked_kwargs)

        try:
            return _validate_data_model(instance.session, sig.return_annotation, result)
        except SchemaError as ex:
            log_function("%s: returned value failed validation because %s", wrapped_name, ex)
            if not warn:
                raise
        return result

    return wrapper(fn)
