import copy
import typing
import uuid

import pandas as pd
import pandera as pa
import pytest
from pandas.testing import assert_frame_equal
from pandera.typing import DataFrame, Index, Series

from pycheribenchplot.core.model import DataModel, check_data_model


# Note: Validation should ensure index ordering but not column
# The dynamic model index fields should be prepended to the schema index
class FakeModel(DataModel):
    foo: Index[int] = pa.Field(check_name=True)
    bar: Series[int]


class FakeDatasetTask:
    @check_data_model()
    def transform_noop(self, frame_in: DataFrame[FakeModel]) -> DataFrame[FakeModel]:
        return frame_in

    @check_data_model()
    def transform_extra_args(self, frame_in: DataFrame[FakeModel], foo: int) -> DataFrame[FakeModel]:
        return frame_in

    @check_data_model()
    def transform_return(self, frame_in: typing.Any) -> DataFrame[FakeModel]:
        return frame_in

    @check_data_model()
    def transform_no_return(self, frame_in: DataFrame[FakeModel]):
        return frame_in

    @check_data_model
    def transform_tuple_return(self, frame_in: typing.Any) -> tuple[DataFrame[FakeModel], DataFrame[FakeModel]]:
        return (frame_in, frame_in)

    @check_data_model
    def transform_tuple_return_fail(self, frame_in: typing.Any) -> tuple[DataFrame[FakeModel], DataFrame[FakeModel]]:
        return (frame_in, frame_in.reset_index("foo"))

    @check_data_model
    def transform_list_return(self, frame_in: typing.Any) -> list[DataFrame[FakeModel]]:
        return [frame_in, frame_in]

    @check_data_model
    def transform_list_return_fail(self, frame_in: typing.Any) -> list[DataFrame[FakeModel]]:
        return [frame_in, frame_in.reset_index("foo")]


@pytest.fixture
def session_config(single_benchmark_config):
    conf = copy.deepcopy(single_benchmark_config)
    bench = conf["configurations"][0]
    bench["parameters"] = {"param0": 10, "param1": "param1-value"}
    return conf


@pytest.fixture
def fake_task(session_config, fake_session_factory):
    task = FakeDatasetTask()
    s = fake_session_factory(session_config)
    task.session = s
    task.logger = s.logger
    return task


test_frame = (pd.DataFrame({
    "dataset_id": ["8bc941a3-f6d6-4d37-b193-4738f1da3dae"],
    "dataset_gid": ["2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb"],
    "iteration": [5],
    "param0": [10],
    "param1": ["param1-value"],
    "foo": [0xdead],
    "bar": [0xbeef]
}).astype({
    "iteration": pd.Int64Dtype()
}).set_index(["dataset_id", "dataset_gid", "iteration", "param0", "param1", "foo"]))

# Simple dataframe to expect for each test
expect = (pd.DataFrame({
    "dataset_id": [uuid.UUID("8bc941a3-f6d6-4d37-b193-4738f1da3dae")],
    "dataset_gid": [uuid.UUID("2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb")],
    "iteration": [5],
    "param0": [10],
    "param1": ["param1-value"],
    "foo": [0xdead],
    "bar": [0xbeef]
}).astype({
    "iteration": pd.Int64Dtype()
}).set_index(["dataset_id", "dataset_gid", "iteration", "param0", "param1", "foo"]))


def test_dynamic_model_fields(session_config, fake_session_factory):
    s = fake_session_factory(session_config)
    m = FakeModel.to_schema(session=s)

    assert type(m) == pa.DataFrameSchema

    result = m.validate(test_frame.copy())
    assert_frame_equal(result, expect)


def test_dynamic_model_fields_extra_cols(session_config, fake_session_factory):
    s = fake_session_factory(session_config)
    m = FakeModel.to_schema(session=s)

    assert type(m) == pa.DataFrameSchema

    test_data = test_frame.copy()
    test_data["useless_column"] = ["should-not-be-here"]

    result = m.validate(test_data)
    assert_frame_equal(result, expect)


def test_dynamic_model_fields_extra_index(session_config, fake_session_factory):
    s = fake_session_factory(session_config)
    m = FakeModel.to_schema(session=s)

    assert type(m) == pa.DataFrameSchema

    test_data = test_frame.copy()
    test_data["useless_index"] = ["should-not-be-here"]
    test_data.set_index(["useless_index"], append=True, inplace=True)

    with pytest.raises(ValueError):
        m.validate(test_data)


def test_dynamic_model_fields_index_order(session_config, fake_session_factory):
    s = fake_session_factory(session_config)
    m = FakeModel.to_schema(session=s)

    assert type(m) == pa.DataFrameSchema

    test_data = test_frame.reset_index()
    test_data = test_data.set_index(["dataset_id", "dataset_gid", "iteration", "foo", "param0", "param1"])

    with pytest.raises(pa.errors.SchemaError):
        m.validate(test_data)


def test_dynamic_model_fields_column_order(session_config, fake_session_factory):
    s = fake_session_factory(session_config)
    m = FakeModel.to_schema(session=s)

    assert type(m) == pa.DataFrameSchema

    test_data = test_frame.reset_index()
    test_data = test_data.reindex(columns=["bar", "dataset_id", "param0", "foo", "param1", "dataset_gid", "iteration"])
    test_data = test_data.set_index(["dataset_id", "dataset_gid", "iteration", "param0", "param1", "foo"])

    result = m.validate(test_data)
    assert_frame_equal(result, expect)


def test_data_model_decorator_check_input(fake_task):
    """
    Test dataframe type check and coercion on arguments
    """
    test_data = test_frame.copy()

    result = fake_task.transform_noop(test_data)
    assert_frame_equal(result, expect)

    result = fake_task.transform_extra_args(test_data, 100)
    assert_frame_equal(result, expect)

    result = fake_task.transform_no_return(test_data)
    assert_frame_equal(result, expect)


def test_data_model_decorator_check_output(fake_task):
    """
    Test dataframe type check and coercion on the return path
    """
    test_data = test_frame.copy()

    result = fake_task.transform_return(test_data)
    assert_frame_equal(result, expect)


def test_data_model_decorator_check_tuple_output(fake_task):
    """
    Test dataframe type check and coercion on the return path for tuples/lists
    """
    test_data = test_frame.copy()

    def check(df):
        assert df.columns == expect.columns
        assert df.index == expect.index
        assert (df == expect).all().all()

    result = fake_task.transform_tuple_return(test_data)

    assert len(result) == 2
    check(result[0])
    check(result[1])

    result = fake_task.transform_list_return(test_data)

    assert len(result) == 2
    check(result[0])
    check(result[1])

    with pytest.raises(pa.errors.SchemaError):
        fake_task.transform_tuple_return_fail(test_data)

    with pytest.raises(pa.errors.SchemaError):
        fake_task.transform_list_return_fail(test_data)


def test_model_groupby_uuid_schema(fake_task):
    """
    Test derived dataframe schema generation.

    This checks that the schema index is modified correctly.
    """
    test_data = test_frame.groupby(["dataset_id"]).first()

    derived = FakeModel.as_groupby(["dataset_id"])
    schema = derived.to_schema(fake_task.session)

    result = schema.validate(test_data)

    assert result.columns == expect.columns
    assert result.index.names == ["dataset_id"]
    assert (result.index == expect.index.get_level_values("dataset_id")).all()
