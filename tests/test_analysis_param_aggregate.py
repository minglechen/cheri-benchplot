import numpy as np
import pandas as pd
import pytest
from pandera import Field
from pandera.extensions import register_check_method
from pandera.typing import Series

from pycheribenchplot.core.analysis import (BenchmarkDataLoadTask, StatsByParamGroupTask)
from pycheribenchplot.core.config import AnalysisConfig
from pycheribenchplot.core.model import DataModel, ParamGroupDataModel
from pycheribenchplot.core.task import AnalysisTask, DataFrameTarget


class DummyModel(DataModel):
    count: Series[int]


class DummyStatsModel(ParamGroupDataModel):
    count: Series[float] = Field(alias=("count", "mean|std|median|q25|q75", "sample|delta|norm_delta"),
                                 regex=True,
                                 nullable=True)


@register_check_method(statistics=["sequence", "distance"])
def approx_oneof(series, *, sequence, distance):
    center = np.array(sequence)
    return (center - distance < series) & (center + distance > series)


class ValidateStats(ParamGroupDataModel):
    """
    Schema to validate the stats dataframe output.
    This should be the DummyStatsModel but we call it explicitly to make sure that
    the task actually does the validation.
    """
    count_mean_sample: Series[float] = Field(alias=("count", "mean", "sample"),
                                             approx_oneof=dict(sequence=[10, 100], distance=1))

    count_mean_delta: Series[float] = Field(alias=("count", "mean", "delta"),
                                            approx_oneof=dict(sequence=[0, 90], distance=1))

    count_mean_ndelta: Series[float] = Field(alias=("count", "mean", "norm_delta"),
                                             nullable=True,
                                             approx_oneof=dict(sequence=[0, 9], distance=.8))
    count_std_sample: Series[float] = Field(alias=("count", "std", "sample"),
                                            approx_oneof=dict(sequence=[1, 1], distance=.2))
    count_std_delta: Series[float] = Field(alias=("count", "std", "delta"),
                                           approx_oneof=dict(sequence=[1.4, 1.4], distance=.5))
    count_std_ndelta: Series[float] = Field(alias=("count", "std", "norm_delta"),
                                            nullable=True,
                                            approx_oneof=dict(sequence=[np.nan, 1], distance=.2))
    count_median_sample: Series[float] = Field(alias=("count", "median", "sample"),
                                               approx_oneof=dict(sequence=[10, 100], distance=2))
    count_median_delta: Series[float] = Field(alias=("count", "median", "delta"),
                                              approx_oneof=dict(sequence=[0, 90], distance=2))
    count_median_ndelta: Series[float] = Field(alias=("count", "median", "norm_delta"),
                                               nullable=True,
                                               approx_oneof=dict(sequence=[0, 9], distance=2))
    count_q25_sample: Series[float] = Field(alias=("count", "q25", "sample"))
    count_q25_delta: Series[float] = Field(alias=("count", "q25", "delta"))
    count_q25_ndelta: Series[float] = Field(alias=("count", "q25", "norm_delta"), nullable=True)
    count_q75_sample: Series[float] = Field(alias=("count", "q75", "sample"))
    count_q75_delta: Series[float] = Field(alias=("count", "q75", "delta"))
    count_q75_ndelta: Series[float] = Field(alias=("count", "q75", "norm_delta"), nullable=True)


@pytest.fixture
def register_tasks(mock_task_registry):
    """
    Helper to register the dummy tasks using a fake task registry.
    """
    class DummyLoadTask(BenchmarkDataLoadTask):
        """
        A fake load task. This is never run, we just need it to verify dependency
        generation.
        """
        task_namespace = "test"
        task_name = "fake-load-task"
        model = DummyModel

        def run(self):
            return

    class DummyParamGroupTask(StatsByParamGroupTask):
        """
        Note this is marked public for simplicity but would fail to be instantiated
        by the session analysis entry point unless we override __init__().
        """
        public = True
        task_namespace = "test"
        task_name = "fake-group-task"
        load_task = DummyLoadTask
        model = DummyStatsModel

    return DummyLoadTask, DummyParamGroupTask


@pytest.fixture
def simple_session(register_tasks, multi_benchmark_config, fake_session_factory):
    """
    Session built using the default 2 benchmark run on the same machine ID with a parameter level.
    No need to modify the handler here as we stub the load task logic.
    """
    return fake_session_factory(multi_benchmark_config)


@pytest.fixture
def full_session(register_tasks, fullmatrix_benchmark_config, fake_session_factory):
    """
    Session built using the standard test 2x2 benchmark matrix.
    No need to modify the handler here as we stub the load task logic.
    """
    return fake_session_factory(fullmatrix_benchmark_config)


def test_param_group_task_simple_deps(register_tasks, simple_session):
    """
    Check that the statistics aggregation task generates the expected dependencies.
    UUIDs are from the common standard session fixtures.
    """
    DummyLoadTask, DummyParamGroupTask = register_tasks

    aconf = AnalysisConfig.schema().load({"handlers": [{"handler": "test.fake-group-task"}]})
    task = DummyParamGroupTask(simple_session, aconf, {"param0": "param0-value1"})

    deps = list(task.dependencies())

    assert len(deps) == 1
    assert type(deps[0]) == DummyLoadTask
    assert deps[0].task_id == "test.fake-load-task-c73169b7-5797-41c8-9edc-656d666cb45a"


def test_param_group_task_full_deps(register_tasks, full_session):
    """
    Check that the statistics aggregation task generates the expected dependencies.
    UUIDs are from the common standard session fixtures.
    """
    DummyLoadTask, DummyParamGroupTask = register_tasks

    aconf = AnalysisConfig.schema().load({"handlers": [{"handler": "test.fake-group-task"}]})
    task = DummyParamGroupTask(full_session, aconf, {"param0": "param0-value1"})

    deps = list(task.dependencies())

    assert len(deps) == 2
    assert type(deps[0]) == DummyLoadTask
    assert type(deps[1]) == DummyLoadTask
    dep_ids = [t.task_id for t in deps]
    assert "test.fake-load-task-c73169b7-5797-41c8-9edc-656d666cb45a" in dep_ids
    assert "test.fake-load-task-f011e12b-75ef-4174-ba38-2795c2ca1e30" in dep_ids


def test_param_group_simple_run(mocker, register_tasks, full_session):
    """
    Verify that the statistics aggregation task generates sensible results
    """
    DummyLoadTask, DummyParamGroupTask = register_tasks

    aconf = AnalysisConfig.schema().load({"handlers": [{"handler": "test.fake-group-task"}]})
    task = DummyParamGroupTask(full_session, aconf, {"param0": "param0-value1"})

    # XXX this would be nicer if we could use hypothesis.given()
    # input_df0 = DummyModel.strategy(full_session, override_constraints={}, size=5)
    N = 50
    input_df0 = pd.DataFrame({
        "dataset_id": ["c73169b7-5797-41c8-9edc-656d666cb45a"] * N,
        "dataset_gid": ["2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb"] * N,
        "iteration": list(range(N)),
        "param0": ["param0-value1"] * N,
        "count": np.random.normal(loc=10, size=N)
    }).set_index(["dataset_id", "dataset_gid", "iteration", "param0"])
    target0 = DataFrameTarget(DummyModel.to_schema(full_session))
    target0.assign(input_df0)
    input_df1 = pd.DataFrame({
        "dataset_id": ["f011e12b-75ef-4174-ba38-2795c2ca1e30"] * N,
        "dataset_gid": ["4995a8b2-4852-4310-9b34-26cbd28494f0"] * N,
        "iteration": list(range(N)),
        "param0": ["param0-value1"] * N,
        "count": np.random.normal(loc=100, size=N),
    }).set_index(["dataset_id", "dataset_gid", "iteration", "param0"])
    target1 = DataFrameTarget(DummyModel.to_schema(full_session))
    target1.assign(input_df1)

    # Simulate computed dependencies
    task_load_0 = mocker.Mock(spec=DummyLoadTask)
    task_load_0.completed = mocker.PropertyMock(return_value=True)
    task_load_0.outputs.return_value = [("df", target0)]
    task_load_0.output_map = {"df": target0}
    task_load_1 = mocker.Mock(spec=DummyLoadTask)
    task_load_1.completed = mocker.PropertyMock(return_value=True)
    task_load_1.outputs.return_value = [("df", target1)]
    task_load_1.output_map = {"df": target1}
    task.resolved_dependencies = {task_load_0, task_load_1}

    task.run()

    # Given that we used two normal distributions with sigma=1 we can predict the
    # resulting std_delta and std_norm_delta
    result = task.output_map["df"].get()

    schema = ValidateStats.to_schema(full_session)
    validated = schema.validate(result)
    pd.testing.assert_frame_equal(validated, result)
