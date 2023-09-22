from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pandera.errors import SchemaError
from pandera.typing import Index, Series

from pycheribenchplot.core.artefact import *
from pycheribenchplot.core.model import DataModel, GlobalModel
from pycheribenchplot.core.task import BenchmarkTask, SessionTask


class FakeModel(GlobalModel):
    index: Index[int] = pa.Field(check_name=True)
    value: Series[int]


class FakeBenchmarkModel(DataModel):
    index: Index[int]
    value: Series[int]


@pytest.fixture
def fake_task(mock_task_registry, fake_session):
    class FakeTask(SessionTask):
        task_namespace = "test"
        task_name = "fake-task"

    return FakeTask(fake_session)


@pytest.fixture
def fake_benchmark_task(mock_task_registry, fake_simple_benchmark):
    class FakeTask(BenchmarkTask):
        task_namespace = "test"
        task_name = "fake-task"

    return FakeTask(fake_simple_benchmark)


@pytest.fixture
def sample_content():
    """
    Produce some sample data to store in a test file that conforms to FakeModel
    """
    df = pd.DataFrame({"index": [1, 2, 3], "value": [10, 20, 30]})
    return df.set_index("index")


def test_dataframe_target(fake_task):
    test_df = pd.DataFrame({"index": [1, 2, 3], "value": [10, 20, 30]}).set_index("index")
    target = DataFrameTarget(fake_task, FakeModel)

    assert not target.is_file()

    target.assign(test_df)

    df = target.get()
    assert_frame_equal(df, test_df)

    target2 = DataFrameTarget(fake_task, FakeModel)
    df = target2.get()
    assert_frame_equal(df, test_df)


def test_dataframe_target_without_model(fake_task):
    test_df = pd.DataFrame({"index": [1, 2, 3], "value": [10, 20, 30]}).set_index("index")
    target = DataFrameTarget(fake_task, None)

    assert not target.is_file()

    target.assign(test_df)
    df = target.get()
    assert_frame_equal(df, test_df)


def test_dataframe_target_invalid(fake_task):
    invalid_df = pd.DataFrame({"invalid": [1, 2, 3], "value": ["foo", "bar", "baz"]}).set_index("invalid")
    target = DataFrameTarget(fake_task, FakeModel)

    with pytest.raises(SchemaError):
        target.assign(invalid_df)


def test_session_data_file_target(fake_task):
    with pytest.raises(TypeError):
        DataFileTarget(fake_task)


def test_session_local_file_target(fake_task):
    target = LocalFileTarget(fake_task, ext="foo")

    # assert target.borg_state_id == ""
    assert target.is_file()
    assert len(target.paths()) == 1
    assert target.path.parent == fake_task.session.get_data_root_path()
    assert target.path.name == f"test-fake-task-{fake_task.session.uuid}.foo"

    assert not target.needs_extraction()
    with pytest.raises(TypeError):
        target.remote_path
    with pytest.raises(TypeError):
        target.remote_paths()


def test_session_file_target_with_iterations(fake_task):
    target = LocalFileTarget(fake_task, use_iterations=True)

    with pytest.raises(NotImplementedError):
        target.paths()


def test_benchmark_file_target(fake_benchmark_task):
    target = DataFileTarget(fake_benchmark_task)

    assert target.is_file()
    assert len(target.paths()) == 1
    assert target.path.parent == fake_benchmark_task.benchmark.get_benchmark_data_path()
    assert target.path.name == f"test-fake-task-{fake_benchmark_task.benchmark.uuid}"

    assert target.needs_extraction()
    assert len(target.remote_paths()) == 1
    assert target.remote_path.parent == Path("/root/benchmark-output")
    assert target.remote_path.name == f"test-fake-task-{fake_benchmark_task.benchmark.uuid}"


def test_benchmark_file_target_with_iterations(fake_benchmark_task):
    fake_benchmark_task.benchmark.config.iterations = 2
    target = DataFileTarget(fake_benchmark_task, use_iterations=True)

    assert target.is_file()
    assert len(target.paths()) == 2
    with pytest.raises(ValueError):
        target.path
    expect = [
        fake_benchmark_task.benchmark.get_benchmark_iter_data_path(0),
        fake_benchmark_task.benchmark.get_benchmark_iter_data_path(1)
    ]
    for p, e in zip(target.paths(), expect):
        assert p.parent == e
        assert p.name == f"test-fake-task-{fake_benchmark_task.benchmark.uuid}"

    assert target.needs_extraction()
    assert len(target.remote_paths()) == 2
    with pytest.raises(ValueError):
        target.remote_path
    expect = [Path("/root/benchmark-output/0"), Path("/root/benchmark-output/1")]
    for p, e in zip(target.remote_paths(), expect):
        assert p.parent == e
        assert p.name == f"test-fake-task-{fake_benchmark_task.benchmark.uuid}"


def test_benchmark_file_target_with_prefix(fake_benchmark_task):
    target = DataFileTarget(fake_benchmark_task, prefix="myprefix")

    assert target.is_file()
    assert len(target.paths()) == 1
    assert target.path.parent == fake_benchmark_task.benchmark.get_benchmark_data_path()
    assert target.path.name == f"myprefix-test-fake-task-{fake_benchmark_task.benchmark.uuid}"

    assert target.needs_extraction()
    assert len(target.remote_paths()) == 1
    assert target.remote_path.parent == Path("/root/benchmark-output")
    assert target.remote_path.name == f"myprefix-test-fake-task-{fake_benchmark_task.benchmark.uuid}"


def test_file_target_loader(fake_task, sample_content):
    test_session = fake_task.session.uuid
    # Expect the content to be here
    sample_content.to_csv(fake_task.session.get_data_root_path() / f"test-fake-task-{test_session}.csv")

    target = LocalFileTarget(fake_task, model=FakeModel, ext="csv")

    loader = target.get_loader()
    assert loader.is_session_task()
    assert loader.task_id == f"internal.target-session-load-{test_session}-for-test.fake-task-{test_session}-output"

    loader.run()
    assert_frame_equal(loader.df.get(), sample_content)


def test_benchmark_file_target_loader(fake_benchmark_task, sample_content):
    test_benchmark = fake_benchmark_task.benchmark.uuid
    # Expect the content to be here
    fake_benchmark_task.benchmark.get_benchmark_data_path().mkdir(exist_ok=True)
    sample_content.to_csv(fake_benchmark_task.benchmark.get_benchmark_data_path() /
                          f"test-fake-task-{test_benchmark}.csv")
    # Expect that the loaded data will have the dataset_id, dataset_gid and iteration index levels
    expect_content = sample_content.reset_index()
    expect_content["dataset_id"] = fake_benchmark_task.benchmark.uuid
    expect_content["dataset_gid"] = fake_benchmark_task.benchmark.g_uuid
    expect_content["iteration"] = -1
    expect_content["iteration"] = expect_content["iteration"].astype(pd.Int64Dtype())
    expect_content = expect_content.set_index(["dataset_id", "dataset_gid", "iteration", "index"])

    target = DataFileTarget(fake_benchmark_task, model=FakeBenchmarkModel, ext="csv")

    loader = target.get_loader()
    assert loader.is_benchmark_task()
    assert loader.task_id == f"internal.target-load-{test_benchmark}-for-test.fake-task-{test_benchmark}-output"

    loader.run()
    assert_frame_equal(loader.df.get(), expect_content)
