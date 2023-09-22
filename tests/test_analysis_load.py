import csv

import pandas as pd
import pytest
from pandera.typing import Index, Series

from pycheribenchplot.core.analysis import AnalysisTask, BenchmarkDataLoadTask
from pycheribenchplot.core.artefact import DataFrameTarget, LocalFileTarget
from pycheribenchplot.core.config import AnalysisConfig
from pycheribenchplot.core.model import DataModel
from pycheribenchplot.core.task import ExecutionTask


class DummyExecTask(ExecutionTask):
    public = True
    task_namespace = "test.analysis-load"

    def run(self):
        pass

    def outputs(self):
        yield "test-target", LocalFileTarget(self, file_path="fake-path.csv")


class DummyModel(DataModel):
    number: Index[int]
    name: Series[str]
    surname: Series[str]


class DummyLoadTask(BenchmarkDataLoadTask):
    task_namespace = "test.analysis-load"
    task_name = "fake-load-task"
    exec_task = DummyExecTask
    target_key = "test-target"
    model = DummyModel


class DummyAnalysisRoot(AnalysisTask):
    """
    This is needed for testing as the load tasks are never meant to be public,
    so we need a top-level task with the correct __init__ signature that can be
    instantiated by the session analysis entry point.
    """
    public = True
    task_namespace = "test.analysis-load"
    task_name = "fake-analysis-main"

    def dependencies(self):
        yield DummyLoadTask(self.session.benchmark_matrix.iloc[0, 0], self.analysis_config)

    def run(self):
        return


@pytest.fixture
def csv_file_content(fake_analysis_session):
    # Ensure that the loader will find the csv file where expected
    task = DummyExecTask(fake_analysis_session.benchmark_matrix.iloc[0, 0], None)
    output = task.output_map["test-target"]
    output.path.parent.mkdir(exist_ok=True)
    with open(output.path, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["number", "name", "surname"])
        writer.writerow([0, "Dennis", "Ritchie"])
        writer.writerow([1, "Bjarne", "Stroustroup"])
        writer.writerow([2, "Graydon", "Hoare"])
        writer.writerow([3, "Claude", "Shannon"])
    yield output.path
    output.path.unlink()


@pytest.fixture
def session_config_with_param(single_benchmark_config):
    """
    Modify the single_benchmark_config to add a parameter key
    """
    conf = single_benchmark_config["configurations"][0]
    conf["parameters"] = {"param_key": "param-value"}
    conf["generators"] = [{"handler": "test.analysis-load.exec"}]
    return single_benchmark_config


@pytest.fixture
def fake_analysis_session(session_config_with_param, fake_session_factory):
    session = fake_session_factory(session_config_with_param)
    return session


def test_simple_load_task(csv_file_content, fake_analysis_session):
    """
    Test loading data from a simple CSV file.
    This exercises the common dataframe load task with data validation.
    """
    aconf = AnalysisConfig.schema().load({"tasks": [{"handler": "test.analysis-load.fake-analysis-main"}]})
    fake_analysis_session.analyse(aconf)

    assert not fake_analysis_session.scheduler.failed_tasks
    # Now check that we produced the dataframe output we are looking for
    tasks = fake_analysis_session.scheduler.completed_tasks

    # benchmark UUID is appended for task ID, see conftest single_benchmark_config
    expect_id = "test.analysis-load.fake-load-task-8bc941a3-f6d6-4d37-b193-4738f1da3dae"
    assert expect_id in tasks
    load_task = tasks[expect_id]
    target = load_task.output_map["df"]
    df = target.get()
    assert isinstance(target, DataFrameTarget)
    assert df.index.names == ["dataset_id", "dataset_gid", "iteration", "param_key", "number"]
    assert (df.columns == ["name", "surname"]).all()
    assert len(df) == 4
