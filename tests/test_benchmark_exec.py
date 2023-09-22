import copy

import pytest

from conftest import FakeExecTask
from pycheribenchplot.core.benchmark import BenchmarkExecTask, ExecTaskConfig
from pycheribenchplot.core.task import ExecutionTask


@pytest.fixture
def benchmark_config(single_benchmark_config):
    conf = copy.deepcopy(single_benchmark_config)
    bench_conf = conf["configurations"][0]
    bench_conf["generators"] = [
        {
            "handler": "test-gen-1.exec"
        },
        {
            "handler": "test-gen-2.exec"
        },
    ]
    return conf


def test_benchmark_task_aux_deps(mock_task_registry, benchmark_config, fake_benchmark_factory):
    """
    Check that the benchmark exec task generates the expected dependencies
    """
    class FakeExecTask1(ExecutionTask):
        public = True
        task_namespace = "test-gen-1"

    class FakeExecTask2(ExecutionTask):
        public = True
        task_namespace = "test-gen-2"

    bench = fake_benchmark_factory(benchmark_config["configurations"][0])
    task = BenchmarkExecTask(bench, task_config=ExecTaskConfig())

    deps = list(task.dependencies())
    # Expect only the main test-benchmark.exec task and the instance-boot task
    assert len(deps) == 2
    assert [d for d in deps if isinstance(d, FakeExecTask1)]
    assert [d for d in deps if isinstance(d, FakeExecTask2)]
