import copy
import io
import uuid
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fixtures.task_helpers import mock_task_registry, snap_task_registry

from pycheribenchplot.core.benchmark import Benchmark
from pycheribenchplot.core.config import (AnalysisConfig, BenchmarkRunConfig, BenchplotUserConfig, SessionRunConfig)
from pycheribenchplot.core.session import Session
from pycheribenchplot.core.task import ExecutionTask

fake_benchmark_conf = {
    "name": "selftest0",
    "iterations": 2,
    "generators": [{
        "handler": "test-benchmark"
    }],
    "uuid": "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
    "g_uuid": "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
    "instance": {
        "kernel": "selftest-kernel",
        "baseline": True,
        "name": "selftest-instance",
        "cheribuild_kernel": False
    }
}

fake_session_conf = {
    "uuid": "17856370-2fd1-4597-937a-42b1277da44f",
    "name": "benchplot-selftest",
    "configurations": [fake_benchmark_conf]
}

fake_session_conf_with_params = {
    "uuid":
    "17856370-2fd1-4597-937a-42b1277da440",
    "name":
    "benchplot-selftest-model",
    "configurations": [{
        "name": "selftest0",
        "iterations": 2,
        "parameters": {
            "param0": 10,
            "param1": "param1-value"
        },
        "generators": [{
            "handler": "test-benchmark"
        }],
        "uuid": "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
        "g_uuid": "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
        "instance": {
            "kernel": "selftest-kernel",
            "baseline": True,
            "name": "selftest-instance"
        }
    }]
}


class FakeExecTask(ExecutionTask):
    public = True
    task_namespace = "test-benchmark"


@pytest.fixture
def empty_session_config():
    """
    Produces a sample skeleton for a valid session configuration.
    """
    conf = {"uuid": "17856370-2fd1-4597-937a-42b1277da44f", "name": "benchplot-selftest", "configurations": []}
    return conf


@pytest.fixture
def single_benchmark_config(empty_session_config):
    """
    Produces a session configuration with a single benchmark entry.
    No parameterization.
    """
    conf = copy.deepcopy(empty_session_config)
    conf["configurations"].append({
        "name": "selftest0",
        "iterations": 2,
        "generators": [{
            "handler": "test-benchmark"
        }],
        "uuid": "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
        "g_uuid": "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
        "instance": {
            "kernel": "selftest-kernel",
            "baseline": True,
            "name": "selftest-instance",
            "cheribuild_kernel": False
        }
    })
    return conf


@pytest.fixture
def multi_benchmark_config(empty_session_config):
    """
    Produces a session configuration with 2 parameterized benchmark entries.
    """
    conf = copy.deepcopy(empty_session_config)
    conf["configurations"].append({
        "name": "selftest0",
        "iterations": 2,
        "parameters": {
            "param0": "param0-value0"
        },
        "generators": [{
            "handler": "test-benchmark"
        }],
        "uuid": "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
        "g_uuid": "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
        "instance": {
            "kernel": "selftest-kernel",
            "baseline": True,
            "name": "selftest-instance",
            "cheribuild_kernel": False
        }
    })
    conf["configurations"].append({
        "name": "selftest0",
        "iterations": 2,
        "parameters": {
            "param0": "param0-value1"
        },
        "generators": [{
            "handler": "test-benchmark"
        }],
        "uuid": "c73169b7-5797-41c8-9edc-656d666cb45a",
        "g_uuid": "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
        "instance": {
            "kernel": "selftest-kernel",
            "baseline": True,
            "name": "selftest-instance",
            "cheribuild_kernel": False
        }
    })
    return conf


@pytest.fixture
def fullmatrix_benchmark_config(multi_benchmark_config):
    """
    Produces a session configuration with a 2x2 benchmark matrix.
    This will have 2 parameterized runs for each of the 2 machine configurations ID.
    """
    conf = copy.deepcopy(multi_benchmark_config)
    conf["configurations"].append({
        "name": "selftest0",
        "iterations": 2,
        "parameters": {
            "param0": "param0-value0"
        },
        "generators": [{
            "handler": "test-benchmark"
        }],
        "uuid": "05a4ca0d-c659-4f40-bb18-6f8ead5d2ec3",
        "g_uuid": "4995a8b2-4852-4310-9b34-26cbd28494f0",
        "instance": {
            "kernel": "selftest-kernel",
            "baseline": False,
            "name": "selftest-instance",
            "cheribuild_kernel": False
        }
    })
    conf["configurations"].append({
        "name": "selftest0",
        "iterations": 2,
        "parameters": {
            "param0": "param0-value1"
        },
        "generators": [{
            "handler": "test-benchmark"
        }],
        "uuid": "f011e12b-75ef-4174-ba38-2795c2ca1e30",
        "g_uuid": "4995a8b2-4852-4310-9b34-26cbd28494f0",
        "instance": {
            "kernel": "selftest-kernel",
            "baseline": False,
            "name": "selftest-instance",
            "cheribuild_kernel": False
        }
    })
    return conf


@pytest.fixture
def benchplot_user_config(pytestconfig):
    user_config = BenchplotUserConfig.load_json(pytestconfig.getoption("--benchplot-user-config"))
    return user_config


@pytest.fixture
def fake_session_factory(tmp_path):
    cleanup = []

    def factory(config: dict = None, user_config: BenchplotUserConfig = None):
        if config is None:
            config = fake_session_conf
        sess_config = SessionRunConfig.schema().load(config)
        if user_config is None:
            user_config = BenchplotUserConfig()
        session = Session(user_config, sess_config, session_path=tmp_path)
        session.analysis_config = AnalysisConfig()
        cleanup.append(session)
        return session

    yield factory

    for s in cleanup:
        s.clean_all()


@pytest.fixture
def fake_benchmark_factory(fake_session):
    def factory(config: dict = None, randomize_uuid: bool = False):
        if config is None:
            config = fake_benchmark_conf
        bench_config = BenchmarkRunConfig.schema().load(config)
        if randomize_uuid:
            bench_config.uuid = uuid.uuid4()
        return Benchmark(fake_session, bench_config)

    return factory


@pytest.fixture
def fake_session(fake_session_factory):
    """
    A fake session instance with the default user configuration.
    This uses an empty session configuration.
    """
    return fake_session_factory()


@pytest.fixture
def fake_session_with_params(fake_session_factory):
    """
    Create a fake session instance with a single parameterized benchmark configuration.
    """
    return fake_session_factory(fake_session_conf_with_params)


@pytest.fixture
def fake_simple_benchmark(fake_benchmark_factory):
    """
    A fake benchmark instance with the default user configuration
    """
    return fake_benchmark_factory(fake_benchmark_conf)


@pytest.fixture
def fake_existing_path():
    path = Path("fake/output/file")
    mock_path = MagicMock(wraps=path)
    mock_path.exists.return_value = True
    mock_path.is_file.return_value = True
    mock_path.is_dir.return_value = False
    return mock_path
