import copy
from collections import defaultdict

import pytest

from pycheribenchplot.core.borg import Borg
from pycheribenchplot.core.task import (ExecutionTask, TaskRegistry, TaskScheduler)


def task_registry_snapshot(mocker):
    public_tasks = defaultdict(dict)
    all_tasks = defaultdict(dict)
    fake_borg_registry = {}
    mock_public_tasks = mocker.patch.object(TaskRegistry, "public_tasks", new=public_tasks)
    mock_all_tasks = mocker.patch.object(TaskRegistry, "all_tasks", new=all_tasks)
    mocker.patch.object(Borg, "_borg_registry", new=fake_borg_registry)

    return (public_tasks, all_tasks)


@pytest.fixture
def mock_task_registry(mocker):
    """
    Mock the task registry lists so that the tasks that are declared within the test
    are guaranteed to never conflict with other tests.

    The borg task registry must also be mocked temporarily to avoid unintentional
    collision between test tasks.
    """
    public_tasks, all_tasks = task_registry_snapshot(mocker)

    # Always define at least this as it is used by the common session fixtures
    class FakeExecTask(ExecutionTask):
        public = True
        task_namespace = "test-benchmark"

    return (public_tasks, all_tasks)


@pytest.fixture
def snap_task_registry(mocker):
    """
    Mock the task registry so that any updates are reverted after the benchmark run,
    but preserve the tasks that have been registered by benchplot.
    """
    old_registry_public = copy.deepcopy(TaskRegistry.public_tasks)
    old_registry_all = copy.deepcopy(TaskRegistry.all_tasks)
    public_tasks, all_tasks = task_registry_snapshot(mocker)

    public_tasks.update(old_registry_public)
    all_tasks.update(old_registry_all)

    return (public_tasks, all_tasks)
