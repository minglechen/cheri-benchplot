import io
import re
from pathlib import Path
from unittest.mock import ANY

import pytest

from pycheribenchplot.core.task import TaskScheduler
from pycheribenchplot.netperf.task import NetperfExecTask


@pytest.fixture
def netperf_config():
    return {
        "uuid":
        "17856370-2fd1-4597-937a-42b1277da44f",
        "name":
        "benchplot-selftest",
        "configurations": [{
            "name": "selftest0",
            "iterations": 2,
            "benchmark": {
                "handler": "netperf"
            },
            "command_hooks": {
                "pre-benchmark": ["sysctl hw.qemu_trace_perthread=1"]
            },
            "uuid": "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
            "g_uuid": "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
            "instance": {
                "kernel": "selftest-kernel",
                "baseline": True,
                "name": "selftest-instance",
                "cheribuild_kernel": False
            }
        }]
    }


@pytest.fixture
def netperf_session(snap_task_registry, fake_session_factory, netperf_config):
    session = fake_session_factory(netperf_config)
    return session


@pytest.fixture
def netperf_args_session(snap_task_registry, fake_session_factory, netperf_config):
    bench = netperf_config["configurations"][0]["benchmark"]
    # Add netperf options
    bench["task_options"] = {"netperf_options": ["--foo", "--bar"], "netserver_options": ["-x", "-y"]}
    session = fake_session_factory(netperf_config)
    return session


@pytest.fixture
def netperf_qemu_session(snap_task_registry, fake_session_factory, netperf_config):
    bench = netperf_config["configurations"][0]["benchmark"]
    # Add netperf options and qemu tracing
    bench["task_options"] = {"profile": {"qemu_trace": "perfetto", "qemu_trace_categories": ["instructions"]}}
    session = fake_session_factory(netperf_config)
    return session


@pytest.fixture
def mock_qemu_instance(mocker):
    mock_class = mocker.patch("pycheribenchplot.core.instance.QEMUInstance")
    return mock_class


@pytest.fixture
def mock_run_script(mocker):
    """
    Produces a StringIO buffer connected to the run-script output file.
    """
    mock_open = mocker.mock_open()
    mocker.patch("pycheribenchplot.core.benchmark.open", mock_open)
    iobuffer = io.StringIO()
    # Prevent closing on this StringIO
    mocker.patch.object(iobuffer, "close")

    # We still need to touch the output file because we try to do things with it,
    # instead of mocking everything just make it so the target exist.
    def side_effect(path, *args, **kwargs):
        Path(path).touch(exist_ok=False)
        return iobuffer

    mock_open.side_effect = side_effect
    return iobuffer


def check_lines_match(lines, expect_lines):
    last_checked = 0
    for expect in expect_lines:
        for i, line in enumerate(lines[last_checked:]):
            if re.match(expect, line):
                last_checked += i
                break
        else:
            pytest.fail(f"Lines do not match, last match was {lines[last_checked]}. lines:\n" + "\n".join(lines))


@pytest.mark.timeout(5)
def test_netperf_exec_task_default(netperf_session, mock_qemu_instance, mock_run_script):
    """
    Check netperf execution task handler for a default configuration with no netperf-specific options
    """
    netperf_session.run()

    assert not netperf_session.scheduler.failed_tasks

    # Verify the default run script
    expect_lines = [
        r"STATCOUNTERS_NO_AUTOSAMPLE=1 /opt/{instance.cheri_target}/netperf/bin/netserver &", r"PID_5=\$!",
        r"sysctl hw.qemu_trace_perthread=1",
        r"STATCOUNTERS_NO_AUTOSAMPLE=1 /opt/{instance.cheri_target}/netperf/bin/netperf >> /root/benchmark-output/0/netperf-exec-8bc941a3-f6d6-4d37-b193-4738f1da3dae.csv",
        r"STATCOUNTERS_NO_AUTOSAMPLE=1 /opt/{instance.cheri_target}/netperf/bin/netperf >> /root/benchmark-output/1/netperf-exec-8bc941a3-f6d6-4d37-b193-4738f1da3dae.csv",
        r"kill -TERM \$PID_5"
    ]
    check_lines_match(mock_run_script.getvalue().split("\n"), expect_lines)

    # Verify the instance configuration
    mock_qemu_instance.assert_called_once()
    instance_config = mock_qemu_instance.call_args.args[1]
    assert instance_config.platform_options.qemu_trace == "no"
    assert instance_config.platform_options.qemu_trace_file is None
    assert instance_config.platform_options.qemu_interceptor_trace_file is None
    assert instance_config.platform_options.qemu_trace_categories is None

    # Verify that we tried to extract the correct files
    qemu_instance_object = mock_qemu_instance.return_value
    assert qemu_instance_object.extract_file.call_count == 2
    qemu_instance_object.extract_file.assert_any_call(
        Path("/root/benchmark-output/0/netperf-exec-8bc941a3-f6d6-4d37-b193-4738f1da3dae.csv"), ANY)
    qemu_instance_object.extract_file.assert_any_call(
        Path("/root/benchmark-output/1/netperf-exec-8bc941a3-f6d6-4d37-b193-4738f1da3dae.csv"), ANY)


@pytest.mark.timeout(5)
def test_netperf_exec_task_args(netperf_args_session, mock_qemu_instance, mock_run_script):
    """
    Check netperf execution task handler for a configuration with some dummy netperf options
    """
    netperf_args_session.run()

    assert not netperf_args_session.scheduler.failed_tasks

    # Verify the default run script
    expect_lines = [
        r"STATCOUNTERS_NO_AUTOSAMPLE=1 /opt/{instance.cheri_target}/netperf/bin/netserver -x -y &", r"PID_5=\$!",
        r"sysctl hw.qemu_trace_perthread=1",
        r"STATCOUNTERS_NO_AUTOSAMPLE=1 /opt/{instance.cheri_target}/netperf/bin/netperf --foo --bar >> /root/benchmark-output/0/netperf-exec-8bc941a3-f6d6-4d37-b193-4738f1da3dae.csv",
        r"STATCOUNTERS_NO_AUTOSAMPLE=1 /opt/{instance.cheri_target}/netperf/bin/netperf --foo --bar >> /root/benchmark-output/1/netperf-exec-8bc941a3-f6d6-4d37-b193-4738f1da3dae.csv",
        r"kill -TERM \$PID_5"
    ]
    check_lines_match(mock_run_script.getvalue().split("\n"), expect_lines)

    # Verify the instance configuration
    mock_qemu_instance.assert_called_once()
    instance_config = mock_qemu_instance.call_args.args[1]
    assert instance_config.platform_options.qemu_trace == "no"
    assert instance_config.platform_options.qemu_trace_file is None
    assert instance_config.platform_options.qemu_interceptor_trace_file is None
    assert instance_config.platform_options.qemu_trace_categories is None

    # Verify that we tried to extract the correct files
    qemu_instance_object = mock_qemu_instance.return_value
    assert qemu_instance_object.extract_file.call_count == 2
    qemu_instance_object.extract_file.assert_any_call(
        Path("/root/benchmark-output/0/netperf-exec-8bc941a3-f6d6-4d37-b193-4738f1da3dae.csv"), ANY)
    qemu_instance_object.extract_file.assert_any_call(
        Path("/root/benchmark-output/1/netperf-exec-8bc941a3-f6d6-4d37-b193-4738f1da3dae.csv"), ANY)


@pytest.mark.timeout(5)
def test_netperf_exec_task_qemu(netperf_qemu_session, mock_qemu_instance, mock_run_script):
    """
    Check netperf execution task handler for a default configuration with no netperf-specific options
    """
    netperf_qemu_session.run()

    assert not netperf_qemu_session.scheduler.failed_tasks

    # Verify the default run script
    expect_lines = [
        r"STATCOUNTERS_NO_AUTOSAMPLE=1 /opt/{instance.cheri_target}/netperf/bin/netserver &", r"PID_5=\$!",
        r"sysctl hw.qemu_trace_perthread=1",
        r"STATCOUNTERS_NO_AUTOSAMPLE=1 /opt/{instance.cheri_target}/netperf/bin/netperf >> /root/benchmark-output/0/netperf-exec-8bc941a3-f6d6-4d37-b193-4738f1da3dae.csv",
        r"STATCOUNTERS_NO_AUTOSAMPLE=1 /opt/{instance.cheri_target}/netperf/bin/netperf >> /root/benchmark-output/1/netperf-exec-8bc941a3-f6d6-4d37-b193-4738f1da3dae.csv",
        r"kill -TERM \$PID_5"
    ]
    check_lines_match(mock_run_script.getvalue().split("\n"), expect_lines)

    # Verify the instance configuration
    mock_qemu_instance.assert_called_once()
    instance_config = mock_qemu_instance.call_args.args[1]
    assert instance_config.platform_options.qemu_trace == "perfetto"
    assert re.match(
        r".*run/selftest0-8bc941a3-f6d6-4d37-b193-4738f1da3dae/qemu-perfetto-8bc941a3-f6d6-4d37-b193-4738f1da3dae.pb$",
        str(instance_config.platform_options.qemu_trace_file))
    assert instance_config.platform_options.qemu_interceptor_trace_file is None
    assert instance_config.platform_options.qemu_trace_categories == {
        "instructions",
    }

    # Verify that we tried to extract the correct files
    qemu_instance_object = mock_qemu_instance.return_value
    assert qemu_instance_object.extract_file.call_count == 2
    qemu_instance_object.extract_file.assert_any_call(
        Path("/root/benchmark-output/0/netperf-exec-8bc941a3-f6d6-4d37-b193-4738f1da3dae.csv"), ANY)
    qemu_instance_object.extract_file.assert_any_call(
        Path("/root/benchmark-output/1/netperf-exec-8bc941a3-f6d6-4d37-b193-4738f1da3dae.csv"), ANY)
