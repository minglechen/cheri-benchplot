import io
import logging
import os
import signal
from queue import Queue

import pytest
import pytest_cases

from pycheribenchplot.core.config import InstanceConfig, PlatformOptions
from pycheribenchplot.core.instance import (HostCommandRunner, Instance, InstanceManager, InstancePlatform,
                                            InstanceState, Msg, MsgType, QEMUInstance)


class DummyInstance(Instance):
    def _boot(self):
        # Immediately signal boot done
        self._signal_state(InstanceState.SETUP)

    def _shutdown(self):
        pass

    def _ssh_connect(self):
        pass


@pytest.fixture
def manager(pytestconfig, mocker, fake_session):
    if hasattr(pytestconfig.option, "quiet"):
        logger = logging.getLogger("cheri-benchplot")
        logger.setLevel(logging.CRITICAL)
    mock_resolve_instance = mocker.patch.object(InstanceManager, "_find_instance_type")
    mock_resolve_instance.return_value = DummyInstance
    manager = InstanceManager(fake_session, limit=1)
    yield manager
    manager.sched_shutdown()


@pytest.fixture
def fake_qemu_instance_config():
    conf = InstanceConfig(kernel="selftest-kernel")
    conf.platform_options = PlatformOptions()
    conf.platform = InstancePlatform.QEMU
    return conf


@pytest.fixture
def instance_request(fake_session):
    """
    Build a valid instance request
    """
    bench = fake_session.benchmark_matrix.iloc[0, 0]
    pool_uuid = bench.g_uuid
    req = InstanceManager.request(pool_uuid, instance_config=bench.config.instance)
    return req


@pytest.fixture
def pipes():
    """
    Build two pairs of pipes for stdout and stderr
    """
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    f = {
        "out": (os.fdopen(out_r, "rb", buffering=0), os.fdopen(out_w, "wb", buffering=0)),
        "err": (os.fdopen(err_r, "rb", buffering=0), os.fdopen(err_w, "wb", buffering=0))
    }
    yield f
    if not f["out"][0].closed:
        f["out"][0].close()
    if not f["out"][1].closed:
        f["out"][1].close()
    if not f["err"][0].closed:
        f["err"][0].close()
    if not f["err"][1].closed:
        f["err"][1].close()


@pytest_cases.fixture(unpack_into="qemu_instance, mock_host_command, mock_ssh_command, mock_ssh_client")
def qemu_instance_mocks(mocker, fake_qemu_instance_config):
    mock_manager = mocker.MagicMock()
    mock_host_command = mocker.patch("pycheribenchplot.core.instance.HostCommandRunner")
    mock_ssh_command = mocker.patch("pycheribenchplot.core.instance.SSHCommandRunner")
    mock_ssh_client = mocker.patch("pycheribenchplot.core.instance.SSHClient")
    # Just avoid waiting during tests
    mock_time_sleep = mocker.patch("pycheribenchplot.core.instance.time.sleep")

    # Ensure that SSHClient.exec_command returns a 3-tuple
    mock_ssh_client.return_value.exec_command.return_value = (mocker.Mock(), mocker.Mock(), mocker.Mock())

    instance = QEMUInstance(mock_manager, fake_qemu_instance_config)
    yield instance, mock_host_command.return_value, mock_ssh_command.return_value, mock_ssh_client.return_value
    try:
        # This may fail if we already shut down
        instance.signal_shutdown()
    except Exception as ex:
        pass
    instance.wait_dead()


def test_instance_manager_invalid_request(manager, fake_session):
    """
    Test that we do not allow malformed instance requests
    """
    bench = fake_session.benchmark_matrix.iloc[0, 0]

    with pytest.raises(ValueError, match=r"[iI]nvalid resource"):
        req = InstanceManager.request("invalid-uuid", instance_config=bench.config.instance)
        with manager.acquire(req) as i:
            assert False

    with pytest.raises(ValueError, match=r"[iI]nvalid resource"):
        req = InstanceManager.request(bench.g_uuid)
        with manager.acquire(req) as i:
            assert False


@pytest.mark.timeout(5)
def test_instance_lifecycle(manager, instance_request):
    """
    Test the lifecycle of a noop instance to verify common
    functionality in the instance implementation
    """
    with manager.acquire(instance_request) as instance:
        assert instance.state == InstanceState.READY
    # In theory it would be nice to see it go to shutdown or idle queue here


@pytest.mark.timeout(5)
def test_instance_failure_lifecycle(manager, instance_request):
    """
    Test the lifecycle of a noop instance to verify common
    functionality in the instance implementation
    """
    with pytest.raises(Exception, match=r"DEBUG ERROR"):
        with manager.acquire(instance_request) as instance:
            raise Exception("DEBUG ERROR")


@pytest.mark.timeout(5)
def test_instance_exec_cmd(mocker, fake_session, pipes):
    """
    Test the command runner execution helper.
    We fake a popen object with two os pipes to simulate stdout/err.
    We should see the expected set of messages in the given queue
    """
    mock_popen = mocker.patch("subprocess.Popen")
    mock_process = mocker.Mock()
    mock_process.stdout = pipes["out"][0]
    mock_process.stderr = pipes["err"][0]
    mock_process.pid = os.getpid()
    mock_process.args = ["fake-command", "arg0", "arg1"]
    mock_process.returncode = None
    mock_popen.return_value = mock_process
    msg_queue = Queue()
    stdout = pipes["out"][1]
    stderr = pipes["err"][1]

    cmd = HostCommandRunner(fake_session.logger, msg_queue)
    cmd.run("fake-command", ["arg0", "arg1"])

    stdout.write("stdout-test-1\n".encode("utf-8"))
    stdout.write("stdout-test-2\n".encode("utf-8"))
    stderr.write("stderr-test-1\n".encode("utf-8"))
    stderr.write("stderr-test-2\n".encode("utf-8"))
    stdout.close()
    stderr.close()
    mock_process.returncode = 0

    cmd.wait()

    msg_found = []
    while not msg_queue.empty():
        msg_found.append(msg_queue.get_nowait())
    assert len(msg_found) == 5
    assert Msg(MsgType.OUT, "fake-command", "stdout-test-1") in msg_found
    assert Msg(MsgType.OUT, "fake-command", "stdout-test-2") in msg_found
    assert Msg(MsgType.ERR, "fake-command", "stderr-test-1") in msg_found
    assert Msg(MsgType.ERR, "fake-command", "stderr-test-2") in msg_found
    assert Msg(MsgType.EXITED, "fake-command", cmd) in msg_found
    # We expect stop to not have called send_signal() because the process exited cleanly
    mock_process.send_signal.assert_not_called()
    mock_process.wait.assert_called()


@pytest.mark.timeout(5)
def test_instance_exec_cmd_broken_lines(mocker, fake_session, pipes):
    """
    Test the command runner execution helper.
    We fake a popen object with two os pipes to simulate stdout/err,
    this time we provide simulated output that does not respect line boundaries
    """
    mock_popen = mocker.patch("subprocess.Popen")
    mock_process = mocker.Mock()
    mock_process.stdout = pipes["out"][0]
    mock_process.stderr = pipes["err"][0]
    mock_process.pid = os.getpid()
    mock_process.args = ["fake-command", "arg0", "arg1"]
    mock_process.returncode = None
    mock_popen.return_value = mock_process
    msg_queue = Queue()
    stdout = pipes["out"][1]
    stderr = pipes["err"][1]

    cmd = HostCommandRunner(fake_session.logger, msg_queue)
    cmd.run("fake-command", ["arg0", "arg1"])

    stdout.write("stdout-test-1\nstdout-".encode("utf-8"))
    stderr.write("stderr-test".encode("utf-8"))
    stdout.write("test-2\n".encode("utf-8"))
    stderr.write("-1\nstderr-test-2\n".encode("utf-8"))
    stdout.close()
    stderr.close()
    mock_process.returncode = 0

    cmd.wait()

    msg_found = []
    while not msg_queue.empty():
        msg_found.append(msg_queue.get_nowait())
    assert len(msg_found) == 5
    assert Msg(MsgType.OUT, "fake-command", "stdout-test-1") in msg_found
    assert Msg(MsgType.OUT, "fake-command", "stdout-test-2") in msg_found
    assert Msg(MsgType.ERR, "fake-command", "stderr-test-1") in msg_found
    assert Msg(MsgType.ERR, "fake-command", "stderr-test-2") in msg_found
    assert Msg(MsgType.EXITED, "fake-command", cmd) in msg_found
    # We expect stop to not have called send_signal() because the process exited cleanly
    mock_process.send_signal.assert_not_called()
    mock_process.wait.assert_called()


@pytest.mark.timeout(5)
def test_qemu_instance_exec(qemu_instance, mock_host_command, mock_ssh_command, mock_ssh_client):
    """
    Simulate a qemu instance run.
    """
    # instance = QEMUInstance(mock_manager, fake_qemu_instance_config)
    qemu_instance.signal_boot()
    # The host command would trigger a transition to SETUP
    qemu_instance._signal_state(InstanceState.SETUP)
    qemu_instance.wait_ready()
    # Check the first interactions, the host run should run before the setup is issued but
    # it's hard to wait for Instance._boot() to run. This is close enough
    mock_host_command.run.assert_called_once()
    mock_ssh_client.connect.assert_called_once()

    qemu_instance.run_cmd("test-cmd", ["arg0", "arg1"])
    mock_ssh_command.run.assert_called_once_with("test-cmd", ["arg0", "arg1"], None)
    mock_ssh_command.wait.assert_called_once()
    # The command runner for cheribuild would signal exit at some point
    qemu_instance.signal_shutdown()
    qemu_instance.wait_dead()
    # Expect the clean shutdown path
    mock_host_command.stop.assert_not_called()
    mock_ssh_client.exec_command.assert_called_once_with("poweroff")


@pytest.mark.timeout(5)
def test_qemu_instance_exec_stop_ssh(qemu_instance, mock_host_command, mock_ssh_command, mock_ssh_client):
    """
    Simulate a qemu instance run that is aborted while running an ssh command
    """
    qemu_instance.signal_boot()
    # The host command would trigger a transition to SETUP
    qemu_instance._signal_state(InstanceState.SETUP)
    qemu_instance.wait_ready()
    # Check the first interactions, the host run should run before the setup is issued but
    # it's hard to wait for Instance._boot() to run. This is close enough
    mock_host_command.run.assert_called_once()
    mock_ssh_client.connect.assert_called_once()

    qemu_instance._active_commands.append(mock_ssh_command)
    # The command runner for cheribuild would signal exit at some point
    qemu_instance.signal_shutdown()
    qemu_instance.wait_dead()

    # We expect to see a cancellation request to the remote command
    mock_ssh_command.stop.assert_called()
    # Expect the clean shutdown path
    mock_host_command.stop.assert_not_called()
    mock_ssh_client.exec_command.assert_called_once_with("poweroff")


@pytest.mark.timeout(5)
def test_qemu_instance_exec_boot_failure(mocker, qemu_instance, mock_host_command, mock_ssh_command, mock_ssh_client):
    """
    Verify the instance state machine when a boot failure occurs
    """
    spy_shutdown = mocker.spy(qemu_instance, "_shutdown")

    qemu_instance.signal_boot()
    # Instead of triggering SETUP we simulate an EXITED message
    qemu_instance._msg_queue.put(Msg(MsgType.EXITED, "cheribuild", mock_host_command))
    # Check that a wait_ready() now fails
    with pytest.raises(RuntimeError, match=r"[Ii]nstance died"):
        qemu_instance.wait_ready()

    # Check that the instance responds with a call to Instance._shutdown
    spy_shutdown.assert_called_once()
    # Check that we did not try to stop the command as it is already dead
    mock_host_command.stop.assert_not_called()
    # Check that we did not try to run any SSH stuff
    mock_ssh_client.connect.assert_not_called()
    mock_ssh_client.exec_command.assert_not_called()
