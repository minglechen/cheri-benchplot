import gzip
import logging
import shutil
import subprocess
import time
import typing
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from threading import Lock, Thread

import termcolor


class LogColorFormatter(logging.Formatter):
    colors = {
        logging.DEBUG: "white",
        logging.INFO: "blue",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "red"
    }

    def __init__(self, *args, use_colors=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = use_colors

    def format(self, record):
        msg = super().format(record)
        if self.use_colors:
            color = self.colors[record.levelno]
            return termcolor.colored(msg, color)
        return msg


def setup_logging(verbose: bool = False, logfile: Path = None):
    log_fmt = "[%(levelname)s] %(name)s: %(message)s"
    date_fmt = None
    default_level = logging.INFO
    if verbose:
        default_level = logging.DEBUG
    logger = logging.getLogger("cheri-benchplot")
    logger.setLevel(default_level)
    logger.propagate = False
    # Console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(LogColorFormatter(fmt=log_fmt))
    logger.addHandler(console_handler)
    # File logging
    if logfile:
        file_handler = logging.FileHandler(logfile, mode="w")
        file_handler.setFormatter(logging.Formatter(fmt=log_fmt))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    # Silence logging for paramiko SSH events
    pko_logger = logging.getLogger("paramiko")
    pko_logger.setLevel(logging.ERROR)
    pko_logger.addHandler(console_handler)
    if logfile:
        pko_logger.addHandler(file_handler)
    return logger


def new_logger(name, parent=None):
    if parent is None:
        parent = logging.getLogger("cheri-benchplot")
    return parent.getChild(name)


@contextmanager
def timing(name, level=logging.INFO, logger=None):
    if logger is None:
        logger = logging
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        logger.log(level, "%s in %.2fs", name, end - start)


def resolve_system_command(name: str, logger: logging.Logger | None = None):
    if logger is None:
        logger = logging.getLogger("cheri-benchplot")
    path = shutil.which(name)
    if path is None:
        logger.critical("Missing dependency %s, should be in your $PATH", name)
        raise RuntimeError("Missing dependency")
    return Path(path)


@contextmanager
def gzopen(path: Path, mode: str) -> typing.IO:
    if path.suffix == ".gz":
        openfn = gzip.open
        if "b" not in mode and "t" not in mode:
            mode += "t"
    else:
        openfn = open
    with openfn(path, mode) as fileio:
        yield fileio


@contextmanager
def gzopen(path: Path, mode: str) -> typing.IO:
    if path.suffix == ".gz":
        openfn = gzip.open
        if "b" not in mode and "t" not in mode:
            mode += "t"
    else:
        openfn = open
    with openfn(path, mode) as fileio:
        yield fileio


class SubprocessHelper:
    """
    Helper to run a subprocess and live-capture the output into our logger.

    XXX This can probably be consolidated with the instance command runners.
    """
    def __init__(self, executable: Path, args: list, env: dict = None, logger: logging.Logger = None):
        """
        :param executable: Path to the executable command
        :param args: Arguments list
        :param env: Environment variables
        """
        self.logger = logger or new_logger(f"subcommand-{executable.name}")
        self.executable = executable
        self.args = args
        self.env = env
        #: Worker thread that looks at the data I/O
        self._out_worker = None
        #: Output observers to perform live actions on the stdout data
        self._out_observers = []
        #: Worker thread that looks at the err I/O
        self._err_worker = None
        #: Output observers to perform live actions on the stderr data
        self._err_observers = []
        #: Subprocess or worker died with error
        self._failed = False
        #: Subprocess instance
        self._subprocess = None
        #: Workers lock
        self._lock = Lock()
        #: Level at which to log subprocess stderr
        self._stderr_loglevel = logging.WARNING

    def __bool__(self):
        with self._lock:
            if self._failed:
                return False
            return self._subprocess.returncode == 0

    def _do_io(self, pipe: typing.IO[bytes], observers: list[typing.Callable[[str], None]]):
        self.logger.debug("Subcommand %s I/O loop", self.executable)
        try:
            for raw_line in pipe:
                line = raw_line.decode("utf-8").strip()
                for callback in observers:
                    callback(line)
        except Exception as ex:
            self.logger.error("Subcommand %s I/O loop failed: %s", self.executable, ex)
            with self._lock:
                self._failed = True
            self._subprocess.kill()
        self._subprocess.wait()
        with self._lock:
            self._failed = (self._subprocess.returncode != 0)
        self.logger.debug("Subcommand %s I/O loop done", self.executable)

    def observe_stdout(self, handler: typing.Callable[[str], None]):
        """
        Add callback invoked on every stdout line.
        Note that this is unsynchronized, so should only be called before starting the process.
        """
        assert self._subprocess is None, "Can not add observers after starting"
        self._out_observers.append(handler)

    def observe_stderr(self, handler: typing.Callable[[str], None]):
        """
        Add callback invoked on every stderr line.
        Note that this is unsynchronized, so should only be called before starting the process.
        """
        assert self._subprocess is None, "Can not add observers after starting"
        self._err_observers.append(handler)

    def set_stderr_loglevel(self, level: int):
        """
        Set level at which to log the subprocess stderr.

        By default this is set to WARNING.
        """
        self._stderr_loglevel = level

    def run(self, **popen_kwargs):
        """
        Syncrhonously run the command and process output.
        """
        self.start(**popen_kwargs)
        self.wait()
        if self._subprocess.returncode != 0:
            self.logger.error("Failed to run %s: %d", self.executable, self._subprocess.returncode)
            raise RuntimeError("Subprocess failed")

    def start(self, **popen_kwargs):
        assert self._subprocess is None, "multiple start()"
        self.logger.debug("Subcommand: %s %s", self.executable, " ".join(map(str, self.args)))
        self._subprocess = subprocess.Popen([self.executable] + self.args,
                                            env=self.env,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            **popen_kwargs)
        self._out_observers.append(lambda line: self.logger.debug(line.strip()))
        self._err_observers.append(lambda line: self.logger.log(self._stderr_loglevel, line.strip()))
        self._out_worker = Thread(target=self._do_io, args=(self._subprocess.stdout, self._out_observers))
        self._out_worker.start()
        self._err_worker = Thread(target=self._do_io, args=(self._subprocess.stderr, self._err_observers))
        self._err_worker.start()

    def stop(self):
        assert self._subprocess is not None, "stop() before start()"
        self._subprocess.terminate()

    def wait(self):
        assert self._subprocess is not None, "wait() before start()"
        self._out_worker.join()
        self._err_worker.join()
        self._subprocess.wait()
