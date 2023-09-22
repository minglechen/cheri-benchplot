import shlex
import typing
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from .artefact import Target
from .util import new_logger


@dataclass
class ScriptCommand:
    """
    Represent a single command executed by the generated run script.
    """
    #: The command to execute
    cmd: str

    #: CLI arguments as a list
    args: list[str] = field(default_factory=list)

    #: Env variables as a dict ENV_VAR => value
    env: dict[str, str] = field(default_factory=dict)

    #: path to the output file we should redirect stdout to
    stdout_path: Path | None = None

    #: Run the command in background
    background: bool = False

    #: Bind command to CPU
    cpu: int | None = None

    #: Debug mode prints the commands before executing
    debug: bool = False

    #: Unique name of the PID reference variable. This is useful to kill background processes during teardown
    pid_reference_id: str | None = None

    def generate(self, fd: typing.IO[str]):
        # Base command bits
        cmd_string = " ".join([str(self.cmd)] + [str(v) for v in self.args])
        env_parts = [f"{name}=" + shlex.quote(str(value)) for name, value in self.env.items()]
        env_string = " ".join(env_parts)

        # Modifiers
        if self.stdout_path:
            stdout = f">> {self.stdout_path}"
        else:
            stdout = ""
        if self.cpu is not None:
            cpuset = ["cpuset", "-c", "-l", str(self.cpu)] + [cmd_string]
            cmd_string = shlex.join(cpuset)

        full_command = f"{env_string} {cmd_string}".strip()
        if self.background:
            background = "&"
        else:
            background = ""
        lines = []

        exec_command = f"{full_command} {stdout}".strip()
        exec_command = f"{exec_command} {background}".strip()
        lines.append(exec_command)

        if self.background and self.pid_reference_id:
            # We want to capture the pid
            lines.append(f"PID_{self.pid_reference_id}=$!")

        if self.debug:
            lines.append(f"echo {full_command}")

        for line in lines:
            fd.write(line + "\n")


class ScriptBuilder:
    """
    Generate a shell script that runs the benchmark steps as they are scheduled by
    datasets.

    The script is organised into sections.
    Each benchmark execution task can take its turn to add commands to each section.
    The sections are separated in steps, so that the order of the commands within a single
    section should not matter. It is inadvisable anyway to perform too many operations
    in between benchmark iterations steps so the number of operations per section should
    generally stay on the low side.

    Currently the section are as follow:
    - pre-benchmark -> common preparatory activity
    - benchmark -> benchmark activity
    - post-benchmark -> system stats extraction steps and data massaging
    - last -> anything that depends on something in post_benchmark

    The benchmark section is further split into iterations:
    - pre-benchmark-iter -> sampling before a benchmark iteration
    - benchmark-iter -> benchmark iteration activity (generally a single command)
    - post-benchmark-iter -> sampling after a benchmark iteration
    """
    class Section:
        """
        A section containing commands.
        """
        def __init__(self, script, name):
            #: Parent script
            self._script = script
            #: Name of the section, mostly for debug output
            self.name = name
            #: Command sequence
            self._commands = []
            #: Command sequence lock
            self._section_lock = Lock()

        def add_cmd(self,
                    command: str | Path,
                    args: list[str] = None,
                    env: dict[str, str] = None,
                    output: Path | Target = None,
                    cpu: int | None = None,
                    background: bool = False) -> ScriptCommand:
            """
            Add a foreground command to the run script.

            :param cmd_path: Path to the executable (guest path)
            :param args: Command arguments as a list of strings
            :param output: Target specifying a file to which redirect the command output
            :param cpu: Bind to the given CPU
            :param background: Run as a background process
            :return: An opaque command reference, this is useful to terminate background commands
            """
            cmd = ScriptCommand(cmd=command, args=args or [], env=env or {}, background=background)
            if output and isinstance(output, Target):
                # Must assume that the target points to a single file
                cmd.stdout_path = output.remote_path
            else:
                cmd.stdout_path = output
            if cpu is not None:
                cmd.cpu = cpu
            cmd.pid_reference_id = str(self._script._next_sequence())
            with self._section_lock:
                self._commands.append(cmd)
            return cmd

        def add_kill_cmd(self, target: ScriptCommand):
            cmd = ScriptCommand(cmd="kill", args=["-TERM", f"$PID_{target.pid_reference_id}"])
            with self._section_lock:
                self._commands.append(cmd)

        def add_sleep(self, nsecs: int):
            self.add_cmd("sleep", [str(nsecs)])

        def generate(self, fd: typing.IO[str], logger):
            logger.debug("Populating script section %s", self.name)
            with self._section_lock:
                for cmd in self._commands:
                    cmd.generate(fd)

    def __init__(self, benchmark):
        #: Benchmark that owns this script
        self.benchmark = benchmark
        #: Shell generator logger
        self.logger = new_logger("shellgen", parent=benchmark.logger)
        #: Path to the benchmark data output directory on the guest machine, relative to guest $HOME
        self._guest_output_path = benchmark.config.remote_output_dir
        #: Global command sequence number, protected by the script lock
        self._sequence = 0
        #: Script command list lock
        self._script_lock = Lock()
        #: Sections structure
        self._sections = {
            "pre-benchmark":
            self.Section(self, "pre-benchmark"),
            "benchmark": [{
                "pre-benchmark": self.Section(self, f"pre-benchmark-{i}"),
                "benchmark": self.Section(self, f"benchmark-{i}"),
                "post-benchmark": self.Section(self, f"post-benchmark-{i}")
            } for i in range(benchmark.config.iterations)],
            "post-benchmark":
            self.Section(self, "post-benchmark"),
            "last":
            self.Section(self, "last")
        }
        self._prepare_guest_output_dirs()

    def _prepare_guest_output_dirs(self):
        """
        Prepare the guest output environment to store the files to mirror the local benchmark instance
        output directory.
        This makes it easier to share file paths by just using directories relative to the benchmark
        output_path.
        """
        section = self.sections["pre-benchmark"]
        section.add_cmd("mkdir", ["-p", self._guest_output_path])
        section.add_cmd("mkdir", ["-p", self._guest_output_path])
        for i in range(self.benchmark.config.iterations):
            section.add_cmd("mkdir", ["-p", self._guest_output_path / str(i)])
        # section.add_cmd("touch", [self.local_to_remote_path(self.command_history_path())])

    def _next_sequence(self):
        with self._script_lock:
            self._sequence += 1
            return self._sequence

    @property
    def sections(self):
        """Note: The self._sections container should be immutable (only the contents of sections can change)"""
        return self._sections

    @property
    def benchmark_sections(self):
        """
        Shorthand to access the per-iteration sections list
        Note: The self._sections container should be immutable (only the contents of sections can change)
        """
        return self._sections["benchmark"]

    def generate(self, fd: typing.IO[str]):
        fd.write("#!/bin/sh\n\n")
        fd.write("set -e\n")
        # Syncronization here is not really needed since we only do this
        # in the top-level benchmark task, where the script is private to each
        # benchmark.
        with self._script_lock:
            self.sections["pre-benchmark"].generate(fd, self.logger)
            for group in self.sections["benchmark"]:
                group["pre-benchmark"].generate(fd, self.logger)
                group["benchmark"].generate(fd, self.logger)
                group["post-benchmark"].generate(fd, self.logger)
            self.sections["post-benchmark"].generate(fd, self.logger)
            self.sections["last"].generate(fd, self.logger)
