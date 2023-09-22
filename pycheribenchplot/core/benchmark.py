import shlex
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Type

import pandas as pd

from .config import AnalysisConfig, Config, ExecTargetConfig
from .dwarf import DWARFManager
from .elf import AddressSpaceManager
from .error import TaskNotFound
from .instance import InstanceManager
from .shellgen import ScriptBuilder
from .task import (ExecutionTask, SessionExecutionTask, Task, TaskRegistry, TaskScheduler)
from .util import new_logger, timing

AnyExecTask = ExecutionTask | SessionExecutionTask


class BenchmarkExecMode(Enum):
    """
    Run mode determines the type of run strategy to use.
    This is currently used for partial or pretend runs.
    """
    FULL = "full"
    SHELLGEN = "shellgen"


@dataclass
class ExecTaskConfig(Config):
    """
    Configuration object for :class:`BenchmarkExecTask`
    """
    mode: BenchmarkExecMode = BenchmarkExecMode.FULL


class BenchmarkExecTask(Task):
    """
    Task that dynamically gathers dependencies for all the execution task required
    by a benchmark run instance.
    """
    task_namespace = "internal.exec"
    task_name = "benchmark-root"

    def __init__(self, benchmark, task_config: ExecTaskConfig):
        #: Associated benchmark context
        self.benchmark = benchmark
        #: Script builder for this benchmark context
        self.script = ScriptBuilder(benchmark)
        #: Whether we need to request a VM instance, this becomes valid after dependency scanning
        self._need_instance = None

        # Borg initialization occurs here
        super().__init__(task_config=task_config)

    def _fetch_task(self, config: ExecTargetConfig) -> Task:
        task_klass = TaskRegistry.resolve_exec_task(config.handler)
        if task_klass is None:
            self.logger.error("Invalid task name specification exec.%s", task_name)
            raise ValueError("Invalid task name")
        if task_klass.require_instance:
            self._need_instance = True
        if issubclass(task_klass, ExecutionTask):
            return task_klass(self.benchmark, self.script, task_config=config.task_options)
        elif issubclass(task_klass, SessionExecutionTask):
            return task_klass(self.session, task_config=config.task_options)
        else:
            self.logger.error("Invalid task %s, must be ExecutionTask or SessionExecutionTask", task_klass)
            raise TypeError("Invalid task type")

    def _handle_config_command_hooks_for_section(self, section_name: str, cmd_list: list[str | dict]):
        """
        Helper to handle config command hooks for a specific section.
        """
        global_section = self.script.sections[section_name]
        # Accumulate here any non-global command hook specification
        iter_sections = defaultdict(list)
        for cmd_spec in cmd_list:
            if isinstance(cmd_spec, str):
                parts = shlex.split(cmd_spec)
                global_section.add_cmd(parts[0], parts[1:])
            else:
                iter_sections.update({k: iter_sections[k] + v for k, v in cmd_spec.items()})
        # Now resolve the merged iteration spec
        for iter_section_spec, iter_cmd_list in iter_sections.items():
            if iter_section_spec == "*":
                # Wildcard for all iterations
                for group in self.script.benchmark_sections:
                    for cmd in iter_cmd_list:
                        parts = shlex.split(cmd)
                        group[section_name].add_cmd(cmd[0], cmd[1:])
            else:
                # Must be an integer index
                index = int(iteration_spec)
                for cmd in iter_cmd_list:
                    parts = shlex.split(cmd)
                    self.script.benchmark_sections[index][section_name].add_cmd(parts[0], parts[1:])

    def _handle_config_command_hooks(self):
        """
        Add the extra commands from :attr:`BenchmarkRunConfig.command_hooks`
        See the :class:`BenchmarkRunConfig` class for a description of the format we expect.
        """
        for section_name, cmd_list in self.benchmark.config.command_hooks.items():
            if section_name not in self.script.sections:
                self.logger.warning(
                    "Configuration file wants to add hooks to script" +
                    " section %s but the section does not exist: %s", section_name, self.script.sections.keys())
                continue
            self._handle_config_command_hooks_for_section(section_name, cmd_list)

    def _extract_files(self, instance, task):
        """
        Extract remote files for a given task.
        """
        for output_key, output in task.outputs():
            if not output.is_file() or not output.needs_extraction():
                continue
            for remote_path, host_path in zip(output.remote_paths(), output.paths()):
                self.logger.debug("Extract %s guest: %s => host: %s", output_key, remote_path, host_path)
                instance.extract_file(remote_path, host_path)

    @property
    def session(self):
        return self.benchmark.session

    @property
    def task_id(self):
        return f"{self.task_namespace}-{self.task_name}-{self.benchmark.uuid}"

    def resources(self):
        assert self._need_instance is not None, "need_instance must have been set, did not scan dependencies?"
        if self._need_instance:
            self.instance_req = InstanceManager.request(self.benchmark.g_uuid,
                                                        instance_config=self.benchmark.config.instance)
            yield self.instance_req

    def dependencies(self):
        # Note that dependencies are guaranteed to be scanned on schedule resolution,
        # so before resource requests are resolved. We can determine here how many deps
        # require an instance
        self._need_instance = False
        for exec_target in self.benchmark.config.generators:
            yield self._fetch_task(exec_target)

    def run(self):
        """
        Run the benchmark and extract the results.
        This involves the steps:
        1. Emit the remote run script from the benchmark state
        2. Ask the instance manager for an instance that we can run on
        3. Connect to the instance, upload the run script and run it to completion.
        4. Extract results
        """
        assert self._need_instance is not None, "need_instance must have been set, did not scan dependencies?"

        if not self._need_instance:
            # We are done, nothing to run on a remote instance, any task to execute has been completed by deps
            self.logger.info("Benchmark run completed")
            return

        self._handle_config_command_hooks()
        script_path = self.benchmark.get_run_script_path()
        remote_script_path = Path(f"{self.benchmark.config.name}-{self.benchmark.uuid}.sh")
        with open(script_path, "w+") as script_file:
            self.script.generate(script_file)
        script_path.chmod(0o755)
        if self.config.mode == BenchmarkExecMode.SHELLGEN:
            # Just stop after shell script generation
            return

        instance = self.instance_req.get()
        self.logger.debug("Import script file host: %s => guest: %s", script_path, remote_script_path)
        instance.import_file(script_path, remote_script_path)
        self.logger.info("Execute benchmark script")
        with timing("Benchmark script completed", logger=self.logger):
            instance.run_cmd("sh", [remote_script_path])

        # Extract all output files
        self.logger.info("Extract output files")
        for dep in self.resolved_dependencies:
            self._extract_files(instance, dep)
        self.logger.info("Benchmark run completed")


class Benchmark:
    """
    Benchmark context

    :param session: The parent session
    :param config: The benchmark run configuration allocated to this handler.
    """
    def __init__(self, session: "Session", config: "BenchmarkRunConfig"):
        self.session = session
        self.config = config
        self.logger = new_logger(f"{self.config.name}:{self.config.instance.name}")

        # Symbol mapping handler for this benchmark instance
        self.sym_resolver = AddressSpaceManager(self)
        # Dwarf information extraction helper
        self.dwarf = DWARFManager(self)

    def __str__(self):
        return f"{self.config.name}({self.uuid})"

    def __repr__(self):
        return str(self)

    @property
    def uuid(self):
        return self.config.uuid

    @property
    def g_uuid(self):
        return self.config.g_uuid

    @property
    def parameters(self) -> dict[str, any]:
        """
        Helper to inspect the user-definied parameterization for this benchmark instance.
        This is the index in the datagen matrix corresponding to this benchmark's row.
        """
        return self.config.parameters

    @property
    def user_config(self):
        return self.session.user_config

    @property
    def analysis_config(self):
        return self.session.analysis_config

    @property
    def cheribsd_rootfs_path(self):
        rootfs_path = self.user_config.rootfs_path / f"rootfs-{self.config.instance.cheri_target}"
        if not rootfs_path.exists() or not rootfs_path.is_dir():
            raise ValueError(f"Invalid rootfs path {rootfs_path} for benchmark instance")
        return rootfs_path

    def get_run_script_path(self) -> Path:
        """
        :return: The path to the run script to import to the guest for this benchmark
        """
        return self.get_benchmark_data_path() / "runner-script.sh"

    def get_benchmark_data_path(self) -> Path:
        """
        :return: The output directory for run data corresponding to this benchmark configuration.
        """
        return self.session.get_data_root_path() / f"{self.config.name}-{self.uuid}"

    def get_benchmark_asset_path(self) -> Path:
        """
        :return: Path for archiving binaries used in the run
        """
        return self.session.get_asset_root_path() / f"{self.config.name}-{self.uuid}"

    def get_instance_asset_path(self) -> Path:
        """
        :return: Path for archiving binaries used in the run but belonging to the instance.
            This avoids duplication.
        """
        return self.session.get_asset_root_path() / f"{self.config.instance.name}-{self.g_uuid}"

    def get_plot_path(self):
        """
        Get base plot path for the current benchmark instance.
        Every plot should use this path as the base path to generate plots.
        """
        return self.session.get_plot_root_path() / self.config.name

    def get_benchmark_iter_data_path(self, iteration: int) -> Path:
        """
        :return: The output directory for run data corresponding to this benchmark configuration and
            a specific benchmark iteration.
        """
        iter_path = self.get_benchmark_data_path() / str(iteration)
        return iter_path

    def build_exec_task(self, exec_config: ExecTaskConfig) -> BenchmarkExecTask:
        """
        Create an instance of the top-level benchmark execution task for this benchmark.

        The task is not scheduled and may be used to obtain dependency and output
        information for loading data from generators during analysis.

        :param exec_config: The execution task configuration.
        :return: The top-level execution task
        """
        exec_task = BenchmarkExecTask(self, task_config=exec_config)
        return exec_task

    def find_exec_task(self, task_class: Type[AnyExecTask]) -> AnyExecTask:
        """
        Find the given execution task configured for the current benchmark context.

        This is used during analysis to resolve generators as dependencies.
        :param task_class: The type of the task that needs to be resolved.
        :return: An instance of the given task type, bound to the current benchmark context.
        """
        main_task = self.build_exec_task(ExecTaskConfig())
        deps = list(main_task.dependencies())
        for task in deps:
            if (task_class.task_namespace == task.task_namespace and task_class.task_name == task.task_name):
                if not isinstance(task, task_class):
                    self.logger.error("Found generator for %s but it is not an instance of %s", task.task_id,
                                      task_class)
                    raise RuntimeError("Invalid exec task instance")
                return task
        # XXX Recursively look into dependencies
        raise TaskNotFound("Task %s can not be found among the session generators", task_class)

    def schedule_exec_tasks(self, scheduler: TaskScheduler, exec_config: ExecTaskConfig):
        """
        Generate the top-level benchmark execution task for this benchmark run and schedule it.

        :param scheduler: The scheduler for this session
        :param exec_config: The execution task configuration. This is used to control partial
            execution runs and behaviour outside of the session configuration file.
        """
        # Ensure that all the data paths are initialized
        self.get_benchmark_data_path().mkdir(exist_ok=True)
        for i in range(self.config.iterations):
            self.get_benchmark_iter_data_path(i).mkdir(exist_ok=True)

        exec_task = self.build_exec_task(exec_config)
        self.logger.debug("Initialize top-level benchmark task %s", exec_task)
        scheduler.add_task(exec_task)
