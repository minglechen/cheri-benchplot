import shlex
import typing
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

from .config import AnalysisConfig, Config, ExecTargetConfig
from .elf import AddressSpaceManager, DWARFHelper
from .instance import InstanceManager
from .shellgen import ScriptBuilder
from .task import (ExecutionTask, SessionExecutionTask, Task, TaskRegistry, TaskScheduler)
from .util import new_logger, timing


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
        super().__init__(task_config=task_config)
        #: Associated benchmark context
        self.benchmark = benchmark
        #: Script builder for this benchmark context
        self.script = ScriptBuilder(benchmark)
        #: Whether we need to request a VM instance, this becomes valid after dependency scanning
        self._need_instance = None

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
            for remote_path, host_path in zip(output.remote_paths, output.paths):
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
        yield self._fetch_task(self.benchmark.config.benchmark)
        for aux_config in self.benchmark.config.aux_tasks:
            yield self._fetch_task(aux_config)

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
        self.dwarf_helper = DWARFHelper(self)

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

        exec_task = BenchmarkExecTask(self, task_config=exec_config)
        self.logger.info("Initialize top-level benchmark task %s", exec_task)
        scheduler.add_task(exec_task)


# class _Benchmark:
#     """
#     New benchmark handler implementation

#     :param session: The parent session
#     :param config: The benchmark run configuration allocated to this handler.
#     """
#     def __init__(self, session: "Session", config: "BenchmarkRunConfig"):
#         self.session = session
#         self.config = config
#         self.logger = new_logger(f"{self.config.name}:{self.config.instance.name}")

#         # Map uuids to benchmarks that have been merged into the current instance (which is the baseline)
#         # so that we can look them up if necessary
#         self.merged_benchmarks = {}
#         # Cross-benchmark parameterisation merged instances
#         self.cross_merged = {}
#         # Symbol mapping handler for this benchmark instance
#         self.sym_resolver = AddressSpaceManager(self)
#         # Dwarf information extraction helper
#         self.dwarf_helper = DWARFHelper(self)

#         self.get_benchmark_data_path().mkdir(exist_ok=True)
#         # self._dataset_modules = self._collect_dataset_modules()
#         # self._dataset_runner_modules = self._collect_runner_modules()
#         # self._configure_datasets()
#         # PID/TID mapper XXX this should be a separate thing like sym_resolver
#         # self.pidmap = self.get_dataset_by_artefact(DatasetArtefact.PIDMAP)

#     def __str__(self):
#         return f"{self.config.name}({self.uuid})"

#     def __repr__(self):
#         return str(self)

#     def _find_dataset_module(self, config: DatasetConfig):
#         """Resolve the parser for the given dataset"""
#         ds_name = DatasetName(config.handler)
#         handler_class = DatasetRegistry.resolve_name(ds_name)
#         handler = handler_class(self, config)
#         return handler

#     def _collect_dataset_modules(self) -> typing.Dict[DatasetArtefact, DataSetContainer]:
#         """
#         Helper to resolve all dataset handlers for this benchmark setup.
#         This will return a dictionary mapping DatasetName => :class:`DataSetContainer`

#         :return: A mapping of dataset ID to the dataset module responsible to analyse the data.
#         """
#         modules = {}
#         # The main benchmark dataset
#         # XXX we should mark these somehow, to avoid confusion
#         modules[self.config.benchmark.handler] = self._find_dataset_module(self.config.benchmark)
#         # Implicit auxiliary datasets
#         modules[DatasetName.PIDMAP] = self._find_dataset_module(DatasetConfig(handler=DatasetName.PIDMAP))
#         # Extra datasets configured
#         for aux_config in self.config.aux_dataset_handlers:
#             assert aux_config.handler not in modules, "Duplicate AUX dataset handler"
#             modules[aux_config.handler] = self._find_dataset_module(aux_config)
#         return modules

#     def _collect_runner_modules(self) -> typing.Dict[DatasetArtefact, DataSetContainer]:
#         """
#         Helper to resolve all dataset handlers that will produce data during a benchmark run.
#         This is necessary to deduplicate any handler that require the same input data but we
#         only need to generate the data once.
#         If multiple datasets have the same dataset_source_id, assume it to be equivalent and
#         just pick one of them (this is the case for commands that produce multiple datasets).

#         :return: A mapping of dataset artefact ID to the dataset module responsible to generate the data
#         """
#         runners = {}
#         for dset in self._dataset_modules.values():
#             if dset.dataset_source_id not in runners:
#                 runners[dset.dataset_source_id] = dset
#         return runners

#     def _collect_analysers(self, analysis_config: AnalysisConfig, cross_analysis: bool):
#         """
#         Helper to resolve all analysis steps that need to be run for this benchmark set.
#         Note that this will be called only on the instance where the merged dataframes are built.
#         """
#         analysers = []
#         datasets = set(self._dataset_modules.keys())

#         for conf in analysis_config.handlers:
#             try:
#                 handler_class = BenchmarkAnalysisRegistry.analysis_steps[conf.name]
#             except KeyError:
#                 self.logger.warn("No analysis handler with name %s, skipping", conf.name)
#                 continue
#             if not handler_class.require.issubset(datasets):
#                 self.logger.warn("analysis handler %s requires but only %s available", handler_class.require, datasets)
#                 continue
#             if handler_class.cross_analysis != cross_analysis:
#                 continue
#             if handler_class.analysis_options_class:
#                 options = handler_class.analysis_options_class.schema().load(conf.options)
#             else:
#                 options = conf.options
#             analysers.append(handler_class(self, options))

#         if len(analysis_config.handlers) == 0:
#             # This mode should probably be deprecated and go away
#             for handler_class in BenchmarkAnalysisRegistry.analysis_steps.values():
#                 if handler_class.cross_analysis != cross_analysis:
#                     continue
#                 # First check if we have the minimal requirements
#                 if callable(handler_class.require):
#                     checker = handler_class.require
#                 else:
#                     checker = lambda dss, conf: handler_class.require.issubset(dss)
#                 if not checker(datasets, analysis_config):
#                     continue
#                 if handler_class.analysis_options_class:
#                     options = handler_class.analysis_options_class.schema().load({})
#                 else:
#                     options = {}
#                 analysers.append(handler_class(self, options))
#         return analysers

#     def _configure_datasets(self):
#         """Resolve platform options for the instance configuration and finalize dataset configuration"""
#         opts = self.config.instance.platform_options
#         for dset in self._dataset_modules.values():
#             opts = dset.configure(opts)

#     def _build_remote_script(self):
#         """
#         Helper to build a new benchmark run script based on the current configuration.

#         :return: The local path to the runner script
#         """
#         script = ShellScriptBuilder(self)

#         bench_handler = self.get_dataset(DatasetName(self.config.benchmark.handler))
#         assert bench_handler, "Missing benchmark dataset"
#         pre_generators = sorted(self._dataset_runner_modules.values(), key=lambda ds: ds.dataset_run_order)
#         post_generators = sorted(self._dataset_runner_modules.values(),
#                                  key=lambda ds: ds.dataset_run_order,
#                                  reverse=True)
#         self.logger.info("Generate benchmark script")

#         self._gen_hooks(script, "pre_benchmark")
#         for dset in pre_generators:
#             dset.gen_pre_benchmark(script)

#         for i in range(self.config.iterations):
#             self.logger.info("Generate benchmark iteration %d", i)
#             for dset in pre_generators:
#                 dset.configure_iteration(script, i)
#             self._gen_hooks(script, "pre_benchmark_iter")
#             for dset in pre_generators:
#                 dset.gen_pre_benchmark_iter(script, i)
#             # Only run the benchmark step for the given benchmark_dataset
#             bench_handler.gen_benchmark(script, i)
#             self._gen_hooks(script, "post_benchmark_iter")
#             for dset in post_generators:
#                 dset.gen_post_benchmark_iter(script, i)

#         self._gen_hooks(script, "post_benchmark")
#         for dset in post_generators:
#             dset.gen_post_benchmark(script)
#         for dset in post_generators:
#             dset.gen_pre_extract_results(script)

#         script_path = self.get_run_script_path()
#         with open(script_path, "w+") as script_file:
#             script.to_shell_script(script_file)
#         script_path.chmod(0o755)
#         return script

#     def _gen_hooks(self, script: "ShellScriptBuilder", name: str):
#         """
#         Generate script commands from configured hooks. The given name is the
#         hook key in the configuration dictionary.

#         :param name: Hook name (as appears in the configuration file)
#         """
#         commands = self.config.command_hooks.get(name, [])
#         for cmd in commands:
#             cmd_args = shlex.split(cmd)
#             script.gen_cmd(cmd_args[0], cmd_args[1:])

#     async def _extract_results(self, script: "ShellScriptBuilder", instance: "InstanceInfo"):
#         """
#         Helper to extract results from a connected instance

#         :param script: The script generator
#         :param instance: Instance handler from the :class:`InstanceManager`
#         """
#         for remote_path, local_path, custom_extract_fn in script.get_extract_files():
#             self.logger.debug("Extract %s -> %s", remote_path, local_path)
#             if custom_extract_fn:
#                 await custom_extract_fn(instance, script, remote_path, local_path)
#             else:
#                 await instance.extract_file(remote_path, local_path)

#         # Extract also the implicit command history
#         cmd_history = script.command_history_path()
#         remote_cmd_history = script.local_to_remote_path(cmd_history)
#         self.logger.debug("Extract %s -> %s", remote_cmd_history, cmd_history)
#         await instance.extract_file(remote_cmd_history, cmd_history)

#     @property
#     def uuid(self):
#         return self.config.uuid

#     @property
#     def g_uuid(self):
#         return self.config.g_uuid

#     @property
#     def user_config(self):
#         return self.session.user_config

#     @property
#     def analysis_config(self):
#         return self.session.analysis_config

#     @property
#     def cheribsd_rootfs_path(self):
#         rootfs_path = self.user_config.rootfs_path / f"rootfs-{self.config.instance.cheri_target}"
#         if not rootfs_path.exists() or not rootfs_path.is_dir():
#             raise ValueError(f"Invalid rootfs path {rootfs_path} for benchmark instance")
#         return rootfs_path

#     def get_run_script_path(self) -> Path:
#         """
#         :return: The path to the run script to import to the guest for this benchmark
#         """
#         return self.get_benchmark_data_path() / "runner-script.sh"

#     def get_benchmark_data_path(self) -> Path:
#         """
#         :return: The output directory for run data corresponding to this benchmark configuration.
#         """
#         return self.session.get_data_root_path() / f"{self.config.name}-{self.uuid}"

#     def get_benchmark_asset_path(self) -> Path:
#         """
#         :return: Path for archiving binaries used in the run
#         """
#         return self.session.get_asset_root_path() / f"{self.config.name}-{self.uuid}"

#     def get_instance_asset_path(self) -> Path:
#         """
#         :return: Path for archiving binaries used in the run but belonging to the instance.
#         This avoids duplication.
#         """
#         return self.session.get_asset_root_path() / f"{self.config.instance.name}-{self.g_uuid}"

#     def get_plot_path(self):
#         """
#         Get base plot path for the current benchmark instance.
#         Every plot should use this path as the base path to generate plots.
#         """
#         return self.session.get_plot_root_path() / self.config.name

#     def get_benchmark_iter_data_path(self, iteration: int) -> Path:
#         """
#         :return: The output directory for run data corresponding to this benchmark configuration and
#         a specific benchmark iteration.
#         """
#         iter_path = self.get_benchmark_data_path() / str(iteration)
#         iter_path.mkdir(exist_ok=True)
#         return iter_path

#     def get_dataset(self, name: DatasetName) -> typing.Optional[DataSetContainer]:
#         """
#         :return: The instance within the current benchmark for given dataset module name.
#         """
#         return self._dataset_modules.get(name, None)

#     def get_dataset_by_artefact(self, ds_id: DatasetArtefact) -> typing.Optional[DataSetContainer]:
#         """
#         Lookup a generic dataset by the artefact ID.
#         Note that this will fail if there are multiple matches
#         XXX do we need this?
#         """
#         match = [dset for dset in self._dataset_modules.values() if dset.dataset_source_id == ds_id]
#         if len(match) > 1:
#             raise KeyError("Multiple matching dataset for artefact %s", ds_id)
#         if len(match):
#             return match[0]
#         return None

#     def get_merged_benchmarks(self) -> typing.Dict[UUID, "Benchmark"]:
#         """
#         Find the set of benchmarks that have been merged or will be merged to the current benchmark
#         instance. This is relevant only for baseline instances, as other instances will be merged
#         to the corresponding baseline :class:`Benchmark` instance.

#         :return: a dict mapping the UUID to the benchmark instances that are merged to the current
#         instance. This includes the current instance.
#         """
#         assert self.g_uuid == self.session.baseline_g_uuid,\
#             "Benchmark groups are meaningful only on the baseline instance"
#         row = self.session.benchmark_matrix[self.g_uuid] == self
#         assert row.sum() == 1
#         return {b.uuid: b for b in self.session.benchmark_matrix.loc[row, :]}

#     def get_benchmark_groups(self) -> typing.Dict[UUID, "Benchmark"]:
#         """
#         Return the benchmark matrix columns as a mapping of group UUIDs to the list of benchmarks
#         for each group (instance configuration).
#         """
#         assert self.g_uuid == self.session.baseline_g_uuid,\
#             "Benchmark groups are meaningful only on the baseline instance"
#         groups = self.session.benchmark_matrix.to_dict(orient="list")
#         return groups

#     def register_mapped_binary(self, map_addr: int, path: Path, pid: int):
#         """
#         Add a binary to the symbolizer for this benchmark.
#         The symbols will be associated with the given PID address space

#         :param map_addr: Base load address
#         :param path: Host path to the binary ELF file
#         :param pid: PID of the process associated to the binary
#         """
#         # bench_dset = self.get_dataset(self.config.benchmark_dataset.type)
#         # addrspace = bench_dset.get_addrspace_key()
#         pidmap = self.get_dataset_by_artefact(DatasetArtefact.PIDMAP)
#         assert pidmap is not None
#         cmd = pidmap.df[pidmap.df["pid"] == pid]
#         if len(cmd) > 1:
#             # XXX PID reuse is not supported yet
#             self.logger.warning("Multiple commands for pid %d (%s), PID reuse not yet supported", pid, cmd)
#             return
#         elif len(cmd) == 0:
#             self.logger.warning("No commands for pid %d, can not register binary", pid)
#             return
#         cmd = Path(cmd["command"][0])
#         addrspace_key = cmd.name
#         self.logger.debug("Register binary mapping %s at 0x%x for %s", path, map_addr, addrspace_key)
#         # XXX We should use both PID and command name to distinguish between different runs?
#         self.sym_resolver.register_address_space(cmd.name)
#         self.sym_resolver.add_mapped_binary(cmd.name, map_addr, path)
#         self.dwarf_helper.register_object(path.name, path)

#     async def build_run_script(self):
#         """
#         Generate the benchmark run script for the benchmark.

#         This is asynchronous as it is reused as a partial run step.
#         """
#         return self._build_remote_script()

#     async def run(self, instance_manager: InstanceManager) -> bool:
#         """
#         Run the benchmark and extract the results.
#         This involves the steps:
#         1. Build the remote run script
#         2. Ask the instance manager for an instance that we can run on
#         3. Connect to the instance, upload the run script and run it to completion.
#         4. Extract results

#         :param instance_manager: The global instance manager
#         :return: True if the benchmark completed successfully
#         """
#         script_path = self.get_run_script_path()
#         remote_script = Path(f"{self.config.name}-{self.uuid}.sh")
#         script = await self.build_run_script()
#         self.logger.info("Waiting for instance")

#         for dset in sorted(self._dataset_runner_modules.values(), key=op.attrgetter("dataset_run_order")):
#             dset.before_run()

#         instance = await instance_manager.request_instance(self.uuid, self.config.instance)
#         try:
#             await instance.connect()
#             self.logger.debug("Import script file host:%s => guest:%s", script_path, remote_script)
#             await instance.import_file(script_path, remote_script, preserve=True)
#             self.logger.info("Execute benchmark script verbose=%s", self.user_config.verbose)
#             with timing("Benchmark script completed", logger=self.logger):
#                 await instance.run_cmd("sh", [remote_script], verbose=self.user_config.verbose)
#             await self._extract_results(script, instance)

#             self.logger.info("Generate extra datasets")
#             for dset in sorted(self._dataset_runner_modules.values(), key=op.attrgetter("dataset_run_order")):
#                 await dset.after_extract_results(script, instance)

#             return True
#         except Exception as ex:
#             self.logger.exception("Benchmark run failed: %s", ex)
#         finally:
#             self.logger.info("Benchmark run completed")
#             await instance_manager.release_instance(self.uuid)
#         return False

#     def load(self):
#         """
#         Setup benchmark metadata and load results into datasets from the currently assigned run configuration.
#         Note: this runs in the aio loop executor asyncronously
#         """
#         # XXX move this to a pluggable dataset
#         # Always load the pidmap dataset first
#         self.logger.info("Loading pidmap data")
#         self._dataset_modules[DatasetName.PIDMAP].load()
#         for i in range(self.config.iterations):
#             self._dataset_modules[DatasetName.PIDMAP].load_iteration(i)
#         for dset in self._dataset_modules.values():
#             if dset.dataset_config_name == DatasetName.PIDMAP:
#                 # We loaded it before
#                 continue
#             self.logger.info("Loading %s data, dropping first %d iterations", dset.name, self.config.drop_iterations)
#             for i in range(self.config.iterations):
#                 try:
#                     if i >= self.config.drop_iterations:
#                         dset.load_iteration(i)
#                 except Exception:
#                     self.logger.exception("Failed to load data for %s iteration %d", dset.name, i)
#             dset.load()

#     def pre_merge(self):
#         """
#         Perform pre-processing step for all datasets. This may generate derived fields.
#         Note: this runs in the aio loop executor asyncronously
#         """
#         for dset in self._dataset_modules.values():
#             self.logger.info("Pre-process %s", dset.name)
#             dset.pre_merge()

#     def load_and_pre_merge(self):
#         """Shortcut to perform both the load and pre-merge steps"""
#         self.load()
#         self.pre_merge()

#     def merge(self, others: typing.List["Benchmark"]):
#         """
#         Merge datasets from compatible runs into a single dataset.
#         Note that this is called only on the baseline benchmark instance
#         """
#         self.logger.debug("Merge datasets %s onto baseline %s", [str(b) for b in others], self.uuid)
#         for dset in self._dataset_modules.values():
#             dset.init_merge()
#         for bench in others:
#             self.logger.debug("Merge %s(%s) instance='%s'", bench.config.name, bench.uuid, bench.config.instance.name)
#             self.merged_benchmarks[bench.uuid] = bench
#             for parser_id, dset in bench._dataset_modules.items():
#                 self._dataset_modules[parser_id].merge(dset)
#         for dset in self._dataset_modules.values():
#             dset.post_merge()

#     def aggregate(self):
#         """
#         Generate dataset aggregates (e.g. mean and quartiles)
#         Note that this is called only on the baseline benchmark instance
#         """
#         self.logger.debug("Aggregate datasets %s", self.config.name)
#         for dset in self._dataset_modules.values():
#             dset.aggregate()
#             dset.post_aggregate()

#     def analyse(self, run_context: "SessionAnalysisContext", analysis_config: AnalysisConfig):
#         """
#         Run analysis steps on this benchmark. This includes plotting.
#         Currently there is no ordering guarantee among analysis steps.
#         Note that this is called only on the baseline benchmark instance
#         """
#         self.logger.debug("Analise %s", self.config.name)
#         analysers = self._collect_analysers(analysis_config, cross_analysis=False)
#         self.logger.debug("Resolved analysis steps %s", [str(a) for a in analysers])
#         for handler in analysers:
#             run_context.schedule_task(handler.process_datasets())

#     def cross_merge(self, others: typing.List["Benchmark"]):
#         """
#         Merge the aggregated frames from all other benchmark parametrized variants.
#         """
#         self.logger.debug("Cross merge parameterized variants %s onto %s", [b for b in others], self.uuid)
#         for dset in self._dataset_modules.values():
#             dset.init_cross_merge()
#         for bench in others:
#             self.logger.debug("Merge %s(%s) param-set: %s", bench.config.name, bench.uuid, bench.config.parameters)
#             self.cross_merged[bench.uuid] = bench
#             for parser_id, dset in bench._dataset_modules.items():
#                 self._dataset_modules[parser_id].cross_merge(dset)
#         for dset in self._dataset_modules.values():
#             dset.post_cross_merge()

#     def cross_analysis(self, run_context: "SessionAnalysisContext", analysis_config: AnalysisConfig):
#         """
#         Perform any analysis steps on the merged frame with all the parameterized
#         benchmark variants.
#         """
#         self.logger.debug("Cross-analise")
#         analysers = self._collect_analysers(analysis_config, cross_analysis=True)
#         self.logger.debug("Resolved cross analysis steps %s", [str(a) for a in analysers])
#         for handler in analysers:
#             run_context.schedule_task(handler.process_datasets())

#     ############# New task API ############
#     def build_exec_task(self):
#         return BenchmarkExecutionTask(self)
