import asyncio as aio
import io
import json
import re
import typing
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import asyncssh
import pandas as pd

from ..pmc import PMCStatData
from .analysis import BenchmarkAnalysisRegistry
from .config import TemplateConfig, TemplateConfigContext
from .dataset import (DatasetArtefact, DataSetContainer, DatasetName, DatasetRegistry)
from .elf import DWARFHelper, Symbolizer
from .instance import InstanceConfig, InstanceInfo, PlatformOptions
from .pidmap import PidMapDataset
from .plot import BenchmarkPlot
from .procstat import ProcstatDataset
from .util import new_logger, timing


class BenchmarkError(Exception):
    def __init__(self, benchmark, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark = benchmark
        self.benchmark.logger.error(str(self))

    def __str__(self):
        msg = super().__str__()
        return f"BenchmarkError: {msg} on benchmark instance {self.benchmark.uuid}"


@dataclass
class BenchmarkDataSetConfig(TemplateConfig):
    """
    Define a dataset generated by the benchmark.

    Attributes
    type: identifier of the dataset handler that generates and imports the dataset
    run_options: dataset-specific options to produce the dataset
    """
    type: DatasetName
    run_options: dict[str, any] = field(default_factory=dict)


@dataclass
class BenchmarkRunConfig(TemplateConfig):
    """
    Define a benchmark to run.
    The benchmark configuration will be associated to an instance configuration
    for running the benchmark.
    Note that dataset names generate template replacements for each dataset file.

    Attributes
    name: display name for the benchmark setup
    iterations: the number of iterations of the benchmark to run
    desc: human-readable benchmark setup description
    benchmark_dataset: Configuration for the dataset handler that runs the actual benchmark
    datasets: Additional datasets configuration. Each entry describes a dataset handler that
      is used to generate additional information about the benchmark and process it.
    drop_iterations: Number of iterations to drop from the beginning. These are considered
    unreliable as the benchmark is priming the system caches/buffers.
    """
    name: str
    iterations: int
    benchmark_dataset: BenchmarkDataSetConfig
    datasets: dict[str, BenchmarkDataSetConfig]
    drop_iterations: int = 0
    desc: str = ""


@dataclass
class BenchmarkRunRecord:
    """
    Record the execution of a benchmark for post-processing. This archives the instance and
    benchmark configurations bound to a specific run of the benchmark, with template parameters
    fully resolved.
    """
    uuid: uuid.UUID
    instance: InstanceConfig
    run: BenchmarkRunConfig


@dataclass
class CommandHistoryEntry:
    """Internal helper to bind commands to PIDs of running tasks"""
    cmd: str
    pid: int = None


@dataclass
class BenchmarkScriptCommand:
    """
    Represent a single command executed by the benchmark script
    """
    cmd: str
    args: list[str]
    env: dict[str, str]
    background: bool = False
    collect_pid: bool = True
    remote_output: Path = None
    local_output: Path = None
    extra_output: dict[Path, Path] = field(default_factory=dict)
    extractfn: typing.Callable[["BenchmarkBase", Path, Path], None] = None

    def build_sh_command(self, cmd_index: int):
        cmdargs = f"{self.cmd} " + " ".join(map(str, self.args))
        env_vars = [f"{name}={value}" for name, value in self.env.items()]
        if self.remote_output:
            output_redirect = f" >> {self.remote_output}"
        else:
            output_redirect = " >> /dev/null"
        envstr = " ".join(env_vars)
        if self.background:
            cmdline = f"{envstr} {cmdargs} {output_redirect} &\n"
            if self.collect_pid:
                cmdline += f"PID_{cmd_index}=$!"
        else:
            cmdline = f"{cmdargs} {output_redirect}"
            if self.collect_pid:
                cmdline = f"PID_{cmd_index}=`sh -c \"echo \\\\$\\\\$; {envstr} exec {cmdline}\"`"
            else:
                cmdline = f"{envstr} {cmdline}"
        return cmdline


class BenchmarkScript:
    """
    Generate a shell script that runs the benchmark steps as they are scheduled by
    datasets.
    When the script is done, data is extracted from the guest as needed and the
    datasets get a chance to perform a post-processing step.
    """
    @dataclass
    class VariableRef:
        name: str

        def __str__(self):
            return f"${{{self.name}}}"

    def __init__(self, benchmark):
        self.benchmark = benchmark
        # Main command list
        self._commands = []
        # Intentionally relative to the run-script location.
        # We may want to add a knob to be able to store these in tmpfs or other places.
        self._guest_output = Path("benchmark-output")

        self._prepare_guest_output_dirs()

    def _add_command(self, command, args, env=None, collect_pid=False):
        env = env or {}
        cmd = BenchmarkScriptCommand(command, args, env)
        cmd.collect_pid = collect_pid
        self._commands.append(cmd)

    def _prepare_guest_output_dirs(self):
        """
        Prepare the guest output environment to store the files to mirror the local benchmark instance
        output directory.
        This makes it easier to share file paths by just using directories relative to the benchmark
        output_path.
        """
        self._add_command("mkdir", [self._guest_output])
        for i in range(self.benchmark.config.iterations):
            self._add_command("mkdir", [self._guest_output / str(i)])

    def local_to_remote_path(self, host_path: Path) -> Path:
        assert host_path.is_absolute(), "Ensure host_path is absolute"
        assert host_path.is_relative_to(self.benchmark.get_output_path()), "Ensure host_path is in benchmark output"
        return self._guest_output / host_path.relative_to(self.benchmark.get_output_path())

    def command_history_path(self):
        return self.benchmark.get_output_path() / "command-history.csv"

    def get_commands_with_pid(self):
        """Return a list of commands for which we recorded the PID in the command history"""
        commands = []
        for cmd in self._commands:
            if cmd.collect_pid:
                commands.append(cmd.cmd)
        return commands

    def get_variable(self, name: str) -> VariableRef:
        return self.VariableRef(name)

    def gen_cmd(self,
                command: str,
                args: list,
                outfile: Path = None,
                env: dict[str, str] = None,
                extractfn=None,
                extra_outfiles: list[Path] = []):
        """
        Add a foreground command to the run script.
        If the output is to be captured, the outfile argument specifies the host path in which it will be
        extracted. The host path must be within the benchmark instance output path
        (see BenchmarkBase.get_output_path()), the guest output path will be derived automatically from it.
        If extra post-processing should be performed upon file extraction, a callback can be given via
        extractfn. This function will be called to extract the remote file to the output file as
        `extractfn(benchmark, remote_path, host_path)`.
        """
        env = env or {}
        cmd = BenchmarkScriptCommand(command, args, env)
        if outfile is not None:
            cmd.remote_output = self.local_to_remote_path(outfile)
        cmd.local_output = outfile
        cmd.extra_output = {self.local_to_remote_path(p): p for p in extra_outfiles}
        cmd.extractfn = extractfn
        self._commands.append(cmd)

    def gen_bg_cmd(self,
                   command: str,
                   args: list,
                   outfile: Path = None,
                   env: dict[str, str] = None,
                   extractfn=None) -> VariableRef:
        """
        Similar to add_cmd() but will return an handle that can be used at a later time to terminate the
        background process.
        """
        env = env or {}
        cmd = BenchmarkScriptCommand(command, args, env)
        if outfile is not None:
            cmd.remote_output = self.local_to_remote_path(outfile)
        cmd.local_output = outfile
        cmd.extractfn = extractfn
        cmd.background = True
        cmd.collect_pid = True
        self._commands.append(cmd)
        cmd_index = len(self._commands) - 1
        return self.VariableRef(f"PID_{cmd_index}")

    def gen_stop_bg_cmd(self, command_handle: VariableRef):
        self._add_command("kill", ["-TERM", command_handle])

    def gen_sleep(self, seconds: int):
        self._add_command("sleep", [seconds])

    def to_shell_script(self, fd: typing.IO[str]):
        fd.write("#!/bin/sh\n\n")
        for i, cmd in enumerate(self._commands):
            fd.write(cmd.build_sh_command(i))
            fd.write("\n")
        # Dump all the collected PIDs in the command history metadata file
        command_history = self.command_history_path()
        pid_history_path = self.local_to_remote_path(command_history)
        for i, cmd in enumerate(self._commands):
            if cmd.collect_pid:
                var = self.get_variable(f"PID_{i}")
                fd.write(f"echo {var} >> {pid_history_path}\n")

    def get_extract_files(self) -> list[tuple[Path, Path, typing.Optional[typing.Callable]]]:
        """
        Get a list of all files to extract for the benchmark.
        Each list item is a tuple of the form (remote_file, local_file, extract_fn)
        If a custom extract function is given, it will be passed along as the
        last tuple item for each file.
        """
        entries = []
        for cmd in self._commands:
            if cmd.remote_output:
                assert cmd.local_output, "Missing local output file"
                entries.append((cmd.remote_output, cmd.local_output, cmd.extractfn))
            if cmd.extra_output:
                for remote_path, local_path in cmd.extra_output.items():
                    entries.append((remote_path, local_path, cmd.extractfn))
        return entries


class BenchmarkBase(TemplateConfigContext):
    """
    Base class for all the benchmarks
    """
    def __init__(self,
                 manager: "BenchmarkManager",
                 config: BenchmarkRunConfig,
                 instance_config: InstanceConfig,
                 run_id=None):
        super().__init__()
        if run_id:
            self.uuid = run_id
        else:
            self.uuid = uuid.uuid4()
        self.logger = new_logger(f"{config.name}:{instance_config.name}")
        self.manager = manager
        self.instance_manager = manager.instance_manager
        self.manager_config = manager.config
        self.instance_config = instance_config
        self.config = config

        # InstanceInfo of the instance we have allocated
        self._reserved_instance = None
        # Connection to the CheriBSD instance
        self._conn = None
        # Benchmark script assembler
        self._script = BenchmarkScript(self)
        # Map uuids to benchmarks that have been merged into the current instance (which is the baseline)
        # so that we can look them up if necessary
        self.merged_benchmarks = {}
        # Symbol mapping handler for this benchmark instance
        self.sym_resolver = Symbolizer(self)
        # Dwarf information extraction helper
        self.dwarf_helper = DWARFHelper(self)

        self._bind_early_configs()
        self.datasets = {}
        self._collect_datasets()
        self._bind_configs()

        rootfs_path = self.manager_config.sdk_path / f"rootfs-{self.instance_config.cheri_target}"
        rootfs_path = rootfs_path.expanduser()
        if not rootfs_path.exists() or not rootfs_path.is_dir():
            raise Exception(f"Invalid rootfs path {rootfs_path} for benchmark instance")
        self.rootfs = rootfs_path
        # Ensure that we have the main output directory
        self._result_path = self.manager.session_output_path / str(self.uuid)
        self._result_path.mkdir(parents=True, exist_ok=True)
        # Ensure that we have the plot output directory
        self.get_plot_path().mkdir(parents=True, exist_ok=True)

        self._configure_datasets()
        self.logger.info("Benchmark instance with UUID=%s", self.uuid)
        # Dataset setup summary
        for did, dset in self.datasets.items():
            role = "benchmark" if did == self.config.benchmark_dataset.type else "aux"
            dataset_artefact = dset.dataset_source_id
            generator = self.datasets_gen[dataset_artefact]
            if generator == dset:
                generator_status = f"generator for {dataset_artefact}"
            else:
                generator_status = f"depends on {dataset_artefact}"
            self.logger.debug("Configured %s dataset: %s (%s) %s", role, dset.name, dset.__class__.__name__,
                              generator_status)

    def _bind_early_configs(self):
        """First pass of configuration template subsitution"""
        self.register_template_subst(uuid=self.uuid,
                                     cheri_target=self.instance_config.cheri_target,
                                     session=self.manager.session)
        self.instance_config = self.instance_config.bind(self)
        self.config = self.config.bind(self)

    def _bind_configs(self):
        """Second pass of configuration template substitution"""
        # template_params = {dset.name.replace("-", "_"): dset.output_file() for dset in self.datasets.values()}
        # self.register_template_subst(**template_params)
        self.instance_config = self.instance_config.bind(self)
        self.config = self.config.bind(self)

    def _configure_datasets(self):
        """Resolve platform options for the instance configuration and finalize dataset configuration"""
        opts = PlatformOptions()
        for dset in self.datasets.values():
            opts = dset.configure(opts)
        self.instance_config.platform_options = opts

    def _collect_datasets(self):
        """
        Initialize dataset instances from the configuration.
        Note that this must happen before we perform the second pass of configuration template resolution
        XXX-AM: do we really care about template substitution in config anymore?
        """
        # The main dataset for the benchmark
        self.datasets[self.config.benchmark_dataset.type] = self._get_dataset_handler(
            "benchmark", self.config.benchmark_dataset)
        # Implicit auxiliary datasets
        # Procstat dataset should be added in configuration file as it depends on the benchmark
        # self.datasets[DatasetArtefact.PROCSTAT] = self._get_dataset_handler(
        #     "procstat", BenchmarkDataSetConfig(type=DatasetName.PROCSTAT))
        self.datasets[DatasetArtefact.PIDMAP] = self._get_dataset_handler(
            "pidmap", BenchmarkDataSetConfig(type=DatasetName.PIDMAP))
        # Extra datasets configured
        for name, config in self.config.datasets.items():
            assert config.type not in self.datasets, "Duplicate dataset name"
            self.datasets[config.type] = self._get_dataset_handler(name, config)

        # Collect the datasets that are supposed to generate output.
        # These are only used for running the benchmark.
        # If multiple datasets have the same dataset_source_id, assume it to be equivalent and
        # just pick one of them (this is the case for commands that produce multiple datasets).
        self.datasets_gen = {}
        for dset in self.datasets.values():
            if dset.dataset_source_id not in self.datasets_gen:
                self.datasets_gen[dset.dataset_source_id] = dset

    def _get_dataset_handler(self, dset_key: str, config: BenchmarkDataSetConfig):
        """Resolve the parser for the given dataset"""
        ds_name = DatasetName(config.type)
        handler_class = DatasetRegistry.resolve_name(ds_name)
        handler = handler_class(self, dset_key, config)
        return handler

    def _dataset_generators_sorted(self, reverse=False) -> list[DataSetContainer]:
        return sorted(self.datasets_gen.values(), key=lambda ds: ds.dataset_run_order, reverse=reverse)

    def get_output_path(self):
        """
        Get base output path for the current benchmark instance.
        This can be used as a base path or to store data that is not specific to a single iteration
        of the benchmark.
        """
        return self._result_path

    def get_plot_path(self):
        """
        Get base plot path for the current benchmark instance.
        Every plot should use this path as the base path to generate plots.
        """
        group_name = self.config.name
        return self.manager.plot_output_path / group_name

    def get_iter_output_path(self, iteration: int):
        """
        Return the output directory path for the current benchmark iteration.
        This should be used by all datasets to store the output files.
        """
        iter_path = self.get_output_path() / str(iteration)
        iter_path.mkdir(exist_ok=True)
        return iter_path

    def get_dataset(self, name: DatasetName):
        return self.datasets.get(name, None)

    def get_dataset_by_artefact(self, ds_id: DatasetArtefact):
        """
        Lookup a generic dataset by the artefact ID.
        Note that this will fail if there are multiple matches
        """
        match = [dset for dset in self.datasets.values() if dset.dataset_source_id == ds_id]
        if len(match) > 1:
            raise KeyError("Multiple matching dataset for artefact %s", ds_id)
        if len(match):
            return match[0]
        return None

    def get_script_builder(self) -> BenchmarkScript:
        return self._script

    def _record_benchmark_run(self):
        record = BenchmarkRunRecord(uuid=self.uuid, instance=self.instance_config, run=self.config)
        self.manager.record_benchmark(record)

    def _build_remote_script(self) -> Path:
        """
        Build the remote benchmark script to run
        """
        bench_dset = self.get_dataset(self.config.benchmark_dataset.type)
        assert bench_dset, "Missing benchmark dataset"
        pre_generators = self._dataset_generators_sorted()
        post_generators = self._dataset_generators_sorted(reverse=True)
        self.logger.info("Generate benchmark script")

        for dset in pre_generators:
            dset.gen_pre_benchmark()

        for i in range(self.config.iterations):
            self.logger.info("Generate benchmark iteration %d", i)
            for dset in pre_generators:
                dset.configure_iteration(i)
            for dset in pre_generators:
                dset.gen_pre_benchmark_iter(i)
            # Only run the benchmark step for the given benchmark_dataset
            bench_dset.gen_benchmark(i)
            for dset in post_generators:
                dset.gen_post_benchmark_iter(i)

        for dset in post_generators:
            dset.gen_post_benchmark()

        script_path = self._result_path / "runner-script.sh"
        with open(self._result_path / "runner-script.sh", "w+") as script:
            self._script.to_shell_script(script)
        script_path.chmod(0o755)
        return script_path

    async def run_ssh_cmd(self, command: str, args: list = [], env: dict = {}):
        self.logger.debug("SH exec: %s %s", command, args)
        cmdline = f"{command} " + " ".join(map(str, args))
        result = await self._conn.run(cmdline)
        if result.returncode != 0:
            self.logger.error("Failed to run %s: %s", command, result.stderr)
        else:
            self.logger.debug("%s done: %s", command, result.stdout)

    async def run_nohup_ssh_cmd(self, command: str, args: list = [], env: dict = {}):
        self.logger.debug("NOHUP ssh command not yet implemented")
        await self.run_ssh_cmd(command, args, env)

    async def extract_file(self, guest_src: Path, host_dst: Path, **kwargs):
        """Extract file from instance"""
        src = (self._conn, guest_src)
        await asyncssh.scp(src, host_dst, **kwargs)

    async def import_file(self, host_src: Path, guest_dst: Path, **kwargs):
        """Import file into instance"""
        dst = (self._conn, guest_dst)
        await asyncssh.scp(host_src, dst, **kwargs)

    async def read_remote_file(self, guest_src: Path, target_fd: typing.TextIO):
        result = await self._conn.run(f"cat {guest_src}")
        if result.returncode != 0:
            self.logger.error("Failed to read remote file %s: %s", guest_src, result.stderr)
        else:
            self.logger.debug("Done reading remote file %s", guest_src)
            # Do this here so that we don't automatically close() the target_fd
            target_fd.write(result.stdout)

    async def _connect_instance(self, info: InstanceInfo):
        conn = await asyncssh.connect(info.ssh_host,
                                      port=info.ssh_port,
                                      known_hosts=None,
                                      client_keys=[self.manager_config.ssh_key],
                                      username="root",
                                      passphrase="")
        self.logger.debug("Connected to instance")
        return conn

    async def _extract_results(self):
        for remote_path, local_path, custom_extract_fn in self._script.get_extract_files():
            self.logger.debug("Extract %s -> %s", remote_path, local_path)
            if custom_extract_fn:
                await custom_extract_fn(remote_path, local_path)
            else:
                await self.extract_file(remote_path, local_path)

        # Extract also the implicit command history
        cmd_history = self._script.command_history_path()
        remote_cmd_history = self._script.local_to_remote_path(cmd_history)
        self.logger.debug("Extract %s -> %s", remote_cmd_history, cmd_history)
        await self.extract_file(remote_cmd_history, cmd_history)

    async def run(self):
        remote_script = Path(f"{self.config.name}-{self.uuid}.sh")
        script_path = self._build_remote_script()
        self.logger.info("Waiting for instance")
        self._reserved_instance = await self.instance_manager.request_instance(self.uuid, self.instance_config)
        try:
            self._conn = await self._connect_instance(self._reserved_instance)
            await self.import_file(script_path, remote_script, preserve=True)
            self.logger.info("Execute benchmark script")
            if self.manager_config.verbose:
                # run inline
                with timing("Benchmark script completed", logger=self.logger):
                    await self.run_ssh_cmd("sh", [remote_script])
            else:
                await self.run_nohup_ssh_cmd("sh", [remote_script])

            await self._extract_results()

            self.logger.info("Generate extra datasets")
            for dset in self._dataset_generators_sorted():
                await dset.after_extract_results()

            # Record successful run and cleanup any pending background task
            self._record_benchmark_run()
        except Exception as ex:
            self.logger.exception("Benchmark run failed: %s", ex)
            self.manager.failed_benchmarks.append(self)
        finally:
            self.logger.info("Benchmark run completed")
            await self.instance_manager.release_instance(self.uuid)

    def _load_kernel_symbols(self):
        kernel = self.rootfs / "boot" / f"kernel.{self.instance_config.kernel}" / "kernel.full"
        if not kernel.exists():
            self.logger.warning("Kernel name not found in kernel.<CONF> directories, using the default kernel")
            kernel = self.rootfs / "kernel" / "kernel.full"
        self.sym_resolver.register_sym_source(0, "kernel.full", kernel, shared=True)
        arch_pointer_size = self.instance_config.kernel_pointer_size
        self.dwarf_helper.register_object("kernel.full", kernel, arch_pointer_size)

    def register_mapped_binary(self, map_addr: int, path: Path):
        """
        Add a binary to the symbolizer for this benchmark.
        The symbols will be associated with the current benchmark address-space.
        """
        bench_dset = self.get_dataset(self.config.benchmark_dataset.type)
        addrspace = bench_dset.get_addrspace_key()
        self.sym_resolver.register_sym_source(map_addr, addrspace, path)
        self.dwarf_helper.register_object(path.name, path)

    def get_benchmark_group(self):
        """
        Return dictionary of the baseline and merged benchmarks in this group
        Note: Only sensible after the merge step on the baseline instance.
        """
        assert self.instance_config.baseline
        group = {self.uuid: self}
        group.update(self.merged_benchmarks)
        return group

    def load(self):
        """
        Setup benchmark metadata and load results into datasets from the currently assigned run configuration.
        Note: this runs in the aio loop executor asyncronously
        """
        self._load_kernel_symbols()
        for dset in self.datasets.values():
            self.logger.info("Loading %s data, dropping first %d iterations", dset.name, self.config.drop_iterations)
            for i in range(self.config.iterations):
                if i >= self.config.drop_iterations:
                    dset.load_iteration(i)
            dset.load()

    def pre_merge(self):
        """
        Perform pre-processing step for all datasets. This may generate derived fields.
        Note: this runs in the aio loop executor asyncronously
        """
        for dset in self.datasets.values():
            self.logger.info("Pre-process %s", dset.name)
            dset.pre_merge()

    def load_and_pre_merge(self):
        """Shortcut to perform both the load and pre-merge steps"""
        self.load()
        self.pre_merge()

    def merge(self, others: list["BenchmarkBase"]):
        """
        Merge datasets from compatible runs into a single dataset.
        Note that this is called only on the baseline benchmark instance
        """
        self.logger.debug("Merge datasets %s onto baseline %s", [str(b) for b in others], self.uuid)
        for dset in self.datasets.values():
            dset.init_merge()
        for bench in others:
            self.logger.debug("Merge %s(%s) instance='%s'", bench.config.name, bench.uuid, bench.instance_config.name)
            self.merged_benchmarks[bench.uuid] = bench
            for parser_id, dset in bench.datasets.items():
                self.datasets[parser_id].merge(dset)
        for dset in self.datasets.values():
            dset.post_merge()

    def aggregate(self):
        """
        Generate dataset aggregates (e.g. mean and quartiles)
        Note that this is called only on the baseline benchmark instance
        """
        self.logger.debug("Aggregate datasets %s", self.config.name)
        for dset in self.datasets.values():
            dset.aggregate()
            dset.post_aggregate()

    def analyse(self):
        """
        Run analysis steps on this benchmark. This includes plotting.
        Currently there is no ordering guarantee among analysis steps.
        Note that this is called only on the baseline benchmark instance
        """
        self.logger.debug("Analise %s", self.config.name)
        analysers = []
        for handler_class in BenchmarkAnalysisRegistry.analysis_steps:
            if handler_class.check_required_datasets(self.datasets.keys()):
                analysers.append(handler_class(self))
        self.logger.debug("Resolved analysys steps %s", [str(a) for a in analysers])
        for handler in analysers:
            handler.process_datasets()

    def __str__(self):
        return f"{self.config.name}({self.uuid})"
