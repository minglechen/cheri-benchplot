import logging
import io
import re
import time
import uuid
import typing
import json
import asyncio as aio
import traceback
from contextlib import contextmanager
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

import pandas as pd
import asyncssh

from .procstat import ProcstatDataset
from .pidmap import PidMapDataset
from .instanced import InstanceConfig, BenchmarkInfo
from .config import TemplateConfig, TemplateConfigContext
from .dataset import DataSetParser
from .elf import SymResolver
from ..pmc import PMCStatData
from ..qemu_stats import (QEMUStatsBBHistogramDataset, QEMUStatsBranchHistogramDataset)


@contextmanager
def timing(name, logger=None):
    if logger is None:
        logger = logging
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        logger.info("%s in %.2fs", name, end - start)


class BenchmarkError(Exception):
    def __init__(self, benchmark, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark = benchmark
        self.benchmark.logger.error(str(self))

    def __str__(self):
        msg = super().__str__()
        return f"BenchmarkError: {msg} on benchmark instance {self.benchmark.uuid}"


class BenchmarkType(Enum):
    NETPERF = "netperf"
    TEST = "TEST"  # Reserved for tests

    def __str__(self):
        return self.value


@dataclass
class BenchmarkDataSetConfig(TemplateConfig):
    """
    Define a dataset generated by the benchmark.

    Attributes
    name: name of the output file to generate
    parser: the dataset parser that imports the dataset for plotting
    """
    name: str
    parser: DataSetParser


@dataclass
class BenchmarkRunConfig(TemplateConfig):
    """
    Define a benchmark to run.
    The benchmark configuration will be associated to an instance configuration
    for running the benchmark.

    Attributes
    name: display name for the benchmark setup
    type: type of the benchmark. This selects the factory for the benchmark driver task
    output_file: main benchmark output file XXX may merge this with extra_files...
    benchmark_options: benchmark-specific options, depend on the benchmark type
    datasets: description of the outputs of the benchmark. Note that these will generate
      template replacements for the current benchmark but will not otherwise impact the
      benchmark configuration. They still need to be placed in the command line options,
      extra_files and ouput_file as needed.
    desc: human-readable benchmark setup description
    env: extra environment variables to set
    extra_files: extra files that the benchmark generates and need to be extracted from the
      guest
    """
    name: str
    type: BenchmarkType
    benchmark_options: dict[str, any]
    datasets: dict[str, BenchmarkDataSetConfig]
    output_file: typing.Optional[Path] = None
    desc: str = ""
    env: dict = field(default_factory=dict)
    extra_files: list[str] = field(default_factory=list)


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


class BenchmarkBase(TemplateConfigContext):
    """
    Base class for all the benchmarks
    """
    def __init__(self, manager, config, instance_config, run_id=None):
        super().__init__()
        if run_id:
            self.uuid = run_id
        else:
            self.uuid = uuid.uuid4()
        self.manager = manager
        self.daemon = manager.instance_manager
        self.manager_config = manager.config
        self.instance_config = instance_config
        self.config = config
        self._bind_configs()

        rootfs_path = self.manager_config.sdk_path / f"rootfs-{self.instance_config.cheri_target}"
        rootfs_path = rootfs_path.expanduser()
        if not rootfs_path.exists() or not rootfs_path.is_dir():
            raise Exception(f"Invalid rootfs path {rootfs_path} for benchmark instance")
        self.rootfs = rootfs_path

        self.result_path = self.manager.session_output_path / str(self.uuid)
        self.result_path.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"{config.name}:{instance_config.name}:{self.uuid}")
        self._reserved_instance = None  # BenchmarkInfo of the instance the daemon has reserved us
        self._conn = None  # Connection to the CheriBSD instance
        self._command_tasks = []  # Commands being run on the instance
        # Datasets loaded for the benchmark
        self.datasets = {}
        # Map uuids to benchmarks that have been merged into the current instance (which is the baseline)
        # so that we can look them up if necessary
        self.merged_benchmarks = {}
        # Plots to show for this benchmark. Note: this is only relevant for baseline instances
        self.plots = []
        self.sym_resolver = SymResolver(self)
        # Extra implicit dataset to extract procstat -v mappings
        self.procstat_output = self.result_path / f"procstat-{self.uuid}.csv"
        # Extra implicit dataset to extract PID to command mapping
        self.pid_map_output = self.result_path / f"pid-map-{self.uuid}.json"

    def _bind_configs(self):
        """
        Bind configurations to the current benchmark instance.
        This is done in two passes: First we resolve all the base template parameters and datasets,
        the second pass replaces the template parameters generated by the datasets.
        """
        self.register_template_subst(uuid=self.uuid,
                                     cheri_target=self.instance_config.cheri_target,
                                     session=self.manager.session)
        # First pass
        instance_config = self.instance_config.bind(self)
        config = self.config.bind(self)
        template_params = {param.replace("-", "_"): dataset.name for param, dataset in config.datasets.items()}
        self.register_template_subst(**template_params)
        # Second pass
        self.instance_config = instance_config.bind(self)
        self.config = config.bind(self)
        self._remote_task_pid = {}

    def _record_benchmark_run(self):
        record = BenchmarkRunRecord(uuid=self.uuid, instance=self.instance_config, run=self.config)
        self.manager.record_benchmark(record)

    def register_plot(self, plotter):
        self.plots.append(plotter)

    def _build_remote_command(self, cmd: str, args: list, env={}):
        """
        asyncss does not return the pid of the process so we need to print it.
        We also use this to work around any restriction on the environment variables
        we are able to set on the guest.
        The first thing that the remote command will print will contain the process PID,
        this is handled by the _cmd_io() loop.
        """
        cmdline = f"{cmd} " + " ".join(args)
        exports = [f"export {name}={value};" for name, value in env.items()]
        export_line = " ".join(exports)
        sh_cmd = ["sh", "-c", f"'echo $$;{export_line} exec {cmdline}'"]
        return sh_cmd

    def _parse_pid_line(self, ssh_task, line: str) -> int:
        """
        Parse the first line of the output with the process PID number
        """
        try:
            self.logger.debug("Attempt to resolve PID %s", line)
            pid = int(line)
            self._remote_task_pid[ssh_task] = pid
            self.logger.debug("Bind remote command %s to PID %d", ssh_task.command, pid)
        except ValueError:
            raise BenchmarkError(self, f"Can not determine running process pid for {ssh_task.command}, bailing out.")

    async def _cmd_io(self, proc_task, callback):
        try:
            while proc_task.returncode is None and not proc_task.stdout.at_eof():
                out = await proc_task.stdout.readline()
                if not out:
                    continue
                # Expect to receive the PID of the command in the first output line
                if proc_task not in self._remote_task_pid:
                    try:
                        self._parse_pid_line(proc_task, out)
                    except BenchmarkError as ex:
                        proc_task.terminate()
                        raise ex
                try:
                    if callback:
                        callback(out)
                except aio.CancelledError as ex:
                    raise ex
                except Exception as ex:
                    self.logger.error("Error while processing output for %s: %s", proc_task.command, ex)
                if out:
                    self.logger.debug(out)
        except aio.CancelledError as ex:
            proc_task.terminate()
            raise ex
        finally:
            self.logger.debug("Background task %s done", proc_task.command)

    async def _run_bg_cmd(self, command: str, args: list, env={}, iocallback=None):
        """Run a background command without waiting for termination"""
        remote_cmd = self._build_remote_command(command, args, env)
        self.logger.debug("SH exec background: %s", remote_cmd)
        proc_task = await self._conn.create_process(" ".join(remote_cmd))
        self._command_tasks.append(aio.create_task(self._cmd_io(proc_task, iocallback)))
        return proc_task

    async def _stop_bg_cmd(self, task):
        """
        Work around the unreliability of asyncssh/openssh signal delivery
        """
        try:
            pid = self._remote_task_pid[task]
            result = await self._conn.run(f"kill -TERM {pid}")
            if result.returncode != 0:
                self.logger.error("Failed to stop remote process %d: %s", pid, task.command)
            else:
                await task.wait()
        except KeyError:
            self.logger.error("Can not stop %s, missing pid", task.command)

    async def _run_cmd(self, command: str, args: list, env={}, outfile=None):
        """Run a command and wait for the process to complete"""
        remote_cmd = self._build_remote_command(command, args, env)
        self.logger.debug("SH exec: %s", remote_cmd)
        result = await self._conn.run(" ".join(remote_cmd))
        if result.returncode != 0:
            self.logger.error("Failed to run %s: %s", command, result.stderr)
        else:
            self.logger.debug("%s done: %s", command, result.stdout)
        stdout = io.StringIO(result.stdout)
        # Expect to receive the PID of the command in the first output line
        self._parse_pid_line(result, stdout.readline())
        if outfile:
            outfile.write(stdout.read())
        return result

    async def _extract_file(self, guest_src: Path, host_dst: Path):
        """Extract file from instance"""
        src = (self._conn, guest_src)
        await asyncssh.scp(src, host_dst)

    async def _import_file(self, host_src: Path, guest_dst: Path):
        """Import file into instance"""
        dst = (self._conn, guest_dst)
        await asyncssh.scp(host_src, dst)

    async def _connect_instance(self, info: BenchmarkInfo):
        conn = await asyncssh.connect(info.ssh_host,
                                      port=info.ssh_port,
                                      known_hosts=None,
                                      client_keys=[self.manager_config.ssh_key],
                                      username="root",
                                      passphrase="")
        self.logger.debug("Connected to instance")
        return conn

    async def _extract_pid_mappings(self):
        """
        Extract system PID to process mappings for dataset to use
        """
        self.logger.info("Extract system PIDs")
        pid_raw_data = io.StringIO()
        await self._run_cmd("ps", ["-a", "-x", "-o", "uid,pid,command", "--libxo", "json"], outfile=pid_raw_data)
        # Append the PIDs for all processes executed by the benchmark to the pid map
        pid_json = json.loads(pid_raw_data.getvalue())
        proc_list = pid_json["process-information"]["process"]
        for ssh_proc_task, pid in self._remote_task_pid.items():
            proc_list.append({"uid": 1, "pid": pid, "command": ssh_proc_task.command})
        with open(self.pid_map_output, "w+") as pid_fd:
            json.dump(pid_json, pid_fd)

    async def _run_benchmark(self):
        """
        Run the actual benchmark sequence.
        """
        self.logger.info("Running benchmark")

    async def _run_procstat(self):
        """
        Try to get virtual memory map for the benchmark
        """
        self.logger.info("Collect procstat info")

    async def run(self):
        self.logger.info("Waiting for instance")
        self._reserved_instance = await self.daemon.request_instance(self.uuid, self.instance_config)
        if self._reserved_instance is None:
            self.logger.error("Can not reserve instance, bailing out...")
            return
        try:
            self._conn = await self._connect_instance(self._reserved_instance)
            await self._run_procstat()
            with timing("Benchmark completed", self.logger):
                await self._run_benchmark()
            await self._extract_pid_mappings()
            self._record_benchmark_run()
            # Stop all pending background processes
            for t in self._command_tasks:
                t.cancel()
            await aio.gather(*self._command_tasks, return_exceptions=True)
        except Exception as ex:
            self.logger.error("Benchmark run failed: %s", ex)
            traceback.print_tb(ex.__traceback__)
            self.manager.failed_benchmarks.append(self)
        finally:
            await self.daemon.release_instance(self.uuid, self._reserved_instance)

    def _get_dataset_parser(self, dset_key: str, dset: BenchmarkDataSetConfig):
        """Resolve the parser for the given dataset"""
        if dset.parser == DataSetParser.PMC:
            parser = PMCStatData.get_parser(self, dset_key)
        elif dset.parser == DataSetParser.QEMU_STATS_BB_HIST:
            parser = QEMUStatsBBHistogramDataset.get_parser(self, dset_key)
        elif dset.parser == DataSetParser.QEMU_STATS_CALL_HIST:
            parser = QEMUStatsBranchHistogramDataset.get_parser(self, dset_key)
        else:
            self.logger.error("No parser for dataset %s", dset.name)
            raise Exception("No parser")
        return parser

    def _load_dataset(self, dset_key: str, dset: BenchmarkDataSetConfig):
        """Resolve the parser for the given dataset and import the target file"""
        parser = self._get_dataset_parser(dset_key, dset)
        parser.load(self.result_path / dset.name)
        return parser

    def _load_extra_data(self):
        kernel = self.rootfs / "boot" / f"kernel.{self.instance_config.kernel}" / "kernel.full"
        if not kernel.exists():
            self.logger.warning("Kernel name not found in kernel.<CONF> directories, using the default kernel")
            kernel = self.rootfs / "kernel" / "kernel.full"
        self.sym_resolver.import_symbols(kernel, 0)
        if self.procstat_output.exists():
            # If we have procstat output, import all the symbols
            self.logger.debug("Load implicit procstat dataset")
            pstat = ProcstatDataset(self, "procstat")
            pstat.load(self.procstat_output)
            self.datasets[DataSetParser.PROCSTAT] = pstat
            for base, guest_path in pstat.mapped_binaries(self.uuid):
                local_path = self.rootfs / guest_path.relative_to("/")
                self.sym_resolver.import_symbols(local_path, base)
        if self.pid_map_output.exists():
            # If we have the process PID mapping, import the dataset
            self.logger.debug("Load implicit PID dataset")
            pidmap = PidMapDataset(self, "pidmap")
            self.datasets[DataSetParser.PIDMAP] = pidmap

    def get_dataset(self, parser_id: DataSetParser):
        return self.datasets.get(parser_id, None)

    def load(self):
        """
        Setup benchmark metadata and load results into datasets from the currently assigned run configuration.
        """
        self._load_extra_data()
        for name, dset in self.config.datasets.items():
            self.logger.info("Loading %s from %s", name, dset.name)
            self.datasets[dset.parser] = self._load_dataset(name, dset)
        for dset in self.datasets.values():
            dset.pre_merge()

    def merge(self, others: list["BenchmarkBase"]):
        """
        Merge datasets from compatible runs into a single dataset.
        Note that this is called only on the baseline benchmark instance
        """
        self.logger.debug("Merge datasets %s onto baseline %s", [str(b) for b in others], self.uuid)
        for dset in self.datasets.values():
            dset.init_merge()
        for bench in others:
            self.merged_benchmarks[bench.uuid] = bench
            for parser_id, dset in bench.datasets.items():
                self.datasets[parser_id].merge(dset)
            for parser_id, dset in bench.datasets.items():
                self.datasets[parser_id].post_merge()

    def aggregate(self):
        """
        Generate dataset aggregates (e.g. mean and quartiles)
        Note that this is called only on the baseline benchmark instance
        """
        self.logger.debug("Aggregate datasets %s", self.config.name)
        for dset in self.datasets.values():
            dset.aggregate()
            dset.post_aggregate()

    def verify(self):
        """
        Verify the integrity of the aggregate / post-processed data
        """
        self.logger.debug("Verify dataset integrity for %s", self.config.name)

    def plot(self):
        """
        Plot the data from the generated datasets
        Note that this is called only on the baseline benchmark instance
        """
        self.logger.debug("Plot datasets")
        for plot in self.plots:
            plot.prepare()
            plot.draw()

    def __str__(self):
        return f"{self.config.name}:{self.uuid}"


class _BenchmarkBase:
    def plot(self):
        # Common libpmc input
        self.pmc = PMCStatData.get_pmc_for_cpu(self.cpu, self.options, self)
        """Entry point for plotting benchmark results or analysis files"""
        for dirpath in self.options.stats:
            if not dirpath.exists():
                fatal("Source directory {} does not exist".format(dirpath))
            self._load_dir(dirpath)
        logging.info("Process data")
        self._process_data_sources()
        self.merged_raw_data = self._merge_raw_data()
        self.merged_stats = self._merge_stats()
        logging.info("Generate relative data for baseline %s", self.options.baseline)
        self._compute_relative_stats()
        logging.info("Generate plots")
        self._draw()

    def pmc_map_index(self, df):
        """
        Map the progname and archname columns from statcounters
        to new columns to be used as part of the index, or an empty
        dataframe.
        """
        return pd.DataFrame()

    def _merge_raw_data(self):
        return self.pmc.df

    def _merge_stats(self):
        return self.pmc.stats_df

    def _process_data_sources(self):
        self.pmc.process()

    def _load_dir(self, path):
        for fpath in path.glob("*.csv"):
            self._load_file(path, fpath)

    def _load_file(self, dirpath, filepath):
        logging.info("Loading %s", filepath)
        if self._is_pmc_input(filepath):
            self._load_pmc(filepath)
