
import logging
import time
import pandas as pd
import uuid
import typing
import asyncio as aio
import asyncssh
from contextlib import contextmanager
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from subprocess import PIPE

from .instanced import InstanceConfig, BenchmarkInfo
from .options import TemplateConfig, TemplateConfigContext
from .dataset import DataSetParser
from ..netperf.config import NetperfBenchmarkRunConfig

from .cpu import BenchmarkCPU
from ..pmc import PMCStatData
from ..elf import ELFInfo, SymResolver


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

class BenchmarkType(Enum):
    NETPERF = "netperf"

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
    plots: the plots to generate for the benchmark
    """
    name: str
    type: BenchmarkType
    output_file: str
    benchmark_options: typing.Union[NetperfBenchmarkRunConfig]
    datasets: dict[str, BenchmarkDataSetConfig]
    desc: str = ""
    env: dict = field(default_factory=dict)
    plots: list[str] = field(default_factory=list)
    extra_files: list[str] = field(default_factory=list)


class BenchmarkBase(TemplateConfigContext):
    """
    Base class for all the benchmarks
    """

    def __init__(self, manager, config, instance_config):
        super().__init__()
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

        self.result_path = self.manager_config.output_path / str(self.uuid)
        self.result_path.mkdir(parents=True)

        self.logger = logging.getLogger(f"{config.name}:{instance_config.name}:{self.uuid}")
        self._reserved_instance = None  # BenchmarkInfo of the instance the daemon has reserved us
        self._conn = None  # Connection to the CheriBSD instance
        self._command_tasks = []  # Commands being run on the instance

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

    async def _cmd_io(self, proc_task, callback):
        try:
            while proc_task.returncode is None:
                out = await proc_task.stdout.readline()
                try:
                    if callback:
                        callback(out)
                except aio.CancelledError as ex:
                    raise ex
                except Exception as ex:
                    self.logger.error("Error while processing output for %s: %s",
                                      proc_task.command, ex)
                self.logger.debug(out)
        except aio.CancelledError as ex:
            proc_task.terminate()
            raise ex
        finally:
            self.logger.debug("Background task %s done", proc_task.command)


    async def _run_bg_cmd(self, command: str, args: list, env={}, iocallback=None):
        """Run a background command without waiting for termination"""
        cmdline = f"{command} " + " ".join(args)
        env_str = [f"{k}={v}" for k,v in env.items()]
        self.logger.debug("exec background: %s env=%s", cmdline, env)
        proc_task = await self._conn.create_process(cmdline, env=env_str)
        self._command_tasks.append(aio.create_task(self._cmd_io(proc_task, iocallback)))
        return proc_task

    async def _run_cmd(self, command: str, args: list, env={}, outfile=PIPE):
        """Run a command and wait for the process to complete"""
        cmdline = f"{command} " + " ".join(args)
        env_str = [f"{k}={v}" for k,v in env.items()]
        self.logger.debug("exec: %s env=%s", cmdline, env)
        result = await self._conn.run(cmdline, env=env_str, stdout=outfile)
        if result.returncode != 0:
            if outfile:
                cmdline += f" >> {outfile}"
            self.logger.error("Failed to run %s: %s", command, result.stderr)
        else:
            self.logger.debug("%s done: %s", command, result.stdout)
        return result.returncode

    async def _extract_file(self, guest_src: Path, host_dst: Path):
        """Extract file from instance"""
        src = (self._conn, guest_src)
        await asyncssh.scp(src, host_dst)

    async def _connect_instance(self, info: BenchmarkInfo):
        conn = await asyncssh.connect(info.ssh_host, port=info.ssh_port, known_hosts=None,
                                      client_keys=[self.manager_config.ssh_key], username="root",
                                      passphrase="")
        self.logger.debug("Connected to instance")
        return conn

    async def _run_benchmark(self):
        self.logger.info("Running benchmark")

    async def run(self):
        self.logger.info("Waiting for instance")
        self._reserved_instance = await self.daemon.request_instance(self.uuid, self.instance_config)
        if self._reserved_instance is None:
            self.logger.error("Can not reserve instance, bailing out...")
            return
        try:
            self._conn = await self._connect_instance(self._reserved_instance)
            with timing("Benchmark completed", self.logger):
                await self._run_benchmark()
            # Stop all pending background processes
            for t in self._command_tasks:
                t.cancel()
            await aio.gather(*self._command_tasks, return_exceptions=True)
        except Exception as ex:
            self.logger.error("Benchmark run failed: %s", ex)
        finally:
            await self.daemon.release_instance(self.uuid, self._reserved_instance)

    def plot(self):
        pass


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
        logging.info("Generate relative data for baseline %s",
                     self.options.baseline)
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

