import logging
import uuid
import json
import asyncio as aio
import itertools as it
import traceback
import typing
import argparse as ap
import shutil
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum

import termcolor
import asyncssh

from .util import new_logger
from .config import Config, TemplateConfig, TemplateConfigContext, path_field
from .benchmark import BenchmarkRunConfig, BenchmarkRunRecord, BenchmarkType
from .instance import InstanceConfig, InstanceManager
from .dataset import DatasetRegistry
from .analysis import BenchmarkAnalysisRegistry
from .html import HTMLSurface
from .excel import SpreadsheetSurface

@dataclass
class BenchplotUserConfig(Config):
    """
    User-environment configuration.
    This defines system paths for programs and source code we use
    """
    sdk_path: Path = path_field("~/cheri/cherisdk")
    perfetto_path: Path = path_field("~/cheri/cheri-perfetto/build")
    cheribuild_path: Path = path_field("~/cheri/cheribuild/cheribuild.py")


@dataclass
class BenchmarkManagerPlotConfig(TemplateConfig):
    """
    Plot analysis configuration.
    """
    backends: list[str] = field(default_factory=lambda: ["html"])


@dataclass
class BenchmarkSessionConfig(TemplateConfig):
    """
    Describe the benchmarks to run in the current benchplot session.
    """
    verbose: bool = False
    ssh_key: Path = Path("~/.ssh/id_rsa")
    output_path: Path = field(default_factory=Path.cwd)
    instances: list[InstanceConfig] = field(default_factory=list)
    benchmarks: list[BenchmarkRunConfig] = field(default_factory=list)
    plot_options: BenchmarkManagerPlotConfig = field(default_factory=BenchmarkManagerPlotConfig)

@dataclass
class BenchmarkManagerConfig(BenchmarkSessionConfig, BenchplotUserConfig):
    """
    Internal configuration object merging user and session top-level configurations.
    This is the top-level configuration object passed around as manager_config.
    """
    pass


@dataclass
class BenchmarkManagerRecord(Config):
    session: uuid.UUID
    records: list[BenchmarkRunRecord] = field(default_factory=list)


class BenchmarkManager(TemplateConfigContext):
    benchmark_runner_map = {}
    records_filename = "benchplot-run.json"

    @classmethod
    def register_benchmark(cls, type_: BenchmarkType, bench_class):
        cls.benchmark_runner_map[type_] = bench_class

    def __init__(self, user_config: BenchplotUserConfig, config: BenchmarkSessionConfig):
        super().__init__()
        # Merge configurations
        merged_conf = asdict(user_config)
        merged_conf.update(asdict(config))
        manager_config = BenchmarkManagerConfig(**merged_conf)
        # Cached copy of the configuration template
        self._config_template = manager_config
        # The ID for this benchplot session
        self.session = uuid.uuid4()
        self.logger = new_logger("manager")
        self.loop = aio.get_event_loop()
        self.instance_manager = InstanceManager(self.loop, manager_config)
        self.benchmark_instances = {}
        self.failed_benchmarks = []
        self.queued_tasks = []

        self._init_session()
        self.logger.debug("Assign initial session %s", self.session)
        # Adjust libraries log level
        if not self.config.verbose:
            ssh_logger = logging.getLogger("asyncssh")
            ssh_logger.setLevel(logging.WARNING)
        matplotlib_logger = logging.getLogger("matplotlib")
        matplotlib_logger.setLevel(logging.WARNING)

        self.logger.debug("Registered datasets %s", [str(k) for k in DatasetRegistry.dataset_types.keys()])
        self.logger.debug("Registered analysis %s", BenchmarkAnalysisRegistry.analysis_steps)

    def record_benchmark(self, record: BenchmarkRunRecord):
        self.benchmark_records.records.append(record)

    def get_requested_plot_backends(self):
        """Return a list of plot surface classes to use for plots"""
        backends = []
        for name in self.config.plot_options.backends:
            if name == "html":
                backends.append(HTMLSurface)
            elif name == "excel":
                backends.append(SpreadsheetSurface)
            elif name == "matplotlib":
                raise NotImplementedError("Not yet implemented")
                # backends.append(MatplotlibSurface)
        return backends

    def _init_session(self):
        """Session-ID dependant initialization"""
        # Note: this will only bind the manager-specific options, the rest of the template arguments
        # will remain as they will need to be bound to specific benchmark instances.
        self.register_template_subst(session=self.session)
        self.config = self._config_template.bind(self)
        self.session_output_path = self.config.output_path / f"benchplot-session-{str(self.session)}"
        self.benchmark_records = BenchmarkManagerRecord(session=self.session)
        self.benchmark_records_path = self.session_output_path / self.records_filename

    def _resolve_recorded_session(self, session: typing.Optional[uuid.UUID]):
        """
        Find recorded session to use for benchmark analysis.
        If a session ID is given, we try to locate the benchmark records for that session.
        If no session is given, default to the most recent session
        (by last-modified time of the record file).
        The resolved session is set as the current session ID and any dependent session state is
        re-initialized.
        If a session can not be resolved, raise an exception
        """
        self.logger.debug("Lookup session records in %s", self.config.output_path)
        resolved = None
        resolved_mtime = 0
        sessions = []
        for next_dir in self._iter_output_session_dirs():
            record_file = next_dir / self.records_filename
            record = BenchmarkManagerRecord.load_json(record_file)
            if session is not None and session == record.session:
                resolved = record
                break
            fstat = record_file.stat()
            mtime = datetime.fromtimestamp(fstat.st_mtime, tz=timezone.utc)
            sessions.append((mtime, record))
        if resolved is None and len(sessions):
            recent_session = sorted(sessions, key=lambda tup: tup[0], reverse=True)[0]
            resolved = recent_session[1]
        if resolved is None:
            self.logger.error("Can not resolve benchmark session %s in %s",
                              session if session is not None else "DEFAULT", self.config.output_path)
            raise Exception("Benchmark session not found")
        self.session = resolved.session
        self._init_session()
        # Overwrite benchmark records with the resolved data
        self.benchmark_records = resolved

    def _iter_output_session_dirs(self):
        if not self.config.output_path.is_dir():
            self.logger.error("Output directory %s does not exist", self.config.output_path)
            raise OSError("Output directory not found")
        for next_dir in self.config.output_path.iterdir():
            if not next_dir.is_dir() or not (next_dir / self.records_filename).exists():
                continue
            yield next_dir

    def _emit_records(self):
        self.logger.debug("Emit benchmark records")
        with open(self.benchmark_records_path, "w") as record_file:
            record_file.write(self.benchmark_records.to_json(indent=4))

    def create_benchmark(self, bench_config: BenchmarkRunConfig, instance: InstanceConfig, uid: uuid.UUID = None):
        """Create a benchmark run on an instance"""
        try:
            bench_class = self.benchmark_runner_map[bench_config.type]
        except KeyError:
            self.logger.error("Invalid benchmark type %s", bench_config.type)
        bench = bench_class(self, bench_config, instance, run_id=uid)
        self.benchmark_instances[bench.uuid] = bench
        self.logger.debug("Created benchmark run %s on %s id=%s", bench_config.name, instance.name, bench.uuid)
        return bench

    async def _run_tasks(self):
        await aio.gather(*self.queued_tasks)
        await self.instance_manager.shutdown()

    async def _shutdown_tasks(self):
        for t in self.queued_tasks:
            t.cancel()
        await aio.gather(*self.queued_tasks, return_exceptions=True)
        await self.instance_manager.shutdown()

    async def _list_task(self):
        self.logger.debug("List recorded sessions at %s", self.config.output_path)
        sessions = []
        for next_dir in self._iter_output_session_dirs():
            record_file = next_dir / self.records_filename
            records = BenchmarkManagerRecord.load_json(record_file)
            fstat = record_file.stat()
            mtime = datetime.fromtimestamp(fstat.st_mtime, tz=timezone.utc)
            sessions.append((mtime, records))
        sessions = sorted(sessions, key=lambda tup: tup[0], reverse=True)
        for mtime, records in sessions:
            if records == sessions[0][1]:
                is_default = " (default)"
            else:
                is_default = ""
            print(termcolor.colored(f"SESSION {records.session} [{mtime:%d-%m-%Y %H:%M}]{is_default}", "blue"))
            for bench_record in records.records:
                print(f"\t{bench_record.run.type}:{bench_record.run.name} on instance " +
                      f"{bench_record.instance.name} ({bench_record.uuid})")

    async def _clean_task(self):
        self.logger.debug("Clean all sessions from the output directory")
        for next_dir in self._iter_output_session_dirs():
            shutil.rmtree(next_dir)

    async def _plot_task(self):
        # Find all benchmark variants we were supposed to run
        # Note: this assumes that we aggregate to compare the same benchmark across OS configs,
        # it can be easily changed to also support comparison of different benchmark runs on
        # the same instance configuration if needed.
        # Load all datasets and for each benchmark and find the baseline instance for
        # each benchmark variant
        aggregate_baseline = {}
        aggregate_groups = defaultdict(list)

        for record in self.benchmark_records.records:
            bench = self.create_benchmark(record.run, record.instance, record.uuid)
            if record.instance.baseline:
                if record.run.name in aggregate_baseline:
                    self.logger.error("Multiple baseline instances?")
                    raise Exception("Too many baseline specifiers")
                aggregate_baseline[record.run.name] = bench
            else:
                # Must belong to a group
                aggregate_groups[record.run.name].append(bench)
        if len(aggregate_baseline) != len(self.config.benchmarks):
            self.logger.error("Number of benchmark variants does not match " + "number of runs marked as baseline")
            raise Exception("Missing baseline")
        self.logger.debug("Benchmark aggregation groups: %s",
                          {k: map(lambda b: b.uuid, v)
                           for k, v in aggregate_groups.items()})
        # Load datasets concurrently
        loading_tasks = []
        self.logger.info("Loading datasets")
        try:
            # XXX in theory we can run the whole aggregation steps concurrently, is it worth it though?
            for bench in it.chain(aggregate_baseline.values(), it.chain.from_iterable(aggregate_groups.values())):
                bench.task = self.loop.run_in_executor(None, bench.load)
                loading_tasks.append(bench.task)
                # Wait for everything to have loaded
            await aio.gather(*loading_tasks)
        except aio.CancelledError as ex:
            # Cancel any pending loading
            for task in loading_tasks:
                task.cancel()
            await aio.gather(*loading_tasks, return_exceptions=True)
            raise ex
        self.logger.info("Merge datasets")
        self.logger.debug("Benchmark aggregation baselines: %s", {k: b.uuid for k, b in aggregate_baseline.items()})
        # Merge compatible benchmark datasets into the baseline instance
        for name, baseline_bench in aggregate_baseline.items():
            baseline_bench.merge(aggregate_groups[name])
        # From now on we ony operate on the merged data
        self.logger.info("Aggregate datasets")
        for bench in aggregate_baseline.values():
            bench.aggregate()
            bench.verify()
        # Now we have processed all the input data, do the plotting
        self.logger.info("Generate plots")
        for bench in aggregate_baseline.values():
            bench.plot()

    def _handle_run_command(self, args: ap.Namespace):
        self.session_output_path.mkdir(parents=True)
        self.logger.info("Start benchplot session %s", self.session)
        for conf in self.config.benchmarks:
            self.logger.debug("Found benchmark %s", conf.name)
            for inst_conf in self.config.instances:
                bench = self.create_benchmark(conf, inst_conf)
                bench.task = self.loop.create_task(bench.run())
                self.queued_tasks.append(bench.task)

    def _handle_plot_command(self, args: ap.Namespace):
        self._resolve_recorded_session(args.session)
        self.logger.info("Using recorded session %s", self.session)
        self.queued_tasks.append(self.loop.create_task(self._plot_task()))

    def _handle_list_command(self, args: ap.Namespace):
        self.queued_tasks.append(self.loop.create_task(self._list_task()))

    def _handle_clean_command(self, args: ap.Namespace):
        self.queued_tasks.append(self.loop.create_task(self._clean_task()))

    def run(self, args: ap.Namespace):
        """Main entry point to execute benchmark tasks."""
        command = args.command
        if command == "run":
            self._handle_run_command(args)
        elif command == "plot":
            self._handle_plot_command(args)
        elif command == "list":
            self._handle_list_command(args)
        elif command == "clean":
            self._handle_clean_command(args)
        else:
            self.logger.error("Fatal: invalid command")
            exit(1)

        try:
            self.loop.run_until_complete(self._run_tasks())
            if command == "run":
                self._emit_records()
        except KeyboardInterrupt:
            self.logger.error("Shutdown")
            self.loop.run_until_complete(self._shutdown_tasks())
        except Exception as ex:
            self.logger.error("Died %s", ex)
            traceback.print_tb(ex.__traceback__)
            self.loop.run_until_complete(self._shutdown_tasks())
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
            self.logger.info("All tasks finished")
