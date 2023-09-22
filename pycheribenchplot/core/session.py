import logging
import re
import shutil
from collections import defaultdict, namedtuple
from contextlib import AbstractContextManager
from dataclasses import asdict, fields
from enum import Enum
from pathlib import Path
from typing import Type
from uuid import UUID

import pandas as pd
from marshmallow.exceptions import ValidationError
from tabulate import tabulate
from typing_extensions import Self

from .analysis import AnalysisTask
from .benchmark import Benchmark, BenchmarkExecMode, ExecTaskConfig
from .config import (AnalysisConfig, BenchplotUserConfig, Config, ConfigContext, ExecTargetConfig, PipelineConfig,
                     SessionRunConfig, TaskTargetConfig)
from .instance import InstanceManager
from .model import UNPARAMETERIZED_INDEX_NAME
from .task import (ExecutionTask, SessionExecutionTask, TaskRegistry, TaskScheduler)
from .util import new_logger

#: Constant name of the generated session configuration file
SESSION_RUN_FILE = "session-run.json"
benchplot_logger = logging.getLogger("cheri-benchplot")


class Session:
    """
    Represent a benchmarking session.
    A session is the primary container for the benchmark assets, results and
    intermediary files.
    When created, a session has at minimum a session descriptor JSON file.
    The session descriptor encodes the session configuration expanded from the
    main benchplot configuration. This instructs the session about the components
    to run and how.
    """

    @classmethod
    def make_new(cls, user_config: BenchplotUserConfig, config: PipelineConfig, session_path: Path) -> Self:
        """
        Create a new session and initialize the directory hierarchy

        :param user_config: The user configuration file for local configuration options
        :param config: The session configuration
        :param session_path: The session_path of the new session
        :return: A new session instance
        """
        if session_path.exists():
            benchplot_logger.error("Session directory already exists for session %s", session_path)
            raise ValueError("New session path already exists")
        run_config = SessionRunConfig.generate(user_config, config)
        run_config.name = session_path.name
        session_path.mkdir()
        with open(session_path / SESSION_RUN_FILE, "w") as runfile:
            runfile.write(run_config.emit_json())
        return Session(user_config, run_config, session_path=session_path)

    @classmethod
    def is_session(cls, path: Path) -> bool:
        """
        Check if the given path corresponds to a session.
        This currently checks for the session runfile.

        :param path: The path to a directory
        :return: True if the path contains a session information
        """
        if (path / SESSION_RUN_FILE).exists():
            return True
        return False

    @classmethod
    def from_path(cls, user_config: BenchplotUserConfig, path: Path) -> Self | None:
        """
        Load a session from the given path.

        :param mgr: The parent pipeline manager
        :param path: The session directory path
        :return: The corresponding :class:`Session`
        """
        benchplot_logger.debug("Scanning %s for valid session", path)

        if not path.exists() or not cls.is_session(path):
            benchplot_logger.debug("Session for %s not found", path)
            return None

        config = SessionRunConfig.load_json(path / SESSION_RUN_FILE)
        session = Session(user_config, config, session_path=path)

        benchplot_logger.debug("Resolved session %s => %s", path, session)
        return session

    def __init__(
        self,
        user_config: BenchplotUserConfig,
        config: SessionRunConfig,
        session_path: Path = None,
    ):
        super().__init__()
        # Pre-initialization logger
        self.logger = new_logger(f"session-init")
        #: The local user configuration
        self.user_config = user_config
        #: Main session configuration
        self.config = self._resolve_config_template(config)
        #: Root logger for the session, initialized as soon as we know we have a stable session name
        self.logger = new_logger(f"session-{self.config.name}")
        if session_path is None:
            session_root_path = self.user_config.session_path / self.config.name
        else:
            session_root_path = session_path
        #: Session root path, where all the session data will be stored
        self.session_root_path = session_root_path.expanduser().absolute()
        self._ensure_dir_tree()
        #: Mapping from g_uuid to platform configurations. Note that this should be readonly.
        self.platform_map = {}
        #: A dataframe that organises the set of benchmarks to run or analyse.
        self.benchmark_matrix = self._resolve_benchmark_matrix()
        #: Benchmark baseline instance group UUID.
        self.baseline_g_uuid = self._resolve_baseline()

        # Before using the workers configuration, check if we are overriding it
        if self.user_config.concurrent_workers:
            self.logger.debug(
                "Overriding maximum workers count from user configuration (max=%d)",
                self.user_config.concurrent_workers,
            )
            self.config.concurrent_workers = self.user_config.concurrent_workers

        #: Task scheduler
        self.scheduler = TaskScheduler(self)

    def __str__(self):
        return f"Session({self.uuid}) [{self.name}]"

    def _resolve_config_template(self, config: SessionRunConfig) -> SessionRunConfig:
        """
        Resolves the templates for the given session run configuration,
        using the current user configuration.
        """
        ctx = ConfigContext()
        ctx.add_namespace(self.user_config, "user")
        return config.bind(ctx)

    def _resolve_baseline(self):
        """
        Resolve the baseline benchmark run group ID.
        This is necessary to identify the benchmark run that we compare against
        (actually the column in the datarun matrix we compare against).

        :return: The baseline group ID.
        """
        baseline = None
        for benchmark_config in self.config.configurations:
            if benchmark_config.instance.baseline:
                if baseline and baseline != benchmark_config.g_uuid:
                    self.logger.error("Multiple baseline instances?")
                    raise RuntimeError("Too many baseline specifiers")
                baseline = benchmark_config.g_uuid
        if not baseline:
            self.logger.error("Missing baseline instance")
            raise RuntimeError("Missing baseline")
        baseline_conf = self.benchmark_matrix[baseline].iloc[0].config
        self.logger.debug(
            "Benchmark baseline %s (%s)", baseline_conf.instance.name, baseline
        )
        return baseline

    def _resolve_benchmark_matrix(self) -> tuple[pd.DataFrame, UUID]:
        """
        Generate the datarun matrix from the generators configurations.
        In the resulting dataframe:
         - rows are different parameterizations of the benchmark, indexed by param keys.
         - columns are different instances on which the benchmark is run, indexed by dataset_gid.

        :return: The benchmark matrix as a pandas dataframe
        """
        # First pass, just collect instances and parameters
        instances = {}
        parameters = defaultdict(list)
        for benchmark_config in self.config.configurations:
            self.logger.debug("Found benchmark run config %s", benchmark_config)
            instances[benchmark_config.g_uuid] = benchmark_config.instance.name
            self.platform_map[benchmark_config.g_uuid] = benchmark_config.instance
            for k, p in benchmark_config.parameters.items():
                parameters[k].append(p)
        if parameters:
            index_frame = pd.DataFrame(parameters)
            index = pd.MultiIndex.from_frame(index_frame)
            if not index.is_unique:
                index = index.unique()
        else:
            # If there is no parameterization, just use a flat index, there will be only
            # one row in the benchmark matrix.
            index = pd.Index([0], name=UNPARAMETERIZED_INDEX_NAME)

        # Second pass, fill the matrix
        bench_matrix = pd.DataFrame(index=index, columns=instances.keys())
        for index_name in bench_matrix.index.names:
            if not re.fullmatch(r"[a-zA-Z0-9_]+", index_name):
                self.logger.exception(
                    "Benchmark parameter keys must be valid python property names, only [a-zA-Z0-9_] allowed"
                )
                raise ValueError("Invalid parameter key")
        BenchParams = namedtuple("BenchParams", bench_matrix.index.names)
        for benchmark_config in self.config.configurations:
            benchmark = Benchmark(self, benchmark_config)
            if benchmark_config.parameters:
                # Note: config.parameters is unordered, use namedtuple to ensure
                # the key ordering
                i = BenchParams(**benchmark_config.parameters)
                bench_matrix.loc[i, benchmark_config.g_uuid] = benchmark
            else:
                bench_matrix[benchmark_config.g_uuid] = benchmark
            benchmark.get_plot_path().mkdir(exist_ok=True)

        show_matrix = tabulate(bench_matrix, tablefmt="github", headers="keys")
        self.logger.debug("Benchmark matrix:\n%s", show_matrix)

        assert not bench_matrix.isna().any().any(), "Incomplete benchmark matrix"
        if bench_matrix.shape[0] * bench_matrix.shape[1] != len(self.config.configurations):
            self.logger.error("Malformed benchmark matrix")
            raise RuntimeError("Malformed benchmark matrix")

        return bench_matrix

    def _ensure_dir_tree(self):
        """
        Build the session directory tree.
        """
        self.get_metadata_root_path().mkdir(exist_ok=True)
        self.get_data_root_path().mkdir(exist_ok=True)
        self.get_asset_root_path().mkdir(exist_ok=True)
        self.get_plot_root_path().mkdir(exist_ok=True)

    @property
    def name(self):
        return self.config.name

    @property
    def uuid(self):
        return self.config.uuid

    @property
    def parameter_keys(self) -> list[str]:
        """
        The set of parameter keys that index the rows of the benchmark matrix.
        """
        names = self.benchmark_matrix.index.names
        if len(names) == 1 and names[0] == UNPARAMETERIZED_INDEX_NAME:
            return []
        else:
            return list(names)

    def get_public_tasks(self) -> list[Type[AnalysisTask]]:
        """
        Return the public tasks available for this session.

        :return: A list of :class:`AnalysisTask` classes.
        """
        namespaces = set()
        for bench_config in self.config.configurations:
            for exec_task_config in bench_config.generators:
                exec_task = TaskRegistry.resolve_exec_task(exec_task_config.handler)
                namespaces.add(exec_task.task_namespace)
        # Now group all public tasks from the namespaces
        tasks = []
        for ns in namespaces:
            tasks += TaskRegistry.public_tasks[ns].values()
        return [t for t in tasks if issubclass(t, AnalysisTask)]

    def delete(self):
        """
        Delete this session and all associated files.
        """
        self.logger.info("Remove session %s (%s)", self.name, self.uuid)
        shutil.rmtree(self.session_root_path)

    def bundle(self):
        """
        Produce a compressed archive with all the session output.
        """
        bundle_file = self.session_root_path.with_suffix(".tar.xz")
        self.logger.info("Generate %s bundle", self.session_root_path)
        if bundle_file.exists():
            self.logger.info("Replacing old bundle %s", bundle_file)
            bundle_file.unlink()
        result = subprocess.run(
            ["tar", "-J", "-c", "-f", bundle_file, self.session_root_path]
        )
        if result.returncode:
            self.logger.error("Failed to produce bundle")
        self.logger.info("Archive created at %s", bundle_file)

    def machine_configuration_name(self, g_uuid: UUID) -> str:
        """
        Retrieve the human-readable form of the machine configuration identified by a given UUID.
        This is useful for producing readable output.
        """
        col = self.benchmark_matrix[g_uuid]
        # column shares that same instance configuration, just grab the first
        instance_config = col.iloc[0].config.instance
        return instance_config.name

    def get_metadata_root_path(self) -> Path:
        """
        :return: The metadata directory, used to store benchplot task metadata.
        """
        return self.session_root_path / ".metadata"

    def get_data_root_path(self) -> Path:
        """
        :return: The root raw data directory, used to store artifacts from benchmark runs.
        """
        return self.session_root_path / "run"

    def get_plot_root_path(self) -> Path:
        """
        :return: The root path for the generated plots and tables
        """
        return self.session_root_path / "plots"

    def get_asset_root_path(self) -> Path:
        """
        :return: The root path for binary assets e.g. kernel images
        """
        return self.session_root_path / "assets"

    def all_benchmarks(self) -> list[Benchmark]:
        """
        Helper method to iterate over benchmark contexts.

        :return: A list containing all benchmark contexts from the
        benchmark matrix.
        """
        return list(self.benchmark_matrix.to_numpy().ravel())

    def clean_all(self):
        """
        Clean all output files, including benchmark data.
        """
        meta_root = self.get_metadata_root_path()
        data_root = self.get_data_root_path()
        plot_root = self.get_plot_root_path()
        asset_root = self.get_asset_root_path()
        if meta_root.exists():
            shutil.rmtree(meta_root)
        if data_root.exists():
            shutil.rmtree(data_root)
        if plot_root.exists():
            shutil.rmtree(plot_root)
        if asset_root.exists():
            shutil.rmtree(asset_root)
        self._ensure_dir_tree()

    def clean_analysis(self):
        """
        Clean all output analysis files.
        This retains the benchmark run data and assets.
        """
        plot_root = self.get_plot_root_path()
        if plot_root.exists():
            shutil.rmtree(plot_root)
        self._ensure_dir_tree()

    def run(self, mode: str = "full"):
        """
        Run the session benchmark execution task stack

        :param mode: Alternate run mode for partial or pretend runs.
        """
        # Cleanup previous run, to avoid any confusion
        self.clean_all()

        exec_config = ExecTaskConfig(mode=BenchmarkExecMode(mode))
        instance_manager = InstanceManager(self)
        self.scheduler.register_resource(instance_manager)

        for col in self.benchmark_matrix:
            for bench in self.benchmark_matrix[col]:
                bench.schedule_exec_tasks(self.scheduler, exec_config)
        self.logger.info("Session %s start run", self.name)
        self.scheduler.run()
        self.logger.info("Session %s run finished", self.name)

    def analyse(self, analysis_config: AnalysisConfig | None):
        """
        Run the session analysis tasks requested.
        The analysis pipeline is slighly different from the execution pipeline.
        This is because the same data can be used for different analysis targets and
        the user should be able to specify the set of targets we need to run.
        Here the analysis configuration should specify a set of public task names that
        we collect and schedule

        :param analysis_config: The analysis configuration
        """
        if analysis_config is None:
            # Load analysis configuration from the session
            analysis_config = self.config.analysis_config

        # Override the baseline ID if configured
        if (
            analysis_config.baseline_gid is not None
            and analysis_config.baseline_gid != self.baseline_g_uuid
        ):
            self.baseline_g_uuid = analysis_config.baseline_gid
            try:
                baseline_conf = (
                    self.benchmark_matrix[self.baseline_g_uuid].iloc[0].config
                )
            except KeyError:
                self.logger.error(
                    "Invalid baseline instance ID %s", self.baseline_g_uuid
                )
                raise
            self.logger.info(
                "Using alternative baseline %s (%s)",
                baseline_conf.instance.name,
                self.baseline_g_uuid,
            )

        for task_spec in analysis_config.tasks:
            resolved = TaskRegistry.resolve_task(task_spec.handler)
            if not resolved:
                self.logger.error(
                    "Invalid task name specification %s", task_spec.handler
                )
                raise ValueError("Invalid task name")
            for task_klass in resolved:
                if not issubclass(task_klass, AnalysisTask):
                    self.logger.warning("Analysis process only supports scheduling of AnalysisTasks, skipping %s",
                                        task_klass)
                if task_klass.task_config_class and isinstance(task_spec.task_options, dict):
                    options = task_klass.task_config_class.schema().load(task_spec.task_options)
                else:
                    options = task_spec.task_options
                task = task_klass(self, analysis_config, task_config=options)
                self.logger.debug("Schedule analysis task %s with opts %s", task, options)
                self.scheduler.add_task(task)
        self.logger.info("Session %s start analysis", self.name)
        self.scheduler.run()
        self.logger.info("Session %s analysis finished", self.name)

    def find_exec_task(self, task_class: Type[SessionExecutionTask]) -> SessionExecutionTask:
        """
        Find a session-wide execution task and return a task instance.

        This can be used by analysis tasks to load generator dependencies.
        """
        task = self.all_benchmarks()[0].find_exec_task(task_class)
        if not task.is_session_task() or not task.is_exec_task():
            self.logger.error("The task %s is not a SessionExecutionTask", task_class)
            raise TypeError("Invalid task type")
        return task

    def find_all_exec_tasks(self, task_class: Type[ExecutionTask]) -> list[ExecutionTask]:
        """
        Find all execution tasks of a given type and return them as a list.

        This can be used by analysis tasks to load generator dependencies.
        """
        tasks = []
        for bench in self.all_benchmarks():
            task = bench.find_exec_task(task_class)
            if not task.is_benchmark_task() or not task.is_exec_task():
                self.logger.error("The task %s is not an ExecutionTask", task_class)
                raise TypeError("Invalid task type")
            tasks.append(task)
        return tasks
