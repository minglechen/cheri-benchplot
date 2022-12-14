import logging
import re
import shutil
import typing
from collections import defaultdict, namedtuple
from contextlib import AbstractContextManager
from dataclasses import asdict, fields
from enum import Enum
from pathlib import Path
from uuid import UUID

import pandas as pd

from .benchmark import Benchmark, BenchmarkExecMode, ExecTaskConfig
from .config import (
    AnalysisConfig,
    BenchplotUserConfig,
    Config,
    ExecTargetConfig,
    PipelineConfig,
    SessionRunConfig,
    TaskTargetConfig,
    TemplateConfigContext,
)
from .instance import InstanceManager
from .model import UUIDType
from .task import AnalysisTask, TaskRegistry, TaskScheduler
from .util import new_logger

#: Constant name of the generated session configuration file
SESSION_RUN_FILE = "session-run.json"
#: Constant name to mark the benchmark matrix index as unparameterized
UNPARAMETERIZED_INDEX_NAME = "RESERVED__unparameterized_index"
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
    def make_new(
        cls,
        user_config: BenchplotUserConfig,
        config: PipelineConfig,
        session_path: Path,
    ) -> "Session":
        """
        Create a new session and initialize the directory hierarchy

        :param user_config: The user configuration file for local configuration options
        :param config: The session configuration
        :param session_path: The session_path of the new session
        :return: A new session instance
        """
        if session_path.exists():
            benchplot_logger.logger.error(
                "Session directory already exists for session %s", session_path
            )
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
    def from_path(
        cls, user_config: BenchplotUserConfig, path: Path
    ) -> typing.Optional["Session"]:
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

    def _resolve_exec_task_options(
        self, config: ExecTargetConfig
    ) -> typing.Optional[Config]:
        """
        Resolve the configuration type for a :class:`ExecTargetConfig` containing run options.

        :param config: A single dataset configuration
        :return: A Config object to be used as the new :attr:`ExecTargetConfig.task_options`.
        If the exec task does not specify a task_config_class, return a dict with the same content as
        the original run_options dict.
        """
        # Existence of the exec task is ensured by configuration validation
        exec_task = TaskRegistry.resolve_exec_task(config.handler)
        if exec_task.task_config_class:
            return exec_task.task_config_class.schema().load(config.task_options)
        return config.task_options

    def _resolve_config_template(self, config: SessionRunConfig) -> SessionRunConfig:
        """
        Resolves the templates for the given session run configuration,
        using the current user configuration.
        """
        ctx = TemplateConfigContext()
        ctx.register_template_subst(**asdict(self.user_config))
        # Register substitutions for known stable fields in the run configuration
        ctx.register_template_subst(session=config.uuid)
        ctx.register_template_subst(session_name=config.name)
        new_config = config.bind(ctx)
        # Now scan through all the configurations and subsitute per-benchmark/instance fields
        new_bench_conf = []
        for bench_conf in new_config.configurations:
            ctx.register_template_subst(
                uuid=bench_conf.uuid,
                g_uuid=bench_conf.g_uuid,
                iterations=bench_conf.iterations,
                drop_iterations=bench_conf.drop_iterations,
                remote_ouptut_dir=bench_conf.remote_output_dir,
            )
            ctx.register_template_subst(**bench_conf.parameters)
            inst_conf = bench_conf.instance
            ctx.register_template_subst(
                kernel=inst_conf.kernel,
                baseline=inst_conf.baseline,
                platform=inst_conf.platform.value,
                cheri_target=inst_conf.cheri_target.value,
                kernelabi=inst_conf.kernelabi.value,
            )
            # Resolve run_options for each TaskTargetConfig
            bench_conf.benchmark.task_options = self._resolve_exec_task_options(
                bench_conf.benchmark
            )
            for aux_conf in bench_conf.aux_tasks:
                aux_conf.task_options = self._resolve_exec_task_options(aux_conf)
            new_bench_conf.append(bench_conf.bind(ctx))
        new_config.configurations = new_bench_conf
        return new_config

    def _resolve_baseline(self):
        """
        Resolve the baseline benchmark run group ID.
        This is necessary to identify the benchmark run that we compare against
        (actually the column in the benchmark matrix we compare against).

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

    def _resolve_benchmark_matrix(self) -> pd.DataFrame:
        """
        Generate the benchmark matrix from the benchmark configurations.
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

        assert not bench_matrix.isna().any().any(), "Incomplete benchmark matrix"
        for i, row in bench_matrix.iterrows():
            if isinstance(i, tuple):
                i = BenchParams(*i)
            row_str = [str(bench_ctx) for bench_ctx in row.values]
            self.logger.debug("Benchmark matrix %s = %s", i, row_str)
        if bench_matrix.shape[0] * bench_matrix.shape[1] != len(
            self.config.configurations
        ):
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

    def get_public_tasks(self) -> list[typing.Type[AnalysisTask]]:
        """
        Return the public tasks available for this session.

        :return: A list of :class:`AnalysisTask` classes.
        """
        namespaces = set()
        for bench_config in self.config.configurations:
            exec_task = TaskRegistry.resolve_exec_task(bench_config.benchmark.handler)
            namespaces.add(exec_task.task_namespace)
            for aux_config in bench_config.aux_tasks:
                aux_task = TaskRegistry.resolve_exec_task(aux_config.handler)
                namespaces.add(aux_task.task_namespace)
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

    def analyse(self, analysis_config: AnalysisConfig):
        """
        Run the session analysis tasks requested.
        The analysis pipeline is slighly different from the execution pipeline.
        This is because the same data can be used for different analysis targets and
        the user should be able to specify the set of targets we need to run.
        Here the analysis configuration should specify a set of public task names that
        we collect and schedule

        :param analysis_config: The analysis configuration
        """
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

        for task_spec in analysis_config.handlers:
            resolved = TaskRegistry.resolve_task(task_spec.handler)
            if not resolved:
                self.logger.error(
                    "Invalid task name specification %s", task_spec.handler
                )
                raise ValueError("Invalid task name")
            for task_klass in resolved:
                if not issubclass(task_klass, AnalysisTask):
                    self.logger.warning(
                        "Analysis process only supports scheduling of AnalysisTasks, skipping %s",
                        task_klass,
                    )
                if task_klass.task_config_class:
                    options = task_klass.task_config_class.schema().load(
                        task_spec.task_options
                    )
                else:
                    options = task_spec.task_options
                task = task_klass(self, analysis_config, task_config=options)
                self.scheduler.add_task(task)
        self.logger.info("Session %s start analysis", self.name)
        self.scheduler.run()
        self.logger.info("Session %s analysis finished", self.name)
