import argparse as ap
import collections
import dataclasses as dc
import functools as ft
import itertools as it
import json
import logging
import shutil
from collections import OrderedDict
from datetime import date, datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

import marshmallow.fields as mfields
import numpy as np
from marshmallow import Schema, ValidationError, validates_schema
from marshmallow.validate import And, OneOf, Predicate
from marshmallow_dataclass import NewType, class_schema
from marshmallow_enum import EnumField
from typing_inspect import get_args, get_origin, is_generic_type


def _template_safe(temp: str, **kwargs):
    try:
        return temp.format(**kwargs)
    except KeyError:
        return temp


def lazy_nested_config_field():
    """
    Create a field marked as lazily-resolved configuration type.
    This is used for parts of configuration objects that have a configuration type
    assigned at runtime from a Task.
    This prevents the data from being modified when binding template values.
    """
    # Note that we need to levels of metadata: the first is for the dataclass field
    # that is picked up by marshmallow_dataclass, the second is the keyword argument
    # 'metadata' passed to the underlying marshmallow field constructor.
    # return dc.field(default_factory=dict, metadata={"metadata": {"late_config_bind": True}})
    return dc.field(default_factory=dict, metadata={"late_config_bind": True})


def is_lazy_nested_config(f: dc.Field):
    return f.metadata.get("late_config_bind", False)


class PathField(mfields.Field):
    """
    Simple wrapper for pathlib.Path fields
    """

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return ""
        return str(value)

    def _deserialize(self, value, attr, data, **kwargs):
        if value == "":
            return None
        try:
            return Path(value)
        except TypeError as ex:
            raise ValidationError(f"Invalid path {value}") from ex


#: Helper to validate that a PathField points to an existing regular file
validate_path_exists = And(
    Predicate("exists", error="File does not exist"),
    Predicate("is_file", error="File is not regular file"),
)


class TaskSpecField(mfields.Field):
    """
    Field used to validate a public task specifier.

    See :meth:`TaskRegistry.resolve_task` for details on the format.
    """

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return ""
        return str(value)

    def _validate_taskspec(self, value):
        from .task import TaskRegistry

        task = TaskRegistry.resolve_task(value)
        if not task:
            raise ValidationError(
                f"Task specifier {value} does not name any public tasks"
            )

    def _deserialize(self, value, attr, data, **kwargs):
        value = str(value)
        if value == "":
            raise ValidationError("Task specifier can not be blank")

        self._validate_taskspec(value)
        return value


class ExecTaskSpecField(TaskSpecField):
    """
    Field used to validate a public execution task name.

    See :meth:`TaskRegistry.resolve_exec_task` for details on the format.
    """

    def _validate_taskspec(self, value):
        from .task import TaskRegistry

        task = TaskRegistry.resolve_exec_task(value)
        if not task:
            raise ValidationError(
                f"Task specifier {value} does not name any public tasks"
            )


# Helper type for dataclasses to use the PathField
ConfigTaskSpec = NewType("ConfigTaskSpec", str, field=TaskSpecField)
ConfigExecTaskSpec = NewType("ConfigExecTaskSpec", str, field=ExecTaskSpecField)
ConfigPath = NewType("ConfigPath", Path, field=PathField)
ConfigAny = NewType("ConfigAny", any, field=mfields.Raw)


@dc.dataclass
class Config:
    """
    Base class for JSON-based configuration file parsing.
    Each field in this dataclass is deserialized from a json file, with support to
    nested configuration dataclasses.
    Types of the fields are normalized to the type annotation given in the dataclass.
    """

    class Meta:
        ordered = True

    @classmethod
    def schema(cls):
        return class_schema(cls)()

    @classmethod
    def copy(cls, other):
        return cls.schema().load(cls.schema().dump(other))

    @classmethod
    def load_json(cls, jsonpath):
        with open(jsonpath, "r") as jsonfile:
            data = json.load(jsonfile)
        return cls.schema().load(data)

    def emit_json(self) -> str:
        """
        Custom logic to emit json.
        This is required as in older python version pathlib objects are not serializable.
        """
        schema = self.schema()
        return json.dumps(schema.dump(self), indent=4)

    def __post_init__(self):
        return


class TemplateConfigContext:
    """
    Base class for context that can be used to bind TemplateConfig to.
    """

    def __init__(self):
        self._template_params = {}

    def register_template_subst(self, **kwargs):
        for key, value in kwargs.items():
            self._template_params[key] = value

    def conf_template_params(self):
        return dict(self._template_params)


@dc.dataclass
class TemplateConfig(Config):
    def _bind_one(self, context, dtype, value):
        if value is None:
            return value
        params = context.conf_template_params()
        if dtype == str:
            return _template_safe(value, **params)
        elif dc.is_dataclass(dtype) and issubclass(dtype, TemplateConfig):
            return value.bind(context)
        elif dtype == Path or dtype == ConfigPath:
            str_path = _template_safe(str(value), **params)
            return Path(str_path)
        return value

    def bind_field(self, context, f: dc.Field, value):
        """
        Run the template substitution on a config field.
        If the field is a collection or a nested TemplateConfig, we recursively bind
        each value.
        """
        origin = get_origin(f.type)
        if dc.is_dataclass(f.type):
            # Forward the nested bind if the dataclass is a TemplateConfig
            return self._bind_one(context, f.type, value)
        elif f.type == ConfigPath:
            return self._bind_one(context, f.type, value)
        elif is_lazy_nested_config(f):
            # Signals that the field contains a lazily resolved TemplateConfig, if there is one
            # recurse the binding, else skip it.
            if dc.is_dataclass(value):
                return self._bind_one(context, type(value), value)
            return value
        elif origin is Union:
            args = get_args(f.type)
            if len(args) == 2 and args[1] == type(None):
                # If we have an optional field, bind with the type argument instead
                return self._bind_one(context, args[0], value)
            else:
                # Common union field, use whatever type we have as the argument as we do not
                # know how to parse it
                return self._bind_one(context, type(value), value)
        elif origin is List or origin is list:
            arg_type = get_args(f.type)[0]
            return [self._bind_one(context, arg_type, v) for v in value]
        elif origin is Dict or origin is dict:
            arg_type = get_args(f.type)[1]
            return {
                key: self._bind_one(context, arg_type, v) for key, v in value.items()
            }
        else:
            return self._bind_one(context, f.type, value)

    def bind(self, context):
        """
        Run a template substitution pass with the given substitution context.
        This will resolve template strings as "{foo}" for template key/value
        substitutions that have been registerd in the contexv via
        TemplateConfigContext.register_template_subst() and leave the missing
        template parameters unchanged for later passes.
        """
        changes = {}
        for f in dc.fields(self):
            if not f.init:
                continue
            try:
                replaced = self.bind_field(context, f, getattr(self, f.name))
            except Exception as ex:
                raise ValueError(
                    f"Failed to bind {f.name} with value {getattr(self, f.name)}: {ex}"
                )
            if replaced:
                changes[f.name] = replaced
        return dc.replace(self, **changes)


@dc.dataclass
class BenchplotUserConfig(Config):
    """
    User-environment configuration.
    This defines system paths for programs and source code we use.
    The main point of the user configuration is to make sessions portable,
    so that a session that is run on a machine can be analysed on another.
    """

    #: Path to write the cheri-benchplot sessions to
    session_path: ConfigPath = dc.field(default_factory=Path.cwd)

    #: CHERI sdk path
    sdk_path: ConfigPath = Path("~/cheri/cherisdk")

    #: CHERI projects build directory, expects the format from cheribuild
    build_path: ConfigPath = Path("~/cheri/build")

    #: git repositories path
    src_path: ConfigPath = Path("~/cheri")

    #: Path to the CHERI perfetto fork build directory
    perfetto_path: ConfigPath = Path("~/cheri/cheri-perfetto/build")

    #: Path to openocd, will be inferred if missing (only relevant when running FPGA)
    openocd_path: ConfigPath = Path("/usr/bin/openocd")

    #: Path to flamegraph.pl flamegraph generator
    flamegraph_path: ConfigPath = Path("flamegraph.pl")

    #: CHERI rootfs path
    rootfs_path: Optional[ConfigPath] = None

    #: Hybrid rootfs path
    hybrid_rootfs_path: Optional[ConfigPath] = None

    #: Path to the CheriBSD extra files
    cheribsd_extra_files_path: Optional[ConfigPath] = dc.field(default=None)

    #: Path to cheribuild, inferred from :attr:`src_path` if missing
    cheribuild_path: Optional[ConfigPath] = dc.field(init=False, default=None)

    #: Path to the CheriBSD sources, inferred from :attr:`src_path` if missing
    cheribsd_path: Optional[ConfigPath] = dc.field(init=False, default=None)

    #: Path to the qemu sources, inferred from :attr:`src_path` if missing
    qemu_path: Optional[ConfigPath] = dc.field(init=False, default=None)

    #: Override the maximum number of workers to use
    concurrent_workers: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.sdk_path = self.sdk_path.expanduser().absolute()
        self.build_path = self.build_path.expanduser().absolute()
        self.src_path = self.src_path.expanduser().absolute()
        if self.rootfs_path is None:
            self.rootfs_path = self.sdk_path
        self.cheribuild_path = self.src_path / "cheribuild"
        self.cheribsd_path = self.src_path / "cheribsd"
        self.qemu_path = self.src_path / "qemu"
        # Try to autodetect openocd
        if self.openocd_path is None:
            self.openocd_path = shutil.which("openocd")

        self.session_path = self.session_path.expanduser().absolute()
        if not self.session_path.is_dir():
            raise ValueError("Session path must be a directory")


@dc.dataclass
class CommonPlatformOptions(TemplateConfig):
    """
    Base class for platform-specific options.
    This is internally used during benchmark dataset collection to
    set options for the instance that is to be run.
    """

    #: Number of cores in the system
    cores: int = 1

    #: The trace file used by default unless one of the datasets overrides it
    qemu_trace_file: Optional[ConfigPath] = None

    #: The trace file generated by interceptor
    qemu_interceptor_trace_file: Optional[ConfigPath] = None

    #: Run qemu with tracing enabled ('no', "perfetto", "perfetto-dynamorio")
    qemu_trace: str = dc.field(
        default="no",
        metadata={"validate": OneOf(["no", "perfetto", "perfetto-dynamorio"])},
    )

    #: Trace categories to enable for qemu-perfetto
    qemu_trace_categories: Set[str] = dc.field(default_factory=set)

    #: VCU118 bios
    vcu118_bios: Optional[ConfigPath] = None

    #: IP to use for the VCU118 board
    vcu118_ip: str = "10.88.88.2"

    def __post_init__(self):
        assert self.qemu_trace in ["no", "perfetto", "perfetto-dynamorio"]


@dc.dataclass
class PlatformOptions(TemplateConfig):
    """
    Platform options for an instance.
    This accepts the same fields as :class:`CommonPlatformOptions`, but
    will keep track of which are actively set in the configuration file,
    so that we can go look them up in the common options before setting
    a default value.
    """

    #: Number of cores in the system
    cores: Optional[int] = None

    #: The trace file used by default unless one of the datasets overrides it
    qemu_trace_file: Optional[ConfigPath] = None

    #: The trace file generated by interceptor
    qemu_interceptor_trace_file: Optional[ConfigPath] = None

    #: Run qemu with tracing enabled
    qemu_trace: str = dc.field(
        default="no",
        metadata={"validate": OneOf(["no", "perfetto", "perfetto-dynamorio"])},
    )

    #: Trace categories to enable for qemu-perfetto
    qemu_trace_categories: Optional[Set[str]] = None

    #: VCU118 bios
    vcu118_bios: Optional[ConfigPath] = None

    #: IP to use for the VCU118 board
    vcu118_ip: Optional[str] = None

    def replace_common(self, common: CommonPlatformOptions):
        for f in dc.fields(self):
            if getattr(self, f.name) is None:
                setattr(self, f.name, getattr(common, f.name))


@dc.dataclass
class ProfileConfig(TemplateConfig):
    """
    Common profiling options.
    These are inteded to be embedded into benchmark task_options for those benchmarks
    that support some form of profiling.
    """

    #: Run qemu with tracing enabled
    qemu_trace: Optional[str] = dc.field(
        default=None,
        metadata={"validate": OneOf([None, "perfetto", "perfetto-dynamorio"])},
    )

    #: Trace categories to enable for qemu-perfetto
    qemu_trace_categories: Optional[Set[str]] = None

    #: HWPMC performance counters modes
    hwpmc_trace: Optional[str] = dc.field(
        default=None, metadata={"validate": OneOf([None, "pmc", "profclock"])}
    )


class InstancePlatform(Enum):
    QEMU = "qemu"
    VCU118 = "vcu118"

    def __str__(self):
        return self.value


class InstanceCheriBSD(Enum):
    RISCV64_PURECAP = "riscv64-purecap"
    RISCV64_HYBRID = "riscv64-hybrid"
    MORELLO_PURECAP = "morello-purecap"
    MORELLO_HYBRID = "morello-hybrid"

    def is_riscv(self):
        return (
            self == InstanceCheriBSD.RISCV64_PURECAP
            or self == InstanceCheriBSD.RISCV64_HYBRID
        )

    def is_morello(self):
        return (
            self == InstanceCheriBSD.MORELLO_PURECAP
            or self == InstanceCheriBSD.MORELLO_HYBRID
        )

    def freebsd_kconf_dir(self):
        if self.is_riscv():
            arch = "riscv"
        elif self.is_morello():
            arch = "arm64"
        else:
            assert False, "Unknown arch"
        return Path("sys") / arch / "conf"

    def __str__(self):
        return self.value


class InstanceKernelABI(Enum):
    NOCHERI = "nocheri"
    HYBRID = "hybrid"
    PURECAP = "purecap"

    def __str__(self):
        return self.value


@dc.dataclass
class InstanceConfig(TemplateConfig):
    """
    Configuration for a CheriBSD instance to run benchmarks on.
    XXX-AM May need a custom __eq__() if iterable members are added
    """

    kernel: str
    baseline: bool = False
    name: Optional[str] = None
    platform: InstancePlatform = dc.field(
        default=InstancePlatform.QEMU, metadata={"by_value": True}
    )
    cheri_target: InstanceCheriBSD = dc.field(
        default=InstanceCheriBSD.RISCV64_PURECAP, metadata={"by_value": True}
    )
    kernelabi: InstanceKernelABI = dc.field(
        default=InstanceKernelABI.HYBRID, metadata={"by_value": True}
    )
    # Is the kernel config name managed by cheribuild or is it an extra one
    # specified via --cheribsd/extra-kernel-configs?
    cheribuild_kernel: bool = True
    # Internal fields, should not appear in the config file and are missing by default
    platform_options: PlatformOptions = dc.field(default_factory=PlatformOptions)

    @property
    def user_pointer_size(self):
        if (
            self.cheri_target == InstanceCheriBSD.RISCV64_PURECAP
            or self.cheri_target == InstanceCheriBSD.MORELLO_PURECAP
        ):
            return 16
        elif (
            self.cheri_target == InstanceCheriBSD.RISCV64_HYBRID
            or self.cheri_target == InstanceCheriBSD.MORELLO_HYBRID
        ):
            return 8
        assert False, "Not reached"

    @property
    def kernel_pointer_size(self):
        if (
            self.cheri_target == InstanceCheriBSD.RISCV64_PURECAP
            or self.cheri_target == InstanceCheriBSD.MORELLO_PURECAP
        ):
            if self.kernelabi == InstanceKernelABI.PURECAP:
                return self.user_pointer_size
            else:
                return 8
        elif (
            self.cheri_target == InstanceCheriBSD.RISCV64_HYBRID
            or self.cheri_target == InstanceCheriBSD.MORELLO_HYBRID
        ):
            if self.kernelabi == InstanceKernelABI.PURECAP:
                return 16
            else:
                return self.user_pointer_size
        assert False, "Not reached"

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = (
                f"{self.platform}-{self.cheri_target}-{self.kernelabi}-{self.kernel}"
            )

    def __str__(self):
        return f"{self.name}"


@dc.dataclass
class TaskTargetConfig(TemplateConfig):
    """
    Specify an analysis task and associated options.
    """

    #: Task specifier with format indicated by :meth:`TaskRegistry.resolve_task`
    handler: ConfigTaskSpec

    #: Extra options for the dataset handler, depend on the handler
    task_options: Dict[str, ConfigAny] = lazy_nested_config_field()


@dc.dataclass
class ExecTargetConfig(TemplateConfig):
    """
    Specify an execution task name.
    Note that the namespace of this task is also used to resolve compatible analysis tasks.
    """

    #: Task specifier with format indicated by :meth:`TaskRegistry.resolve_exec_task`
    handler: ConfigExecTaskSpec

    #: Extra options for the dataset handler, depend on the handler
    task_options: Dict[str, ConfigAny] = lazy_nested_config_field()


@dc.dataclass
class PipelineInstanceConfig(Config):
    """
    Describe the instances on which the benchmarks will be run.
    This is used to generate the interal :class:`InstanceConfig` objects.
    """

    #: Common platform options, depend on the platforms used in the instances
    platform_options: CommonPlatformOptions = dc.field(
        default_factory=CommonPlatformOptions
    )
    #: Instance descriptors for each instance to run
    instances: List[InstanceConfig] = dc.field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if len(self.instances) == 0:
            raise ValueError("There must be at least one instance configuration")


@dc.dataclass
class CommonBenchmarkConfig(TemplateConfig):
    """
    Common benchmark configuration parameters.
    This is shared between the user-facing configuration file and the internal
    benchmark description.
    """

    #: The name of the benchmark
    name: str
    #: The number of iterations to run
    iterations: int
    #: Benchmark configuration
    benchmark: ExecTargetConfig
    #: Auxiliary data generators
    aux_tasks: List[ExecTargetConfig] = dc.field(default_factory=list)
    #: Number of iterations to drop to reach steady-state
    drop_iterations: int = 0
    #: Benchmark description, used for plot titles (can contain a template), defaults to :attr:`name`.
    desc: Optional[str] = None
    #: Name of the benchmark output directory in the Guest instance OS filesystem
    remote_output_dir: ConfigPath = Path("/root/benchmark-output")
    #: Extra commands to run in the benchmark script.
    #: Keys in the dictionary are shell generator sections (see :class:`ScriptBuilder.Section`)
    #: 'pre_benchmark', 'benchmark', 'post_benchmark', 'last'. Each key maps to a list containing either
    #: commmands as strings or a dictionary. Commands are added directyl to the corresponding global section,
    #: dictionaries map an iteration index to the corresponding list of commands.
    #:
    #: .. code-block:: python
    #:
    #:    { "pre_benchmark": [
    #:        "cmd1", "cmd2", # added to the global pre_benchmark section
    #:        {
    #:            "*": ["cmd3"], # added to all iterations pre_benchmark section
    #:            1: ["cmd4"], # added only to the first iteration pre_benchmark section
    #:        }
    #:    ]}
    #:
    command_hooks: Dict[str, List[Union[str, dict]]] = dc.field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.remote_output_dir.is_absolute()
        ), f"Remote output path must be absolute {self.remote_output_dir}"

    @classmethod
    def from_common_conf(cls, other: "CommonBenchmarkConfig"):
        """
        Initialize a child config common fields.
        """
        initializer = {}
        for f in dc.fields(CommonBenchmarkConfig):
            initializer[f.name] = getattr(other, f.name)
        return cls(**initializer)


@dc.dataclass
class PipelineBenchmarkConfig(CommonBenchmarkConfig):
    """
    User-facing benchmark configuration.
    """

    #: Parameterized benchmark generator instructions. This should map (param_name => [values]).
    parameterize: Dict[str, List[ConfigAny]] = dc.field(default_factory=dict)


@dc.dataclass
class BenchmarkRunConfig(CommonBenchmarkConfig):
    """
    Internal benchmark configuration.
    This represents a resolved benchmark run, associated to an ID and set of parameters.
    """

    #: Unique benchmark run identifier
    uuid: UUID = dc.field(default_factory=uuid4)

    #: Unique benchmark group identifier, links benchmarks that run on the same instance
    g_uuid: Optional[UUID] = None

    #: Benchmark parameters
    parameters: Dict[str, ConfigAny] = dc.field(default_factory=dict)

    #: Instance configuration
    instance: Optional[InstanceConfig] = None

    def __str__(self):
        common_info = (
            f"params={self.parameters} ns={self.benchmark.handler} aux={self.aux_tasks}"
        )
        if self.g_uuid and self.instance:
            return (
                f"{self.name} ({self.uuid}/{self.g_uuid}) on {self.instance} "
                + common_info
            )
        else:
            return f"unallocated {self.name} ({self.uuid}) " + common_info


@dc.dataclass
class CommonSessionConfig(TemplateConfig):
    """
    Common session configuration.
    This is shared between the user-facing configuration file format and the
    internal session runfile.
    """

    #: Path to the SSH private key to use to access instances
    ssh_key: ConfigPath = Path("~/.ssh/id_rsa")

    #: Failure policy for workers. If true, when an worker encounters an error, it
    #: causes the scheduler to stop executing tasks and cleaup. If false failures
    #: are tolerated and the tasks that depend on the failed one will be removed
    #: from the schedule.
    abort_on_failure: bool = True

    #: Maximum number of concurrent instances that can be run (0 means unlimted)
    concurrent_instances: int = 0

    #: Maximum number of concurrent workers
    concurrent_workers: int = 0

    #: Allow reusing instances for multiple benchmark runs
    reuse_instances: bool = False

    #: Extract symbols with elftools instead of llvm
    use_builtin_symbolizer: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.ssh_key = self.ssh_key.resolve()

    @classmethod
    def from_common_conf(cls, other: "CommonSessionConfig"):
        """
        Initialize a child config common fields.
        """
        initializer = {}
        for f in dc.fields(CommonSessionConfig):
            initializer[f.name] = getattr(other, f.name)
        return cls(**initializer)


@dc.dataclass
class PipelineConfig(CommonSessionConfig):
    """
    Describe the benchmarks to run in the current benchplot session.
    Note that this configuration does not allow template substitution,
    the templates will be retained in the session instructions file so that
    the substitution can be replicated with a different user configuration every time.
    """

    #: Instances configuration, required
    instance_config: PipelineInstanceConfig = dc.field(
        default_factory=PipelineInstanceConfig
    )

    #: Benchmark configuration, required
    benchmark_config: List[PipelineBenchmarkConfig] = dc.field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.ssh_key = self.ssh_key.resolve()
        if self.instance_config is None:
            raise ValueError("Missing instance_config")
        if len(self.benchmark_config) == 0:
            raise ValueError("Missing benchmark_config")


@dc.dataclass
class SessionRunConfig(CommonSessionConfig):
    """
    Internal session configuration file, autogenerated from the pipeline configuration.
    This unwraps the benchmark parameterization and generates the full set of benchmarks
    to run with the associated instance configurations.
    """

    #: Session unique ID
    uuid: UUID = dc.field(default_factory=uuid4)

    #: Session name, defaults to the session UUID
    name: Optional[str] = None

    #: Benchmark run configuration, this is essentially the flattened benchmark matrix
    configurations: List[BenchmarkRunConfig] = dc.field(default_factory=list)

    #: User configuration
    user_config: BenchplotUserConfig = dc.field(default_factory=BenchplotUserConfig)

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = str(self.uuid)

    @validates_schema
    def validate_configuration(self, data, **kwargs):
        # Check that the concurrent instances configuration is compatible with the platform
        if data["concurrent_instances"] != 1:
            for bench_config in data["configurations"]:
                if bench_config.instance.platform == InstancePlatform.VCU118:
                    raise ValidationError(
                        "Running on VCU118 instances requires concurrent_instances=1",
                        "concurrent_instances",
                    )

    @classmethod
    def generate(
        cls, user_config: BenchplotUserConfig, config: PipelineConfig
    ) -> "SessionRunConfig":
        """
        Generate a new :class:`SessionRunConfig` from a :class:`PipelineConfig`.
        Manual benchmark parameterization is supported by specifying multiple benchmarks with
        the same set of parameterize keys and different values.
        We support 3 types of parameterization:

        1. there is a single benchmark_config and no parametrization
        2. there is a single benchmark_config with parametrization and template substitution
        3. there are multiple benchmark_configs with the same set of parametrization keys but
           disjoint sets of values

        :param user_config: The user configuration for the local machine setup.
        :param config: The :class:`PipelineConfig` to use.
        :return: A new session runfile configuration
        """
        session = SessionRunConfig.from_common_conf(config)
        logger = logging.getLogger("cheri-benchplot")
        logger.info("Create new session %s", session.uuid)

        # Collect benchmarks and the instances
        all_conf = []
        # Case (1)
        if ft.reduce(
            lambda noparams, conf: noparams or len(conf.parameterize) == 0,
            config.benchmark_config,
            False,
        ):
            if len(config.benchmark_config) > 1:
                logger.error(
                    "Multiple benchmark configurations must have a parameterization key"
                )
                raise ValueError("Invalid configuration")
            conf = config.benchmark_config[0]
            logger.debug("Found benchmark %s", conf.name)
            all_conf = [BenchmarkRunConfig.from_common_conf(conf)]
        else:
            param_keys = None
            # Case (2) (3)
            for conf in config.benchmark_config:
                if len(conf.parameterize) == 0:
                    logger.error("Missing benchmark parameterization?")
                    raise ValueError("Invalid configuration")
                sorted_params = OrderedDict(conf.parameterize)
                keys = set(conf.parameterize.keys())
                if param_keys and param_keys != keys:
                    logger.error(
                        "Mismatching parameterization keys %s != %s", param_keys, keys
                    )
                    raise ValueError("Invalid configuration")
                elif param_keys is None:
                    param_keys = keys
                # XXX TODO Ensure parameter set disjointness
                logger.debug("Found parameterized benchmark '%s'", conf.name)
                for param_combination in it.product(*sorted_params.values()):
                    parameters = dict(zip(sorted_params.keys(), param_combination))
                    run_conf = BenchmarkRunConfig.from_common_conf(conf)
                    run_conf.parameters = parameters
                    all_conf.append(run_conf)

        # Map instances to dataset group IDs
        group_uuids = [uuid4() for _ in config.instance_config.instances]

        # Now create all the full configurations by combining instance and benchmark configurations
        common_platform_options = config.instance_config.platform_options
        for run_conf in all_conf:
            for gid, inst_conf in zip(group_uuids, config.instance_config.instances):
                final_run_conf = BenchmarkRunConfig.copy(run_conf)
                # Resolve merged platform options
                final_inst_conf = InstanceConfig.copy(inst_conf)
                final_inst_conf.platform_options.replace_common(common_platform_options)
                # Generate new unique ID for the parameterized run
                final_run_conf.uuid = uuid4()
                final_run_conf.g_uuid = gid
                final_run_conf.instance = final_inst_conf
                session.configurations.append(final_run_conf)
        return session


@dc.dataclass
class PlotConfig(Config):
    """
    Plotting configuration.
    This is separated in case it needs to be propagated separately.
    """

    #: Generate multiple split plots instead of combining
    split_subplots: bool = False
    #: Output formats
    plot_output_format: List[str] = dc.field(default_factory=lambda: ["pdf"])


@dc.dataclass
class AnalysisConfig(Config):
    #: General plot configuration
    plot: PlotConfig = dc.field(default_factory=PlotConfig)

    #: Constants to show in various plots, depending on the X and Y axes.
    # The dictionary maps parameters of the benchmark parameterisation to a dict
    # mapping description -> constant value
    parameter_constants: Dict[str, dict] = dc.field(default_factory=dict)

    #: Baseline dataset group id, defaults to the baseline instance uuid
    baseline_gid: Optional[UUID] = None

    #: Use builtin symbolizer instead of addr2line
    use_builtin_symbolizer: bool = True

    #: Specify analysis passes to run
    handlers: List[TaskTargetConfig] = dc.field(default_factory=list)
