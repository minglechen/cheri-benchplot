import collections
import dataclasses as dc
import functools as ft
import itertools as it
import json
import logging
import re
import shutil
from collections import OrderedDict
from datetime import date, datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Type, Union
from uuid import UUID, uuid4

import marshmallow.fields as mfields
from git import Repo
from marshmallow import Schema, ValidationError, validates_schema
from marshmallow.validate import And, OneOf, Predicate
from marshmallow_dataclass import NewType, class_schema
from typing_extensions import Self
from typing_inspect import get_args, get_origin, is_generic_type, is_union_type

from .error import ConfigurationError

# Global configuration logger
# We do not drag around a custom logger during configuration
logger = logging.getLogger("cheri-benchplot")


def resolve_task_options(task_spec: str, task_options: dict, is_exec: bool = False) -> Type["Task"]:
    """
    Helper to lazily coerce task options to the correct type.
    """
    # Need to lazily import this to avoid circular dependencies
    from .task import TaskRegistry

    if is_exec:
        task_class = TaskRegistry.resolve_exec_task(task_spec)
        if not task_class:
            raise ConfigurationError(f"Invalid task spec: {task_spec}")
    else:
        matches = TaskRegistry.resolve_task(task_spec)
        if not matches:
            raise ConfigurationError(f"Invalid task spec: {task_spec}")
        if len(matches) > 1:
            raise ConfigurationError(f"Task handler should be unique: {task_spec}")
        task_class = matches[0]
    if task_class.task_config_class:
        conf_class = task_class.task_config_class
        try:
            return conf_class.schema().load(task_options)
        except ValidationError as err:
            logger.error("Invalid task options, %s validation failed: %s", conf_class, err.normalized_messages())
            raise err
    return task_options


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
        matches = TaskRegistry.resolve_task(value)
        if not matches:
            raise ValidationError(f"Task specifier {value} does not name any public tasks")
        if len(matches) > 1:
            raise ValidationError(f"Task specifier {value} must identify an unique task")

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
        matches = TaskRegistry.resolve_exec_task(value)
        if not matches:
            raise ValidationError(f"Task specifier {value} does not name any public tasks")


class LazyNestedConfigField(mfields.Field):
    """
    Field used to mark lazily-resolved nested configurations.

    This is used for task-dependent configuration types that are not known statically.
    This fields treats its data as a dictionary, however, upon serialization, it accepts
    an arbitrary dataclass that is converted to a dict.
    Note that this automatically defaults to an empty dictionary.
    """
    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return {}
        if dc.is_dataclass(value) and isinstance(value, Config):
            return type(value).schema().dump(value)
        return value

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return {}
        return value


# Helper type for dataclasses to use the PathField
ConfigTaskSpec = NewType("ConfigTaskSpec", str, field=TaskSpecField)
ConfigExecTaskSpec = NewType("ConfigExecTaskSpec", str, field=ExecTaskSpecField)
ConfigPath = NewType("ConfigPath", Path, field=PathField)
ConfigAny = NewType("ConfigAny", any, field=mfields.Raw)
LazyNestedConfig = NewType("LazyNestedConfig", dict[str, any], field=LazyNestedConfigField)


class ConfigContext:
    """
    Base class for context that can be used to bind Config to.

    The context describes the mapping between keys and values.
    The substitution occurs in two steps:
    1. lookup the key in the static parameters
    2. search for any namespace matching the first part of a hierarchical key
       and recursively resolve the following parts of a dotted name.

    The "None" namespace is considered the default namespace in which to look up
    substitution keys.
    """
    def __init__(self):
        self._template_params = {}
        self._namespaces = {}
        self._nresolved = 0

    @property
    def resolved_count(self) -> int:
        """
        Return the number of keys that have been resolved.

        This can be used to detect when we are done with recursive
        resolution.
        """
        return self._nresolved

    def add_values(self, **kwargs):
        """
        Add one or more static template substitution values.

        :param **kwargs: Key-value pairs for template substitution
        """
        self._template_params.update(kwargs)

    def add_namespace(self, config: "Config", name: str | None = None):
        """
        Add a source configuration object that is used to look up referenced keys.

        :param config: An existing configuration object, may contain
            unresolved templates.
        :param name: Name to use in the template keys to refer to this namespace.
            For example, if we set name = "user" for :class:`BenchplotUserConfig`, the value for
            :attr:`BenchplotUserConfig.sdk_path` can be referenced as "{user.sdk_path}".
        """
        self._namespaces[name] = config

    def find(self, key: str) -> str | None:
        """
        Resolve a dot-separated key to the corresponding substitution value.

        :param key: The key to lookup.
        :return: The template substitution or None
        """
        try:
            value = self._template_params[key]
            self._nresolved += 1
            return value
        except KeyError:
            pass

        parts = key.split(".")
        try:
            ns = self._namespaces[parts[0]]
        except KeyError:
            try:
                ns = self._namespaces[None]
            except KeyError:
                return None
        for name in parts[1:]:
            try:
                ns = getattr(ns, name)
            except AttributeError:
                return None
        if isinstance(ns, Config):
            return None
        self._nresolved += 1
        return str(ns)


@dc.dataclass
class Config:
    """
    XXX merge with the basic config.
    Base class for configuration data structure that support template substitution.
    Note that this should be used across the whole hierarchy of nested configurations
    to have the expected behaviour.
    Each field in this dataclass is deserialized from a json file, with support to
    nested configuration dataclasses.
    Types of the fields are normalized to the type annotation given in the dataclass.

    Note that the substitution process is designed to be incremental.
    Some template substitutions may become available later during initialization, therefore
    any unmatched template string will be retained unchanged after a call to
    :meth:`Config.bind`.
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

    def __post_init__(self):
        return

    def _bind_value(self, context: ConfigContext, dtype: Type, value: any) -> any:
        """
        Resolve all template keys in this configuration value and produce the substituted value.

        :param context: The template configuration context
        :param dtype: Type annotation of the field
        :param value: Current value of the field. At this point, this is assumed to be
        a string-like object or a nested configuration.
        :return: The substituted value
        """
        if value is None:
            return None
        if dtype == ConfigAny or dtype == any:
            dtype = type(value)

        if dc.is_dataclass(dtype) and issubclass(dtype, Config):
            return value.bind(context)

        if dtype == str:
            template = value
        elif dtype == Path or dtype == ConfigPath:
            # Path-like object expected here
            template = str(value)
            dtype = Path
        else:
            # Anything else is assumed to never contain things to bind
            return value

        # Now we can query the context for substitution keys to resolve.
        chunks = []
        last_match = 0
        for m in re.finditer(r"\{([a-zA-Z0-9_.]+)\}", template):
            key = m.group(1)
            subst = context.find(key)
            if subst is None:
                continue
            chunks.append(template[last_match:m.start()])
            chunks.append(str(subst))
            last_match = m.end()

        chunks.append(template[last_match:])
        return dtype("".join(chunks))

    def _bind_field(self, context: ConfigContext, dtype: Type, value: any, metadata: dict | None = None) -> any:
        """
        Run the template substitution on a config field.
        If the field is a collection or a nested Config, we recursively bind
        each value.
        """
        if dc.is_dataclass(dtype) and issubclass(dtype, Config):
            # If it is a nested dataclass, just forward it
            return value.bind(context)
        elif dtype == LazyNestedConfig:
            # If it is a lazy_nested_config field, treat this either as a dataclass or a dict
            if dc.is_dataclass(value):
                return value.bind(context)
            else:
                return self._bind_generic(context, dict, value)
        elif is_generic_type(dtype) or is_union_type(dtype):
            # Recurse through the type
            return self._bind_generic(context, dtype, value)
        else:
            # Handle as a single value
            return self._bind_value(context, dtype, value)

    def _bind_generic(self, context: ConfigContext, dtype: Type, value: any) -> any:
        """
        Recursively bind values in generic container types.
        """
        origin = get_origin(dtype)
        if is_union_type(dtype):
            args = get_args(dtype)
            if len(args) == 2 and args[1] == type(None):
                # If we have an optional field, bind with the type argument
                if value is None:
                    return value
                return self._bind_field(context, args[0], value)
            else:
                # Common union field, use whatever type we have as the argument as we do not
                # know how to parse it
                return self._bind_field(context, type(value), value)
        elif type(value) == list:
            if origin is List or origin is list:
                inner_type_fn = lambda v: get_args(dtype)[0]
            else:
                inner_type_fn = lambda v: type(v)
            return [self._bind_field(context, inner_type_fn(v), v) for v in value]
        elif type(value) == dict:
            if origin is Dict or origin is dict:
                inner_type_fn = lambda v: get_args(dtype)[1]
            else:
                inner_type_fn = lambda v: type(v)
            return {key: self._bind_field(context, inner_type_fn(v), v) for key, v in value.items()}
        else:
            return self._bind_value(context, dtype, value)

    def _bind_config(self, source: Self, context: ConfigContext) -> Self:
        """
        Run a template substitution pass with the given substitution context.
        This will resolve template strings as "{foo}" for template key/value
        substitutions that have been registerd in the contexv via
        ConfigContext.register_template_subst() and leave the missing
        template parameters unchanged for later passes.
        """
        changes = {}
        for f in dc.fields(self):
            if not f.init:
                continue
            try:
                metadata = f.metadata.get("metadata", {})
                replaced = self._bind_field(context, f.type, getattr(source, f.name), metadata)
            except Exception as ex:
                raise ConfigurationError(f"Failed to bind {f.name} with value "
                                         f"{getattr(source, f.name)}: {ex}")
            if replaced:
                changes[f.name] = replaced
        return dc.replace(source, **changes)

    def bind(self, context: ConfigContext) -> Self:
        """
        Substitute all templates until there is nothing else that we can substitute.

        :param context: The substitution context
        :return: A new configuration instance with the substituted values.
        """
        bound = self
        max_steps = 10
        last_matched = context.resolved_count
        for _ in range(max_steps):
            bound = self._bind_config(bound, context)
            if context.resolved_count == last_matched:
                break
            last_matched = context.resolved_count
        else:
            raise RuntimeError("LOOP")
            logger.warning("Configuration template binding exceeded recursion depth limit")
        return bound

    def emit_json(self) -> str:
        """
        Custom logic to emit json.
        This is required as in older python version pathlib objects are not serializable.
        """
        schema = self.schema()
        return json.dumps(schema.dump(self), indent=4)


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

    #: Path to the Cheri LLVM sources, inferred from :attr:`src_path` if missing
    llvm_path: Optional[ConfigPath] = dc.field(init=False, default=None)

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
        self.llvm_path = self.src_path / "llvm-project"
        # Try to autodetect openocd
        if self.openocd_path is None:
            self.openocd_path = shutil.which("openocd")

        self.session_path = self.session_path.expanduser().absolute()
        if not self.session_path.is_dir():
            raise ValueError("Session path must be a directory")


@dc.dataclass
class CommonPlatformOptions(Config):
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
class PlatformOptions(Config):
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
class ProfileConfig(Config):
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
    LOCAL = "local"

    def __str__(self):
        return self.value

    def is_fpga(self):
        return self == InstancePlatform.VCU118


class InstanceCheriBSD(Enum):
    LOCAL_NATIVE = "native"
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
class InstanceConfig(Config):
    """
    Configuration for a CheriBSD instance to run benchmarks on.
    XXX-AM May need a custom __eq__() if iterable members are added
    """
    #: Name of the kernel configuration file used
    kernel: str
    #: Is this the baseline reference platform for analysis?
    baseline: bool = False
    #: Optional name used for user-facing output such as plot legends
    name: Optional[str] = None
    #: Platform identifier, this affects the strategy used to run execution tasks
    platform: InstancePlatform = dc.field(default=InstancePlatform.QEMU, metadata={"by_value": True})
    #: Userspace ABI identifier
    cheri_target: InstanceCheriBSD = dc.field(default=InstanceCheriBSD.RISCV64_PURECAP, metadata={"by_value": True})
    #: Kernel ABI identifier
    kernelabi: InstanceKernelABI = dc.field(default=InstanceKernelABI.HYBRID, metadata={"by_value": True})
    #: Is the kernel config name managed by cheribuild or is it an extra one
    #: specified via --cheribsd/extra-kernel-configs?
    cheribuild_kernel: bool = True
    #: Additional key-value parameters that can be used in benchmark templates to 'parameterize' on
    #: the platform axis. These are otherwise unused.
    parameters: Dict[str, ConfigAny] = dc.field(default_factory=dict)
    #: Internal fields, should not appear in the config file and are missing by default
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
            self.name = (f"{self.platform} UserABI:{self.cheri_target} "
                         f"KernABI:{self.kernelabi} KernConf:{self.kernel}")

    def __str__(self):
        return f"{self.name}"


@dc.dataclass
class TaskTargetConfig(Config):
    """
    Specify an analysis task and associated options.
    """

    #: Task specifier with format indicated by :meth:`TaskRegistry.resolve_task`
    handler: ConfigTaskSpec

    #: Extra options for the dataset handler, depend on the handler
    task_options: LazyNestedConfig = dc.field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        # Resolve the lazy task options if this is not already a Config
        if dc.is_dataclass(self.task_options):
            assert isinstance(self.task_options, Config), f"Task options must inherit from Config"
        else:
            self.task_options = resolve_task_options(self.handler, self.task_options, is_exec=False)


@dc.dataclass
class ExecTargetConfig(Config):
    """
    Specify an execution task name.
    Note that the namespace of this task is also used to resolve compatible analysis tasks.
    """

    #: Task specifier with format indicated by :meth:`TaskRegistry.resolve_exec_task`
    handler: ConfigExecTaskSpec

    #: Extra options for the dataset handler, depend on the handler
    task_options: LazyNestedConfig = dc.field(default_factory=dict)  # Dict[str, ConfigAny] = lazy_nested_config_field()

    def __post_init__(self):
        super().__post_init__()
        # Resolve the lazy task options if this is not already a Config
        if dc.is_dataclass(self.task_options):
            assert isinstance(self.task_options, Config), f"Task options must inherit from Config"
        else:
            self.task_options = resolve_task_options(self.handler, self.task_options, is_exec=True)


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
    tasks: List[TaskTargetConfig] = dc.field(default_factory=list)


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


@dc.dataclass
class CommonBenchmarkConfig(Config):
    """
    Common benchmark configuration parameters.
    This is shared between the user-facing configuration file and the internal
    benchmark description.
    """

    #: The name of the benchmark
    name: str
    #: The number of iterations to run
    iterations: int = 1
    #: Benchmark configuration
    #: .. deprecated:: 1.2
    #:    Use :attr:`generators` instead.
    benchmark: Optional[ExecTargetConfig] = None
    #: Auxiliary data generators
    #: .. deprecated:: 1.2
    #:    Use :attr:`generators` instead.
    aux_tasks: List[ExecTargetConfig] = dc.field(default_factory=list)
    #: Data generator tasks.
    #: These are used to produce the benchmark data and loading it during the analysis phase.
    generators: List[ExecTargetConfig] = dc.field(default_factory=list)
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
        assert self.remote_output_dir.is_absolute(), f"Remote output path must be absolute {self.remote_output_dir}"
        conftype = self.__class__.__name__
        if self.benchmark is None:
            if self.aux_tasks:
                raise ValueError(f"Can not use `{conftype}.aux_tasks` and `{conftype}.generators` at the same time")
            if not self.generators:
                raise ValueError(f"At least one `{conftype}.generators` must be specified.")
        else:
            if self.generators:
                raise ValueError(f"Can not use both `{conftype}.benchmark` and `{conftype}.generators`")
            self.generators.append(self.benchmark)
            self.generators.extend(self.aux_tasks)
        # Wipe deprecated fields
        self.benchmark = None
        self.aux_tasks = []

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
        common_info = f"params={self.parameters} gen={self.generators}"
        if self.g_uuid and self.instance:
            return (
                f"{self.name} ({self.uuid}/{self.g_uuid}) on {self.instance} "
                + common_info
            )
        else:
            return f"unallocated {self.name} ({self.uuid}) " + common_info


@dc.dataclass
class CommonSessionConfig(Config):
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

    #: Default analysis task configuration
    analysis_config: AnalysisConfig = dc.field(default_factory=AnalysisConfig)

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

    #: Snapshots of the relevant git repositories that should be syncronised to this session
    #: These are taken when the session is created.
    git_sha: Dict[str, str] = dc.field(default_factory=dict)

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
    def _resolve_template(cls, session_config: Self) -> Self:
        """
        Resolve all template substitutions, except the user_config substitutions that must
        be resolved at session-load time.

        This produces a new session configuration with the substituted values.
        Note that this resolves first the common session template keys, then per-dataset
        configuration keys are introduced and resolved.
        """
        ctx = ConfigContext()
        # Default namespace
        ctx.add_namespace(session_config)
        new_config = session_config.bind(ctx)

        # Replace the default namespace as we have an updated version
        ctx.add_namespace(new_config)
        # Now scan through all the configurations and subsitute per-dataset fields
        new_bench_conf = []
        for bench_conf in new_config.configurations:
            ctx.add_namespace(bench_conf, "benchmark")
            # Shorthand for the corresponding instance configuration
            ctx.add_namespace(bench_conf.instance, "instance")
            # Register parameterization keys, note that these may happen shadow
            # other names, but let it be for now
            ctx.add_values(**bench_conf.parameters)
            # Finally do the binding
            new_bench_conf.append(bench_conf.bind(ctx))
        new_config.configurations = new_bench_conf
        return new_config

    @classmethod
    def generate(cls, user_config: BenchplotUserConfig, config: PipelineConfig) -> Self:
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

        # If there is no instance, use the local instance
        if not config.instance_config.instances:
            config.instance_config.instances.append(
                InstanceConfig(kernel="unknown",
                               baseline=True,
                               name="local",
                               platform=InstancePlatform.LOCAL,
                               cheri_target=InstanceCheriBSD.LOCAL_NATIVE,
                               kernelabi=InstanceKernelABI.NOCHERI,
                               cheribuild_kernel=False))

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

        # Snapshot all repositories we care about, if they are present.
        # Note that we should support snapshot hooks in the configured tasks.
        def snap_head(repo_path, key):
            if repo_path.exists():
                session.git_sha[key] = Repo(repo_path).head.commit.hexsha
            else:
                logger.warning("No %s repository, skip SHA snapshot", key)

        snap_head(user_config.cheribuild_path, "cheribuild")
        snap_head(user_config.cheribsd_path, "cheribsd")
        snap_head(user_config.qemu_path, "qemu")
        snap_head(user_config.llvm_path, "llvm")

        # Now that we are done with generating the configuration, resolve all
        # templates that do not involve the user configuration
        return cls._resolve_template(session)
