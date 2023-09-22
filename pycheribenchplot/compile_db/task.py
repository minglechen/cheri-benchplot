import platform
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

from ..core.artefact import LocalFileTarget
from ..core.config import Config
from ..core.task import DataGenTask, dependency, output
from ..core.util import SubprocessHelper, resolve_system_command
from .model import CompilationDBModel


class Builder(Enum):
    CheriBuild = "cheribuild"
    CheriBSDPorts = "cheribsd-ports"


@dataclass
class BuildConfig(Config):
    """
    Common options for tasks that build source code to collect statistics
    """
    #: How to build the target, required
    builder: Builder = field(default=None, metadata={"by_value": True})
    #: Target to build, depends on the builder, required
    target: str = None
    #: Always do a clean build
    clean_build: bool = False
    #: Use a temporary build directory instead of the default one
    ephemeral_build_root: bool = False
    # Hidden debug option to skip the build step and only extract
    debug_skip_build: bool = False


@dataclass
class CompilationDBConfig(Config):
    targets: list[BuildConfig] = field(default_factory=list)


class BuildTask(DataGenTask):
    """
    Base task to build some binary.

    This is generally designed to collect statistics from the build,
    not to use the artefacts of the build.
    """
    task_namespace = "cdb"
    task_name = None
    task_config_class = BuildConfig

    def __init__(self, benchmark, script, task_config=None):
        assert task_config.target is not None, "missing target"
        assert task_config.builder is not None, "missing builder"
        # Required for task_id
        self._target = task_config.target
        super().__init__(benchmark, script, task_config=task_config)

        self._cheribuild = self.session.user_config.cheribuild_path / "cheribuild.py"
        if not hasattr(self, "build_root"):
            self.build_root = self.session.user_config.build_path
            if self.config.ephemeral_build_root:
                self._tmp_handle = TemporaryDirectory()
                self.build_root = Path(self._tmp_handle.name)

    @property
    def task_id(self):
        return super().task_id + "-" + self._target

    def _make_subprocess(self, build_root: Path, build_opts: list) -> SubprocessHelper:
        if self.config.builder == Builder.CheriBuild:
            return SubprocessHelper(self._cheribuild, build_opts)
        elif self.config.builder == Builder.CheriBSDPorts:
            raise NotImplementedError("Can't build ports for now")
        raise ValueError(f"Invalid builder {config.builder}")

    def _cheribuild_options(self, target_prefix) -> list[str]:
        opts = ["--skip-update"]
        if self.config.clean_build:
            opts += ["--clean"]
        return opts

    def _do_build_cheribuild(self, build_root: Path):
        instance_config = self.benchmark.config.instance
        cbuild_target = self.config.target + "-" + str(instance_config.cheri_target)

        cbuild_opts = self._cheribuild_options(cbuild_target)
        cbuild_opts += ["--build-root", build_root]
        cbuild_opts += [cbuild_target]
        build_cmd = self._make_subprocess(build_root, cbuild_opts)
        if not self.config.debug_skip_build:
            build_cmd.run()

    def _do_build(self, build_root: Path):
        self.logger.info("Build %s with %s config %s", self.config.target, self.config.builder,
                         self.benchmark.config.instance.kernel)
        if self.config.builder == Builder.CheriBuild:
            self._do_build_cheribuild(build_root)
        elif self.config.builder == Builder.CheriBSDPorts:
            raise NotImplementedError("Can't build ports for now")
        else:
            raise ValueError(f"Invalid builder {self.config.builder}")
        self._extract(build_root)

    def _extract(self, build_root: Path):
        raise NotImplementedError("Must override")

    def run(self):
        self._do_build(self.build_root)


class CheriBSDBuild(BuildTask):
    """
    Base task for building a CheriBSD kernel
    """
    def _cheribuild_options(self, target_prefix) -> list[str]:
        opts = super()._cheribuild_options(target_prefix)
        instance_config = self.benchmark.config.instance
        kconfig = (self.session.user_config.cheribsd_path / instance_config.cheri_target.freebsd_kconf_dir() /
                   instance_config.kernel)
        opts += ["--skip-buildworld", "--skip-install", f"--{target_prefix}/kernel-config", instance_config.kernel]
        if instance_config.platform.is_fpga():
            mfs_image = (self.session.user_config.sdk_path / f"cheribsd-mfs-root-{instance_config.cheri_target}.img")
            opts += [f"--{target_prefix}/mfs-root-image", mfs_image]
        return opts

    def kernel_build_path(self, build_root: Path) -> Path:
        """
        Retrieve the kernel build directory from the build root directory
        """
        instance_config = self.benchmark.config.instance
        build_base = build_root / f"cheribsd-{instance_config.cheri_target.value}-build"
        path_match = list(build_base.glob(f"**/sys/{instance_config.kernel}"))
        if len(path_match) == 0:
            self.logger.error("No kernel build directory for %s in %s", instance_config.kernel, build_root)
            raise FileNotFoundError("Missing kernel build directory")
        assert len(path_match) == 1
        return path_match[0]


class CheriBSDBuildHelper(CheriBSDBuild):
    """
    Common internal task for building the CheriBSD kernel
    """
    task_name = "cheribsd-common"

    def _extract(self, build_root):
        return


class CheriBSDCompilationDB(CheriBSDBuild):
    """
    Internal task that builds a compilation database for CheriBSD builds.

    Note that we use strace on linux and truss on FreeBSD to observe the
    build process. This allows us to catch both source code files and headers
    that are part of the build.
    """
    task_namespace = "cdb"
    task_name = "cheribsd"

    def __init__(self, benchmark, script, task_config=None):
        super().__init__(benchmark, script, task_config=task_config)
        #: strace or truss is required for this
        if platform.system() == "Linux":
            self._tracer = resolve_system_command("strace", self.logger)
        elif platform.system() == "FreeBSD":
            self._tracer = resolve_system_command("truss", self.logger)
        else:
            raise NotImplementedError("Unsupported OS")
        # Force clean builds
        if not self.config.clean_build:
            self.logger.warning("Task configuration does not enable clean builds "
                                "but a clean build is required to build the compilation DB. "
                                "Automatically switching on the clean_build flag.")
            self.config.clean_build = True

    def _build_trace_output(self, build_root: Path) -> Path:
        return build_root / "cdb-trace-output.txt"

    def _make_subprocess(self, build_root, cbuild_opts):
        out_path = self._build_trace_output(build_root)
        if platform.system() == "Linux":
            # Truss arguments
            tracer_args = ["-o", out_path, "-qqq", "--signal=!SIGCHLD", "-f", "--trace=open,openat", "-z"]
        elif platform.system() == "FreeBSD":
            tracer_args = []
        else:
            raise NotImplementedError("Unsupported OS")
        cbuild_cmd = [self._cheribuild] + cbuild_opts
        return SubprocessHelper(self._tracer, tracer_args + cbuild_cmd)

    def _extract_strace(self, out_path: Path) -> pd.DataFrame:
        open_re = re.compile(r"openat\(.*, ?\"([a-zA-Z0-9_/.-]+\.[hcmS])\"")
        file_set = set()
        with open(out_path, "r") as strace_fd:
            for line in strace_fd:
                m = open_re.search(line)
                if not m:
                    continue
                p = Path(m.group(1))
                try:
                    rp = p.relative_to(self.session.user_config.cheribsd_path)
                    file_set.add(str(rp))
                except ValueError:
                    # not relative to cheribsd_path
                    continue
        df = pd.DataFrame({"file": list(file_set)})
        df["target"] = "cheribsd"
        return df.set_index("target")

    def _extract_truss(self, out_path: Path) -> pd.DataFrame:
        raise NotImplementedError("TODO")

    def _extract(self, build_root):
        out_path = self._build_trace_output(build_root)
        if platform.system() == "Linux":
            df = self._extract_strace(out_path)
        elif platform.system() == "FreeBSD":
            df = self._extract_truss(out_path)
        else:
            raise NotImplementedError("Unsupported OS")
        df.to_csv(self.compilation_db.path)

    @output
    def compilation_db(self):
        return LocalFileTarget(self, ext="csv", model=CompilationDBModel)


class CompilationDB(DataGenTask):
    """
    Public target that generates compilation databases for any group of repositories.
    """
    public = True
    task_namespace = "cdb"
    task_name = "generic"
    task_config_class = CompilationDBConfig

    @dependency
    def targets(self):
        """
        Produce a build task for each target configured.
        """
        for build_config in self.config.targets:
            if build_config.target == "cheribsd" and build_config.builder == Builder.CheriBuild:
                # Special case, we have a cheribuild target for this
                yield CheriBSDCompilationDB(self.benchmark, self.script, task_config=build_config)
            else:
                raise NotImplementedError("Invalid target or builder combination")

    def run(self):
        pass

    @output
    def cdb(self):
        """
        Compilation DBs for each target. The data files are in the order of the configuration.
        """
        for target in self.targets:
            yield target.compilation_db
