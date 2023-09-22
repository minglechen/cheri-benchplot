from dataclasses import dataclass, field

from ..core.config import Config


@dataclass
class BuildConfig(Config):
    """
    Options for a build task.
    """
    #: Always make a fresh build, this is slow but required for accurate compilation DB data
    clean_build: bool = True
    #: Use an ephemeral build root in tmpfs instead of the default one
    ephemeral_build_root: bool = False
    #: Trigger the build on a remote machine
    remote_build: bool = True


@dataclass
class PortsBuildConfig(BuildConfig):
    #: Use poudriere to setup a jail
    use_jail: bool = True


class CDBGenBase(DataGenTask):
    """
    Generate the compilation database for a build.
    The build targets the platform associated to the current benchmark instance and may
    use parameters specified for this benchmark.
    """
    task_config_class = BuildConfig

    def __init__(self, benchmark, script, task_config=None):
        super().__init__(benchmark, script, task_config=task_config)


class CheriBSDPortsBuild(CDBGenBase):
    """
    Generate the compilation database for specific CheriBSD ports package(s)
    """
    task_config_class = PortsBuildConfig
