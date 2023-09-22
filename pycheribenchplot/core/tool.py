import argparse as ap
import inspect
import logging
from json.decoder import JSONDecodeError
from pathlib import Path
from traceback import print_exception
from typing import Callable, Type

from marshmallow.exceptions import ValidationError
from typing_extensions import Self

from .config import BenchplotUserConfig, Config
from .error import ToolArgparseError
from .session import Session
from .util import setup_logging


class SubCommand:
    """
    Represent a subcommand in the command line tool.

    Base class, this should be derived to provide specific implementations.
    The comments in the subclass will be used to generate help messages.
    """
    name: str = None

    def __init__(self):
        #: Parent handler, initialized when the subcommand is added to the parent object
        self.parent: Self | None = None

    def _parse_config(self, path: Path | None, config_model: Type[Config], use_default: bool = False) -> Config:
        """
        Try to load a configuration file.

        If this fails, emit an error message and raise the appropriate exception.
        """
        if not path or not path.exists():
            if use_default:
                return config_model()
            self.logger.error("Missing configuration file %s", path)
            raise FileNotFoundError(f"Missing {path}")

        try:
            config = config_model.load_json(path)
        except JSONDecodeError as ex:
            self.logger.error("Malformed configuration %s: %s", path, ex)
            raise
        except ValidationError as ex:
            self.logger.error("Invalid configuration %s: %s", path, ex)
            raise
        return config

    def _get_session(self, user_config: BenchplotUserConfig, args: ap.Namespace, missing_ok: bool = False):
        """
        Helper to produce a session from argument parser information.

        Note that this assumes that the session path is stored in `args.target`.
        """
        session = Session.from_path(user_config, args.target)
        if not session and not missing_ok:
            self.logger.error("Session %s does not exist", args.target)
            raise FileNotFoundError(f"Session not found at {args.target}")
        return session

    def _register_session_arg(self, parser):
        """
        Helper to add the session ID argument to the given parser.
        """
        parser.add_argument("target",
                            type=Path,
                            default=Path.cwd(),
                            nargs="?",
                            help="Path to the target session, defaults to the current working directory.")

    @property
    def logger(self):
        return self.parent.logger

    def register_options(self, parser):
        """
        Register options for this command to the parser.
        """
        pass

    def handle(self, user_config: BenchplotUserConfig, args: ap.Namespace):
        """
        Handle the invocation of this subcommand
        """
        raise NotImplementedError("Must override")


class CommandLineTool:
    """
    Helper class to manage command line parsing and benchplot initialization.
    """
    def __init__(self, name: str):
        self.parser = ap.ArgumentParser(description=f"CHERI plot and data analysis tool {name}")
        #: User-friendly name for the tool help
        self.name = name
        #: SubCommand that handles setup if no subcommand is registered or no subcommand is specified on the CLI.
        self._default_handler = None
        #: Subcommands registered
        self._subcommands = {}
        #: Subcommands parser builder
        self._subp = None
        #: Root logger
        self.logger = None

        self._register_common_options()

    def _register_common_options(self):
        """
        Register common options to the command line parser.
        """
        self.parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
        self.parser.add_argument("-l", "--logfile", type=Path, help="logfile", default=None)
        self.parser.add_argument("-w",
                                 "--workers",
                                 type=int,
                                 help="Override max number of workers from configuration file",
                                 default=None)
        self.parser.add_argument("-c",
                                 "--config",
                                 type=Path,
                                 help="User environment configuration file. Defaults to ~/.config/cheri-benchplot.json",
                                 default=Path("~/.config/cheri-benchplot.json").expanduser())

    def _parse_user_config(self, args) -> BenchplotUserConfig:
        """
        Load the user environment configuration file.
        """
        try:
            if args.config.exists():
                user_config = BenchplotUserConfig.load_json(args.config)
            else:
                self.logger.debug("Missing user config, using defaults")
                user_config = BenchplotUserConfig()
            user_config.verbose = args.verbose
        except JSONDecodeError as ex:
            self.logger.error("Malformed user configuration %s: %s", args.config, ex)
            if args.verbose:
                print_exception(ex)
            exit(1)
        except ValidationError as ex:
            self.logger.error("Invalid user configuration %s: %s", args.config, ex)
            if args.verbose:
                print_exception(ex)
            exit(1)
        return user_config

    def _handle_command(self, user_config: BenchplotUserConfig, args: ap.Namespace):
        """
        Dispatch the command specified by the command line to the respective handler.

        If no subcommand is specified, run the default handler
        """
        if self._subcommands:
            try:
                subcmd = self._subcommands[args.command]
            except KeyError:
                if self._default_handler:
                    self._default_handler.handle(user_config, args)
                    return
                else:
                    self.parser.print_help()
                    exit(1)
            subcmd.handle(user_config, args)
        else:
            assert self._default_handler, "Default handler must be set if no subcommand is registered"
            self._default_handler.handle(user_config, args)

    def set_default_handler(self, cmd: SubCommand):
        """
        Set the default handler that will be called when no subcommand are registered
        or no command is given on the command line.
        """
        cmd.parent = self
        cmd.register_options(self.parser)
        self._default_handler = cmd

    def add_subcommand(self, cmd: SubCommand):
        """
        Enable subcommands for this tool and register a new subcommand.
        """
        if not self._subcommands:
            self._subp = self.parser.add_subparsers(help="command", dest="command")

        assert self._subp is not None, "Subcommand missing name"
        assert cmd.name not in self._subcommands, "Duplicate subcommand"
        cmd.parent = self
        self._subcommands[cmd.name] = cmd

        cmd_help = inspect.cleandoc(inspect.getdoc(cmd))
        sub = self._subp.add_parser(cmd.name, help=cmd_help)
        cmd.register_options(sub)

    def setup(self):
        """
        Parse arguments and setup the common options
        """
        args = self.parser.parse_args()

        self.logger = setup_logging(args.verbose, args.logfile)
        self.logger.debug("Loading user config %s", args.config)
        if not args.config.exists():
            self.logger.error("Missing user configuration file %s", args.config)
            self.parser.print_help()
            exit(1)

        user_config = self._parse_user_config(args)

        if args.workers:
            # The argument takes precedence over every other source
            # of the concurrent_workers option
            user_config.concurrent_workers = args.workers

        # Adjust log level
        if user_config.verbose:
            self.logger.setLevel(logging.DEBUG)

        # Now we can handle the command that is being selected
        try:
            self._handle_command(user_config, args)
        except ToolArgparseError as ex:
            self.logger.error("Invalid arguments: %s", ex)
            self.parser.print_help()
            exit(1)
        except Exception as ex:
            if self._subcommands:
                self.logger.error("Failed to run command '%s': %s", args.command, ex)
            else:
                self.logger.error("Failed to run %s", self.name)
            if args.verbose:
                print_exception(ex)
            exit(1)
