#!/usr/bin/env python

import argparse as ap
from pathlib import Path

from pycheribenchplot.core.config import BenchplotUserConfig
from pycheribenchplot.core.gui import GUIManager
from pycheribenchplot.core.session import Session
from pycheribenchplot.core.tool import CommandLineTool, SubCommand


class GUISubCommand(SubCommand):
    def register_options(self, parser):
        super().register_options(parser)
        parser.add_argument("target",
                            type=Path,
                            default=Path.cwd(),
                            nargs="?",
                            help="Target session path, defaults to the current working directory.")

    def handle(self, user_config, args):
        """
        Initialize the GUI subsystem and startup everything.
        """
        session = self._get_session(user_config, args)
        gui = GUIManager(session)
        gui.run()


def main():
    """
    This is the entry point for the GUI interface to cheri-benchplot.

    Morally this should build onto the same infrastructure as benchplot-cli and use
    the results of the scheduled tasks.
    Currently this is very experimental and only implements a simple demo.
    """
    cli = CommandLineTool("benchplot-gui")
    cli.set_default_handler(GUISubCommand())
    cli.setup()


if __name__ == "__main__":
    main()
