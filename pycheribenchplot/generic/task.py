import shlex
from dataclasses import dataclass

from pycheribenchplot.core.artefact import DataFileTarget
from pycheribenchplot.core.config import Config
from pycheribenchplot.core.task import ExecutionTask


@dataclass
class GenericTaskConfig(Config):
    #: The command to execute
    command: str


class GenericExecTask(ExecutionTask):
    """
    This is a simple generic executor that uses the run_options configuration entry to run
    a custom command and collect the output to the output file.
    """
    public = True
    task_namespace = "generic"
    task_config_class = GenericTaskConfig

    def run(self):
        for i in range(self.benchmark.config.iterations):
            output = DataFileTarget.from_task(self, iteration=i)
            section = self.script.benchmark_sections[i]["benchmark"]
            parts = shlex.split(self.config.command)
            section.add_cmd(parts[0], parts[1:], output=output)

    def outputs(self):
        yield "stdout", DataFileTarget.from_task(self, use_iterations=True, ext="txt")
