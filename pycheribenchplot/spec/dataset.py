import typing
from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import ConfigPath, TemplateConfig
from ..core.dataset import DatasetArtefact, DataSetContainer, DatasetName


@dataclass
class SpecRunConfig(TemplateConfig):
    #: Path to the Spec2006 suite in the guest
    spec_path: ConfigPath = Path("/opt/spec2006")

    #: Spec workload to run (test/train/ref)
    spec_workload: str = "test"

    #: Spec benchmark name to run
    spec_benchmark: str = "471.omnetpp"

    #: Extra options to the benchmark
    spec_benchmark_options: typing.List[str] = field(default_factory=list)

    def __post_init__(self):
        assert self.spec_path.is_absolute(), "Remote SPEC suite path must be absolute"


class SpecDataset(DataSetContainer):
    dataset_config_name = DatasetName.SPEC
    dataset_source_id = DatasetArtefact.SPEC
    run_options_class = SpecRunConfig

    def __init__(self, benchmark, config):
        super().__init__(benchmark, config)
        # Remote path to spec benchmark
        remote_benchmark_dir = self.config.spec_path / self.config.spec_benchmark
        self.spec_benchmark_bin = remote_benchmark_dir / self.config.spec_benchmark
        self.spec_workload_dir = remote_benchmark_dir / "data" / self.config.spec_workload / "input"

    def iteration_output_file(self, iteration):
        # For now do not commit to an extension, not sure what they emit
        return super().iteration_output_file(iteration).with_suffix(".txt")

    def gen_pre_benchmark(self, script):
        script.gen_cmd("cd", [self.spec_workload_dir])

    def gen_benchmark(self, script, iteration):
        super().gen_benchmark(script, iteration)
        outpath = self.iteration_output_file(iteration)
        script.gen_cmd(self.spec_benchmark_bin, self.config.spec_benchmark_options.copy(), outfile=outpath)
