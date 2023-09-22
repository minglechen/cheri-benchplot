import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from pycheribenchplot.core.plot import PlotTask
from pycheribenchplot.core.config import Config, ConfigPath
from matplotlib import pyplot as plt

@dataclass
class CSetBoundsSizesPlotConfig(Config):
    input_file: ConfigPath = field(default_factory=Path)
    output_file: ConfigPath = field(default_factory=Path)

class CSetBoundsSizesPlotTask(PlotTask):

    task_name = "csetbounds-sizes-plot"
    task_config_class = CSetBoundsSizesPlotConfig
    public = True

    def run(self):
        df = pd.read_csv(self.config.input_file)
        df = df["size"].value_counts().sort_index()
        ax = df.plot.hist(bins=20, logy=True)
        ax.set_xlabel("Size(bytes)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of CSetBounds sizes in SPEC2006")
        plt.tight_layout()
        plt.savefig(self.config.output_file, bbox_inches="tight")
