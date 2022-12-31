import typing
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

from ..core.config import ConfigPath, ProfileConfig, TemplateConfig
from ..core.task import Task, AnalysisTask
from subprocess import Popen, PIPE, CompletedProcess
from uuid import uuid4


@dataclass
class Addr2LineConfig(TemplateConfig):
    obj_path: ConfigPath = Path(
        "root/spec_static/spec-riscv64-purecap/471.omnetpp/471.omnetpp"
    )
    output_path: ConfigPath = field(default_factory=None)


class Addr2LineTask(Task):
    """Run addr2line on a binary file"""

    task_name = "addr2line"
    task_namespace = "addr2line"
    task_config_class = Addr2LineConfig

    def __init__(self, session, task_config=None):
        super().__init__(task_config)
        self.uuid = uuid4()
        self._session = session

    @property
    def session(self):
        return self._session

    @property
    def task_id(self) -> typing.Hashable:
        return f"{self.task_namespace}-{self.task_name}- {self.uuid}"

    def run(self):
        if self.config.output_path.is_file():
            return
        with ObjdumpResolver(
            self.session.user_config.sdk_path, self.config.obj_path
        ) as resolver:
            df = resolver.load_to_df()
            df.to_csv(self.config.output_path, index=False)


class ObjdumpResolver:
    def __init__(self, sdk_path: Path, obj_path: Path):
        self.objdump_bin = sdk_path / "bin" / "llvm-objdump"
        self.path = obj_path
        self.objdump = None

    def __enter__(self):
        self.objdump = Popen(
            [self.objdump_bin, "-l", "-C", str(self.path)],
            stdout=PIPE,
            text=True,
            encoding="utf-8",
        )
        return self

    def load_to_df(self) -> pd.DataFrame:
        path: Path = None
        addr: int = None
        line_num: int = None
        symbol: str = None
        addr_line_info = []
        it = iter(self.objdump.stdout.readline, "")
        # skip first 2 lines
        next(it)
        next(it)
        for line in it:
            line = line.strip()
            if line.startswith("Disassembly of section") or line == "...":
                continue
            if line == "":
                path = ""
                line_num = 0
                symbol = ""
                continue
            if line.startswith(";"):
                if line.endswith(":"):
                    continue
                segs = line.strip("; ").split(":")
                path = segs[0]
                line_num = segs[1]
            else:
                if line.endswith(":"):
                    symbol = line.strip().split("<")[-1].strip(">:")
                else:
                    segs = line.split(":")
                    addr = int(segs[0], 16)
                    # print(f"addr: {addr}, symbol: {symbol}, path: {path}, line: {line_num}")
                    addr_line_info.append([addr, symbol, path, line_num])
        return pd.DataFrame.from_records(
            addr_line_info, columns=["addr", "symbol", "path", "line"]
        )

    def __exit__(self, type_, value, traceback):
        self.objdump.terminate()
        self.objdump.wait()
