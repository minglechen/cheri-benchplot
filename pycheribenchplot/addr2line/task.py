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
    output_dir: ConfigPath = field(default_factory=None)
    raw_output_path: ConfigPath = field(default_factory=None)


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
        addr2line_file:Path = self.config.output_dir / "addr2line.csv"
        symbol_file:Path = self.config.output_dir / "symbol.csv"
        # skip if both files are present
        if (
            addr2line_file.is_file()
            and symbol_file.is_file()
            and self.config.raw_output_path.is_file()
        ):
            return
        self.logger.info("Running addr2line on %s", self.config.obj_path)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with ObjdumpResolver(
            self.session.user_config.sdk_path, self.config.obj_path
        ) as resolver:
            addr_df, symbol_df = resolver.load_to_df()
            with addr2line_file.open("w") as f:
                f.write(addr_df.to_csv(index=False))
            with symbol_file.open("w") as f:
                f.write(symbol_df.to_csv(index=False))
            # addr_df.to_csv(path_or_buf=addr2line_file, index=False)
            # symbol_df.to_csv(path_or_buf=symbol_file, index=False)
            if self.config.raw_output_path:
                resolver.write_to_file(self.config.raw_output_path)


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
        self.text = self.objdump.stdout.readlines()
        return self

    def load_to_df(self):
        path: Path = None
        addr: int = None
        line_num: int = None
        symbol: str = None
        addr_line_info = []
        symbol_info = []
        it = iter(self.text)
        # skip first 2 lines
        next(it)
        next(it)
        for line in it:
            line = line.strip()
            if line.startswith("Disassembly of section") or line == "...":
                continue
            # new block
            if line == "":
                # keep the last path and symbol in case of basic block
                # path = ""
                line_num = 0
                # symbol = ""
                continue
            if line.startswith(";"):
                # alternative symbol name line (not used)
                if line.endswith(":"):
                    continue
                # new path and line number
                segs = line.strip("; ").split(":")
                path = segs[0]
                line_num = segs[1]
            else:
                if line.endswith(":"):
                    # new symbol
                    new_symbol = line[line.index("<") + 1 : line.rindex(">")]
                    # clear path and symbol if not basic block
                    if not new_symbol.startswith(".LBB"):
                        # new start of a function
                        symbol = new_symbol
                        path = ""
                        # Add end address of previous function
                        if symbol_info:
                            symbol_info[-1].append(addr)
                        addr = int(line.split()[0], 16)
                        symbol_info.append([symbol, addr])
                    # else:
                    #     print(f"basic block: {new_symbol}")
                else:
                    # new address
                    segs = line.split(":")
                    addr = int(segs[0], 16)
                    # print(f"addr: {addr}, symbol: {symbol}, path: {path}, line: {line_num}")
                    addr_line_info.append([addr, symbol, path, line_num])
        # Add end address of last function
        symbol_info[-1].append(addr)
        return pd.DataFrame.from_records(
            addr_line_info, columns=["addr", "symbol", "path", "line"]
        ), pd.DataFrame.from_records(
            symbol_info, columns=["symbol", "addr", "end_addr"]
        )

    def write_to_file(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as f:
            f.writelines(self.text)

    def __exit__(self, type_, value, traceback):
        self.objdump.terminate()
        self.objdump.wait()
