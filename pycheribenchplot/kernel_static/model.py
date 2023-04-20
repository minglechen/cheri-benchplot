from enum import Enum

import pandas as pd
from pandera import Field, SchemaModel, check, dataframe_check
from pandera.typing import Index, Series

from ..core.model import DataModel, GlobalModel


class SetboundsKind(Enum):
    STACK = "s"
    HEAP = "h"
    SUBOBJECT = "o"
    GLOBAL = "g"
    CODE = "c"
    UNKNOWN = "?"


class SubobjectBoundsModel(DataModel):
    """
    Representation of subobject bounds statistics emitted by the compiler.
    """
    alignment_bits: Series[pd.Int64Dtype] = Field(nullable=True)
    size: Series[pd.Int64Dtype] = Field(nullable=True)
    src_module: Series[str]
    kind: Series[str] = Field(isin=["o", "s", "c", "h", "g", "?"])
    source_loc: Series[str]
    compiler_pass: Series[str]
    details: Series[str] = Field(nullable=True)


class SubobjectBoundsUnionModel(GlobalModel):
    source_loc: Index[str]
    compiler_pass: Index[str]
    details: Index[str] = Field(nullable=True)
    src_module: Series[str]
    size: Series[pd.Int64Dtype] = Field(nullable=True)
    kind: Series[str] = Field(isin=["o", "s", "c", "h", "g", "?"])
    alignment_bits: Series[pd.Int64Dtype] = Field(nullable=True)
