from collections import namedtuple
from enum import Enum

import pandas as pd
from pandera import Field, SchemaModel, check, dataframe_check
from pandera.typing import Index, Series

from ..core.dwarf import DWARFStructLayoutModel
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


class ImpreciseSubobjectInfoModel(DataModel):
    """
    Concise information about structures with imprecise sub-object bounds.

    This is used to generate more detailed information about the members layout.
    Note that the index mirrors :class:`DWARFStructLayoutModel`.
    """
    file: Index[str]
    line: Index[int]
    type_name: Index[str]
    member_name: Index[str]
    member_offset: Index[int]

    type_id: Series[int]
    member_size: Series[int]
    member_aligned_base: Series[int]
    member_aligned_top: Series[int]


#: Helper to keep records for :class:`ImpreciseSubobjectInfoModel` consistent
ImpreciseSubobjectInfoModelRecord = namedtuple("ImpreciseSubobjectInfoModelRecord", [
    "type_id", "file", "line", "type_name", "member_offset", "member_name", "member_size", "member_aligned_base",
    "member_aligned_top"
])


class ImpreciseSubobjectLayoutModel(DWARFStructLayoutModel):
    """
    Augment the structure layout with capability aliasing information
    """
    #: Identifies a struct member as being the owner of an alias group.
    alias_group_id: Series[pd.Int64Dtype] = Field(nullable=True)
    #: Identifies the aligned base of a member that owns an alias group.
    alias_aligned_base: Series[pd.Int64Dtype] = Field(nullable=True)
    #: Idnetifies the aligned top of a member that owns an alias group.
    alias_aligned_top: Series[pd.Int64Dtype] = Field(nullable=True)
    #: Describe groups of fields that are aliased with the same imprecise capability.
    #: Alias groups are incrementally numbered and are unique for each type_id,
    #: i.e. for each root structure type.
    alias_groups: Series[object] = Field(nullable=True)
