from collections import namedtuple

import pandas as pd
from pandera import Field, SchemaModel, check, dataframe_check
from pandera.typing import Index, Series

from ..core.dwarf import DWARFStructLayoutModel
from ..core.model import DataModel, GlobalModel


class ImpreciseSubobjectInfoModel(DataModel):
    """
    Concise information about structures with imprecise sub-object bounds.

    This is used to generate more detailed information about the members layout.
    Note that the index mirrors :class:`DWARFStructLayoutModel`.
    """
    file: Index[str]
    line: Index[int]
    base_name: Index[str]
    member_name: Index[str]
    member_offset: Index[int]

    member_size: Series[int]
    member_aligned_base: Series[int]
    member_aligned_top: Series[int]


#: Helper to keep records for :class:`ImpreciseSubobjectInfoModel` consistent
ImpreciseSubobjectInfoModelRecord = namedtuple("ImpreciseSubobjectInfoModelRecord", [
    "file", "line", "base_name", "member_offset", "member_name", "member_size", "member_aligned_base",
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
