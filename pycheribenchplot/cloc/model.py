from pandera import Field
from pandera.typing import Index, Series

from ..core.model import GlobalModel


class LoCCountModel(GlobalModel):
    """
    Model for dataframes representing LoC count for a repository snapshot
    """
    repo: Index[str]
    filename: Index[str] = Field(alias="file")
    code: Series[int]
    comment: Series[int]
    blank: Series[int]


class LoCDiffModel(GlobalModel):
    """
    Model for dataframes representing the diff of LoC count between
    two repositories or commits.
    """
    repo: Index[str]
    filename: Index[str] = Field(alias="file")
    how: Index[str] = Field(isin=["added", "same", "modified", "removed"])
    code: Series[int]
    comment: Series[int]
    blank: Series[int]
