import pandas as pd
from pandera import Field, check, dataframe_check
from pandera.typing import Index, Series

from pycheribenchplot.core.model import BaseDataModel
from pycheribenchplot.core.util import new_logger


class CheriBSDChangesModel(BaseDataModel):
    """
    CheriBSD changes are not identified by a benchmark UUID, instead they exist as a
    session-global dataset.
    We omit dataset_id/gid and iteration indexes.
    """
    filename: Index[str] = Field(alias="file", check_name=True)
    target_type: Series[str] = Field(isin=["header", "kernel", "lib", "prog"])
    updated: Series[str]
    changes: Series[object]
    changes_purecap: Series[object]
    hybrid_specific: Series[bool]
    change_comment: Series[str] = Field(nullable=True)

    @check("target_type")
    def check_warn_target_null(cls, target_type: Series[str]):
        logger = new_logger("cheribsd-changes-model")
        nulls = target_type.isna()
        offending_files = target_type.index.get_level_values("file")[nulls]
        for fname in offending_files:
            logger.warning("%s has CHERI changes annotation but misses target_type field", fname)
        return pd.Series(True, index=target_type.index)

    @dataframe_check
    def changes_nonnull(cls, df: pd.DataFrame) -> Series[bool]:
        """
        Check that the changes + changes_purecap columns are collectively non-null.
        """
        logger = new_logger("cheribsd-changes-model")
        nulls = (df["changes"].isna() & df["changes_purecap"].isna())
        offending_files = df.index.get_level_values("file")[nulls]
        for fname in offending_files:
            logger.error("%s has CHERI changes annotation but have no changes category", fname)
        return ~nulls

    @classmethod
    def dynamic_index_position(cls):
        return None
