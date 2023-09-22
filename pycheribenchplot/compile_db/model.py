from pandera.typing import Index, Series

from ..core.model import DataModel


class CompilationDBModel(DataModel):
    """
    The list of files touched during the compilation process.
    This is specific to a 'benchmark' run.
    """
    target: Index[str]
    file: Series[str]
