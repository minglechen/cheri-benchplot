import numpy as np
import pandas as pd
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QLayout, QTreeView

from ..core.gui import GUIAnalysisTaskMixin


class C18NTraceInspector(GUIAnalysisTaskMixin):
    """
    GUI application to display trace data.

    Note that this detaches to a separate thread
    """
    def _fill_from_df(self, root, df):
        """
        Fill intems into the model.

        Note that we take advantage of the fact that sequence numbers are
        linearly increasing.
        """
        current = root
        for _, row in df.iterrows():
            if row["op"] == "enter":
                item = QStandardItem(f"{row.address} {row.op} => {row.compartment}:{row.symbol}".strip())
                current.appendRow(item)
                current = item
            else:
                current = current.parent()
                if current is None:
                    current = root

    def _fill_items(self, model):
        """
        Grab items from the annotated tree and add them to the model.

        XXX we should have different tabs for different traces
        """
        root = model.invisibleRootItem()

        for annotated_trace in self.deps:
            df = annotated_trace.trace_df.get()
            self._fill_from_df(root, df)

    def populate(self, layout: QLayout):
        """
        Build the tab widgets and attach them to the GUI.
        """
        tv = QTreeView()
        model = QStandardItemModel()
        tv.setModel(model)
        layout.insertWidget(layout.count() - 1, tv)

        layout.takeAt(layout.count() - 1)

        self._fill_items(model)
