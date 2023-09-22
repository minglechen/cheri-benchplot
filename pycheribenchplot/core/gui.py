import dataclasses as dc
import json
from functools import partial
from typing import NewType, Type, get_origin

import pandas as pd
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (QApplication, QCheckBox, QFrame, QGridLayout,
                             QHeaderView, QLabel, QLayout, QLineEdit,
                             QMainWindow, QPushButton, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout,
                             QWidget)

from .analysis import AnalysisTask
from .config import AnalysisConfig, Config, TaskTargetConfig
from .session import UNPARAMETERIZED_INDEX_NAME, Session
from .util import new_logger


class GUIAnalysisTaskMixin:
    """
    Mixin class for all analysis tasks that provide a GUI interface

    These will be rendered as independent tabs on the main benchplot application.
    """
    def populate(self, layout: QLayout):
        raise NotImplementedError("Must override")


class RunContext(QObject):
    #: Signal emitted when the task is completed
    done = pyqtSignal(GUIAnalysisTaskMixin)

    def __init__(self, task_type, task_options=None):
        super().__init__()
        self.task_type = task_type
        self.task_options = task_options


class SessionRunner(QObject):
    """
    Thread worker used to run the session scheduler loop.
    There can only be one loop running at a time.

    XXX currently this finishes running after all tasks are done.
    This should be converted to a proper thread pool.
    """

    def __init__(self, session: Session):
        super().__init__()
        self.session = session

    def run_session(self, ctx: RunContext):
        target = f"{ctx.task_type.task_namespace}.{ctx.task_type.task_name}"
        analysis_config = AnalysisConfig()
        analysis_config.tasks = [
            TaskTargetConfig(handler=target, task_options=dc.asdict(ctx.task_options))
        ]
        self.session.analyse(analysis_config)

        for task in self.session.scheduler.completed_tasks.values():
            if isinstance(task, ctx.task_type):
                ctx.done.emit(task)
        self.session.scheduler.completed_tasks.clear()


class GUITaskWidget(QWidget):
    """
    Widget that holds a task instance and will be used to host its GUI interface.
    """
    run_trigger = pyqtSignal(RunContext)

    def __init__(self, manager, task_type: Type[AnalysisTask], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = manager.logger
        self._task_type = task_type
        self._task_config_fields = []
        self._task_config = {}

        wl = QVBoxLayout()
        self.run_button = QPushButton("Run analysis task")
        self.run_button.clicked.connect(self.trigger_run)
        wl.addWidget(self.run_button)

        # Build form from task configuration
        if self._task_type.task_config_class:
            self._task_config_fields = {f.name: f for f in dc.fields(self._task_type.task_config_class)}
            self.options_form = self._config_to_form()
            wl.addWidget(self.options_form)
        else:
            self.options_form = None

        self.run_status = QLabel("Running...")
        self.run_status.hide()
        wl.addWidget(self.run_status)

        wl.addStretch()
        self.setLayout(wl)

    def _config_to_form(self) -> QWidget:
        w = QFrame()
        wl = QGridLayout()

        for idx, field in enumerate(self._task_config_fields.values()):
            if field.type is bool:
                input_widget = QCheckBox(field.name)
                input_widget.stateChanged.connect(
                    lambda value: self._update_config(field.name, bool(value)))
                input_widget.setChecked(bool(field.default))
                wl.addWidget(input_widget, idx, 0)
            else:
                # Treat everything else as a string for now
                label = QLabel(field.name + ":")
                wl.addWidget(label, idx, 0)
                input_widget = QLineEdit()
                input_widget.textChanged.connect(
                    partial(self._update_config, field.name))
                wl.addWidget(input_widget, idx, 1)
        w.setLayout(wl)
        return w

    def _update_config(self, field_name: str, value: any):
        self._task_config[field_name] = value

    def trigger_run(self):
        self.run_button.hide()
        if self.options_form:
            self.options_form.hide()
        self.run_status.show()

        config = None
        if self._task_type.task_config_class:
            schema = self._task_type.task_config_class.schema()
            config = schema.load(self._task_config)
        ctx = RunContext(self._task_type, config)
        ctx.done.connect(self.run_done)
        self.run_trigger.emit(ctx)

    def run_done(self, finished_task: GUIAnalysisTaskMixin):
        self.run_status.hide()
        try:
            finished_task.populate(self.layout())
        except Exception as ex:
            self.logger.error("Failed to populate GUI tab for %s: %s",
                              finished_task.task_id, ex)


class GUIManager:
    """
    Main Qt application manager for graphical UI handling.

    This is a top-level interface to manage tasks interactions with GUI tools.
    """

    def __init__(self, session: Session):
        self.session = session
        self.logger = new_logger("gui", self.session.logger)

        #: The root application, this must be created in the main session thread
        self._app = QApplication([])
        self._app.setApplicationName("cheri-benchplot")

        self._session_worker = SessionRunner(session)
        self._session_thread = QThread()
        self._session_worker.moveToThread(self._session_thread)
        self._session_thread.finished.connect(self._session_worker.deleteLater)

    def _create_main_tab(self, task_tabs: QTabWidget):
        """
        Initialize the "home" tab
        """
        w = QWidget()
        wl = QVBoxLayout()
        path_label = QLabel(f"Session location: {self.session.session_root_path}")
        path_label.setFrameStyle(QFrame.Shape.Panel)
        wl.addWidget(path_label)

        info_label = QLabel("Session datagen configuration:")
        wl.addWidget(info_label)
        # Fill the session matrix view
        rows, cols = self.session.benchmark_matrix.shape
        datagen_matrix = QTableWidget(rows, cols + self.session.benchmark_matrix.index.nlevels)
        header = []
        for name in self.session.benchmark_matrix.index.names:
            if name == UNPARAMETERIZED_INDEX_NAME:
                header.append("index")
            else:
                header.append(name)
        for g_uuid in self.session.benchmark_matrix.columns:
            conf = self.session.platform_map[g_uuid]
            header.append(conf.name + "\n" + str(g_uuid))
        datagen_matrix.setHorizontalHeaderLabels(header)
        # Ensure that the headers are sized automatically
        header_widget = datagen_matrix.horizontalHeader()
        header_widget.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        for rownum, (index, row) in enumerate(self.session.benchmark_matrix.iterrows()):
            if not isinstance(index, tuple):
                index = (index,)
            for colnum, value in enumerate(index):
                item = QTableWidgetItem(str(value))
                datagen_matrix.setItem(rownum, colnum, item)
            for colnum, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                datagen_matrix.setItem(rownum, colnum + len(index), item)

        wl.addWidget(datagen_matrix)

        wl.addStretch()
        w.setLayout(wl)
        task_tabs.addTab(w, self.session.name)

    def _collect_gui_components(self, task_tabs: QTabWidget):
        """
        Collect GUI components compatible with this session
        """
        for task_type in self.session.get_public_tasks():
            if issubclass(task_type, GUIAnalysisTaskMixin):
                self._create_task_tab(task_tabs, task_type)

    def _create_task_tab(self, task_tabs: QTabWidget, task_type: Type[GUIAnalysisTaskMixin]):
        """
        Create a widget to host the GUI for the given task
        """
        if not task_type.is_session_task():
            self.logger.warning("Skipping task %s as we only support direct scheduling of session analysis tasks.", task_type)
            return

        w = GUITaskWidget(self, task_type)
        w.run_trigger.connect(self._session_worker.run_session)
        task_tabs.addTab(w, f"{task_type.task_namespace}.{task_type.task_name}" )


    def run(self):
        window = QMainWindow()
        task_tabs = QTabWidget()
        self._create_main_tab(task_tabs)
        window.setCentralWidget(task_tabs)
        window.show()

        self._collect_gui_components(task_tabs)
        self._session_thread.start()

        # XXX We should have a daemon-mode for the job scheduler to start here
        # but I'm lazy and so I will just have a new scheduler run every time.

        self._app.exec()

