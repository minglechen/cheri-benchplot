import dataclasses as dc
import multiprocessing as mp
import re
import typing
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from queue import Queue
from threading import Condition, Event, Lock, Semaphore, Thread

import networkx as nx
import pandas as pd

from .config import AnalysisConfig, Config
from .util import new_logger


class WorkerShutdown(Exception):
    """
    Special exception that triggers the worker thread loop shutdown.
    """

    pass


class TaskRegistry(type):
    """
    The task registry maintains a global list of tasks.
    This is used to resolve tasks from names in configuration files and CLI.

    :attribute public_tasks: Map task_namespace -> { task_name -> Task } for publicly name-able tasks
    :attribute all_tasks: Map task_namespace -> { task_name -> Task } for all tasks
    """

    public_tasks = defaultdict(dict)
    all_tasks = defaultdict(dict)

    def __init__(self, name: str, bases: typing.Tuple[typing.Type], kdict: dict):
        super().__init__(name, bases, kdict)
        if self.task_name is None or self.task_namespace is None:
            # Skip, this is an abstract task
            return
        ns = TaskRegistry.all_tasks[self.task_namespace]
        if self.task_name in ns:
            raise ValueError(
                f"Multiple tasks with the same name: {self}, {ns[self.task_name]}"
            )
        ns[self.task_name] = self
        if self.public:
            ns = TaskRegistry.public_tasks[self.task_namespace]
            ns[self.task_name] = self

    def __str__(self):
        return f"<Task {self.__name__}: spec={self.task_namespace}.{self.task_name}>"

    @classmethod
    def resolve_exec_task(cls, task_spec: str) -> typing.Type["Task"] | None:
        """
        Find the exec task named by the given task specifier.

        :param task_spec:
            Can be one of the following:

            1. The fully qualified name of the task (e.g. <task_namespace>.<task_name>)
            2. The namespace of an exec task (e.g. <task_spec>.exec exists)
        :return: The matching exec task or None
        """
        parts = task_spec.split(".")
        ns = cls.public_tasks[".".join(parts[:-1])]
        # If it is a full name, we are done
        task = ns.get(parts[-1])
        if task:
            return task
        # Do we have an exec task?
        exec_ns = TaskRegistry.public_tasks[".".join(parts)]
        task = exec_ns.get("exec")
        return task

    @classmethod
    def resolve_task(cls, task_spec: str) -> list[typing.Type["Task"]]:
        """
        Find the task named by the given task specifier.

        :param task_spec:
            Can be one of the following:

            1. The fully qualified name of the task (e.g. <task_namespace>.<task_name>)
            2. A wildcard task (e.g. <task_namespace>.*)
        :return: A list of matching tasks
        """
        parts = task_spec.split(".")
        ns = cls.public_tasks[".".join(parts[:-1])]
        if parts[-1] == "*":
            return list(ns.values())
        task = ns.get(parts[-1])
        if task:
            return [task]
        return []


class Target:
    """
    Helper to represent output artifacts of a task.
    This is the base class to also support non-file targets if necessary.
    """

    def is_file(self):
        """
        When true, the target should be a subclass of :class:`FileTarget`
        """
        return False


class DataFrameTarget(Target):
    """
    Target wrapping an output dataframe from a task.
    """

    def __init__(self, model: "DataModel", df: pd.DataFrame):
        self.model = model
        self.df = df


class PlotTarget(Target):
    """
    Target pointing to a plot path
    """

    def __init__(self, path):
        self.path = path


class FileTarget(Target):
    """
    Base class for a target output file.
    """

    @classmethod
    def from_task(cls, task: "Task", prefix: str = None, ext: str = None, **kwargs):
        """
        Create a task path using the task identifier and the parent session data root path.

        :param task: The task that generates this file
        :param prefix: Additional name used to generate the filename, in case the task generates
        multiple files
        :param ext: Optional file extension
        :param `**kwargs`: Forwarded arguments to the :class:`FileTarget` constructor.
        """
        name = re.sub(r"\.", "-", task.task_id)
        if prefix:
            name = f"{prefix}-{name}"
        path = Path(name)

        if ext:
            if not ext.startswith("."):
                ext = "." + ext
            path = path.with_suffix(ext)
        return cls(task.benchmark, path, **kwargs)

    def __init__(
        self,
        benchmark,
        path: Path,
        has_iteration_path: bool = False,
        use_data_root: bool = False,
    ):
        """
        :param benchmark: The benchmark context this file belongs to
        :param path: The file relative path from the benchmark output data directory.
        :param per_iteration: Whether the file is replicated for each iteration
        :param use_data_root: Whether to use the session data root path instead of the per-benchmark data path.
            This is useful for files that do not belong to a single benchmark.
        """
        self.benchmark = benchmark
        self.has_iteration_path = has_iteration_path
        self.use_data_root = use_data_root
        self._path = path

    @property
    def path(self) -> Path:
        """
        Shorthand to return the path for targets that do not depend on iterations.
        If the path depends on the iteration index, this raises a TypeError.
        """
        if self.has_iteration_path:
            raise ValueError(
                "Can not use shorthand path property when multiple paths are present"
            )
        return self.paths[0]

    @property
    def remote_path(self) -> Path:
        """
        Shorthand to return the path for targets that do not depend on iterations.
        If the path depends on the iteration index, this raises a TypeError.
        """
        raise NotImplementedError("Must override")

    @property
    def paths(self) -> list[Path]:
        """
        Return a list of paths belonging to this target.
        If the path depends on the benchmark iteration, this returns all paths sorted
        by iteration number.
        If the path does not depend on iterations, this only returns the path.
        """
        if not self.has_iteration_path:
            if self.use_data_root:
                base = self.benchmark.session.get_data_root_path()
            else:
                base = self.benchmark.get_benchmark_data_path()
            return [base / self._path]
        if self.use_data_root:
            raise NotImplementedError(
                "Using the data root path for iteration output is not yet supported"
            )
        base_paths = map(
            self.benchmark.get_benchmark_iter_data_path,
            range(self.benchmark.config.iterations),
        )
        return [path / self._path for path in base_paths]

    @property
    def remote_paths(self) -> list[Path]:
        """
        Same as path but all paths are coverted rebased to the guest data output directory
        """
        raise NotImplementedError("Must override")

    def is_file(self):
        return True

    def needs_extraction(self):
        raise NotImplementedError("Must override")


class DataFileTarget(FileTarget):
    """
    A target output file that is generated on the guest and needs to be extracted.
    """

    @property
    def remote_path(self):
        base = self.benchmark.get_benchmark_data_path()
        guest_base = self.benchmark.config.remote_output_dir
        return guest_base / self.path.relative_to(base)

    @property
    def remote_paths(self):
        base = self.benchmark.get_benchmark_data_path()
        guest_base = self.benchmark.config.remote_output_dir
        return [guest_base / path.relative_to(base) for path in self.paths]

    def needs_extraction(self):
        return True


class LocalFileTarget(FileTarget):
    """
    A target output file that is generated on the host and does not need to be extracted
    """

    @property
    def remote_path(self):
        raise TypeError("LocalFileTarget does not have a corresponding remote path")

    @property
    def remote_paths(self):
        raise TypeError("LocalFileTarget does not have a corresponding remote path")

    def needs_extraction(self):
        return False


class Task(metaclass=TaskRegistry):
    """
    Abstract base class for dataset operations.
    This can be a task to run a benchmark or perform an analysis step.
    Tasks in the pipeline have determined inputs and outputs that are derived from the session that
    creates the tasks.
    Tasks start as individual entities. When scheduled, if multiple tasks have the same ID, one task instance will be elected as the main task, the others will be attached to it as drones.
    The drones will share the task state with the main task instance to maintain consistency
    when producing stateful task outputs. This is a relatively strange dynamic Borg pattern.
    """

    #: Mark the task as a top-level target
    public = False
    #: Human-readable task namespace, used for task identification
    task_namespace = None
    #: Human-readable task identifier, used for task identification
    task_name = None
    #: If set, use the given Config class to unpack the task configuration
    task_config_class: typing.Type[Config] = None

    def __init__(self, task_config: Config = None):
        assert (
            self.task_name is not None
        ), f"Attempted to use task with uninitialized name {self.__class__.__name__}"
        #: task-specific configuration options, if any
        self.config = task_config
        #: task logger
        self.logger = new_logger(self.task_name)
        #: notify when task is completed
        self._completed = Event()
        #: if the task run() fails, the scheduler will set this to the exception raised by the task
        self.failed = None
        #: set of tasks we resolved that we depend upon, this is filled by the scheduler
        self.resolved_dependencies = set()
        #: collected outputs. This currently caches the results from output_map(). Ideally, however,
        #: this should be filled either by the scheduler or from cached task metadata.
        #: The scheduler fill should occur after all dependencies have completed, but before run(),
        #: so it is possible to access dependencies in the outputs generator. Analysis tasks should be able
        #: to resolve exec tasks outputs from metadata. This would remove the necessity for instantiating tasks
        #: to reference outputs and removes limitations for dynamic output descriptor generation.
        self.collected_outputs = {}

    def __str__(self):
        return str(self.task_id)

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.task_id}>"

    def __hash__(self):
        return hash(self.task_id)

    def __eq__(self, other: "Task"):
        return self.task_id == other.task_id

    @property
    def session(self):
        raise NotImplementedError("Subclasses should override")

    @property
    def task_id(self) -> typing.Hashable:
        """
        Return the unique task identifier.
        """
        raise NotImplementedError("Subclass should override")

    @property
    def completed(self) -> bool:
        """
        Check whether the task has completed. This only notifies whether
        the task is done, to check whether an error occurred, the :attr:`Task.failed`
        should be used.
        """
        return self._completed.is_set()

    @property
    def output_map(self) -> dict[str, Target]:
        """
        Return the output descriptors for the task.
        See note on :attr:`Task.collected_outputs`.
        The only invariant that should be enforced here is that the output map is only ever accessed
        after all dependencies tasks have completed.
        """
        if not self.collected_outputs:
            self.collected_outputs = dict(self.outputs())
        return self.collected_outputs

    def add_drone(self, other: "Task"):
        """
        Register a 'clone' task for this task.
        Drone tasks will share the state of the main task, this includes the completed/failed state and any internal state required for the generation of outputs.
        It is the responsiblity of subclasses to ensure that the proper bits of state
        are shared.
        Note that when this happens it is critical that there are no threads waiting
        on the task.completed event, otherwise they will never be notified.
        """
        assert self.task_id == other.task_id
        assert type(self) == type(other)
        other.__dict__ = self.__dict__

    def wait(self):
        """
        Wait for the task to complete.
        """
        self._completed.wait()

    def notify_done(self):
        """
        Notify that the task is done.
        This is used internally to set the completed event.
        """
        self._completed.set()

    def notify_failed(self, ex: Exception):
        """
        Notify that the task is done but failed.
        """
        self.failed = ex
        self._completed.set()

    def resources(self) -> typing.Iterable["ResourceManager.ResourceRequest"]:
        """
        Produce a set of resources that are consumed by this task.
        Once the resources are available, they will be reserved and
        the resouce object will be availabe via the resource request get()
        method within Task.run().
        """
        yield from []

    def dependencies(self) -> typing.Iterable["Task"]:
        """
        Produce the set of :class:`Task` objects that this task depends upon.

        :return: sequence of dependencies
        """
        yield from []

    def outputs(self) -> typing.Iterable[tuple[str, Target]]:
        """
        Produce the set of :class:`Target` objects that describe the outputs that are produced
        by this task.
        Each target must be associated to a unique name. The name is considered to be the public
        interface for other tasks to fetch targets and outputs of this task.
        The name/target pairs may be used to construct a dictionary to access the return values by key.

        XXX consider making outputs classmethods or recording run tasks outputs somewhere, it is
        annoying to instantiate the tasks just to get the outputs, which may well be dynamic anyway.
        """
        yield from []

    def run(self):
        raise NotImplementedError("Task.run() must be overridden")


class ExecutionTask(Task):
    """
    Base class for tasks that are scheduled as dependencies of the main internal benchmark
    run task.
    These are intended to generate the benchmark run script and perform any instance or benchmark
    configuration before the actual benchmark instance runs.
    Note that these tasks are dynamically resolved by the benchmark context and attached as dependencies
    of the top-level execution tasks :class:`BenchmarkExecTask`.
    Each execution task is associated with an unique benchmark context and a corresponding script builder
    that will generate the script for the associated benchmark.

    The run() method for ExecutionTasks is a good place to perform any of the following:
    1. Extract any static information from benchmark binary files.
    2. Configure the benchmark instance via platform_options.
    3. Add commands to the benchmark run script sections.
    """

    task_name = "exec"
    #: Whether the task requires a running VM instance or not. Instead of changing this use :class:`DataGenTask`.
    require_instance = True

    def __init__(
        self,
        benchmark: "Benchmark",
        script: "ScriptBuilder",
        task_config: Config = None,
    ):
        super().__init__(task_config=task_config)
        #: Associated benchmark context
        self.benchmark = benchmark
        #: Script builder associated to the current benchmark context.
        self.script = script
        #: Make the logger a child of the benchmark logger
        self.logger = new_logger(f"{self.task_name}", parent=self.benchmark.logger)

    @property
    def session(self):
        return self.benchmark.session

    @property
    def uuid(self):
        return self.benchmark.uuid

    @property
    def g_uuid(self):
        return self.benchmark.g_uuid

    @property
    def task_id(self):
        """
        Note that this currently assumes that tasks with the same name are not issued
        more than once for each benchmark run UUID. If this is violated, we need to
        change the task ID generation.
        """
        return f"{self.task_namespace}.{self.task_name}-{self.benchmark.uuid}"


class DataGenTask(ExecutionTask):
    """
    A special type of execution task that does not require a running instance.
    The distinction is made so that the root benchmark execution task knows whether
    to request an instance or not.
    DataGenTasks should not depend on execution tasks, this is to avoid scanning the whole dependency tree
    to determine whether we need to request an instance or not, however the reverse is allowed.
    """

    #: Whether the task requires a running VM instance or not.
    require_instance = False


class AnalysisTask(Task):
    """
    Analysis tasks that perform anythin from plotting to data checks and transformations.
    This is the base class for all public analysis steps that are allocated by the session.
    Analysis tasks are not necessarily associated to a single benchmark. In general they reference the
    current session and analysis configuration, subclasses may be associated to a benchmark context.
    """

    task_namespace = "analysis"

    def __init__(
        self,
        session: "Session",
        analysis_config: AnalysisConfig,
        task_config: Config = None,
    ):
        super().__init__(task_config=task_config)
        #: The current session
        self._session = session
        #: Analysis configuration for this invocation
        self.analysis_config = analysis_config

    @property
    def session(self):
        return self._session

    @property
    def task_id(self):
        """
        Note that this currently assumes that tasks with the same name are not issued
        more than once for each benchmark run UUID. If this is violated, we need to
        change the task ID generation.
        """
        return f"{self.task_namespace}.{self.task_name}-{self.session.uuid}"


class ResourceManager:
    """
    Base class to abstract resource limits on tasks.
    Resouce managers must be registered with the task scheduler before running or adding
    any tasks to the scheduler.
    Resource managers are identified by the resource name, so concrete managers must set
    a resource_name class property and collisions are not allowed.
    """

    resource_name: str = None

    @dc.dataclass
    class ResourceRequest:
        """
        Private object representing a request for this resource.
        This is sortable and comparable.
        """

        name: str
        pool: typing.Hashable | None
        acquire_args: dict[str, typing.Any]
        _resource: typing.Any = dc.field(init=False, default=None)

        def __lt__(self, other):
            return self.name < other.name

        def __eq__(self, other):
            return (
                self.name == other.name
                and self.pool == other.pool
                and self.acquire_args == other.acquire_args
            )

        def set_resource(self, r):
            """
            Internal interface to notify that a resource was given in response to the request
            """
            self._resource = r

        def clear_resource(self):
            """
            Internal interface to notify that a resource is no longer assigned to this request
            """
            self._resource = None

        def get(self):
            if self._resource is None:
                raise RuntimeError(f"Access resource {self.name} but it is not ready")
            return self._resource

    @classmethod
    def request(cls, pool: typing.Hashable = None, **kwargs) -> ResourceRequest:
        assert cls.resource_name is not None, f"{cls} is missing resource name?"
        return cls.ResourceRequest(cls.resource_name, pool, kwargs)

    def __init__(self, session: "Session", limit: int | None):
        assert (
            self.resource_name is not None
        ), f"{self.__class__} is missing resource name?"
        self.session = session
        self.logger = new_logger(f"rman-{self.resource_name}")
        self.limit = limit
        if not self.is_unlimited:
            self._limit_guard = Semaphore(self.limit)
        self.logger.debug(
            "Initialized resource manager %s with limit %s",
            self.resource_name,
            self.limit or "<unlimited>",
        )

    def __str__(self):
        return f"{self.__class__.__name__}[{self.resource_name}]"

    def _acquire(self, pool: typing.Hashable):
        """
        Reserve an item slot in the underlying resource pool
        """
        if self.is_unlimited:
            return
        self._limit_guard.acquire()

    def _release(self, pool: typing.Hashable):
        """
        Release an item slot to the underlying resource pool
        """
        if self.is_unlimited:
            return
        self._limit_guard.release()

    def _get_resource(self, req: ResourceRequest) -> typing.Any:
        """
        Produce a resource item after a slot is acquired.
        """
        raise NotImplementedError("Must override")

    def _put_resource(self, r: typing.Any, req: ResourceRequest):
        """
        Return a resource item to the manager after the client is done with it
        """
        raise NotImplementedError("Must override")

    @property
    def is_unlimited(self) -> bool:
        return self.limit is None or self.limit == 0

    @contextmanager
    def acquire(self, req: ResourceRequest) -> typing.ContextManager[typing.Any]:
        """
        Acquire a resource, optionally specifying an internal pool that
        may be used to improve allocation.
        Resource managers may also require extra arguments to perform the
        resource allocation, there are passed via the ResourceRequest helper
        object.
        """
        self.logger.debug("Waiting for %s", self.resource_name)
        self._acquire(req.pool)
        self.logger.debug("Acquired %s", self.resource_name)
        # Produce the resource
        try:
            r = self._get_resource(req)
            req.set_resource(r)
        except:
            self.logger.error(
                "Failed to produce %s from pool %s", self.resource_name, req.pool
            )
            raise
        finally:
            self._release(req.pool)
        # Wait for somebody to do something with it
        try:
            yield r
        finally:
            req.clear_resource()
            try:
                self._put_resource(r, req)
            except:
                self.logger.error(
                    "Failed to return %s to pool %s", self.resource_name, req.pool
                )
                raise
            finally:
                self._release(req.pool)
        self.logger.debug("Released %s", self.resource_name)

    def sched_shutdown(self):
        """
        The scheduler is shutting down and all activity should
        be moped up. If we need to wait for something to complete cleanup we can
        block here.
        """
        pass


class TaskScheduler:
    """
    Schedule running tasks into workers, handling task dependencies.
    """

    def __init__(self, session: "Session"):
        """
        :param session: The parent session
        """
        #: parent session
        self.session = session
        #: task graph, nodes are Task.task_id, each node has the attribute task, containing the task instance to run.
        self._task_graph = nx.DiGraph()
        #: scheduler logger
        self.logger = new_logger("task-scheduler", self.session.logger)
        #: task runner threads
        self._worker_threads = []
        #: worker wakeup lock, used by the worker wakeup condition
        self._worker_lock = Lock()
        #: worker wakeup event
        self._worker_wakeup = Condition(self._worker_lock)
        #: worker shutdown request
        self._worker_shutdown = False
        #: task queue, this is not a synchronized queue as it shares the lock with _worker_wakeup.
        self._task_queue = []
        #: active tasks set, this contains the set of tasks the workers are currently working on.
        #: Protected by the _worker_wakeup lock.
        self._active_tasks = []
        #: Failed tasks, this does not include tasks that were cancelled as a result of a failure.
        #: Records failed tasks, shares the _worker_wakeup lock.
        self._failed_tasks = []
        #: Records completed tasks by task_id, shares the _worker_wakup lock.
        self._completed_tasks = {}
        # XXX should this share the lock with worker_wakeup?
        #: pending tasks count, protected by _pending_tasks_cv
        self._pending_tasks = 0
        #: task completion barrier
        self._pending_tasks_cv = Condition()
        #: number of worker threads to run
        self.num_workers = session.config.concurrent_workers or (mp.cpu_count() * 5)
        #: resource managers
        self._rman = {}

    def _handle_failure(self, failed_task: Task):
        """
        Handle a task failure.
        Depending on the configuration policy, we either cancel every task or only cancel the subgraph
        linked to the failed task.

        :param failed_task: The task that failed, note that this may already have been cleaned up
        by a previous call to _handle_failure()
        """
        if self.session.config.abort_on_failure:
            # Kill all tasks and signal workers stop
            with self._worker_wakeup:
                self._failed_tasks.append(failed_task)
                for t in self._task_queue:
                    t.notify_failed("cancelled")
                if self._task_queue:
                    self.logger.warning("Cancelling all pending tasks")
                    self._task_queue.clear()
                # XXX Find a nice way to propagate the early-stop request to current tasks.
                # for task in self._active_tasks:
                #     task.cancel()
                self._worker_shutdown = True
                self._worker_wakeup.notify_all()
            # Should this share the lock with worker_wakeup?
            with self._pending_tasks_cv:
                self._pending_tasks = 0
                self._pending_tasks_cv.notify_all()
        else:
            # Find out which tasks we have to kill and unschedule them
            raise NotImplementedError("TODO")

    def _next_task(self) -> Task:
        """
        Pull the next task to work on from the queue.
        If the worker has been asked to shut down we also take this opportunity to bail out.
        """
        with self._worker_wakeup:
            self._worker_wakeup.wait_for(
                lambda: self._worker_shutdown or len(self._task_queue) > 0
            )
            if self._worker_shutdown:
                raise WorkerShutdown()
            return self._task_queue.pop(0)

    def _worker_thread(self, worker_index: int):
        self.logger.debug("Start worker[%d]", worker_index)
        try:
            while True:
                self._worker_loop_one(worker_index)
        except WorkerShutdown:
            self.logger.debug("Caught worker shutdown signal, exit worker loop")
        except Exception as ex:
            # This should never happen, everything should be caught within the worker loop
            self.logger.critical(
                "Critical error in worker thread, scheduler state is undefined: %s", ex
            )
        self.logger.debug("Shutdown worker[%d]", worker_index)

    def _worker_loop_one(self, worker_index: int):
        """
        Main worker loop. This pulls tasks from the task_queue and handles them when their dependencies
        have completed. If the schedule is well-formed, we are guaranteed not to deadlock.

        :param worker_index: The sequential index of the worker in the worker list.
        """
        task = self._next_task()
        self.logger.debug("worker[%d] received task: %s", worker_index, task)
        try:
            # Wait for all dependencies to be done
            for dep_id in self._task_graph.successors(task.task_id):
                dep = self._task_graph.nodes[dep_id]["task"]
                dep.wait()
                assert (
                    dep.completed
                ), f"Dependency wait() returned but task is not completed {dep}"
                # If a dependency failed, we need to do some scheduling changes, the exception triggers it
                if dep.failed:
                    self.logger.error(
                        "Dependency %s failed, bail out from %s", dep, task
                    )
                    raise RuntimeError("Task dependency failed")
                # Before continuing, we should check whether somebody notified worker shutdown
                with self._worker_wakeup:
                    if self._worker_shutdown:
                        raise WorkerShutdown()
            # Grab all resources
            with ExitStack() as resource_stack:
                resources = {}
                for req in sorted(task.resources()):
                    resources[req.name] = resource_stack.enter_context(
                        self._rman[req.name].acquire(req)
                    )
                # Now we can run the task
                task.run()
            # We have finished with the task, if we reach this point the task completed successfully
            # and its resources have been released
            task.notify_done()
            with self._worker_lock:
                self._completed_tasks[task.task_id] = task
            with self._pending_tasks_cv:
                self._pending_tasks -= 1
                self._pending_tasks_cv.notify_all()
        except WorkerShutdown:
            # Just pass it through
            raise
        except Exception as ex:
            self.logger.exception(
                "Error in worker[%d] handling task %s", worker_index, task
            )
            # Notify other workers that may be waiting on this dependency that they should bail out.
            task.notify_failed(ex)
            # Now we need to cancel some tasks
            self._handle_failure(task)

    @property
    def failed_tasks(self) -> list[Task]:
        """
        Verify whether there were failed tasks during the execution.
        This returns a copy of the internal failed tasks list.
        """
        with self._worker_lock:
            return list(self._failed_tasks)

    @property
    def completed_tasks(self) -> dict[str, Task]:
        """
        Maps task-id to completed tasks.
        This is useful to retreive a specific task by ID although generally
        it should not be used by the task pipeline as the task outputs from the task
        instances issued as dependencies can be referenced directly.
        Note that this will not include failed tasks.
        """
        with self._worker_lock:
            return dict(self._completed_tasks)

    def add_task(self, task: Task):
        self.logger.debug("Schedule task %s", task.task_id)
        if task.task_id in self._task_graph:
            # Assume that we can not have tasks with duplicate IDs
            # If a task is already scheduled just skip this, duplicate
            # dependencies are allowed.
            self._task_graph.nodes[task.task_id]["task"].add_drone(task)
            return
        self._task_graph.add_node(task.task_id, task=task)
        for dep in task.dependencies():
            task.resolved_dependencies.add(dep)
            self.add_task(dep)
            self._task_graph.add_edge(task.task_id, dep.task_id)

    def resolve_schedule(self):
        """
        Produce a schedule for the tasks. Note that we use lexicographic sort by g_uuid in
        order to help the instance manager maximise reuse of instances when this is enabled.
        """
        try:
            if self.session.config.reuse_instances:
                sched = nx.lexicographical_topological_sort(
                    self._task_graph, key=lambda t: self._task_graph[t]["task"].g_uuid
                )
            else:
                sched = nx.topological_sort(self._task_graph)
            run_sched = [
                self._task_graph.nodes[t]["task"] for t in reversed(list(sched))
            ]
            self.logger.debug("Resolved benchmark schedule %s", run_sched)
            return run_sched
        except nx.NetworkXUnfeasible:
            for cycle in nx.simple_cycles(self._task_graph):
                cycle_str = [str(self._task_graph.nodes[c]["task"]) for c in cycle]
                self.logger.error(
                    "Impossible task schedule: cyclic dependency %s", cycle_str
                )
            raise RuntimeError("Impossible to create a task schedule")

    def register_resource(self, rman: ResourceManager):
        if rman.resource_name in self._rman:
            self.logger.error(
                "Duplicate resource manager registration %s: given %s found %s",
                rman.resource_name,
                rman,
                self._rman[rman.resource_name],
            )
            raise ValueError(
                "Resource manager with the same name is already registered"
            )
        self._rman[rman.resource_name] = rman

    def run(self):
        """
        Spawn worker threads and run the tasks in the schedule.
        Once all tasks are done, the workers are cleaned up.
        """
        self._worker_shutdown = False
        schedule = self.resolve_schedule()
        try:
            # No need to synchronize this as the workers have not started yet
            self._task_queue.extend(schedule)
            self._pending_tasks = len(self._task_queue)
            self.logger.debug("Queued tasks: pending=%d", self._pending_tasks)
            # only start working if we actually have work
            if self._pending_tasks == 0:
                return
            for i in range(self.num_workers):
                worker = Thread(target=self._worker_thread, args=[i])
                self._worker_threads.append(worker)
                worker.start()
            # notify that we have work
            with self._worker_wakeup:
                self._worker_wakeup.notify_all()
            # wait for all tasks to complete
            with self._pending_tasks_cv:
                self._pending_tasks_cv.wait_for(lambda: self._pending_tasks == 0)
        finally:
            # shutdown all workers
            self.logger.debug("Shutting down workers")
            with self._worker_wakeup:
                self._worker_shutdown = True
                self._worker_wakeup.notify_all()
            for i, worker in enumerate(self._worker_threads):
                self.logger.debug("Wait for worker[%d] shutdown", i)
                worker.join()
            self._worker_threads.clear()
            for name, rman in self._rman.items():
                self.logger.debug("Shutting down resource manager %s", name)
                rman.sched_shutdown()
            self._rman.clear()
