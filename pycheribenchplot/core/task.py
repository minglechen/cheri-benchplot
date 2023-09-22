import dataclasses as dc
import multiprocessing as mp
import re
from collections import defaultdict
from collections.abc import Iterable
from contextlib import ExitStack, contextmanager
from queue import Queue
from threading import Condition, Event, Lock, Semaphore, Thread
from typing import Any, Callable, ContextManager, Hashable, Iterable, Type

import networkx as nx

from .borg import Borg
from .config import Config
from .error import MissingDependency, TaskNotFound
from .util import new_logger


class WorkerShutdown(Exception):
    """
    Special exception that triggers the worker thread loop shutdown.
    """

    pass


@dc.dataclass
class TargetRef:
    """
    Internal helper for targets
    """

    #: The unique key for the target
    name: str
    #: The attribute name for the target in the Task subclass.
    attr: str


class TaskRegistry(type):
    """
    The task registry maintains a global list of tasks.
    This is used to resolve tasks from names in configuration files and CLI.

    :attribute public_tasks: Map task_namespace -> { task_name -> Task } for publicly name-able tasks
    :attribute all_tasks: Map task_namespace -> { task_name -> Task } for all tasks
    """

    public_tasks = defaultdict(dict)
    all_tasks = defaultdict(dict)

    def __new__(mcls, name: str, bases: tuple[Type], kdict: dict):
        """
        Add two registry attributes to each task class.
        The output_registry maintains a set of output generators for the class.
        The deps_registry maintins a set of dependency generators for the class.
        """
        kdict["_output_registry"] = {}
        kdict["_deps_registry"] = []
        return super().__new__(mcls, name, bases, kdict)

    def __init__(self, name: str, bases: tuple[Type], kdict: dict):
        super().__init__(name, bases, kdict)
        if self.task_name is None or self.task_namespace is None:
            # Skip, this is an abstract task
            return
        if self.task_config_class:
            assert dc.is_dataclass(self.task_config_class), "Task configuration must be a dataclass"
            assert issubclass(self.task_config_class, Config), "Task configuration must inherit Config"
        ns = TaskRegistry.all_tasks[self.task_namespace]
        if self.task_name in ns:
            raise ValueError(
                f"Multiple tasks with the same name: {self}, {ns[self.task_name]}"
            )
        ns[self.task_name] = self
        if self.public:
            ns = TaskRegistry.public_tasks[self.task_namespace]
            ns[self.task_name] = self
        for base in bases:
            if hasattr(base, "_deps_registry"):
                self._deps_registry.extend(base._deps_registry)
            if hasattr(base, "_output_registry"):
                self._output_registry.update(base._output_registry)

    def __str__(self):
        return f"<Task {self.__name__}: spec={self.task_namespace}.{self.task_name}>"

    def register_dependency(cls, name):
        cls._deps_registry.append(name)

    def register_output(cls, ref: TargetRef):
        cls._output_registry[ref.name] = ref.attr

    @classmethod
    def resolve_exec_task(cls, task_spec: str) -> Type["Task"] | None:
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
    def resolve_task(cls, task_spec: str) -> list[Type["Task"]]:
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

    @classmethod
    def iter_public(cls) -> "Generator[Type[Task]]":
        for ns, tasks in cls.public_tasks.items():
            for key, t in tasks.items():
                yield t


# Check that a task_namespace and task_name are valid
TASK_NAME_REGEX = re.compile(r"[\S.-]+")


class Task(Borg, metaclass=TaskRegistry):
    """
    Abstract base class for dataset operations.
    This can be a task to run a benchmark or perform an analysis step.
    Tasks in the pipeline have determined inputs and outputs that are derived
    from the session that creates the tasks.
    Tasks start as individual entities. When scheduled, if multiple tasks have
    the same ID, one task instance will be elected as the main task, the others
    will be attached to it as drones.
    The drones will share the task state with the main task instance to
    maintain consistency when producing stateful task outputs.
    This is a relatively strange dynamic Borg pattern.
    """
    #: Mark the task as a top-level target, which can be named in configuration files and from CLI commands.
    public = False
    #: Human-readable task namespace, used for task identification
    task_namespace = None
    #: Human-readable task identifier, used for task identification
    task_name = None
    #: If set, use the given Config class to unpack the task configuration
    task_config_class: Type[Config] = None

    def __init__(self, task_config: Config = None):
        assert self.task_name is not None, f"Attempted to use task with uninitialized name {self.__class__.__name__}"
        assert self.task_namespace is None or TASK_NAME_REGEX.match(
            self.task_namespace), f"Invalid task namespace '{self.task_namespace}'"
        assert TASK_NAME_REGEX.match(self.task_name), f"Invalid task name '{self.task_name}'"

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
        #: This currently caches the results from output_map().
        #: Ideally, however, this should be filled either by the scheduler or
        #: from cached task metadata.
        #: The scheduler fill should occur after all dependencies have completed,
        #: but before run(), so it is possible to access dependencies in the
        #: outputs generator. Analysis tasks should be able to resolve exec
        #: tasks outputs from metadata. This would remove the necessity for
        #: instantiating tasks to reference outputs and removes limitations for
        #: dynamic output descriptor generation.
        self.collected_outputs = {}

        # Note: the whole __dict__ may be replaced by Borg, so no property
        # initialization beyond this point
        super().__init__()

    def __str__(self):
        return str(self.task_id)

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.task_id}>"

    def __hash__(self):
        return hash(self.task_id)

    def __eq__(self, other: "Task"):
        return self.task_id == other.task_id

    @classmethod
    def is_exec_task(cls):
        return False

    @classmethod
    def is_session_task(cls) -> bool:
        """
        Helper to determine whether a Task should be treated like a :class:`SessionTask`.
        """
        return False

    @classmethod
    def is_benchmark_task(cls) -> bool:
        """
        Helper to determine whether a Task should be treated like a :class:`BenchmarkTask`.
        """
        return False

    @property
    def session(self):
        raise NotImplementedError("Subclasses should override")

    @property
    def task_id(self) -> Hashable:
        """
        Return the unique task identifier.
        """
        raise NotImplementedError("Subclass should override")

    @property
    def borg_state_id(self) -> Hashable:
        return self.task_id

    @property
    def completed(self) -> bool:
        """
        Check whether the task has completed. This only notifies whether
        the task is done, to check whether an error occurred, the :attr:`Task.failed`
        should be used.
        """
        return self._completed.is_set()

    @property
    def output_map(self) -> dict[str, "Target"]:
        """
        Return the output descriptors for the task.
        See note on :attr:`Task.collected_outputs`.
        The only invariant that should be enforced here is that the output map is only ever accessed
        after all dependencies tasks have completed.
        XXX this is not strictly true, it can be accessed after the dependencies generator
        has been consumed, so that dependencies are known and their side-effects are known.
        The actual data however will be ready only after the tasks are done.
        """
        if not self.collected_outputs:
            self.collected_outputs = dict(self.outputs())
        return self.collected_outputs

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

    def resources(self) -> Iterable["ResourceManager.ResourceRequest"]:
        """
        Produce a set of resources that are consumed by this task.
        Once the resources are available, they will be reserved and
        the resouce object will be availabe via the resource request get()
        method within Task.run().
        """
        yield from []

    def dependencies(self) -> Iterable["Task"]:
        """
        Produce the set of :class:`Task` objects that this task depends upon.

        :return: sequence of dependencies
        """
        for attr in self._deps_registry:
            deps = getattr(self, attr)
            if not isinstance(deps, Iterable):
                deps = [deps]
            yield from filter(lambda d: d is not None, deps)

    def outputs(self) -> Iterable[tuple[str, "Target"]]:
        """
        Produce the set of :class:`Target` objects that describe the outputs
        that are produced by this task.
        Each target must be associated to a unique name.
        The name is considered to be the public interface for other tasks to
        fetch targets and outputs of this task.
        The name/target pairs may be used to construct a dictionary to access
        the return values by key.

        Note that this method generates output descriptors, not output data.
        The descriptors may be dynamic but must be determined independently of
        :meth:`Task.run`. This allows to use cached outputs in the future, and
        avoid calling :meth:`Task.run` altogether. The :meth:`Task.run` is
        responsible for producing the output data and setting it to the
        output_map, if needed.
        """
        registered = {key: getattr(self, attr) for key, attr in self._output_registry.items()}
        yield from registered.items()

    def run(self):
        raise NotImplementedError("Task.run() must be overridden")


class dependency:
    """
    Decorator for :class:`Task` dependencies.
    Provides a shorthand to declare dependencies of a task.
    Note that this behaves as a property field.
    This should be used in place of the :meth:`Task.dependencies()` and
    :attr:`Task.resolved_dependencies` if there are no special requirements.

    Note that dependencies are Borgs, so there is a shared state for all tasks
    with the same ID, regardless of how many times the dependency property
    is accessed, therefore there is no need to cache here a single instance.
    """
    def __init__(self, fn: Callable | None = None, optional: bool = False):
        self._fn = fn
        self._optional = optional
        self._dependency_name = None

    def __set_name__(self, owner, name):
        """
        Register this descriptor as a dependency generator of this :class:`Task`
        """
        assert issubclass(owner, Task), ("@dependency decorator may be used "
                                         "only within Task classes")
        owner.register_dependency(name)
        self._dependency_name = name

    def __get__(self, instance, owner=None):
        assert instance is not None
        assert self._fn is not None
        try:
            result = self._fn(instance)
            # Normalize any generator or iterable to a list, keep single value unchanged
            if isinstance(result, Iterable) and not isinstance(result, list):
                return list(result)
        except TaskNotFound:
            result = None

        if not self._optional and not result:
            instance.logger.error("Missing required dependency %s", self._dependency_name)
            raise MissingDependency(f"Failed to resolve dependency")
        return result

    def __call__(self, fn):
        """
        Invoked when the decorator is called with arguments.
        Just return the descriptor with the correct arguments.
        """
        return dependency(fn=fn, optional=self._optional)


class output:
    """
    Decorator for :class:`Task` output artefacts.
    Provides a shorthand to declare outputs of a task.
    Note that this behaves as a property field.
    This should be used in place of the :meth:`Task.outputs()` and
    :attr:`Task.output_map` if there are no special requirements.

    This decorator acts as a property decorator.
    There is only one instance for each task output target, which is held here,
    further references to it may be obtained from the :class:`Task` class for
    convenience.
    """
    def __init__(self, fn: Callable | None = None, name: str | None = None):
        """
        :param name: Override the name of the output, the name of the decorated
        function is used otherwise.
        """
        self._fn = fn
        self._key = name

    def __set_name__(self, owner, name):
        """
        Register this descriptor as an output of this :class:`Task`
        """
        assert issubclass(owner, Task), ("@output decorator may be used only "
                                         "within Task classes")
        assert self._fn is not None
        if self._key is None:
            self._key = name
        owner.register_output(TargetRef(self._key, name))

    def __get__(self, instance, owner=None):
        assert instance is not None
        assert self._fn is not None
        result = self._fn(instance)

        # Normalize generators and iterables to a list
        if isinstance(result, Iterable) and not isinstance(result, list):
            result = list(result)
        return result

    def __call__(self, fn):
        """
        Invoked when the decorator is called with arguments.
        Just return the descriptor with the correct arguments.
        """
        return output(fn=fn, name=self._key)


class SessionTask(Task):
    """
    Base class for all tasks that are unique within a session.

    These tasks do not reference a specific benchmark ID or benchmark group ID.
    """
    def __init__(self, session: "Session", task_config: Config = None):
        self._session = session
        #: Task logger is a child of the session logger
        self.logger = new_logger(f"{self.task_name}", parent=session.logger)

        # Borg initialization occurs here
        super().__init__(task_config=task_config)

    @classmethod
    def is_session_task(cls):
        return True

    @property
    def session(self):
        return self._session

    @property
    def task_id(self):
        return f"{self.task_namespace}.{self.task_name}-{self.session.uuid}"


class BenchmarkTask(Task):
    """
    Base class for all tasks that are unique within a session.

    These tasks do not reference a specific benchmark ID or benchmark group ID.
    """
    def __init__(self, benchmark: "Benchmark", task_config: Config = None):
        #: Associated benchmark context
        self.benchmark = benchmark
        #: Task logger is a child of the benchmark logger
        self.logger = new_logger(f"{self.task_name}", parent=self.benchmark.logger)

        # Borg initialization occurs here
        super().__init__(task_config=task_config)

    @classmethod
    def is_benchmark_task(cls):
        return True

    @property
    def session(self):
        return self.benchmark.session

    @property
    def task_id(self):
        return f"{self.task_namespace}.{self.task_name}-{self.benchmark.uuid}"


class ExecutionTask(BenchmarkTask):
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

    Note that the task_id generation currently assumes that tasks with the same name are not issued
    more than once for each benchmark run UUID. If this is violated, we need to
    change the task ID generation.
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
        super().__init__(benchmark, task_config=task_config)
        #: Script builder associated to the current benchmark context.
        self.script = script

    @classmethod
    def is_exec_task(cls):
        return True

    @property
    def uuid(self):
        return self.benchmark.uuid

    @property
    def g_uuid(self):
        return self.benchmark.g_uuid


class SessionExecutionTask(SessionTask):
    """
    Execution task that only exists as a single instance within a session.
    This can be used for data generation that is independent from the benchmark
    configurations.
    """
    @classmethod
    def is_exec_task(cls):
        return True

    @property
    def uuid(self):
        raise TypeError("Task.uuid is invalid on SessionExecutionTask")

    @property
    def g_uuid(self):
        raise TypeError("Task.g_uuid is invalid on SessionExecutionTask")


class DataGenTask(ExecutionTask):
    """
    A special type of execution task that does not require a running instance.
    The distinction is made so that the root benchmark execution task knows whether
    to request an instance or not.
    DataGenTasks should not depend on execution tasks, this is to avoid scanning the whole dependency tree
    to determine whether we need to request an instance or not, however the reverse is allowed.
    """
    require_instance = False


class SessionDataGenTask(SessionExecutionTask):
    """
    Same as a DataGenTask, but is unique within a session.
    """
    require_instance = False


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
        pool: Hashable | None
        acquire_args: dict[str, Any]
        _resource: Any = dc.field(init=False, default=None)

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
    def request(cls, pool: Hashable = None, **kwargs) -> ResourceRequest:
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

    def _acquire(self, pool: Hashable):
        """
        Reserve an item slot in the underlying resource pool
        """
        if self.is_unlimited:
            return
        self._limit_guard.acquire()

    def _release(self, pool: Hashable):
        """
        Release an item slot to the underlying resource pool
        """
        if self.is_unlimited:
            return
        self._limit_guard.release()

    def _get_resource(self, req: ResourceRequest) -> Any:
        """
        Produce a resource item after a slot is acquired.
        """
        raise NotImplementedError("Must override")

    def _put_resource(self, r: Any, req: ResourceRequest):
        """
        Return a resource item to the manager after the client is done with it
        """
        raise NotImplementedError("Must override")

    @property
    def is_unlimited(self) -> bool:
        return self.limit is None or self.limit == 0

    @contextmanager
    def acquire(self, req: ResourceRequest) -> ContextManager[Any]:
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
        self.num_workers = session.config.concurrent_workers or mp.cpu_count()
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
            run_sched = [self._task_graph.nodes[t]["task"] for t in reversed(list(sched))]
            self.logger.debug("Resolved benchmark schedule:\n%s", "\n".join(map(str, run_sched)))
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
