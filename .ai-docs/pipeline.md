# MODULE: pipeline
# CLASSES: LimitKind, PipelineLimit, FlowOptions, RunContext, PipelineFlow, TaskHandle, TaskBatch, DatabaseSpanSink, SpanMetrics, SpanSink, SpanContext, PipelineTask
# DEPENDS: BaseModel, Protocol, StrEnum
# PURPOSE: Pipeline framework primitives.
# VERSION: 0.15.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import FlowOptions, LimitKind, PipelineFlow, PipelineLimit, PipelineTask, RunContext, SpanContext, SpanMetrics, SpanSink, TaskBatch, TaskHandle, as_task_completed, collect_tasks, get_run_id, pipeline_concurrency, pipeline_test_context, run_tasks_until, safe_gather, safe_gather_indexed, traced_operation, track_span
from ai_pipeline_core.pipeline import DatabaseSpanSink
```

## Public API

```python
# Enum
class LimitKind(StrEnum):
    """Kind of concurrency/rate limit.

CONCURRENT: Slots held for duration of operation (lease-based).
    limit=500 means at most 500 simultaneous operations across all runs.

PER_MINUTE: Token bucket with limit/60 decay per second.
    Allows bursting up to `limit` immediately, then refills gradually.
    NOT a sliding window.

PER_HOUR: Token bucket with limit/3600 decay per second. Same burst semantics."""
    CONCURRENT = 'concurrent'
    PER_MINUTE = 'per_minute'
    PER_HOUR = 'per_hour'


@dataclass(frozen=True, slots=True)
class PipelineLimit:
    """Concurrency/rate limit configuration.

Must use names matching ``[a-zA-Z0-9_-]+`` in PipelineDeployment.concurrency_limits (validated at class definition time)."""
    limit: int
    kind: LimitKind = LimitKind.CONCURRENT
    timeout: int = 600

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError(f"limit must be >= 1, got {self.limit}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")


class FlowOptions(BaseModel):
    """Base configuration for pipeline flows.

Use FlowOptions for deployment/environment configuration that may
differ between environments (dev/staging/production).

Never inherit from FlowOptions for task-level options, writer configs,
or programmatically-constructed parameter objects — use BaseModel instead."""
    model_config = ConfigDict(frozen=True, extra='forbid')


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable context for a pipeline run, carried via ContextVar."""
    run_id: str
    execution_id: UUID | None = None


class PipelineFlow:
    """Base class for pipeline flows.

Flows are the unit of resume, progress tracking, and document hand-off in a deployment.
Define ``run`` as an **instance method** (not @classmethod) because flows can carry
per-instance configuration passed via ``build_flows()``::

    class TranslateFlow(PipelineFlow):
        target_language: str = "en"

        async def run(self, documents: tuple[SourceDoc, ...], options: FlowOptions) -> tuple[TranslatedDoc, ...]:
            return await TranslateTask.run(documents, language=self.target_language)

The deployment creates flow instances with constructor kwargs::

    def build_flows(self, options):
        return [TranslateFlow(target_language="fr"), TranslateFlow(target_language="de")]

Each instance runs independently with its own parameters, resume record, and progress.
Constructor kwargs are captured for replay serialization via ``get_params()``.

Signature must be exactly ``(self, documents: tuple[DocType, ...], options: FlowOptions)``
and is validated at class definition time by ``__init_subclass__``.
Use ``get_run_id()`` from ``ai_pipeline_core.pipeline`` to access the run ID inside a flow."""
    name: ClassVar[str]
    estimated_minutes: ClassVar[float] = 1.0
    input_document_types: ClassVar[list[type[Document]]] = []
    output_document_types: ClassVar[list[type[Document]]] = []
    task_graph: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, **kwargs: Any) -> None:
        """Constructor for per-flow instance configuration."""
        cls = type(self)
        known_params: set[str] = set()
        for klass in cls.__mro__:
            known_params.update(_declared_init_annotations(klass))
            known_params.update(
                name
                for name, value in vars(klass).items()
                if not name.startswith("_") and not callable(value) and not isinstance(value, (classmethod, staticmethod, property))
            )
        unknown = sorted(key for key in kwargs if key not in known_params)
        if unknown:
            allowed = ", ".join(sorted(known_params)) or "(none)"
            raise TypeError(f"PipelineFlow '{cls.__name__}' got unknown init parameter(s): {', '.join(unknown)}. Allowed parameters: {allowed}.")
        self._params: dict[str, Any] = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def expected_tasks(cls) -> list[str]:
        """Return expected task names extracted from run() AST."""
        return [name for name, _mode in cls.task_graph]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is PipelineFlow:
            return

        cls._validate_class_config()
        run_fn, hints, params = cls._validate_run_signature()
        input_types, output_types = cls._extract_document_types(hints, params)
        cls.input_document_types = input_types
        cls.output_document_types = output_types
        cls.task_graph = cls._parse_task_graph(run_fn)

    def get_params(self) -> dict[str, Any]:
        """Return constructor params for flow plan serialization."""
        return dict(getattr(self, "_params", {}))

    async def run(self, documents: tuple[Any, ...], options: Any) -> tuple[Any, ...]:
        """Execute the flow.

        Subclasses must provide concrete ``tuple[MyDocument, ...]`` and ``FlowOptions``
        annotations. The base stub stays broad so static type checkers accept
        narrower overrides; ``__init_subclass__`` enforces the real signature.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True, eq=False)
class TaskHandle:
    """Handle for an executing pipeline task."""
    task_class: type[Any] | None
    input_arguments: Mapping[str, Any]

    @property
    def done(self) -> bool:
        """Whether the underlying task has finished."""
        return self._task.done()

    def __await__(self):
        return self._task.__await__()

    def cancel(self) -> None:
        """Cancel the underlying task."""
        self._task.cancel()

    async def result(self) -> T:
        """Await the underlying task result."""
        return await self._task


@dataclass(frozen=True, slots=True)
class TaskBatch:
    """Collected task results and handles that did not complete successfully."""
    completed: list[tuple[Document[Any], ...]]
    incomplete: list[TaskHandle[tuple[Document[Any], ...]]]


class DatabaseSpanSink:
    """Persist tracked span lifecycle updates into the active database backend."""
    def __init__(self, database: DatabaseWriter) -> None:
        self._database = database
        self._started: dict[UUID, _StartedSpanState] = {}

    async def on_span_finished(
        self,
        *,
        span_id: UUID,
        ended_at: datetime,
        output_json: str,
        error_json: str,
        output_document_shas: frozenset[str],
        output_blob_shas: frozenset[str],
        output_preview: Any | None,
        error: BaseException | None,
        metrics: SpanMetrics,
        meta: dict[str, Any],
    ) -> None:
        _ = output_preview
        execution_ctx = get_execution_context()
        if execution_ctx is None:
            return

        started_state = self._started.pop(span_id, None)
        if started_state is None:
            started_state = _StartedSpanState(
                parent_span_id=execution_ctx.parent_span_id,
                kind=SpanKind.OPERATION,
                name="",
                target="",
                started_at=ended_at,
                started_version=_next_span_version(),
                sequence_no=0,
                receiver_json="",
                input_json="",
                input_document_shas=(),
                input_blob_shas=(),
                previous_conversation_id=None,
            )

        deployment_id = execution_ctx.deployment_id or UUID(int=0)
        root_deployment_id = execution_ctx.root_deployment_id or execution_ctx.deployment_id or UUID(int=0)
        cost_usd = metrics.cost_usd or 0.0
        description = ""
        raw_description = meta.pop("description", "")
        raw_cache_key = meta.pop("cache_key", "")
        raw_status = meta.get("_span_status")
        if isinstance(raw_description, str):
            description = raw_description
        cache_key = raw_cache_key if isinstance(raw_cache_key, str) else ""
        meta_json = json_dumps(_strip_control_meta(meta))
        metrics_json = json_dumps(_metrics_to_json(metrics))
        status = _resolve_finished_status(error=error, raw_status=raw_status)

        await self._insert_span_safe(
            SpanRecord(
                span_id=span_id,
                parent_span_id=started_state.parent_span_id,
                deployment_id=deployment_id,
                root_deployment_id=root_deployment_id,
                run_id=execution_ctx.run_id,
                deployment_name=execution_ctx.deployment_name,
                kind=started_state.kind,
                name=started_state.name,
                description=description,
                status=status,
                sequence_no=started_state.sequence_no,
                started_at=started_state.started_at,
                ended_at=ended_at,
                version=_next_span_version(started_state.started_version),
                cost_usd=cost_usd,
                cache_key=cache_key,
                error_type=type(error).__name__ if error is not None else "",
                error_message=str(error) if error is not None else "",
                input_document_shas=started_state.input_document_shas,
                output_document_shas=tuple(sorted(output_document_shas)),
                target=started_state.target,
                receiver_json=started_state.receiver_json,
                input_json=started_state.input_json,
                output_json=output_json,
                error_json=error_json,
                meta_json=meta_json,
                metrics_json=metrics_json,
                input_blob_shas=started_state.input_blob_shas,
                output_blob_shas=tuple(sorted(output_blob_shas)),
                previous_conversation_id=started_state.previous_conversation_id,
            )
        )

    async def on_span_started(
        self,
        *,
        span_id: UUID,
        parent_span_id: UUID | None,
        kind: SpanKind,
        name: str,
        target: str,
        started_at: datetime,
        receiver_json: str,
        input_json: str,
        input_document_shas: frozenset[str],
        input_blob_shas: frozenset[str],
        input_preview: Any | None,
    ) -> None:
        _ = input_preview
        execution_ctx = get_execution_context()
        if execution_ctx is None:
            return

        deployment_id = execution_ctx.deployment_id or UUID(int=0)
        root_deployment_id = execution_ctx.root_deployment_id or execution_ctx.deployment_id or UUID(int=0)
        sequence_no = execution_ctx.next_child_sequence(parent_span_id) if parent_span_id is not None else 0
        started_version = _next_span_version()
        previous_conversation_id = _extract_previous_conversation_id(kind, receiver_json)

        self._started[span_id] = _StartedSpanState(
            parent_span_id=parent_span_id,
            kind=kind,
            name=name,
            target=target,
            started_at=started_at,
            started_version=started_version,
            sequence_no=sequence_no,
            receiver_json=receiver_json,
            input_json=input_json,
            input_document_shas=tuple(sorted(input_document_shas)),
            input_blob_shas=tuple(sorted(input_blob_shas)),
            previous_conversation_id=previous_conversation_id,
        )

        await self._insert_span_safe(
            SpanRecord(
                span_id=span_id,
                parent_span_id=parent_span_id,
                deployment_id=deployment_id,
                root_deployment_id=root_deployment_id,
                run_id=execution_ctx.run_id,
                deployment_name=execution_ctx.deployment_name,
                kind=kind,
                name=name,
                description="",
                status=SpanStatus.RUNNING,
                sequence_no=sequence_no,
                started_at=started_at,
                version=started_version,
                target=target,
                receiver_json=receiver_json,
                input_json=input_json,
                input_document_shas=tuple(sorted(input_document_shas)),
                input_blob_shas=tuple(sorted(input_blob_shas)),
                previous_conversation_id=previous_conversation_id,
            )
        )


@dataclass(frozen=True, slots=True)
class SpanMetrics:
    """Normalized metrics payload for span sinks."""
    time_taken_ms: int
    log_summary: dict[str, Any]
    tokens_input: int | None = None
    tokens_output: int | None = None
    tokens_cache_read: int | None = None
    tokens_reasoning: int | None = None
    cost_usd: float | None = None
    first_token_ms: int | None = None


# Protocol — implement in concrete class
class SpanSink(Protocol):
    """Lifecycle callbacks for execution span transport implementations."""
    async def on_span_finished(
        self,
        *,
        span_id: UUID,
        ended_at: datetime,
        output_json: str,
        error_json: str,
        output_document_shas: frozenset[str],
        output_blob_shas: frozenset[str],
        output_preview: Any | None,
        error: BaseException | None,
        metrics: SpanMetrics,
        meta: dict[str, Any],
    ) -> None: ...

    async def on_span_started(
        self,
        *,
        span_id: UUID,
        parent_span_id: UUID | None,
        kind: SpanKind,
        name: str,
        target: str,
        started_at: datetime,
        receiver_json: str,
        input_json: str,
        input_document_shas: frozenset[str],
        input_blob_shas: frozenset[str],
        input_preview: Any | None,
    ) -> None: ...


class SpanContext:
    """Mutable state populated by a tracked span body."""
    def __init__(self, *, span_id: UUID, parent_span_id: UUID | None, input_preview: Any | None = None) -> None:
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self._input_preview = input_preview
        self._output_preview: Any | None = None
        self._meta: dict[str, Any] = {}
        self._metrics_updates: dict[str, Any] = {}
        self._output_value: Any = None
        self._has_output_value = False
        self._status: str | None = None

    @property
    def input_preview(self) -> Any | None:
        return self._input_preview

    @property
    def output_preview(self) -> Any | None:
        return self._output_preview

    def set_input_preview(self, value: Any) -> None:
        self._input_preview = value

    def set_meta(self, **values: Any) -> None:
        self._meta.update(values)

    def set_metrics(self, **values: Any) -> None:
        valid_fields = {field_name for field_name in SpanMetrics.__dataclass_fields__ if field_name != "log_summary"}
        unknown_fields = sorted(set(values) - valid_fields)
        if unknown_fields:
            names = ", ".join(unknown_fields)
            raise ValueError(f"Unknown span metric field(s): {names}. Use only fields declared on SpanMetrics.")
        self._metrics_updates.update(values)

    def set_output_preview(self, value: Any) -> None:
        self._output_preview = value

    def set_status(self, status: str) -> None:
        self._status = status


class PipelineTask:
    """Base class for pipeline tasks.

Tasks are stateless units of work. Define ``run`` as a **@classmethod** because tasks
carry no per-invocation instance state — all inputs arrive as arguments, all outputs
are returned documents. The framework wraps ``run`` with retries, persistence,
and event emission automatically.

Set ``_abstract_task = True`` on an intermediate base class to skip ``run()``
validation on that class. Concrete subclasses do not inherit that skip; they must
define ``run()`` or inherit a validated implementation from a non-abstract parent.

Minimal example::

    class SummarizeTask(PipelineTask):
        @classmethod
        async def run(cls, documents: tuple[ArticleDocument, ...]) -> tuple[SummaryDocument, ...]:
            conv = Conversation(model="gemini-3-flash").with_context(documents[0])
            conv = await conv.send("Summarize this article.")
            return (SummaryDocument.derive(from_documents=(documents[0],), name="summary.md", content=conv.content),)

Calling ``await SummarizeTask.run((doc,))`` dispatches the full lifecycle. Calling without
``await`` returns a ``TaskHandle`` for parallel execution via ``collect_tasks``."""
    name: ClassVar[str]
    estimated_minutes: ClassVar[float] = 1.0
    retries: ClassVar[int] = 0
    retry_delay_seconds: ClassVar[int] = 20
    timeout_seconds: ClassVar[int | None] = None
    cacheable: ClassVar[bool] = False
    cache_version: ClassVar[int] = 1
    cache_ttl_seconds: ClassVar[int | None] = None
    expected_cost: ClassVar[float | None] = None
    input_document_types: ClassVar[list[type[Document]]] = []
    output_document_types: ClassVar[list[type[Document]]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is PipelineTask:
            return
        if cls.__dict__.get("_abstract_task", False) is True:
            return

        cls._validate_class_config()

        own_run = cls.__dict__.get("run")
        if own_run is None:
            inherited_spec = getattr(cls, "_run_spec", None)
            if inherited_spec is None:
                raise TypeError(f"PipelineTask '{cls.__name__}' must define @classmethod async def run(cls, ...) or inherit a validated run() implementation.")
            cls.input_document_types = list(inherited_spec.input_document_types)
            cls.output_document_types = list(inherited_spec.output_document_types)
            return

        spec = cls._validate_run_signature(own_run)
        cls._run_spec = spec
        cls.input_document_types = list(spec.input_document_types)
        cls.output_document_types = list(spec.output_document_types)
        cls.run = classmethod(cls._build_run_wrapper(spec))


```

## Functions

```python
async def safe_gather[T](
    *coroutines: Coroutine[Any, Any, T],
    label: str = "",
    raise_if_all_fail: bool = True,
) -> list[T]:
    """Execute coroutines in parallel, returning successes and logging failures.

    Uses return_exceptions=True internally. Filters failures with BaseException
    (catches CancelledError). Logs each failure with context.

    Returns:
        List of successful results (failures filtered out). Relative order of
        successes is preserved, but indices shift. Use safe_gather_indexed
        for positional correspondence.
    """
    if not coroutines:
        return []

    results, failures = await _execute_gather(*coroutines, label=label)
    failure_indices = {i for i, _ in failures}
    successes: list[T] = [r for i, r in enumerate(results) if i not in failure_indices]

    if not successes and raise_if_all_fail and failures:
        first_error = failures[0][1]
        raise RuntimeError(f"All {len(failures)} tasks failed{f' in {label!r}' if label else ''}. First error: {first_error}") from first_error

    return successes

async def safe_gather_indexed[T](
    *coroutines: Coroutine[Any, Any, T],
    label: str = "",
    raise_if_all_fail: bool = True,
) -> list[T | None]:
    """Execute coroutines in parallel, preserving positional correspondence.

    Like safe_gather, but returns a list with the same length as the input.
    Failed positions contain None. Useful when results must correspond to
    specific inputs by index.

    Returns:
        List matching input length. Successful results at their original index,
        None at positions where the coroutine failed.
    """
    if not coroutines:
        return []

    results, failures = await _execute_gather(*coroutines, label=label)
    failure_indices = {i for i, _ in failures}
    output: list[T | None] = [None if i in failure_indices else r for i, r in enumerate(results)]

    if len(failures) == len(results) and raise_if_all_fail:
        first_error = failures[0][1]
        raise RuntimeError(f"All {len(failures)} tasks failed{f' in {label!r}' if label else ''}. First error: {first_error}") from first_error

    return output

@asynccontextmanager
async def pipeline_concurrency(
    name: str,
    *,
    timeout: int | None = None,
) -> AsyncGenerator[None]:
    """Acquire a concurrency/rate-limit slot for an operation.

    For CONCURRENT limits: slot held during block, released on exit.
    For PER_MINUTE/PER_HOUR: slot acquired (decays automatically), exit is no-op.

    Proceeds unthrottled when Prefect is unavailable.
    Timeout always raises AcquireConcurrencySlotTimeoutError.
    Logs a warning if slot acquisition takes longer than 120 seconds.
    """
    state = _limits_state.get()
    cfg = state.limits.get(name)
    if cfg is None:
        available = ", ".join(sorted(state.limits)) or "(none)"
        raise KeyError(f"pipeline_concurrency({name!r}) not registered. Available limits: {available}. Declare it on PipelineDeployment.concurrency_limits.")

    # Prefect unavailable — proceed unthrottled
    if not state.status.prefect_available:
        yield
        return

    effective_timeout = timeout if timeout is not None else cfg.timeout
    t0 = time.monotonic()

    def _warn_if_slow() -> None:
        wait_seconds = time.monotonic() - t0
        if wait_seconds > _CACHE_TTL_WARNING_THRESHOLD:
            logger.warning(
                "Slot wait for %r took %.1fs — exceeds %ds threshold. "
                "LLM cache TTL (default 300s) may expire before execution. "
                "Consider increasing concurrency limit or reducing parallelism.",
                name,
                wait_seconds,
                _CACHE_TTL_WARNING_THRESHOLD,
            )

    # Prefect available — use global concurrency/rate limiting
    try:
        match cfg.kind:
            case LimitKind.CONCURRENT:
                async with concurrency(name, occupy=1, timeout_seconds=effective_timeout, strict=False):
                    _warn_if_slow()
                    yield
            case LimitKind.PER_MINUTE | LimitKind.PER_HOUR:
                await rate_limit(name, occupy=1, timeout_seconds=effective_timeout, strict=False)
                _warn_if_slow()
                yield
    except AcquireConcurrencySlotTimeoutError:
        raise
    except ConcurrencySlotAcquisitionError as e:
        logger.warning("Prefect concurrency unavailable for %r, proceeding unthrottled: %s", name, e)
        state.status.prefect_available = False
        yield

def get_run_id() -> str:
    """Return the current run ID from the active execution context."""
    ctx = get_execution_context()
    if ctx is None:
        msg = (
            "get_run_id() called outside execution context. "
            "This function is available inside PipelineFlow.run() and PipelineTask.run() "
            "during deployment execution. "
            "In tests, wrap your code with pipeline_test_context(run_id='...')."
        )
        raise RuntimeError(msg)
    return ctx.run_id

@contextmanager
def pipeline_test_context(
    run_id: str = "test-run",
    publisher: Any | None = None,
    cache_ttl: timedelta | None = None,
) -> Generator[ExecutionContext]:
    """Set up an execution + task context for tests without full deployment wiring.

    Yields:
        The active execution context for the test scope.
    """
    ctx = ExecutionContext(
        run_id=run_id,
        execution_id=None,
        publisher=publisher or _create_noop_publisher(),
        limits=MappingProxyType({}),
        limits_status=_SharedStatus(),
        cache_ttl=cache_ttl,
    )
    with set_execution_context(ctx), set_task_context(TaskContext(scope_kind="test", task_class_name="pipeline_test_context")):
        yield ctx

async def collect_tasks(
    *handles: TaskAwaitableGroup,
    deadline_seconds: float | None = None,
) -> TaskBatch:
    """Await task handles with an optional deadline and split completed/incomplete."""
    ordered_handles = _normalize_handles(handles)
    if not ordered_handles:
        return TaskBatch(completed=[], incomplete=[])

    completed: list[tuple[Document[Any], ...]] = []
    incomplete: list[TaskHandle[tuple[Document[Any], ...]]] = []
    by_task: dict[asyncio.Task[tuple[Document[Any], ...]], TaskHandle[tuple[Document[Any], ...]]] = {handle._task: handle for handle in ordered_handles}
    pending: set[asyncio.Task[tuple[Document[Any], ...]]] = set(by_task.keys())
    deadline_at = (time.monotonic() + deadline_seconds) if deadline_seconds is not None else None

    while pending:
        timeout: float | None = None
        if deadline_at is not None:
            timeout = max(0.0, deadline_at - time.monotonic())
            if timeout <= 0.0:
                break
        done, pending = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if not done:
            break
        for finished in done:
            handle = by_task[finished]
            outcome = (await asyncio.gather(handle.result(), return_exceptions=True))[0]
            if isinstance(outcome, BaseException):
                incomplete.append(handle)
                continue
            completed.append(outcome)

    incomplete.extend(by_task[still_pending] for still_pending in pending)
    return TaskBatch(completed=completed, incomplete=incomplete)

async def as_task_completed(*handles: TaskAwaitableGroup) -> AsyncIterator[TaskHandle[tuple[Document[Any], ...]]]:
    """Yield task handles in completion order."""
    ordered_handles = _normalize_handles(handles)
    if not ordered_handles:
        return

    by_task: dict[asyncio.Task[tuple[Document[Any], ...]], TaskHandle[tuple[Document[Any], ...]]] = {handle._task: handle for handle in ordered_handles}
    pending: set[asyncio.Task[tuple[Document[Any], ...]]] = set(by_task.keys())
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for finished in done:
            yield by_task[finished]

async def run_tasks_until(
    task_cls: type[Any],
    argument_groups: Sequence[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    deadline_seconds: float | None = None,
) -> TaskBatch:
    """Launch ``task_cls.run(*args, **kwargs)`` for each argument group and collect the handles."""
    handles = [task_cls.run(*args, **kwargs) for args, kwargs in argument_groups]
    return await collect_tasks(handles, deadline_seconds=deadline_seconds)

@asynccontextmanager
async def traced_operation(name: str, description: str = "") -> AsyncGenerator[None]:
    """Create a tracked operation span. Outside deployment context this is a no-op."""
    execution_ctx = get_execution_context()
    if execution_ctx is None or execution_ctx.database is None or execution_ctx.deployment_id is None:
        yield
        return

    parent_task = execution_ctx.task_frame
    with contextlib.ExitStack() as stack:
        async with track_span(
            SpanKind.OPERATION,
            name,
            "",
            sinks=get_sinks(),
            db=execution_ctx.database,
            input_preview=None,
        ) as span_ctx:
            task_frame = TaskFrame(
                task_class_name=_TRACED_OPERATION_CLASS_NAME,
                task_id=str(span_ctx.span_id),
                depth=(parent_task.depth + 1) if parent_task is not None else 0,
                parent=parent_task,
            )
            traced_ctx = get_execution_context()
            if traced_ctx is not None:
                stack.enter_context(set_execution_context(traced_ctx.with_task(task_frame)))
            span_ctx.set_meta(description=description)
            try:
                yield
            except Exception, asyncio.CancelledError:
                span_ctx.set_meta(description=description)
                raise

@asynccontextmanager
async def track_span(
    kind: SpanKind,
    name: str,
    target: str,
    *,
    sinks: Sequence[SpanSink],
    span_id: UUID | None = None,
    parent_span_id: UUID | None = None,
    encode_receiver: dict[str, Any] | None = None,
    encode_input: Any = _UNSET,
    db: DatabaseWriter | None = None,
    input_preview: Any | None = None,
) -> AsyncIterator[SpanContext]:
    """Track one span lifecycle and dispatch it to every configured sink.

    Yields:
        SpanContext: Mutable span state for previews, metadata, metrics, and output.
    """
    execution_ctx = get_execution_context()
    span_id = span_id or uuid7()
    effective_parent_span_id = parent_span_id
    if effective_parent_span_id is None and execution_ctx is not None:
        candidate_parent_span_id = execution_ctx.current_span_id or execution_ctx.span_id
        if candidate_parent_span_id != span_id:
            effective_parent_span_id = candidate_parent_span_id
    started_at = datetime.now(UTC)
    context = SpanContext(
        span_id=span_id,
        parent_span_id=effective_parent_span_id,
        input_preview=input_preview,
    )
    codec = UniversalCodec()
    start_payload = await _prepare_start_payload(
        codec=codec,
        encode_receiver=encode_receiver,
        encode_input=encode_input,
        db=db or (execution_ctx.database if execution_ctx is not None else None),
        execution_ctx=execution_ctx,
        kind=kind,
        name=name,
        sinks=tuple(sinks),
    )

    span_execution_ctx = execution_ctx.with_span(span_id, parent_span_id=effective_parent_span_id) if execution_ctx is not None else None

    error: BaseException | None = None
    with ExitStack() as stack:
        if span_execution_ctx is not None:
            stack.enter_context(set_execution_context(span_execution_ctx))

        await _notify_sinks(
            start_payload.active_sinks,
            "on_span_started",
            span_id=span_id,
            parent_span_id=effective_parent_span_id,
            kind=kind,
            name=name,
            target=target,
            started_at=started_at,
            receiver_json=start_payload.receiver_json,
            input_json=start_payload.input_json,
            input_document_shas=start_payload.document_shas,
            input_blob_shas=start_payload.blob_shas,
            input_preview=context.input_preview,
        )

        try:
            yield context
        except BaseException as exc:
            error = exc
            raise
        finally:
            ended_at = datetime.now(UTC)
            log_summary = _consume_log_summary(span_execution_ctx or execution_ctx, span_id)
            metrics = context._build_metrics(ended_at=ended_at, started_at=started_at, log_summary=log_summary)
            finish_payload = await _prepare_finish_payload(
                codec=codec,
                output_value=context._output_value,
                has_output_value=context._has_output_value,
                error=error,
                db=db or (execution_ctx.database if execution_ctx is not None else None),
                execution_ctx=execution_ctx,
                kind=kind,
                name=name,
                sinks=start_payload.active_sinks,
            )

            meta = dict(context._meta)
            if context._status is not None:
                meta["_span_status"] = context._status

            await _notify_sinks(
                finish_payload.active_sinks,
                "on_span_finished",
                span_id=span_id,
                ended_at=ended_at,
                output_json=finish_payload.output_json,
                error_json=finish_payload.error_json,
                output_document_shas=finish_payload.output_document_shas,
                output_blob_shas=finish_payload.output_blob_shas,
                output_preview=context.output_preview,
                error=error,
                metrics=metrics,
                meta=meta,
            )

```

## Examples

**Name with dashes and underscores** (`tests/pipeline/test_limits.py:154`)

```python
def test_name_with_dashes_and_underscores(self):
    raw = {"my-limit_v2": PipelineLimit(10)}
    result = _validate_concurrency_limits("TestDeploy", raw)
    assert "my-limit_v2" in result
```

**Collect tasks accepts list** (`tests/pipeline/test_parallel_primitives.py:124`)

```python
@pytest.mark.asyncio
async def test_collect_tasks_accepts_list() -> None:
    with pipeline_test_context() as ctx:
        with set_execution_context(ctx.with_flow(_make_flow_frame())):
            handles = [_FastTask.run((_make_doc("x"),)) for _ in range(3)]
            batch = await collect_tasks(handles)

    assert len(batch.completed) == 3
    assert batch.incomplete == []
```

**Collect tasks empty returns empty batch** (`tests/pipeline/test_task_constraints.py:67`)

```python
@pytest.mark.asyncio
async def test_collect_tasks_empty_returns_empty_batch() -> None:
    """collect_tasks with no handles returns empty batch."""
    batch = await collect_tasks()
    assert batch.completed == []
    assert batch.incomplete == []
```

**Get run id returns run id from context** (`tests/pipeline/test_execution_context.py:114`)

```python
def test_get_run_id_returns_run_id_from_context() -> None:
    with pipeline_test_context(run_id="ctx-test"):
        assert get_run_id() == "ctx-test"
```

**Pipeline test context sets and restores** (`tests/pipeline/test_execution_context.py:104`)

```python
def test_pipeline_test_context_sets_and_restores() -> None:
    before = get_execution_context()

    with pipeline_test_context(run_id="ctx-test") as ctx:
        assert get_execution_context() is ctx
        assert ctx.run_id == "ctx-test"

    assert get_execution_context() is before
```

**As task completed yields handles** (`tests/pipeline/test_parallel_primitives.py:135`)

```python
@pytest.mark.asyncio
async def test_as_task_completed_yields_handles() -> None:
    with pipeline_test_context() as ctx:
        with set_execution_context(ctx.with_flow(_make_flow_frame())):
            h1 = _FastTask.run((_make_doc("1"),))
            h2 = _FastTask.run((_make_doc("2"),))
            yielded = [handle async for handle in as_task_completed(h1, h2)]

    assert len(yielded) == 2
    assert all(isinstance(handle, TaskHandle) for handle in yielded)
```

**As task completed yields results** (`tests/pipeline/test_flow_resume.py:26`)

```python
@pytest.mark.asyncio
async def test_as_task_completed_yields_results() -> None:
    first = InputDoc.create_root(name="1.txt", content="a", reason="test input")
    second = InputDoc.create_root(name="2.txt", content="b", reason="test input")

    names: list[str] = []
    with pipeline_test_context():
        async for handle in as_task_completed(EchoTask.run((first,)), EchoTask.run((second,))):
            docs = await handle.result()
            names.extend(doc.name for doc in docs)

    assert set(names) == {"out_1.txt", "out_2.txt"}
```

**Collect tasks all complete** (`tests/pipeline/test_parallel_primitives.py:82`)

```python
@pytest.mark.asyncio
async def test_collect_tasks_all_complete() -> None:
    with pipeline_test_context() as ctx:
        with set_execution_context(ctx.with_flow(_make_flow_frame())):
            h1 = _FastTask.run((_make_doc("a"),))
            h2 = _FastTask.run((_make_doc("b"),))
            batch = await collect_tasks(h1, h2)

    assert isinstance(batch, TaskBatch)
    assert len(batch.completed) == 2
    assert batch.incomplete == []
```


## Error Examples

**Invalid name pattern** (`tests/pipeline/test_limits.py:135`)

```python
def test_invalid_name_pattern(self):
    with pytest.raises(TypeError, match="invalid name"):
        _validate_concurrency_limits("TestDeploy", {"bad name!": PipelineLimit(10)})
```

**Get run id outside context raises** (`tests/pipeline/test_execution_context.py:119`)

```python
def test_get_run_id_outside_context_raises() -> None:
    with pytest.raises(RuntimeError, match="pipeline_test_context"):
        get_run_id()
```

**Traced operation marks failed span and reraises** (`tests/pipeline/test_traced_operation.py:135`)

```python
@pytest.mark.asyncio
async def test_traced_operation_marks_failed_span_and_reraises() -> None:
    database = _RecordingSpanDatabase()
    with set_execution_context(_make_context_with_db(database)):
        with pytest.raises(ValueError, match="boom"):
            async with traced_operation("explode"):
                raise ValueError("boom")

    operation_span = next(span for span in database._spans.values() if span.kind == SpanKind.OPERATION)
    assert operation_span.status == SpanStatus.FAILED
    assert operation_span.error_type == "ValueError"
    assert operation_span.error_message == "boom"
```

**Track span captures errors and marks failed status** (`tests/pipeline/test_span_sink.py:218`)

```python
@pytest.mark.asyncio
async def test_track_span_captures_errors_and_marks_failed_status() -> None:
    database = _RecordingMemoryDatabase()
    ctx = _make_context(database)

    with set_execution_context(ctx):
        with pytest.raises(ValueError, match="boom"):
            async with track_span(
                SpanKind.OPERATION,
                "explode",
                "function:test:explode",
                sinks=get_sinks(),
                db=database,
                input_preview=None,
            ):
                raise ValueError("boom")

    span = next(iter(database._spans.values()))
    error_payload = json.loads(span.error_json)
    assert span.status == SpanStatus.FAILED
    assert span.error_type == "ValueError"
    assert error_payload["type_name"] == "ValueError"
    assert "boom" in error_payload["message"]
```

**Base flow options rejects extra** (`tests/pipeline/test_options.py:29`)

```python
def test_base_flow_options_rejects_extra(self):
    """Test that base FlowOptions rejects extra fields (extra='forbid')."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FlowOptions(unknown_field="value")
```

**Flow options is frozen** (`tests/pipeline/test_options.py:34`)

```python
def test_flow_options_is_frozen(self):
    """Test that FlowOptions instances are immutable."""

    class SimpleOptions(FlowOptions):
        core_model: str = "default"

    options = SimpleOptions()
    with pytest.raises(ValidationError):
        options.core_model = "new-model"
```

**Inherited flow options maintains frozen** (`tests/pipeline/test_options.py:111`)

```python
def test_inherited_flow_options_maintains_frozen(self):
    """Test that inherited classes maintain frozen configuration."""

    class CustomFlowOptions(FlowOptions):
        custom_field: str = "default"

    options = CustomFlowOptions()
    with pytest.raises(ValidationError):
        options.custom_field = "new_value"
```

**Invalid kind type** (`tests/pipeline/test_limits.py:143`)

```python
def test_invalid_kind_type(self):
    """Test that kind must be LimitKind enum instance."""
    # Create a PipelineLimit-like object with wrong kind type
    limit = PipelineLimit.__new__(PipelineLimit)
    object.__setattr__(limit, "limit", 10)
    object.__setattr__(limit, "kind", "concurrent")  # str, not LimitKind
    object.__setattr__(limit, "timeout", 600)
    with pytest.raises(TypeError, match="kind must be LimitKind"):
        _validate_concurrency_limits("TestDeploy", {"test": limit})
```
