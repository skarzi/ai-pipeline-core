# MODULE: logger
# CLASSES: LoggingConfig, ExecutionLogBuffer, LogContext, ExecutionLogHandler, LogRecord
# DEPENDS: logging.Handler
# PURPOSE: Logging infrastructure for AI Pipeline Core.
# VERSION: 0.15.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import LoggingConfig, get_pipeline_logger, setup_logging
from ai_pipeline_core.logger import ExecutionLogBuffer, ExecutionLogHandler, LogContext, LogRecord, get_log_context, reset_log_context, set_log_context
```

## Public API

```python
class LoggingConfig:
    """Manages logging configuration for the pipeline.

Provides centralized logging configuration with stdlib logging.

Configuration precedence:
    1. Explicit config_path parameter
    2. AI_PIPELINE_LOGGING_CONFIG environment variable
    3. PREFECT_LOGGING_SETTINGS_PATH environment variable
    4. Default configuration"""
    def __init__(self, config_path: Path | None = None):
        """Initialize logging configuration.

        Args:
            config_path: Optional path to YAML configuration file.
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config: dict[str, Any] | None = None

    def apply(self) -> None:
        """Apply the logging configuration."""
        config = self.load_config()
        logging.config.dictConfig(config)

        # Set Prefect logging environment variables if needed
        if "prefect" in config.get("loggers", {}):
            prefect_level = config["loggers"]["prefect"].get("level", "INFO")
            os.environ.setdefault("PREFECT_LOGGING_LEVEL", prefect_level)

    def load_config(self) -> dict[str, Any]:
        """Load logging configuration from file or defaults.

        Returns:
            Dictionary containing logging configuration.
        """
        if self._config is None:
            if self.config_path and self.config_path.exists():
                with open(self.config_path, encoding="utf-8") as f:
                    self._config = yaml.safe_load(f)
            else:
                self._config = self._get_default_config()
        # self._config cannot be None at this point
        assert self._config is not None
        return self._config


class ExecutionLogBuffer:
    """Thread-safe execution log buffer with per-span ordering and summaries."""
    def __init__(
        self,
        *,
        flush_size: int = DEFAULT_LOG_BUFFER_FLUSH_SIZE,
        max_pending_logs: int = MAX_PENDING_EXECUTION_LOGS,
        request_flush: Callable[[], None] | None = None,
    ) -> None:
        self._flush_size = flush_size
        self._max_pending_logs = max_pending_logs
        self._request_flush = request_flush
        self._lock = Lock()
        self._pending: deque[LogRecord] = deque()
        self._sequence_by_span: dict[UUID, int] = {}
        self._summary_by_span: dict[UUID, dict[str, int | str]] = {}
        self._dropped_count = 0

    def append(self, log: LogRecord) -> None:
        """Assign sequence_no, update summaries, and queue the log for flush."""
        should_request_flush = False
        with self._lock:
            sequence_no = self._sequence_by_span.get(log.span_id, 0)
            self._sequence_by_span[log.span_id] = sequence_no + 1
            stored_log = replace(log, sequence_no=sequence_no)
            self._pending.append(stored_log)
            if len(self._pending) > self._max_pending_logs:
                self._pending.popleft()
                self._dropped_count += 1
            self._update_summary(stored_log)
            should_request_flush = len(self._pending) >= self._flush_size

        if should_request_flush and self._request_flush is not None:
            self._request_flush()

    def consume_dropped_count(self) -> int:
        """Return and reset the count of logs dropped due to local buffer overflow."""
        with self._lock:
            dropped_count = self._dropped_count
            self._dropped_count = 0
        return dropped_count

    def consume_summary(self, span_id: UUID) -> dict[str, int | str]:
        """Return and forget a span summary once its terminal payload has been persisted."""
        with self._lock:
            summary = self._summary_by_span.pop(span_id, None)
            self._sequence_by_span.pop(span_id, None)
            if summary is None:
                return dict(EMPTY_LOG_SUMMARY)
            return dict(summary)

    def drain(self) -> list[LogRecord]:
        """Return all pending logs and clear the buffer."""
        with self._lock:
            drained = list(self._pending)
            self._pending.clear()
        return drained

    def get_summary(self, span_id: UUID) -> dict[str, int | str]:
        """Return lightweight log counters for a span."""
        with self._lock:
            summary = self._summary_by_span.get(span_id)
            if summary is None:
                return dict(EMPTY_LOG_SUMMARY)
            return dict(summary)


@dataclass(frozen=True, slots=True)
class LogContext:
    """Minimal execution state needed by the log handler."""
    log_buffer: ExecutionLogBuffer
    span_id: UUID
    deployment_id: UUID


class ExecutionLogHandler(logging.Handler):
    """Route execution-scoped logs from the root logger into the active log buffer."""
    def emit(self, record: Any) -> None:
        """Append an execution-scoped log record to the active buffer when configured."""
        if getattr(record, _SKIP_EXECUTION_LOG_ATTR, False):
            return

        ctx = _log_context.get()
        if ctx is None:
            return

        category, minimum_level = _classify_record(record)
        if record.levelno < minimum_level:
            return

        timestamp = datetime.fromtimestamp(record.created, tz=UTC)

        try:
            ctx.log_buffer.append(
                LogRecord(
                    deployment_id=ctx.deployment_id,
                    span_id=ctx.span_id,
                    timestamp=timestamp,
                    sequence_no=0,
                    level=record.levelname,
                    category=category,
                    event_type=str(getattr(record, "event_type", "")),
                    logger_name=record.name,
                    message=record.getMessage(),
                    fields_json=_coerce_fields_json(record),
                    exception_text=_format_exception_text(record),
                )
            )
        except AttributeError, OSError, OverflowError, TypeError, ValueError:
            self.handleError(record)


@dataclass(frozen=True, slots=True)
class LogRecord:
    """Row from the logs table."""
    deployment_id: UUID
    span_id: UUID
    timestamp: datetime
    sequence_no: int
    level: str
    category: str
    logger_name: str
    message: str
    event_type: str = ''
    fields_json: str = '{}'
    exception_text: str = ''


```

## Functions

```python
def setup_logging(config_path: Path | None = None, level: str | None = None) -> None:
    """Setup logging for the AI Pipeline Core library.

    Initializes logging configuration for the pipeline system.
    Call once at your application entry point. If not called explicitly,
    ``get_pipeline_logger()`` will auto-initialize with defaults on first use.

    Args:
        config_path: Optional path to YAML logging configuration file.
        level: Optional log level override (INFO, DEBUG, WARNING, etc.).

    """
    global _logging_config  # noqa: PLW0603

    with _setup_lock:
        _logging_config = LoggingConfig(config_path)
        _logging_config.apply()

        # Override level if provided
        if level:
            # Set for our loggers
            for logger_name in _DEFAULT_LOG_LEVELS:
                logger = logging.getLogger(logger_name)
                logger.setLevel(level)

            # Also set for Prefect
            os.environ["PREFECT_LOGGING_LEVEL"] = level

def get_pipeline_logger(name: str) -> logging.Logger:
    """Get a logger for pipeline components.

    Args:
        name: Logger name, typically __name__.

    Returns:
        Configured stdlib logger instance.

    """
    if _logging_config is None:
        setup_logging()

    return logging.getLogger(name)

def get_log_context() -> LogContext | None:
    """Return the active log context for the current coroutine."""
    return _log_context.get()

def set_log_context(ctx: LogContext | None) -> Token[LogContext | None]:
    """Bind the log context for the current scope."""
    return _log_context.set(ctx)

def reset_log_context(token: Token[LogContext | None]) -> None:
    """Restore the previous log context binding."""
    _log_context.reset(token)

```

## Examples

No test examples available.
