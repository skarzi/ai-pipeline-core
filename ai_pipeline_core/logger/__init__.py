"""Logging infrastructure for AI Pipeline Core.

Provides a configured stdlib logging facade and execution-log capture primitives.
Prefer get_pipeline_logger instead of logging.getLogger to ensure consistent setup.
"""

from ._buffer import ExecutionLogBuffer
from ._handler import ExecutionLogHandler, LogContext, get_log_context, reset_log_context, set_log_context
from ._types import LogRecord
from .logging_config import LoggingConfig, get_pipeline_logger, setup_logging

__all__ = [
    "ExecutionLogBuffer",
    "ExecutionLogHandler",
    "LogContext",
    "LogRecord",
    "LoggingConfig",
    "get_log_context",
    "get_pipeline_logger",
    "reset_log_context",
    "set_log_context",
    "setup_logging",
]
