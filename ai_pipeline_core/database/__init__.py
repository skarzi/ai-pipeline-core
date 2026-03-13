"""Unified database module for the span-based schema."""

from ai_pipeline_core.database._factory import Database, create_database, create_database_from_settings
from ai_pipeline_core.database._memory import MemoryDatabase
from ai_pipeline_core.database._protocol import DatabaseReader, DatabaseWriter
from ai_pipeline_core.database._types import (
    BlobRecord,
    CostTotals,
    DocumentRecord,
    HydratedDocument,
    LogRecord,
    SpanKind,
    SpanRecord,
    SpanStatus,
)
from ai_pipeline_core.database.snapshot._download import download_deployment

__all__ = [
    "BlobRecord",
    "CostTotals",
    "Database",
    "DatabaseReader",
    "DatabaseWriter",
    "DocumentRecord",
    "HydratedDocument",
    "LogRecord",
    "MemoryDatabase",
    "SpanKind",
    "SpanRecord",
    "SpanStatus",
    "create_database",
    "create_database_from_settings",
    "download_deployment",
]
