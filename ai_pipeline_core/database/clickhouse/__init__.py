"""ClickHouse database backend."""

from ai_pipeline_core.database.clickhouse._backend import ClickHouseDatabase
from ai_pipeline_core.database.clickhouse._connection import SchemaVersionError, get_async_clickhouse_client
from ai_pipeline_core.database.clickhouse._ddl import (
    BLOBS_DDL,
    BLOBS_TABLE,
    DDL_STATEMENTS,
    DOCUMENTS_DDL,
    DOCUMENTS_TABLE,
    LOGS_DDL,
    LOGS_TABLE,
    SCHEMA_META_DDL,
    SCHEMA_META_TABLE,
    SCHEMA_VERSION,
    SPANS_DDL,
    SPANS_TABLE,
)

__all__ = [
    "BLOBS_DDL",
    "BLOBS_TABLE",
    "DDL_STATEMENTS",
    "DOCUMENTS_DDL",
    "DOCUMENTS_TABLE",
    "LOGS_DDL",
    "LOGS_TABLE",
    "SCHEMA_META_DDL",
    "SCHEMA_META_TABLE",
    "SCHEMA_VERSION",
    "SPANS_DDL",
    "SPANS_TABLE",
    "ClickHouseDatabase",
    "SchemaVersionError",
    "get_async_clickhouse_client",
]
