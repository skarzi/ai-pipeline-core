# MODULE: database
# CLASSES: MemoryDatabase, DatabaseWriter, DatabaseReader, SpanKind, SpanStatus, SpanRecord, DocumentRecord, BlobRecord, CostTotals, HydratedDocument
# DEPENDS: Protocol, StrEnum
# PURPOSE: Unified database module for the span-based schema.
# VERSION: 0.15.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import DatabaseReader
from ai_pipeline_core.database import BlobRecord, CostTotals, Database, DatabaseWriter, DocumentRecord, HydratedDocument, MemoryDatabase, SpanKind, SpanRecord, SpanStatus, create_database, create_database_from_settings, download_deployment
```

## Types & Constants

```python
Database = MemoryDatabase | FilesystemDatabase | ClickHouseDatabase

```

## Public API

```python
class MemoryDatabase:
    """Dict-based backend for tests covering the span schema."""
    supports_remote = False

    def __init__(self) -> None:
        self._spans: dict[UUID, SpanRecord] = {}
        self._documents: dict[str, DocumentRecord] = {}
        self._blobs: dict[str, BlobRecord] = {}
        self._logs: list[LogRecord] = []

    async def flush(self) -> None:
        return None

    async def get_all_document_shas_for_tree(self, root_deployment_id: UUID) -> set[str]:
        shas: set[str] = set()
        for span in self._spans.values():
            if span.root_deployment_id != root_deployment_id:
                continue
            shas.update(span.input_document_shas)
            shas.update(span.output_document_shas)
        return shas

    async def get_blob(self, content_sha256: str) -> BlobRecord | None:
        return self._blobs.get(content_sha256)

    async def get_blobs_batch(self, content_sha256s: list[str]) -> dict[str, BlobRecord]:
        return {sha256: self._blobs[sha256] for sha256 in content_sha256s if sha256 in self._blobs}

    async def get_cached_completion(
        self,
        cache_key: str,
        *,
        max_age: timedelta | None = None,
    ) -> SpanRecord | None:
        now = datetime.now(UTC)
        matches: list[SpanRecord] = []
        for span in self._spans.values():
            if span.cache_key != cache_key or span.status != SpanStatus.COMPLETED:
                continue
            if max_age is not None and (span.ended_at is None or now - span.ended_at > max_age):
                continue
            matches.append(span)
        if not matches:
            return None
        return max(matches, key=lambda span: (span.ended_at or span.started_at, span.version, str(span.span_id)))

    async def get_child_spans(self, parent_span_id: UUID) -> list[SpanRecord]:
        matches = [span for span in self._spans.values() if span.parent_span_id == parent_span_id]
        return sorted(matches, key=child_span_sort_key)

    async def get_deployment_by_run_id(self, run_id: str) -> SpanRecord | None:
        matches = [span for span in self._spans.values() if span.kind == SpanKind.DEPLOYMENT and span.run_id == run_id]
        if not matches:
            return None
        return max(matches, key=deployment_sort_key)

    async def get_deployment_cost_totals(self, root_deployment_id: UUID) -> CostTotals:
        totals = CostTotals()
        for span in self._spans.values():
            if span.root_deployment_id != root_deployment_id or span.kind != SpanKind.LLM_ROUND:
                continue
            metrics = parse_json_object(span.metrics_json, context=f"Span {span.span_id}", field_name="metrics_json")
            totals = CostTotals(
                cost_usd=totals.cost_usd + span.cost_usd,
                tokens_input=totals.tokens_input + get_token_count(metrics, TOKENS_INPUT_KEY),
                tokens_output=totals.tokens_output + get_token_count(metrics, TOKENS_OUTPUT_KEY),
                tokens_cache_read=totals.tokens_cache_read + get_token_count(metrics, TOKENS_CACHE_READ_KEY),
                tokens_reasoning=totals.tokens_reasoning + get_token_count(metrics, TOKENS_REASONING_KEY),
            )
        return totals

    async def get_deployment_logs(
        self,
        deployment_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[LogRecord]:
        return sorted(
            (
                log
                for log in self._logs
                if log.deployment_id == deployment_id and (level is None or log.level == level) and (category is None or log.category == category)
            ),
            key=log_sort_key,
        )

    async def get_deployment_logs_batch(
        self,
        deployment_ids: list[UUID],
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[LogRecord]:
        allowed_ids = set(deployment_ids)
        return sorted(
            (
                log
                for log in self._logs
                if log.deployment_id in allowed_ids and (level is None or log.level == level) and (category is None or log.category == category)
            ),
            key=log_sort_key,
        )

    async def get_deployment_span_count(
        self,
        root_deployment_id: UUID,
        *,
        kinds: list[str] | None = None,
    ) -> int:
        allowed_kinds = set(kinds) if kinds is not None else None
        return sum(
            1 for span in self._spans.values() if span.root_deployment_id == root_deployment_id and (allowed_kinds is None or span.kind in allowed_kinds)
        )

    async def get_deployment_tree(self, root_deployment_id: UUID) -> list[SpanRecord]:
        matches = [span for span in self._spans.values() if span.root_deployment_id == root_deployment_id]
        return sorted(matches, key=span_sort_key)

    async def get_document(self, document_sha256: str) -> DocumentRecord | None:
        return self._documents.get(document_sha256)

    async def get_document_with_content(self, document_sha256: str) -> HydratedDocument | None:
        record = self._documents.get(document_sha256)
        if record is None:
            return None
        blob = self._blobs.get(record.content_sha256)
        if blob is None:
            return None

        attachment_contents: dict[str, bytes] = {}
        missing_attachment_shas: list[str] = []
        for attachment_sha in record.attachment_content_sha256s:
            attachment_blob = self._blobs.get(attachment_sha)
            if attachment_blob is None:
                missing_attachment_shas.append(attachment_sha)
                continue
            attachment_contents[attachment_sha] = attachment_blob.content

        if missing_attachment_shas:
            missing_list = ", ".join(sorted(missing_attachment_shas))
            raise ValueError(
                f"Document {record.document_sha256} references attachment blobs that are missing from storage: {missing_list}. "
                "Persist every attachment blob before reading the document."
            )

        return HydratedDocument(record=record, content=blob.content, attachment_contents=attachment_contents)

    async def get_documents_batch(self, sha256s: list[str]) -> dict[str, DocumentRecord]:
        return {sha256: self._documents[sha256] for sha256 in sha256s if sha256 in self._documents}

    async def get_span(self, span_id: UUID) -> SpanRecord | None:
        return self._spans.get(span_id)

    async def get_span_logs(
        self,
        span_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[LogRecord]:
        return sorted(
            (log for log in self._logs if log.span_id == span_id and (level is None or log.level == level) and (category is None or log.category == category)),
            key=lambda log: (log.sequence_no, log.timestamp, str(log.span_id)),
        )

    async def get_spans_referencing_document(
        self,
        document_sha256: str,
        *,
        kinds: list[str] | None = None,
    ) -> list[SpanRecord]:
        allowed_kinds = set(kinds) if kinds is not None else None
        matches: list[SpanRecord] = []
        for span in self._spans.values():
            if allowed_kinds is not None and span.kind not in allowed_kinds:
                continue
            if document_sha256 in span.input_document_shas or document_sha256 in span.output_document_shas:
                matches.append(span)
                continue
            if document_sha256 in span.input_blob_shas or document_sha256 in span.output_blob_shas:
                matches.append(span)
        return sorted(matches, key=span_sort_key)

    async def insert_span(self, span: SpanRecord) -> None:
        existing = self._spans.get(span.span_id)
        if existing is None or span.version > existing.version:
            self._spans[span.span_id] = span

    async def list_deployments(
        self,
        limit: int,
        *,
        status: str | None = None,
    ) -> list[SpanRecord]:
        matches = [span for span in self._spans.values() if span.kind == SpanKind.DEPLOYMENT]
        if status is not None:
            matches = [span for span in matches if span.status == status]
        return sorted(matches, key=deployment_sort_key, reverse=True)[:limit]

    async def save_blob(self, blob: BlobRecord) -> None:
        self._blobs[blob.content_sha256] = blob

    async def save_blob_batch(self, blobs: list[BlobRecord]) -> None:
        for blob in blobs:
            self._blobs[blob.content_sha256] = blob

    async def save_document(self, record: DocumentRecord) -> None:
        existing = self._documents.get(record.document_sha256)
        if existing is None or record.created_at >= existing.created_at:
            self._documents[record.document_sha256] = record

    async def save_document_batch(self, records: list[DocumentRecord]) -> None:
        for record in records:
            await self.save_document(record)

    async def save_logs_batch(self, logs: list[LogRecord]) -> None:
        self._logs.extend(logs)

    async def shutdown(self) -> None:
        return None

    async def update_document_summary(self, document_sha256: str, summary: str) -> None:
        existing = self._documents.get(document_sha256)
        if existing is None:
            return
        self._documents[document_sha256] = replace(
            existing,
            summary=summary,
            created_at=datetime.now(UTC),
        )


# Protocol — implement in concrete class
@runtime_checkable
class DatabaseWriter(Protocol):
    """Write protocol for the span/document/blob/log schema."""
    @property
    def supports_remote(self) -> bool:
        """Whether this backend supports Prefect-based remote deployment execution."""
        ...

    async def flush(self) -> None:
        """Flush any buffered writes to storage."""
        ...

    async def insert_span(self, span: SpanRecord) -> None:
        """Insert a span row. Lifecycle updates are written as new rows with higher versions."""
        ...

    async def save_blob(self, blob: BlobRecord) -> None:
        """Persist a single blob."""
        ...

    async def save_blob_batch(self, blobs: list[BlobRecord]) -> None:
        """Persist multiple blobs in one operation."""
        ...

    async def save_document(self, record: DocumentRecord) -> None:
        """Persist a single document record."""
        ...

    async def save_document_batch(self, records: list[DocumentRecord]) -> None:
        """Persist multiple document records in one operation."""
        ...

    async def save_logs_batch(self, logs: list[LogRecord]) -> None:
        """Persist multiple log records in one operation."""
        ...

    async def shutdown(self) -> None:
        """Release resources and close connections."""
        ...

    async def update_document_summary(self, document_sha256: str, summary: str) -> None:
        """Update the top-level summary field for a document row."""
        ...


# Protocol — implement in concrete class
@runtime_checkable
class DatabaseReader(Protocol):
    """Read protocol for the span/document/blob/log schema."""
    async def get_all_document_shas_for_tree(self, root_deployment_id: UUID) -> set[str]:
        """Collect all document SHA256s referenced anywhere in a deployment tree."""
        ...

    async def get_blob(self, content_sha256: str) -> BlobRecord | None:
        """Retrieve a blob by content SHA256."""
        ...

    async def get_blobs_batch(self, content_sha256s: list[str]) -> dict[str, BlobRecord]:
        """Retrieve blobs keyed by content SHA256."""
        ...

    async def get_cached_completion(
        self,
        cache_key: str,
        *,
        max_age: timedelta | None = None,
    ) -> SpanRecord | None:
        """Find a completed span matching the cache key within the max age window."""
        ...

    async def get_child_spans(self, parent_span_id: UUID) -> list[SpanRecord]:
        """Retrieve direct child spans ordered by sequence number."""
        ...

    async def get_deployment_by_run_id(self, run_id: str) -> SpanRecord | None:
        """Find the newest deployment span for a run ID."""
        ...

    async def get_deployment_cost_totals(self, root_deployment_id: UUID) -> CostTotals:
        """Aggregate llm_round cost and token totals for a deployment tree."""
        ...

    async def get_deployment_logs(
        self,
        deployment_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[LogRecord]:
        """Retrieve logs for an entire deployment."""
        ...

    async def get_deployment_logs_batch(
        self,
        deployment_ids: list[UUID],
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[LogRecord]:
        """Retrieve logs for multiple deployments in one operation."""
        ...

    async def get_deployment_span_count(
        self,
        root_deployment_id: UUID,
        *,
        kinds: list[str] | None = None,
    ) -> int:
        """Count spans in a deployment tree, optionally filtering by span kind."""
        ...

    async def get_deployment_tree(self, root_deployment_id: UUID) -> list[SpanRecord]:
        """Retrieve every span in a deployment tree as a flat list."""
        ...

    async def get_document(self, document_sha256: str) -> DocumentRecord | None:
        """Retrieve a document record by SHA256."""
        ...

    async def get_document_with_content(
        self,
        document_sha256: str,
    ) -> HydratedDocument | None:
        """Load document metadata plus primary content and attachment blobs."""
        ...

    async def get_documents_batch(self, sha256s: list[str]) -> dict[str, DocumentRecord]:
        """Retrieve multiple document records keyed by SHA256."""
        ...

    async def get_span(self, span_id: UUID) -> SpanRecord | None:
        """Retrieve a span by its ID."""
        ...

    async def get_span_logs(
        self,
        span_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[LogRecord]:
        """Retrieve logs for a specific span."""
        ...

    async def get_spans_referencing_document(
        self,
        document_sha256: str,
        *,
        kinds: list[str] | None = None,
    ) -> list[SpanRecord]:
        """Find spans that reference a SHA in document or blob input/output arrays."""
        ...

    async def list_deployments(
        self,
        limit: int,
        *,
        status: str | None = None,
    ) -> list[SpanRecord]:
        """List deployment spans ordered by newest start time first."""
        ...


# Enum
class SpanKind(StrEnum):
    """Discriminator for span-based execution records."""
    DEPLOYMENT = 'deployment'
    FLOW = 'flow'
    TASK = 'task'
    OPERATION = 'operation'
    CONVERSATION = 'conversation'
    LLM_ROUND = 'llm_round'
    TOOL_CALL = 'tool_call'


# Enum
class SpanStatus(StrEnum):
    """Lifecycle status for span-based execution records."""
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CACHED = 'cached'
    SKIPPED = 'skipped'


@dataclass(frozen=True, slots=True)
class SpanRecord:
    """Row from the span-oriented execution table."""
    span_id: UUID
    parent_span_id: UUID | None
    deployment_id: UUID
    root_deployment_id: UUID
    run_id: str
    kind: str
    name: str
    sequence_no: int
    deployment_name: str = ''
    description: str = ''
    status: str = SpanStatus.RUNNING
    started_at: datetime = field(default_factory=_utcnow)
    ended_at: datetime | None = None
    version: int = 1
    cache_key: str = ''
    previous_conversation_id: UUID | None = None
    cost_usd: float = 0.0
    error_type: str = ''
    error_message: str = ''
    input_document_shas: tuple[str, ...] = ()
    output_document_shas: tuple[str, ...] = ()
    target: str = ''
    receiver_json: str = ''
    input_json: str = ''
    output_json: str = ''
    error_json: str = ''
    meta_json: str = ''
    metrics_json: str = ''
    input_blob_shas: tuple[str, ...] = ()
    output_blob_shas: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_enum_string("kind", self.kind, SpanKind)
        _validate_enum_string("status", self.status, SpanStatus)
        _validate_string_tuple("input_document_shas", self.input_document_shas)
        _validate_string_tuple("output_document_shas", self.output_document_shas)
        _validate_string_tuple("input_blob_shas", self.input_blob_shas)
        _validate_string_tuple("output_blob_shas", self.output_blob_shas)


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Row from the content-addressed documents table."""
    document_sha256: DocumentSha256
    content_sha256: str
    document_type: str
    name: str
    description: str = ''
    mime_type: str = ''
    size_bytes: int = 0
    summary: str = ''
    derived_from: tuple[str, ...] = ()
    triggered_by: tuple[str, ...] = ()
    attachment_names: tuple[str, ...] = ()
    attachment_descriptions: tuple[str, ...] = ()
    attachment_content_sha256s: tuple[str, ...] = ()
    attachment_mime_types: tuple[str, ...] = ()
    attachment_size_bytes: tuple[int, ...] = ()
    created_at: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        _validate_string_tuple("derived_from", self.derived_from)
        _validate_string_tuple("triggered_by", self.triggered_by)
        _validate_string_tuple("attachment_names", self.attachment_names)
        _validate_string_tuple("attachment_descriptions", self.attachment_descriptions)
        _validate_string_tuple("attachment_content_sha256s", self.attachment_content_sha256s)
        _validate_string_tuple("attachment_mime_types", self.attachment_mime_types)
        _validate_int_tuple("attachment_size_bytes", self.attachment_size_bytes)
        attachment_count = len(self.attachment_names)
        attachment_lengths = (
            len(self.attachment_descriptions),
            len(self.attachment_content_sha256s),
            len(self.attachment_mime_types),
            len(self.attachment_size_bytes),
        )
        if any(length != attachment_count for length in attachment_lengths):
            msg = (
                "DocumentRecord attachment fields must have matching lengths. "
                "Provide one name, description, content_sha256, mime_type, and size_bytes entry for each attachment."
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class BlobRecord:
    """Row from the immutable blobs table."""
    content_sha256: str
    content: bytes
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True, slots=True)
class CostTotals:
    """Aggregated cost and token totals for llm_round spans."""
    cost_usd: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_read: int = 0
    tokens_reasoning: int = 0


@dataclass(frozen=True, slots=True)
class HydratedDocument:
    """Document metadata with loaded primary and attachment blob content."""
    record: DocumentRecord
    content: bytes
    attachment_contents: dict[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_bytes_mapping("attachment_contents", self.attachment_contents)


```

## Functions

```python
def create_database(
    *,
    backend: str = "memory",
    base_path: Path | None = None,
    settings: Settings | None = None,
) -> Database:
    """Factory for creating database backends.

    Args:
        backend: Backend type — 'memory', 'filesystem', or 'clickhouse'.
        base_path: Root directory for filesystem backend.
        settings: Application settings (required for 'clickhouse' backend).

    Returns:
        A database backend implementing both DatabaseWriter and DatabaseReader.
    """
    if backend == "memory":
        return MemoryDatabase()

    if backend == "filesystem":
        if base_path is None:
            msg = "FilesystemDatabase requires base_path parameter"
            raise ValueError(msg)
        return FilesystemDatabase(base_path)

    if backend == "clickhouse":
        return ClickHouseDatabase(settings=settings)

    supported = "'memory', 'filesystem', 'clickhouse'"
    msg = f"Unknown database backend: {backend!r}. Supported: {supported}"
    raise ValueError(msg)

def create_database_from_settings(
    settings: Settings,
    base_path: Path | None = None,
) -> Database:
    """Create the right database backend based on application settings.

    Args:
        settings: Application settings with ClickHouse configuration.
        base_path: Root directory for filesystem backend (used when ClickHouse is not configured).

    Returns:
        ClickHouse backend if clickhouse_host is set, filesystem if base_path is provided,
        otherwise in-memory backend.
    """
    if settings.clickhouse_host:
        return create_database(backend="clickhouse", settings=settings)
    if base_path is not None:
        return create_database(backend="filesystem", base_path=base_path)
    return create_database(backend="memory")

async def download_deployment(
    source: DatabaseReader,
    deployment_id: UUID,
    output_path: Path,
) -> None:
    """Export a deployment tree as a FilesystemDatabase snapshot."""
    tree = await source.get_deployment_tree(deployment_id)
    staged_path = await asyncio.to_thread(_create_staging_path, output_path)
    target = await asyncio.to_thread(FilesystemDatabase, staged_path)
    committed = False
    try:
        for span in tree:
            await target.insert_span(span)

        document_shas = await source.get_all_document_shas_for_tree(deployment_id)
        documents = await source.get_documents_batch(sorted(document_shas))
        _raise_if_missing_records(
            record_kind="documents",
            expected_shas=document_shas,
            actual_shas=set(documents),
            deployment_id=deployment_id,
        )
        if documents:
            await target.save_document_batch(list(documents.values()))

        blob_shas = _collect_attachment_blob_shas(list(documents.values()))
        blob_shas.update(_collect_span_blob_shas(tree))
        blobs = await source.get_blobs_batch(sorted(blob_shas))
        _raise_if_missing_records(
            record_kind="blobs",
            expected_shas=blob_shas,
            actual_shas=set(blobs),
            deployment_id=deployment_id,
        )
        if blobs:
            await target.save_blob_batch(list(blobs.values()))

        deployment_ids = sorted({span.deployment_id for span in tree}, key=str)
        logs = await _get_tree_logs(source, deployment_ids)
        if logs:
            await target.save_logs_batch(logs)
        else:
            await asyncio.to_thread((staged_path / "logs.jsonl").write_text, "", encoding="utf-8")

        await generate_run_artifacts(
            target,
            deployment_id,
            staged_path,
            tree=tree,
            documents=documents,
        )
        await target.shutdown()
        validation = await asyncio.to_thread(validate_bundle, staged_path)
        await asyncio.to_thread(
            (staged_path / VALIDATION_FILENAME).write_text,
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        await asyncio.to_thread(_replace_output_path, staged_path, output_path)
        committed = True
    finally:
        if not committed:
            await target.shutdown()
            await asyncio.to_thread(_remove_path, staged_path)

```

## Examples

**Download deployment bundle validation reads attachment fields** (`tests/database/test_span_download.py:202`)

```python
@pytest.mark.asyncio
async def test_download_deployment_bundle_validation_reads_attachment_fields(tmp_path: Path) -> None:
    source, deployment_id = await _seed_source_database()
    output_path = tmp_path / "snapshot"

    await download_deployment(source, deployment_id, output_path)
    validation = validate_bundle(output_path)

    assert validation["valid"] is True
    assert validation["document_count"] == 1
    assert validation["blob_count"] >= 2
```

**Download deployment documents artifact uses promoted fields** (`tests/database/test_span_download.py:215`)

```python
@pytest.mark.asyncio
async def test_download_deployment_documents_artifact_uses_promoted_fields(tmp_path: Path) -> None:
    source, deployment_id = await _seed_source_database()
    output_path = tmp_path / "snapshot"

    await download_deployment(source, deployment_id, output_path)
    documents_md = (output_path / "documents.md").read_text(encoding="utf-8")

    assert "mime=text/markdown" in documents_md
    assert "size=10" in documents_md
    assert "root.md" in documents_md
```

**Download deployment writes llm calls errors and documents artifacts** (`tests/database/test_span_summary.py:272`)

```python
@pytest.mark.asyncio
async def test_download_deployment_writes_llm_calls_errors_and_documents_artifacts(tmp_path: Path) -> None:
    database, deployment_id = await _seed_summary_database()
    output_dir = tmp_path / "download"

    await download_deployment(database, deployment_id, output_dir)

    assert (output_dir / "llm_calls.jsonl").exists()
    assert (output_dir / "documents.md").exists()
    assert (output_dir / "errors.md").exists()
    assert "gpt-5.1" in (output_dir / "llm_calls.jsonl").read_text(encoding="utf-8")
    assert "summary.md" in (output_dir / "documents.md").read_text(encoding="utf-8")
    assert "FailedTask" in (output_dir / "errors.md").read_text(encoding="utf-8")
```

**Download deployment writes portable snapshot** (`tests/database/test_span_download.py:181`)

```python
@pytest.mark.asyncio
async def test_download_deployment_writes_portable_snapshot(tmp_path: Path) -> None:
    source, deployment_id = await _seed_source_database()
    output_path = tmp_path / "snapshot"

    await download_deployment(source, deployment_id, output_path)

    snapshot = FilesystemDatabase(output_path, read_only=True)
    hydrated = await snapshot.get_document_with_content("doc-root")
    deployment = await snapshot.get_deployment_by_run_id(f"run-{deployment_id}")

    assert deployment is not None
    assert hydrated is not None
    assert hydrated.content == b"shared"
    assert hydrated.attachment_contents == {"blob-shared-attachment": b"attachment"}
    assert (output_path / "summary.md").exists()
    assert (output_path / "costs.md").exists()
    assert (output_path / "documents.md").exists()
    assert (output_path / "llm_calls.jsonl").exists()
```

**Download deployment writes summary and cost artifacts** (`tests/database/test_span_summary.py:259`)

```python
@pytest.mark.asyncio
async def test_download_deployment_writes_summary_and_cost_artifacts(tmp_path: Path) -> None:
    database, deployment_id = await _seed_summary_database()
    output_dir = tmp_path / "download"

    await download_deployment(database, deployment_id, output_dir)

    assert (output_dir / "summary.md").exists()
    assert (output_dir / "costs.md").exists()
    assert "span-pipeline / span-run" in (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "# Cost by Model" in (output_dir / "costs.md").read_text(encoding="utf-8")
```

**Database reader is runtime checkable** (`tests/database/test_protocol.py:92`)

```python
def test_database_reader_is_runtime_checkable() -> None:
    assert getattr(DatabaseReader, "_is_runtime_protocol", False) is True
    assert isinstance(_make_reader_stub(), DatabaseReader)
    assert not isinstance(object(), DatabaseReader)
```

**Database writer is runtime checkable** (`tests/database/test_protocol.py:98`)

```python
def test_database_writer_is_runtime_checkable() -> None:
    assert getattr(DatabaseWriter, "_is_runtime_protocol", False) is True
    assert isinstance(_make_writer_stub(), DatabaseWriter)
    assert not isinstance(object(), DatabaseWriter)
```

**Database writer method signatures** (`tests/database/test_protocol.py:160`)

```python
def test_database_writer_method_signatures() -> None:
    _assert_signature(DatabaseWriter, "insert_span", parameter_types={"span": SpanRecord}, return_type=type(None))
    _assert_signature(DatabaseWriter, "save_document", parameter_types={"record": DocumentRecord}, return_type=type(None))
    _assert_signature(DatabaseWriter, "save_blob", parameter_types={"blob": BlobRecord}, return_type=type(None))
    _assert_signature(DatabaseWriter, "save_logs_batch", parameter_types={"logs": list[LogRecord]}, return_type=type(None))
```


## Error Examples

**Blob record defaults and immutability** (`tests/database/test_types.py:124`)

```python
def test_blob_record_defaults_and_immutability() -> None:
    before = datetime.now(UTC)
    blob = BlobRecord(content_sha256="blob-sha", content=b"hello")
    after = datetime.now(UTC)

    assert before <= blob.created_at <= after

    with pytest.raises(dataclasses.FrozenInstanceError):
        blob.content = b"changed"  # type: ignore[misc]
```

**Document record rejects mismatched attachment lengths** (`tests/database/test_types.py:109`)

```python
def test_document_record_rejects_mismatched_attachment_lengths() -> None:
    with pytest.raises(ValueError, match="matching lengths"):
        DocumentRecord(
            document_sha256="doc-sha",
            content_sha256="blob-sha",
            document_type="ExampleDocument",
            name="example.md",
            attachment_names=("a.txt",),
            attachment_descriptions=(),
            attachment_content_sha256s=("blob-a",),
            attachment_mime_types=("text/plain",),
            attachment_size_bytes=(1,),
        )
```

**Ensure schema raises on newer db** (`tests/database/test_clickhouse.py:199`)

```python
@pytest.mark.asyncio
async def test_ensure_schema_raises_on_newer_db() -> None:
    client = _mock_client(table_exists=True, db_version=SCHEMA_VERSION + 1)

    with pytest.raises(SchemaVersionError, match="newer than the framework supports"):
        await _ensure_schema(client, "default")
```

**Ensure schema raises on outdated db** (`tests/database/test_clickhouse.py:191`)

```python
@pytest.mark.asyncio
async def test_ensure_schema_raises_on_outdated_db() -> None:
    client = _mock_client(table_exists=True, db_version=SCHEMA_VERSION - 1)

    with pytest.raises(SchemaVersionError, match="older than the framework expects"):
        await _ensure_schema(client, "default")
```

**Ensure schema raises on zero version in existing table** (`tests/database/test_clickhouse.py:219`)

```python
@pytest.mark.asyncio
async def test_ensure_schema_raises_on_zero_version_in_existing_table() -> None:
    client = _mock_client(table_exists=True, db_version=0)

    with pytest.raises(SchemaVersionError, match="older than the framework expects"):
        await _ensure_schema(client, "default")
```

**Filesystem read only rejects writes** (`tests/database/test_span_filesystem.py:223`)

```python
@pytest.mark.asyncio
async def test_filesystem_read_only_rejects_writes(tmp_path: Path) -> None:
    database, _ = await _seed_database(tmp_path)
    await database.shutdown()

    read_only = FilesystemDatabase(tmp_path, read_only=True)

    with pytest.raises(PermissionError, match="read-only"):
        await read_only.save_document(_make_document())
```

**Get document with content raises for missing attachment blob** (`tests/database/test_span_memory.py:246`)

```python
@pytest.mark.asyncio
async def test_get_document_with_content_raises_for_missing_attachment_blob() -> None:
    database = MemoryDatabase()
    await database.save_document(
        _make_document(
            document_sha256="doc-1",
            content_sha256="blob-1",
            attachment_names=("preview.png",),
            attachment_descriptions=("Preview",),
            attachment_content_sha256s=("blob-missing",),
            attachment_mime_types=("image/png",),
            attachment_size_bytes=(10,),
        )
    )
    await database.save_blob(_make_blob(content_sha256="blob-1", content=b"root"))

    with pytest.raises(ValueError, match="missing from storage"):
        await database.get_document_with_content("doc-1")
```

**Document record defaults and attachment fields** (`tests/database/test_types.py:85`)

```python
def test_document_record_defaults_and_attachment_fields() -> None:
    record = DocumentRecord(
        document_sha256="doc-sha",
        content_sha256="blob-sha",
        document_type="ExampleDocument",
        name="example.md",
    )

    assert record.description == ""
    assert record.mime_type == ""
    assert record.size_bytes == 0
    assert record.summary == ""
    assert record.derived_from == ()
    assert record.triggered_by == ()
    assert record.attachment_names == ()
    assert record.attachment_descriptions == ()
    assert record.attachment_content_sha256s == ()
    assert record.attachment_mime_types == ()
    assert record.attachment_size_bytes == ()

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.name = "changed"  # type: ignore[misc]
```
