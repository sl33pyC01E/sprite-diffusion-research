from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    root_url TEXT NOT NULL,
    adapter_version TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS crawl_runs (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    free_bytes_start INTEGER NOT NULL,
    free_bytes_end INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    external_id TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    title TEXT,
    description TEXT,
    creator_name TEXT,
    creator_url TEXT,
    published_at TEXT,
    metadata_json TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    tombstoned_at TEXT,
    UNIQUE(source_id, external_id)
);

CREATE TABLE IF NOT EXISTS blobs (
    sha256 TEXT PRIMARY KEY,
    size_bytes INTEGER NOT NULL,
    mime_type TEXT,
    storage_path TEXT NOT NULL,
    first_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrievals (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES crawl_runs(id),
    item_id TEXT REFERENCES items(id),
    url TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    completed_at TEXT,
    status_code INTEGER,
    etag TEXT,
    last_modified TEXT,
    mime_type TEXT,
    content_length INTEGER,
    blob_sha256 TEXT REFERENCES blobs(sha256),
    error TEXT
);

CREATE TABLE IF NOT EXISTS item_blobs (
    id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL REFERENCES items(id),
    blob_sha256 TEXT NOT NULL REFERENCES blobs(sha256),
    role TEXT NOT NULL,
    original_url TEXT,
    original_filename TEXT,
    archive_member TEXT,
    observed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rights_observations (
    id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL REFERENCES items(id),
    observed_at TEXT NOT NULL,
    license_raw TEXT,
    license_expression TEXT,
    license_url TEXT,
    attribution_raw TEXT,
    terms_url TEXT,
    terms_blob_sha256 TEXT REFERENCES blobs(sha256),
    robots_url TEXT,
    robots_blob_sha256 TEXT REFERENCES blobs(sha256),
    basis TEXT,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS derivations (
    id TEXT PRIMARY KEY,
    child_blob_sha256 TEXT NOT NULL REFERENCES blobs(sha256),
    parent_blob_sha256 TEXT NOT NULL REFERENCES blobs(sha256),
    operation TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    code_version TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sequences (
    id TEXT PRIMARY KEY,
    item_id TEXT REFERENCES items(id),
    source_blob_sha256 TEXT REFERENCES blobs(sha256),
    extraction_method TEXT NOT NULL,
    extraction_confidence REAL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    loop_mode TEXT,
    action TEXT,
    direction TEXT,
    quality_tier TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS frames (
    sequence_id TEXT NOT NULL REFERENCES sequences(id),
    ordinal INTEGER NOT NULL,
    blob_sha256 TEXT NOT NULL REFERENCES blobs(sha256),
    duration_ms INTEGER,
    bbox_json TEXT,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY(sequence_id, ordinal)
);

CREATE TABLE IF NOT EXISTS captions (
    id TEXT PRIMARY KEY,
    item_id TEXT REFERENCES items(id),
    sequence_id TEXT REFERENCES sequences(id),
    text TEXT NOT NULL,
    structured_json TEXT NOT NULL,
    method TEXT NOT NULL,
    model_revision TEXT,
    prompt_hash TEXT,
    confidence REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_snapshots (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    code_version TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_members (
    snapshot_id TEXT NOT NULL REFERENCES dataset_snapshots(id),
    sequence_id TEXT NOT NULL REFERENCES sequences(id),
    split TEXT NOT NULL,
    sample_weight REAL NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT,
    PRIMARY KEY(snapshot_id, sequence_id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_source ON items(source_id);
CREATE INDEX IF NOT EXISTS idx_retrievals_item ON retrievals(item_id);
CREATE INDEX IF NOT EXISTS idx_item_blobs_item ON item_blobs(item_id);
CREATE INDEX IF NOT EXISTS idx_sequences_item ON sequences(item_id);
CREATE INDEX IF NOT EXISTS idx_frames_blob ON frames(blob_sha256);
"""


SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    external_identity_key TEXT NOT NULL,
    representative_item_id TEXT REFERENCES items(id),
    display_name TEXT,
    entity_class TEXT NOT NULL,
    entity_subclass TEXT,
    species_or_type TEXT,
    taxonomy_version TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(source_id, external_identity_key)
);

CREATE TABLE IF NOT EXISTS sequence_subjects (
    sequence_id TEXT NOT NULL REFERENCES sequences(id),
    entity_id TEXT NOT NULL REFERENCES entities(id),
    role TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY(sequence_id, entity_id, role)
);

CREATE TABLE IF NOT EXISTS motion_annotations (
    sequence_id TEXT PRIMARY KEY REFERENCES sequences(id),
    vocabulary_version TEXT NOT NULL,
    source_action TEXT,
    normalized_action TEXT NOT NULL,
    action_family TEXT NOT NULL,
    view TEXT,
    direction TEXT,
    loopable INTEGER,
    cycle_frames INTEGER,
    phase_zero_frame INTEGER,
    confidence REAL,
    annotation_method TEXT NOT NULL,
    conditioning_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK(loopable IS NULL OR loopable IN (0, 1)),
    CHECK(cycle_frames IS NULL OR cycle_frames > 0),
    CHECK(phase_zero_frame IS NULL OR phase_zero_frame >= 0)
);

CREATE INDEX IF NOT EXISTS idx_entities_source ON entities(source_id);
CREATE INDEX IF NOT EXISTS idx_entities_class ON entities(entity_class, entity_subclass);
CREATE INDEX IF NOT EXISTS idx_sequence_subjects_entity ON sequence_subjects(entity_id);
CREATE INDEX IF NOT EXISTS idx_motion_action ON motion_annotations(normalized_action);
CREATE INDEX IF NOT EXISTS idx_motion_family ON motion_annotations(action_family);
"""


SCHEMA_V3 = """
CREATE TABLE IF NOT EXISTS archive_inventories (
    archive_blob_sha256 TEXT PRIMARY KEY REFERENCES blobs(sha256),
    archive_format TEXT NOT NULL,
    member_count INTEGER NOT NULL,
    file_count INTEGER NOT NULL,
    total_uncompressed_bytes INTEGER NOT NULL,
    total_compressed_bytes INTEGER NOT NULL,
    policy_json TEXT NOT NULL,
    inventory_sha256 TEXT NOT NULL,
    inspected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS archive_members (
    archive_blob_sha256 TEXT NOT NULL REFERENCES blobs(sha256),
    ordinal INTEGER NOT NULL,
    member_path TEXT NOT NULL,
    normalized_path TEXT NOT NULL,
    member_kind TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    compressed_bytes INTEGER NOT NULL,
    crc32 INTEGER,
    compression_method INTEGER,
    modified_at TEXT,
    extracted_blob_sha256 TEXT REFERENCES blobs(sha256),
    selected_role TEXT,
    inspection_status TEXT NOT NULL,
    error TEXT,
    metadata_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    PRIMARY KEY(archive_blob_sha256, ordinal),
    UNIQUE(archive_blob_sha256, normalized_path)
);

CREATE TABLE IF NOT EXISTS media_observations (
    blob_sha256 TEXT NOT NULL REFERENCES blobs(sha256),
    inspector_version TEXT NOT NULL,
    media_format TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    mode TEXT NOT NULL,
    has_alpha INTEGER,
    is_animated INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    loop_count INTEGER,
    total_duration_ms REAL,
    palette_sha256 TEXT,
    pixel_sha256 TEXT,
    metadata_json TEXT NOT NULL,
    inspected_at TEXT NOT NULL,
    PRIMARY KEY(blob_sha256, inspector_version),
    CHECK(has_alpha IS NULL OR has_alpha IN (0, 1)),
    CHECK(is_animated IN (0, 1))
);

CREATE TABLE IF NOT EXISTS duplicate_edges (
    id TEXT PRIMARY KEY,
    left_blob_sha256 TEXT NOT NULL REFERENCES blobs(sha256),
    right_blob_sha256 TEXT NOT NULL REFERENCES blobs(sha256),
    method TEXT NOT NULL,
    distance REAL NOT NULL,
    parameters_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(left_blob_sha256, right_blob_sha256, method)
);

CREATE INDEX IF NOT EXISTS idx_archive_members_path
    ON archive_members(archive_blob_sha256, normalized_path);
CREATE INDEX IF NOT EXISTS idx_archive_members_blob ON archive_members(extracted_blob_sha256);
CREATE INDEX IF NOT EXISTS idx_media_dimensions ON media_observations(width, height);
CREATE INDEX IF NOT EXISTS idx_media_animation ON media_observations(is_animated, frame_count);
CREATE INDEX IF NOT EXISTS idx_media_pixel_hash ON media_observations(pixel_sha256);
"""


SCHEMA_V4 = """
CREATE TABLE IF NOT EXISTS sequence_occurrences (
    sequence_id TEXT NOT NULL REFERENCES sequences(id),
    archive_blob_sha256 TEXT NOT NULL,
    archive_member_ordinal INTEGER NOT NULL,
    occurrence_role TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(
        sequence_id,
        archive_blob_sha256,
        archive_member_ordinal,
        occurrence_role
    ),
    FOREIGN KEY(archive_blob_sha256, archive_member_ordinal)
        REFERENCES archive_members(archive_blob_sha256, ordinal)
);

CREATE TABLE IF NOT EXISTS sequence_frames (
    sequence_id TEXT NOT NULL REFERENCES sequences(id),
    ordinal INTEGER NOT NULL,
    source_blob_sha256 TEXT NOT NULL REFERENCES blobs(sha256),
    source_frame_index INTEGER,
    duration_ms REAL,
    phase REAL,
    direction TEXT,
    view TEXT,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY(sequence_id, ordinal),
    CHECK(source_frame_index IS NULL OR source_frame_index >= 0),
    CHECK(duration_ms IS NULL OR duration_ms >= 0),
    CHECK(phase IS NULL OR (phase >= 0 AND phase <= 1))
);

CREATE INDEX IF NOT EXISTS idx_sequence_occurrences_member
    ON sequence_occurrences(archive_blob_sha256, archive_member_ordinal);
CREATE INDEX IF NOT EXISTS idx_sequence_frames_blob ON sequence_frames(source_blob_sha256);
"""


SCHEMA_V5 = """
CREATE TABLE IF NOT EXISTS sequence_source_keys (
    source_id TEXT NOT NULL REFERENCES sources(id),
    external_sequence_key TEXT NOT NULL,
    sequence_id TEXT NOT NULL UNIQUE REFERENCES sequences(id),
    created_at TEXT NOT NULL,
    PRIMARY KEY(source_id, external_sequence_key)
);

CREATE INDEX IF NOT EXISTS idx_sequence_source_keys_sequence
    ON sequence_source_keys(sequence_id);
"""


class IndexDB:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._transaction_connection: ContextVar[sqlite3.Connection | None] = ContextVar(
            f"spritelab_indexdb_transaction_{id(self)}",
            default=None,
        )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        active = self._transaction_connection.get()
        if active is not None:
            yield active
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Reuse one connection and commit for a synchronous batch of DB calls.

        Existing helpers continue to open and commit their own connection when
        called outside this context. Inside it, their nested ``connect()`` calls
        reuse the outer connection, so a large projection is committed once and
        rolls back as a unit if an exception escapes the context. Nested
        transaction contexts participate in the outer transaction.
        """

        active = self._transaction_connection.get()
        if active is not None:
            yield active
            return

        with self.connect() as connection:
            token = self._transaction_connection.set(connection)
            try:
                yield connection
            finally:
                self._transaction_connection.reset(token)

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA_V1)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (1, utc_now()),
            )
            connection.executescript(SCHEMA_V2)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (2, utc_now()),
            )
            connection.executescript(SCHEMA_V3)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (3, utc_now()),
            )
            connection.executescript(SCHEMA_V4)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (4, utc_now()),
            )
            connection.executescript(SCHEMA_V5)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (5, utc_now()),
            )

    def record_event(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        payload: Any,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        values = (event_type, entity_type, entity_id, json_text(payload), utc_now())
        if connection is not None:
            connection.execute(
                "INSERT INTO events(event_type, entity_type, entity_id, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                values,
            )
            return
        with self.connect() as own_connection:
            own_connection.execute(
                "INSERT INTO events(event_type, entity_type, entity_id, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                values,
            )

    def register_source(
        self,
        *,
        source_id: str,
        kind: str,
        name: str,
        root_url: str,
        adapter_version: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.initialize()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sources(
                    id, kind, name, root_url, adapter_version, config_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind=excluded.kind,
                    name=excluded.name,
                    root_url=excluded.root_url,
                    adapter_version=excluded.adapter_version,
                    config_json=excluded.config_json
                """,
                (
                    source_id,
                    kind,
                    name,
                    root_url,
                    adapter_version,
                    json_text(config or {}),
                    utc_now(),
                ),
            )
            self.record_event(
                "source_registered",
                "source",
                source_id,
                {"kind": kind, "root_url": root_url, "adapter_version": adapter_version},
                connection=connection,
            )

    def counts(self) -> dict[str, int]:
        self.initialize()
        tables = (
            "sources",
            "crawl_runs",
            "items",
            "blobs",
            "entities",
            "sequences",
            "frames",
            "archive_members",
            "media_observations",
            "sequence_frames",
        )
        with self.connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in tables
            }

    def create_crawl_run(
        self,
        *,
        source_id: str,
        parameters: dict[str, Any],
        free_bytes_start: int,
    ) -> str:
        run_id = new_id("run")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO crawl_runs(
                    id, source_id, started_at, status, parameters_json, free_bytes_start
                ) VALUES (?, ?, ?, 'running', ?, ?)
                """,
                (run_id, source_id, utc_now(), json_text(parameters), free_bytes_start),
            )
            self.record_event(
                "crawl_started",
                "crawl_run",
                run_id,
                {"source_id": source_id, "parameters": parameters},
                connection=connection,
            )
        return run_id

    def finish_crawl_run(
        self,
        run_id: str,
        *,
        status: str,
        free_bytes_end: int,
        error: str | None = None,
    ) -> None:
        if status not in {"completed", "failed", "disk_floor", "interrupted"}:
            raise ValueError(f"Invalid terminal crawl status: {status}")
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE crawl_runs
                SET completed_at=?, status=?, free_bytes_end=?, error=?
                WHERE id=?
                """,
                (utc_now(), status, free_bytes_end, error, run_id),
            )
            self.record_event(
                "crawl_finished",
                "crawl_run",
                run_id,
                {"status": status, "error": error},
                connection=connection,
            )

    def upsert_item(
        self,
        *,
        source_id: str,
        external_id: str,
        canonical_url: str,
        title: str | None = None,
        description: str | None = None,
        creator_name: str | None = None,
        creator_url: str | None = None,
        published_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        now = utc_now()
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM items WHERE source_id=? AND external_id=?",
                (source_id, external_id),
            ).fetchone()
            item_id = str(existing["id"]) if existing else new_id("item")
            connection.execute(
                """
                INSERT INTO items(
                    id, source_id, external_id, canonical_url, title, description,
                    creator_name, creator_url, published_at, metadata_json,
                    first_seen_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, external_id) DO UPDATE SET
                    canonical_url=excluded.canonical_url,
                    title=COALESCE(excluded.title, items.title),
                    description=COALESCE(excluded.description, items.description),
                    creator_name=COALESCE(excluded.creator_name, items.creator_name),
                    creator_url=COALESCE(excluded.creator_url, items.creator_url),
                    published_at=COALESCE(excluded.published_at, items.published_at),
                    metadata_json=excluded.metadata_json,
                    last_seen_at=excluded.last_seen_at,
                    tombstoned_at=NULL
                """,
                (
                    item_id,
                    source_id,
                    external_id,
                    canonical_url,
                    title,
                    description,
                    creator_name,
                    creator_url,
                    published_at,
                    json_text(metadata or {}),
                    now,
                    now,
                ),
            )
            self.record_event(
                "item_observed",
                "item",
                item_id,
                {"source_id": source_id, "external_id": external_id},
                connection=connection,
            )
        return item_id

    def start_retrieval(
        self,
        *,
        url: str,
        run_id: str | None = None,
        item_id: str | None = None,
    ) -> str:
        retrieval_id = new_id("fetch")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO retrievals(id, run_id, item_id, url, requested_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (retrieval_id, run_id, item_id, url, utc_now()),
            )
        return retrieval_id

    def finish_retrieval(
        self,
        retrieval_id: str,
        *,
        status_code: int | None,
        etag: str | None = None,
        last_modified: str | None = None,
        mime_type: str | None = None,
        content_length: int | None = None,
        blob_sha256: str | None = None,
        error: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE retrievals SET
                    completed_at=?, status_code=?, etag=?, last_modified=?,
                    mime_type=?, content_length=?, blob_sha256=?, error=?
                WHERE id=?
                """,
                (
                    utc_now(),
                    status_code,
                    etag,
                    last_modified,
                    mime_type,
                    content_length,
                    blob_sha256,
                    error,
                    retrieval_id,
                ),
            )
            self.record_event(
                "retrieval_finished",
                "retrieval",
                retrieval_id,
                {
                    "status_code": status_code,
                    "blob_sha256": blob_sha256,
                    "error": error,
                },
                connection=connection,
            )

    def register_blob(
        self,
        *,
        sha256: str,
        size_bytes: int,
        storage_path: Path,
        mime_type: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO blobs(
                    sha256, size_bytes, mime_type, storage_path, first_seen_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (sha256, size_bytes, mime_type, str(storage_path), utc_now()),
            )

    def link_item_blob(
        self,
        *,
        item_id: str,
        blob_sha256: str,
        role: str,
        original_url: str | None = None,
        original_filename: str | None = None,
        archive_member: str | None = None,
    ) -> str:
        link_id = new_id("itemblob")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO item_blobs(
                    id, item_id, blob_sha256, role, original_url,
                    original_filename, archive_member, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    link_id,
                    item_id,
                    blob_sha256,
                    role,
                    original_url,
                    original_filename,
                    archive_member,
                    utc_now(),
                ),
            )
        return link_id

    def add_rights_observation(
        self,
        *,
        item_id: str,
        license_raw: str | None,
        license_expression: str | None,
        license_url: str | None = None,
        attribution_raw: str | None = None,
        terms_url: str | None = None,
        terms_blob_sha256: str | None = None,
        robots_url: str | None = None,
        robots_blob_sha256: str | None = None,
        basis: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        observation_id = new_id("rights")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO rights_observations(
                    id, item_id, observed_at, license_raw, license_expression,
                    license_url, attribution_raw, terms_url, terms_blob_sha256,
                    robots_url, robots_blob_sha256, basis, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    item_id,
                    utc_now(),
                    license_raw,
                    license_expression,
                    license_url,
                    attribution_raw,
                    terms_url,
                    terms_blob_sha256,
                    robots_url,
                    robots_blob_sha256,
                    basis,
                    json_text(metadata or {}),
                ),
            )
        return observation_id

    def upsert_entity(
        self,
        *,
        source_id: str,
        external_identity_key: str,
        entity_class: str,
        taxonomy_version: str,
        representative_item_id: str | None = None,
        display_name: str | None = None,
        entity_subclass: str | None = None,
        species_or_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create or update a source-scoped identity shared by its action sequences."""
        now = utc_now()
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM entities WHERE source_id=? AND external_identity_key=?",
                (source_id, external_identity_key),
            ).fetchone()
            entity_id = str(existing["id"]) if existing else new_id("entity")
            connection.execute(
                """
                INSERT INTO entities(
                    id, source_id, external_identity_key, representative_item_id,
                    display_name, entity_class, entity_subclass, species_or_type,
                    taxonomy_version, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, external_identity_key) DO UPDATE SET
                    representative_item_id=COALESCE(
                        excluded.representative_item_id, entities.representative_item_id
                    ),
                    display_name=COALESCE(excluded.display_name, entities.display_name),
                    entity_class=excluded.entity_class,
                    entity_subclass=COALESCE(excluded.entity_subclass, entities.entity_subclass),
                    species_or_type=COALESCE(excluded.species_or_type, entities.species_or_type),
                    taxonomy_version=excluded.taxonomy_version,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    entity_id,
                    source_id,
                    external_identity_key,
                    representative_item_id,
                    display_name,
                    entity_class,
                    entity_subclass,
                    species_or_type,
                    taxonomy_version,
                    json_text(metadata or {}),
                    now,
                    now,
                ),
            )
            self.record_event(
                "entity_observed",
                "entity",
                entity_id,
                {
                    "source_id": source_id,
                    "external_identity_key": external_identity_key,
                    "entity_class": entity_class,
                },
                connection=connection,
            )
        return entity_id

    def link_sequence_subject(
        self,
        *,
        sequence_id: str,
        entity_id: str,
        role: str = "primary",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sequence_subjects(sequence_id, entity_id, role, metadata_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(sequence_id, entity_id, role) DO UPDATE SET
                    metadata_json=excluded.metadata_json
                """,
                (sequence_id, entity_id, role, json_text(metadata or {})),
            )

    def annotate_motion(
        self,
        *,
        sequence_id: str,
        vocabulary_version: str,
        normalized_action: str,
        action_family: str,
        annotation_method: str,
        source_action: str | None = None,
        view: str | None = None,
        direction: str | None = None,
        loopable: bool | None = None,
        cycle_frames: int | None = None,
        phase_zero_frame: int | None = None,
        confidence: float | None = None,
        conditioning: dict[str, Any] | None = None,
    ) -> None:
        """Attach steerable motion labels while preserving the source vocabulary."""
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO motion_annotations(
                    sequence_id, vocabulary_version, source_action, normalized_action,
                    action_family, view, direction, loopable, cycle_frames,
                    phase_zero_frame, confidence, annotation_method, conditioning_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sequence_id) DO UPDATE SET
                    vocabulary_version=excluded.vocabulary_version,
                    source_action=excluded.source_action,
                    normalized_action=excluded.normalized_action,
                    action_family=excluded.action_family,
                    view=excluded.view,
                    direction=excluded.direction,
                    loopable=excluded.loopable,
                    cycle_frames=excluded.cycle_frames,
                    phase_zero_frame=excluded.phase_zero_frame,
                    confidence=excluded.confidence,
                    annotation_method=excluded.annotation_method,
                    conditioning_json=excluded.conditioning_json,
                    updated_at=excluded.updated_at
                """,
                (
                    sequence_id,
                    vocabulary_version,
                    source_action,
                    normalized_action,
                    action_family,
                    view,
                    direction,
                    None if loopable is None else int(loopable),
                    cycle_frames,
                    phase_zero_frame,
                    confidence,
                    annotation_method,
                    json_text(conditioning or {}),
                    now,
                    now,
                ),
            )

    def upsert_archive_inventory(
        self,
        *,
        archive_blob_sha256: str,
        archive_format: str,
        member_count: int,
        file_count: int,
        total_uncompressed_bytes: int,
        total_compressed_bytes: int,
        inventory_sha256: str,
        policy: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO archive_inventories(
                    archive_blob_sha256, archive_format, member_count, file_count,
                    total_uncompressed_bytes, total_compressed_bytes, policy_json,
                    inventory_sha256, inspected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(archive_blob_sha256) DO UPDATE SET
                    archive_format=excluded.archive_format,
                    member_count=excluded.member_count,
                    file_count=excluded.file_count,
                    total_uncompressed_bytes=excluded.total_uncompressed_bytes,
                    total_compressed_bytes=excluded.total_compressed_bytes,
                    policy_json=excluded.policy_json,
                    inventory_sha256=excluded.inventory_sha256,
                    inspected_at=excluded.inspected_at
                """,
                (
                    archive_blob_sha256,
                    archive_format,
                    member_count,
                    file_count,
                    total_uncompressed_bytes,
                    total_compressed_bytes,
                    json_text(policy or {}),
                    inventory_sha256,
                    utc_now(),
                ),
            )

    def upsert_archive_members(
        self,
        *,
        archive_blob_sha256: str,
        members: list[dict[str, Any]],
    ) -> int:
        """Bulk-index archive metadata without deleting earlier observations."""
        now = utc_now()
        rows = [
            (
                archive_blob_sha256,
                int(member["ordinal"]),
                str(member["member_path"]),
                str(member["normalized_path"]),
                str(member["member_kind"]),
                int(member["size_bytes"]),
                int(member["compressed_bytes"]),
                member.get("crc32"),
                member.get("compression_method"),
                member.get("modified_at"),
                member.get("extracted_blob_sha256"),
                member.get("selected_role"),
                str(member.get("inspection_status", "listed")),
                member.get("error"),
                json_text(member.get("metadata") or {}),
                now,
            )
            for member in members
        ]
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO archive_members(
                    archive_blob_sha256, ordinal, member_path, normalized_path,
                    member_kind, size_bytes, compressed_bytes, crc32,
                    compression_method, modified_at, extracted_blob_sha256,
                    selected_role, inspection_status, error, metadata_json, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(archive_blob_sha256, ordinal) DO UPDATE SET
                    member_path=excluded.member_path,
                    normalized_path=excluded.normalized_path,
                    member_kind=excluded.member_kind,
                    size_bytes=excluded.size_bytes,
                    compressed_bytes=excluded.compressed_bytes,
                    crc32=excluded.crc32,
                    compression_method=excluded.compression_method,
                    modified_at=excluded.modified_at,
                    extracted_blob_sha256=COALESCE(
                        excluded.extracted_blob_sha256,
                        archive_members.extracted_blob_sha256
                    ),
                    selected_role=COALESCE(excluded.selected_role, archive_members.selected_role),
                    inspection_status=excluded.inspection_status,
                    error=excluded.error,
                    metadata_json=excluded.metadata_json,
                    observed_at=excluded.observed_at
                """,
                rows,
            )
        return len(rows)

    def attach_archive_member_blob(
        self,
        *,
        archive_blob_sha256: str,
        ordinal: int,
        extracted_blob_sha256: str,
        selected_role: str,
    ) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE archive_members SET
                    extracted_blob_sha256=?, selected_role=?,
                    inspection_status='extracted', error=NULL, observed_at=?
                WHERE archive_blob_sha256=? AND ordinal=?
                """,
                (
                    extracted_blob_sha256,
                    selected_role,
                    utc_now(),
                    archive_blob_sha256,
                    ordinal,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown archive member {archive_blob_sha256}:{ordinal}")

    def register_archive_extractions(
        self,
        *,
        archive_blob_sha256: str,
        extracted: list[dict[str, Any]],
        selected_role: str,
        mime_type: str | None = None,
    ) -> int:
        """Register many CAS members and archive links in one transaction."""
        if not extracted:
            return 0
        now = utc_now()
        ordinals = [int(row["ordinal"]) for row in extracted]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("Archive extraction contains duplicate ordinals")
        with self.connect() as connection:
            available = {
                int(row[0])
                for row in connection.execute(
                    """
                    SELECT ordinal FROM archive_members
                    WHERE archive_blob_sha256=?
                    """,
                    (archive_blob_sha256,),
                )
            }
            missing = set(ordinals).difference(available)
            if missing:
                raise KeyError(
                    f"Archive {archive_blob_sha256} is missing selected ordinal(s): "
                    f"{sorted(missing)[:10]}"
                )
            connection.executemany(
                """
                INSERT INTO blobs(
                    sha256, size_bytes, mime_type, storage_path, first_seen_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(sha256) DO UPDATE SET
                    mime_type=COALESCE(blobs.mime_type, excluded.mime_type)
                """,
                [
                    (
                        str(row["sha256"]),
                        int(row["size_bytes"]),
                        mime_type,
                        str(row["storage_path"]),
                        now,
                    )
                    for row in extracted
                ],
            )
            connection.executemany(
                """
                UPDATE archive_members SET
                    extracted_blob_sha256=?, selected_role=?,
                    inspection_status='extracted', error=NULL, observed_at=?
                WHERE archive_blob_sha256=? AND ordinal=?
                """,
                [
                    (
                        str(row["sha256"]),
                        selected_role,
                        now,
                        archive_blob_sha256,
                        int(row["ordinal"]),
                    )
                    for row in extracted
                ],
            )
        return len(extracted)

    def mark_archive_member_inspection(
        self,
        *,
        archive_blob_sha256: str,
        ordinal: int,
        status: str,
        error: str | None = None,
    ) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE archive_members SET
                    inspection_status=?, error=?, observed_at=?
                WHERE archive_blob_sha256=? AND ordinal=?
                """,
                (status, error, utc_now(), archive_blob_sha256, ordinal),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown archive member {archive_blob_sha256}:{ordinal}")

    def mark_archive_member_inspections(
        self,
        *,
        archive_blob_sha256: str,
        inspections: list[dict[str, Any]],
    ) -> int:
        if not inspections:
            return 0
        now = utc_now()
        with self.connect() as connection:
            connection.executemany(
                """
                UPDATE archive_members SET
                    inspection_status=?, error=?, observed_at=?
                WHERE archive_blob_sha256=? AND ordinal=?
                """,
                [
                    (
                        str(inspection["status"]),
                        inspection.get("error"),
                        now,
                        archive_blob_sha256,
                        int(inspection["ordinal"]),
                    )
                    for inspection in inspections
                ],
            )
        return len(inspections)

    def record_media_observation(
        self,
        *,
        blob_sha256: str,
        inspector_version: str,
        media_format: str,
        width: int,
        height: int,
        mode: str,
        is_animated: bool,
        frame_count: int,
        has_alpha: bool | None = None,
        loop_count: int | None = None,
        total_duration_ms: float | None = None,
        palette_sha256: str | None = None,
        pixel_sha256: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO media_observations(
                    blob_sha256, inspector_version, media_format, width, height,
                    mode, has_alpha, is_animated, frame_count, loop_count,
                    total_duration_ms, palette_sha256, pixel_sha256,
                    metadata_json, inspected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(blob_sha256, inspector_version) DO UPDATE SET
                    media_format=excluded.media_format,
                    width=excluded.width,
                    height=excluded.height,
                    mode=excluded.mode,
                    has_alpha=excluded.has_alpha,
                    is_animated=excluded.is_animated,
                    frame_count=excluded.frame_count,
                    loop_count=excluded.loop_count,
                    total_duration_ms=excluded.total_duration_ms,
                    palette_sha256=excluded.palette_sha256,
                    pixel_sha256=excluded.pixel_sha256,
                    metadata_json=excluded.metadata_json,
                    inspected_at=excluded.inspected_at
                """,
                (
                    blob_sha256,
                    inspector_version,
                    media_format,
                    width,
                    height,
                    mode,
                    None if has_alpha is None else int(has_alpha),
                    int(is_animated),
                    frame_count,
                    loop_count,
                    total_duration_ms,
                    palette_sha256,
                    pixel_sha256,
                    json_text(metadata or {}),
                    utc_now(),
                ),
            )

    def record_media_observations(self, observations: list[dict[str, Any]]) -> int:
        """Bulk upsert media inspection facts with one SQLite transaction."""
        if not observations:
            return 0
        now = utc_now()
        rows = [
            (
                str(observation["blob_sha256"]),
                str(observation["inspector_version"]),
                str(observation["media_format"]),
                int(observation["width"]),
                int(observation["height"]),
                str(observation["mode"]),
                (
                    None
                    if observation.get("has_alpha") is None
                    else int(bool(observation["has_alpha"]))
                ),
                int(bool(observation["is_animated"])),
                int(observation["frame_count"]),
                observation.get("loop_count"),
                observation.get("total_duration_ms"),
                observation.get("palette_sha256"),
                observation.get("pixel_sha256"),
                json_text(observation.get("metadata") or {}),
                now,
            )
            for observation in observations
        ]
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO media_observations(
                    blob_sha256, inspector_version, media_format, width, height,
                    mode, has_alpha, is_animated, frame_count, loop_count,
                    total_duration_ms, palette_sha256, pixel_sha256,
                    metadata_json, inspected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(blob_sha256, inspector_version) DO UPDATE SET
                    media_format=excluded.media_format,
                    width=excluded.width,
                    height=excluded.height,
                    mode=excluded.mode,
                    has_alpha=excluded.has_alpha,
                    is_animated=excluded.is_animated,
                    frame_count=excluded.frame_count,
                    loop_count=excluded.loop_count,
                    total_duration_ms=excluded.total_duration_ms,
                    palette_sha256=excluded.palette_sha256,
                    pixel_sha256=excluded.pixel_sha256,
                    metadata_json=excluded.metadata_json,
                    inspected_at=excluded.inspected_at
                """,
                rows,
            )
        return len(rows)

    def create_sequence(
        self,
        *,
        extraction_method: str,
        width: int,
        height: int,
        frame_count: int,
        quality_tier: str,
        item_id: str | None = None,
        source_blob_sha256: str | None = None,
        extraction_confidence: float | None = None,
        loop_mode: str | None = None,
        action: str | None = None,
        direction: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        sequence_id = new_id("sequence")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sequences(
                    id, item_id, source_blob_sha256, extraction_method,
                    extraction_confidence, width, height, frame_count, loop_mode,
                    action, direction, quality_tier, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sequence_id,
                    item_id,
                    source_blob_sha256,
                    extraction_method,
                    extraction_confidence,
                    width,
                    height,
                    frame_count,
                    loop_mode,
                    action,
                    direction,
                    quality_tier,
                    json_text(metadata or {}),
                    utc_now(),
                ),
            )
        return sequence_id

    def update_sequence_facts(
        self,
        *,
        sequence_id: str,
        extraction_method: str,
        width: int,
        height: int,
        frame_count: int,
        quality_tier: str,
        source_blob_sha256: str | None = None,
        extraction_confidence: float | None = None,
        loop_mode: str | None = None,
        action: str | None = None,
        direction: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sequences SET
                    source_blob_sha256=?, extraction_method=?, extraction_confidence=?,
                    width=?, height=?, frame_count=?, loop_mode=?, action=?, direction=?,
                    quality_tier=?, metadata_json=?
                WHERE id=?
                """,
                (
                    source_blob_sha256,
                    extraction_method,
                    extraction_confidence,
                    width,
                    height,
                    frame_count,
                    loop_mode,
                    action,
                    direction,
                    quality_tier,
                    json_text(metadata or {}),
                    sequence_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown sequence: {sequence_id}")
            self.record_event(
                "sequence_facts_updated",
                "sequence",
                sequence_id,
                {
                    "extraction_method": extraction_method,
                    "frame_count": frame_count,
                    "loop_mode": loop_mode,
                    "action": action,
                },
                connection=connection,
            )

    def find_sequence_by_source_key(
        self,
        *,
        source_id: str,
        external_sequence_key: str,
    ) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT sequence_id FROM sequence_source_keys
                WHERE source_id=? AND external_sequence_key=?
                """,
                (source_id, external_sequence_key),
            ).fetchone()
        return str(row["sequence_id"]) if row else None

    def register_sequence_source_key(
        self,
        *,
        source_id: str,
        external_sequence_key: str,
        sequence_id: str,
    ) -> None:
        with self.connect() as connection:
            existing = connection.execute(
                """
                SELECT sequence_id FROM sequence_source_keys
                WHERE source_id=? AND external_sequence_key=?
                """,
                (source_id, external_sequence_key),
            ).fetchone()
            if existing and str(existing["sequence_id"]) != sequence_id:
                raise ValueError(
                    f"Sequence key {source_id}:{external_sequence_key} already maps to "
                    f"{existing['sequence_id']}"
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO sequence_source_keys(
                    source_id, external_sequence_key, sequence_id, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (source_id, external_sequence_key, sequence_id, utc_now()),
            )

    def add_frame(
        self,
        *,
        sequence_id: str,
        ordinal: int,
        blob_sha256: str,
        duration_ms: int | None = None,
        bbox: tuple[int, int, int, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO frames(
                    sequence_id, ordinal, blob_sha256, duration_ms,
                    bbox_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(sequence_id, ordinal) DO UPDATE SET
                    blob_sha256=excluded.blob_sha256,
                    duration_ms=excluded.duration_ms,
                    bbox_json=excluded.bbox_json,
                    metadata_json=excluded.metadata_json
                """,
                (
                    sequence_id,
                    ordinal,
                    blob_sha256,
                    duration_ms,
                    json_text(bbox) if bbox is not None else None,
                    json_text(metadata or {}),
                ),
            )

    def link_sequence_occurrence(
        self,
        *,
        sequence_id: str,
        archive_blob_sha256: str,
        archive_member_ordinal: int,
        occurrence_role: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sequence_occurrences(
                    sequence_id, archive_blob_sha256, archive_member_ordinal,
                    occurrence_role, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    sequence_id, archive_blob_sha256, archive_member_ordinal, occurrence_role
                ) DO UPDATE SET metadata_json=excluded.metadata_json
                """,
                (
                    sequence_id,
                    archive_blob_sha256,
                    archive_member_ordinal,
                    occurrence_role,
                    json_text(metadata or {}),
                    utc_now(),
                ),
            )

    def add_sequence_frame(
        self,
        *,
        sequence_id: str,
        ordinal: int,
        source_blob_sha256: str,
        source_frame_index: int | None,
        duration_ms: float | None,
        phase: float | None,
        direction: str | None = None,
        view: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sequence_frames(
                    sequence_id, ordinal, source_blob_sha256, source_frame_index,
                    duration_ms, phase, direction, view, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sequence_id, ordinal) DO UPDATE SET
                    source_blob_sha256=excluded.source_blob_sha256,
                    source_frame_index=excluded.source_frame_index,
                    duration_ms=excluded.duration_ms,
                    phase=excluded.phase,
                    direction=excluded.direction,
                    view=excluded.view,
                    metadata_json=excluded.metadata_json
                """,
                (
                    sequence_id,
                    ordinal,
                    source_blob_sha256,
                    source_frame_index,
                    duration_ms,
                    phase,
                    direction,
                    view,
                    json_text(metadata or {}),
                ),
            )
