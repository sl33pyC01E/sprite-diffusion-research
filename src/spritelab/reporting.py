from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ProvenanceReportPaths:
    sources_csv: Path
    sources_jsonl: Path
    inventory_jsonl: Path
    attribution_markdown: Path
    corpus_summary_json: Path
    bundle_manifest_json: Path


@contextmanager
def _read_connection(database_path: Path | str) -> Iterator[sqlite3.Connection]:
    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Index database does not exist: {path}")
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("BEGIN")
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


def _atomic_write_chunks(output_path: Path | str, chunks: Iterable[str]) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            for chunk in chunks:
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def _json_line(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _decode_json(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {"_invalid_json": True, "_raw": value}


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _source_records(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            s.*,
            (SELECT COUNT(*) FROM items i WHERE i.source_id=s.id) AS item_count,
            (
                SELECT COUNT(*) FROM retrievals r
                JOIN items i ON i.id=r.item_id
                WHERE i.source_id=s.id
            ) AS retrieval_count,
            (
                SELECT COUNT(*) FROM rights_observations ro
                JOIN items i ON i.id=ro.item_id
                WHERE i.source_id=s.id
            ) AS rights_observation_count,
            (
                SELECT COUNT(*) FROM item_blobs ib
                JOIN items i ON i.id=ib.item_id
                WHERE i.source_id=s.id
            ) AS item_blob_occurrence_count,
            (
                SELECT COUNT(DISTINCT ib.blob_sha256) FROM item_blobs ib
                JOIN items i ON i.id=ib.item_id
                WHERE i.source_id=s.id
            ) AS distinct_linked_blob_count,
            COALESCE((
                SELECT SUM(b.size_bytes) FROM blobs b
                WHERE b.sha256 IN (
                    SELECT ib.blob_sha256 FROM item_blobs ib
                    JOIN items i ON i.id=ib.item_id
                    WHERE i.source_id=s.id
                )
            ), 0) AS distinct_linked_blob_bytes,
            (
                SELECT COUNT(*) FROM entities e WHERE e.source_id=s.id
            ) AS entity_count,
            (
                SELECT COUNT(*) FROM sequence_source_keys ssk WHERE ssk.source_id=s.id
            ) AS sequence_count,
            (
                SELECT COUNT(*) FROM sequence_frames sf
                JOIN sequence_source_keys ssk ON ssk.sequence_id=sf.sequence_id
                WHERE ssk.source_id=s.id
            ) AS sequence_frame_count,
            (
                SELECT COUNT(*) FROM sequence_occurrences so
                JOIN sequence_source_keys ssk ON ssk.sequence_id=so.sequence_id
                WHERE ssk.source_id=s.id
            ) AS sequence_occurrence_count,
            (
                SELECT COUNT(*) FROM sequence_subjects ss
                JOIN sequence_source_keys ssk ON ssk.sequence_id=ss.sequence_id
                WHERE ssk.source_id=s.id
            ) AS sequence_subject_count
        FROM sources s
        ORDER BY s.id COLLATE BINARY
        """
    ).fetchall()
    return [
        {
            "id": row["id"],
            "kind": row["kind"],
            "name": row["name"],
            "root_url": row["root_url"],
            "adapter_version": row["adapter_version"],
            "config": _decode_json(row["config_json"]),
            "created_at": row["created_at"],
            "item_count": row["item_count"],
            "retrieval_count": row["retrieval_count"],
            "rights_observation_count": row["rights_observation_count"],
            "item_blob_occurrence_count": row["item_blob_occurrence_count"],
            "distinct_linked_blob_count": row["distinct_linked_blob_count"],
            "distinct_linked_blob_bytes": row["distinct_linked_blob_bytes"],
            "entity_count": row["entity_count"],
            "sequence_count": row["sequence_count"],
            "sequence_frame_count": row["sequence_frame_count"],
            "sequence_occurrence_count": row["sequence_occurrence_count"],
            "sequence_subject_count": row["sequence_subject_count"],
            "rights_scope": "item_observations_only; source metadata is not inherited",
        }
        for row in rows
    ]


def _sources_csv_text(connection: sqlite3.Connection) -> str:
    columns = (
        "id",
        "kind",
        "name",
        "root_url",
        "adapter_version",
        "config_json",
        "created_at",
        "item_count",
        "retrieval_count",
        "rights_observation_count",
        "item_blob_occurrence_count",
        "distinct_linked_blob_count",
        "distinct_linked_blob_bytes",
        "entity_count",
        "sequence_count",
        "sequence_frame_count",
        "sequence_occurrence_count",
        "sequence_subject_count",
        "rights_scope",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for source in _source_records(connection):
        row = dict(source)
        row["config_json"] = json.dumps(
            row.pop("config"), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        writer.writerow(row)
    return output.getvalue()


def export_sources_csv(database_path: Path | str, output_path: Path | str) -> Path:
    with _read_connection(database_path) as connection:
        contents = _sources_csv_text(connection)
    return _atomic_write_chunks(output_path, (contents,))


def export_sources_jsonl(database_path: Path | str, output_path: Path | str) -> Path:
    with _read_connection(database_path) as connection:
        contents = tuple(_json_line(record) for record in _source_records(connection))
    return _atomic_write_chunks(output_path, contents)


def _license_label(row: sqlite3.Row | Mapping[str, Any]) -> str | None:
    for field in ("license_expression", "license_raw"):
        value = row[field]
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _rights_summaries(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id, item_id, observed_at, license_raw, license_expression
        FROM rights_observations
        ORDER BY item_id COLLATE BINARY, observed_at COLLATE BINARY, id COLLATE BINARY
        """
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["item_id"])].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    item_ids = (
        str(row["id"])
        for row in connection.execute("SELECT id FROM items ORDER BY id COLLATE BINARY")
    )
    for item_id in item_ids:
        observations = grouped.get(item_id, [])
        labels = sorted(
            {label for observation in observations if (label := _license_label(observation))},
            key=lambda value: value.encode("utf-8"),
        )
        unknown_count = sum(_license_label(observation) is None for observation in observations)
        if not observations:
            license_state = "unknown_no_observation"
        elif not labels:
            license_state = "unknown_observed"
        elif len(labels) > 1:
            license_state = "multiple_observed_licenses"
        elif unknown_count:
            license_state = "mixed_known_and_unknown_observations"
        else:
            license_state = "single_observed_license"
        summaries[item_id] = {
            "scope": "item_level_observations_only",
            "observation_state": (
                "none" if not observations else "single" if len(observations) == 1 else "multiple"
            ),
            "license_state": license_state,
            "observation_count": len(observations),
            "unknown_license_observation_count": unknown_count,
            "distinct_observed_license_count": len(labels),
            "observed_license_labels": labels,
            "resolution": "not_resolved_or_inferred",
        }
    return summaries


def _snapshot_reference(sha256: str | None, storage_path: str | None) -> dict[str, Any] | None:
    if sha256 is None:
        return None
    return {"sha256": sha256, "storage_path": storage_path}


def _inventory_records(connection: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    rights_summaries = _rights_summaries(connection)
    for row in connection.execute(
        """
        SELECT i.*, s.kind AS source_kind, s.name AS source_name, s.root_url AS source_root_url
        FROM items i
        JOIN sources s ON s.id=i.source_id
        ORDER BY i.source_id COLLATE BINARY, i.external_id COLLATE BINARY, i.id COLLATE BINARY
        """
    ):
        yield {
            "record_type": "item",
            "id": row["id"],
            "source_id": row["source_id"],
            "external_id": row["external_id"],
            "canonical_url": row["canonical_url"],
            "title": row["title"],
            "description": row["description"],
            "creator_name": row["creator_name"],
            "creator_url": row["creator_url"],
            "published_at": row["published_at"],
            "metadata": _decode_json(row["metadata_json"]),
            "first_seen_at": row["first_seen_at"],
            "last_seen_at": row["last_seen_at"],
            "tombstoned_at": row["tombstoned_at"],
            "source": {
                "id": row["source_id"],
                "kind": row["source_kind"],
                "name": row["source_name"],
                "root_url": row["source_root_url"],
            },
            "rights_summary": rights_summaries[str(row["id"])],
            "provenance_links": {
                "canonical_url": row["canonical_url"],
                "creator_url": row["creator_url"],
                "source_root_url": row["source_root_url"],
            },
        }

    if _table_exists(connection, "archive_inventories"):
        for row in connection.execute(
            "SELECT * FROM archive_inventories ORDER BY archive_blob_sha256 COLLATE BINARY"
        ):
            yield {
                "record_type": "archive_inventory",
                "archive_blob_sha256": row["archive_blob_sha256"],
                "archive_format": row["archive_format"],
                "member_count": row["member_count"],
                "file_count": row["file_count"],
                "total_uncompressed_bytes": row["total_uncompressed_bytes"],
                "total_compressed_bytes": row["total_compressed_bytes"],
                "policy": _decode_json(row["policy_json"]),
                "inventory_sha256": row["inventory_sha256"],
                "inspected_at": row["inspected_at"],
            }

    if _table_exists(connection, "archive_members"):
        for row in connection.execute(
            """
            SELECT am.*, b.storage_path AS extracted_storage_path,
                   b.mime_type AS extracted_mime_type
            FROM archive_members am
            LEFT JOIN blobs b ON b.sha256=am.extracted_blob_sha256
            ORDER BY am.archive_blob_sha256 COLLATE BINARY, am.ordinal
            """
        ):
            yield {
                "record_type": "archive_member",
                "archive_blob_sha256": row["archive_blob_sha256"],
                "ordinal": row["ordinal"],
                "member_path": row["member_path"],
                "normalized_path": row["normalized_path"],
                "member_kind": row["member_kind"],
                "size_bytes": row["size_bytes"],
                "compressed_bytes": row["compressed_bytes"],
                "crc32": row["crc32"],
                "compression_method": row["compression_method"],
                "modified_at": row["modified_at"],
                "extracted_blob": _snapshot_reference(
                    row["extracted_blob_sha256"], row["extracted_storage_path"]
                ),
                "extracted_mime_type": row["extracted_mime_type"],
                "selected_role": row["selected_role"],
                "inspection_status": row["inspection_status"],
                "error": row["error"],
                "metadata": _decode_json(row["metadata_json"]),
                "observed_at": row["observed_at"],
            }

    if _table_exists(connection, "media_observations"):
        for row in connection.execute(
            """
            SELECT mo.*, b.storage_path, b.size_bytes, b.mime_type
            FROM media_observations mo
            JOIN blobs b ON b.sha256=mo.blob_sha256
            ORDER BY mo.blob_sha256 COLLATE BINARY, mo.inspector_version COLLATE BINARY
            """
        ):
            yield {
                "record_type": "media_observation",
                "blob": {
                    "sha256": row["blob_sha256"],
                    "storage_path": row["storage_path"],
                    "size_bytes": row["size_bytes"],
                    "mime_type": row["mime_type"],
                },
                "inspector_version": row["inspector_version"],
                "media_format": row["media_format"],
                "width": row["width"],
                "height": row["height"],
                "mode": row["mode"],
                "has_alpha": (None if row["has_alpha"] is None else bool(row["has_alpha"])),
                "is_animated": bool(row["is_animated"]),
                "frame_count": row["frame_count"],
                "loop_count": row["loop_count"],
                "total_duration_ms": row["total_duration_ms"],
                "palette_sha256": row["palette_sha256"],
                "pixel_sha256": row["pixel_sha256"],
                "metadata": _decode_json(row["metadata_json"]),
                "inspected_at": row["inspected_at"],
            }

    for row in connection.execute("SELECT * FROM blobs ORDER BY sha256 COLLATE BINARY"):
        yield {
            "record_type": "blob",
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "mime_type": row["mime_type"],
            "storage_path": row["storage_path"],
            "first_seen_at": row["first_seen_at"],
            "provenance_key": f"sha256:{row['sha256']}",
        }

    for row in connection.execute(
        """
        SELECT
            ib.*, b.size_bytes, b.mime_type, b.storage_path,
            i.source_id, i.external_id, i.canonical_url, s.root_url AS source_root_url
        FROM item_blobs ib
        JOIN items i ON i.id=ib.item_id
        JOIN sources s ON s.id=i.source_id
        JOIN blobs b ON b.sha256=ib.blob_sha256
        ORDER BY ib.id COLLATE BINARY
        """
    ):
        yield {
            "record_type": "item_blob",
            "id": row["id"],
            "item_id": row["item_id"],
            "source_id": row["source_id"],
            "external_id": row["external_id"],
            "blob": {
                "sha256": row["blob_sha256"],
                "size_bytes": row["size_bytes"],
                "mime_type": row["mime_type"],
                "storage_path": row["storage_path"],
            },
            "role": row["role"],
            "original_url": row["original_url"],
            "original_filename": row["original_filename"],
            "archive_member": row["archive_member"],
            "observed_at": row["observed_at"],
            "provenance_links": {
                "item_canonical_url": row["canonical_url"],
                "original_url": row["original_url"],
                "source_root_url": row["source_root_url"],
            },
        }

    for row in connection.execute(
        """
        SELECT
            r.*, b.size_bytes AS blob_size_bytes, b.mime_type AS blob_registered_mime_type,
            b.storage_path AS blob_storage_path, i.source_id, i.external_id,
            i.canonical_url, s.root_url AS source_root_url
        FROM retrievals r
        LEFT JOIN items i ON i.id=r.item_id
        LEFT JOIN sources s ON s.id=i.source_id
        LEFT JOIN blobs b ON b.sha256=r.blob_sha256
        ORDER BY r.id COLLATE BINARY
        """
    ):
        yield {
            "record_type": "retrieval",
            "id": row["id"],
            "run_id": row["run_id"],
            "item_id": row["item_id"],
            "source_id": row["source_id"],
            "external_id": row["external_id"],
            "url": row["url"],
            "requested_at": row["requested_at"],
            "completed_at": row["completed_at"],
            "status_code": row["status_code"],
            "etag": row["etag"],
            "last_modified": row["last_modified"],
            "mime_type": row["mime_type"],
            "content_length": row["content_length"],
            "blob": _snapshot_reference(row["blob_sha256"], row["blob_storage_path"]),
            "registered_blob_size_bytes": row["blob_size_bytes"],
            "registered_blob_mime_type": row["blob_registered_mime_type"],
            "error": row["error"],
            "provenance_links": {
                "item_canonical_url": row["canonical_url"],
                "retrieval_url": row["url"],
                "source_root_url": row["source_root_url"],
            },
        }

    for row in connection.execute(
        """
        SELECT
            ro.*, i.source_id, i.external_id, i.canonical_url,
            s.root_url AS source_root_url,
            tb.storage_path AS terms_blob_storage_path,
            rb.storage_path AS robots_blob_storage_path
        FROM rights_observations ro
        JOIN items i ON i.id=ro.item_id
        JOIN sources s ON s.id=i.source_id
        LEFT JOIN blobs tb ON tb.sha256=ro.terms_blob_sha256
        LEFT JOIN blobs rb ON rb.sha256=ro.robots_blob_sha256
        ORDER BY ro.item_id COLLATE BINARY, ro.observed_at COLLATE BINARY, ro.id COLLATE BINARY
        """
    ):
        yield {
            "record_type": "rights_observation",
            "id": row["id"],
            "item_id": row["item_id"],
            "source_id": row["source_id"],
            "external_id": row["external_id"],
            "observed_at": row["observed_at"],
            "license_raw": row["license_raw"],
            "license_expression": row["license_expression"],
            "observed_license_label": _license_label(row),
            "license_known": _license_label(row) is not None,
            "license_url": row["license_url"],
            "attribution_raw": row["attribution_raw"],
            "terms_url": row["terms_url"],
            "terms_snapshot": _snapshot_reference(
                row["terms_blob_sha256"], row["terms_blob_storage_path"]
            ),
            "robots_url": row["robots_url"],
            "robots_snapshot": _snapshot_reference(
                row["robots_blob_sha256"], row["robots_blob_storage_path"]
            ),
            "basis": row["basis"],
            "metadata": _decode_json(row["metadata_json"]),
            "scope": "observation_for_this_item_only",
            "resolution": "preserved_without_inference",
            "provenance_links": {
                "item_canonical_url": row["canonical_url"],
                "license_url": row["license_url"],
                "robots_url": row["robots_url"],
                "source_root_url": row["source_root_url"],
                "terms_url": row["terms_url"],
            },
        }


def export_inventory_jsonl(database_path: Path | str, output_path: Path | str) -> Path:
    with _read_connection(database_path) as connection:
        return _atomic_write_chunks(
            output_path, (_json_line(record) for record in _inventory_records(connection))
        )


def _md_text(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    for character in ("\\", "`", "*", "_", "[", "]", "#", "<", ">"):
        text = text.replace(character, f"\\{character}")
    return text.replace("\n", "<br>")


def _md_code(value: Any) -> str:
    return f"`{str(value).replace('`', "'")}`"


def _md_url(value: str | None) -> str:
    if value is None or not value.strip():
        return "unknown"
    url = value.strip()
    if url.startswith(("https://", "http://")) and not any(
        character.isspace() or character in "<>" for character in url
    ):
        return f"<{url}>"
    return _md_code(url)


def _attribution_chunks(connection: sqlite3.Connection) -> Iterator[str]:
    yield "# Sprite corpus provenance and attribution index\n\n"
    yield (
        "This report reproduces source, retrieval, and item-level rights evidence. "
        "A source registry entry is not treated as a license for every asset. "
        "Unknown and conflicting observations remain unresolved.\n\n"
    )
    rights_summaries = _rights_summaries(connection)
    sources = connection.execute("SELECT * FROM sources ORDER BY id COLLATE BINARY").fetchall()
    for source in sources:
        yield f"## {_md_text(source['name'])}\n\n"
        yield f"- Source ID: {_md_code(source['id'])}\n"
        yield f"- Kind: {_md_text(source['kind'])}\n"
        yield f"- Root: {_md_url(source['root_url'])}\n"
        yield "- Rights scope: item observations only; no source-wide inheritance.\n\n"
        items = connection.execute(
            """
            SELECT * FROM items
            WHERE source_id=?
            ORDER BY external_id COLLATE BINARY, id COLLATE BINARY
            """,
            (source["id"],),
        ).fetchall()
        if not items:
            yield "_No indexed items._\n\n"
            continue
        for item in items:
            title = item["title"] or item["external_id"]
            yield f"### {_md_text(title)}\n\n"
            yield f"- Item ID: {_md_code(item['id'])}\n"
            yield f"- External ID: {_md_code(item['external_id'])}\n"
            yield f"- Canonical page: {_md_url(item['canonical_url'])}\n"
            if item["creator_name"] or item["creator_url"]:
                yield f"- Creator: {_md_text(item['creator_name'])}"
                if item["creator_url"]:
                    yield f" ({_md_url(item['creator_url'])})"
                yield "\n"
            summary = rights_summaries[str(item["id"])]
            yield f"- Rights state: {_md_code(summary['license_state'])}\n"
            if summary["observation_state"] == "multiple":
                yield (
                    "- Evidence notice: multiple observations are preserved separately; "
                    "no conflict resolution is inferred.\n"
                )
            observations = connection.execute(
                """
                SELECT * FROM rights_observations
                WHERE item_id=?
                ORDER BY observed_at COLLATE BINARY, id COLLATE BINARY
                """,
                (item["id"],),
            ).fetchall()
            if not observations:
                yield "- License evidence: unknown (no item-level observation recorded).\n"
            for ordinal, observation in enumerate(observations, start=1):
                yield (
                    f"- Rights observation {ordinal}: {_md_code(observation['id'])} "
                    f"at {_md_text(observation['observed_at'])}\n"
                )
                yield (f"  - License expression: {_md_text(observation['license_expression'])}\n")
                yield f"  - License raw: {_md_text(observation['license_raw'])}\n"
                yield f"  - License URL: {_md_url(observation['license_url'])}\n"
                yield f"  - Attribution raw: {_md_text(observation['attribution_raw'])}\n"
                yield f"  - Basis: {_md_text(observation['basis'])}\n"
                yield f"  - Terms: {_md_url(observation['terms_url'])}\n"
                if observation["terms_blob_sha256"]:
                    yield (
                        "  - Terms snapshot SHA-256: "
                        f"{_md_code(observation['terms_blob_sha256'])}\n"
                    )
                yield f"  - Robots: {_md_url(observation['robots_url'])}\n"
                if observation["robots_blob_sha256"]:
                    yield (
                        "  - Robots snapshot SHA-256: "
                        f"{_md_code(observation['robots_blob_sha256'])}\n"
                    )
            blobs = connection.execute(
                """
                SELECT ib.*, b.size_bytes, b.mime_type, b.storage_path
                FROM item_blobs ib
                JOIN blobs b ON b.sha256=ib.blob_sha256
                WHERE ib.item_id=?
                ORDER BY ib.id COLLATE BINARY
                """,
                (item["id"],),
            ).fetchall()
            if blobs:
                yield "- Indexed asset occurrences:\n"
                for blob in blobs:
                    yield (
                        f"  - {_md_text(blob['role'])}: SHA-256 {_md_code(blob['blob_sha256'])}; "
                        f"{blob['size_bytes']} bytes; stored at {_md_code(blob['storage_path'])}"
                    )
                    if blob["original_url"]:
                        yield f"; origin {_md_url(blob['original_url'])}"
                    if blob["archive_member"]:
                        yield f"; archive member {_md_code(blob['archive_member'])}"
                    yield "\n"
            yield "\n"


def export_attribution_markdown(database_path: Path | str, output_path: Path | str) -> Path:
    with _read_connection(database_path) as connection:
        return _atomic_write_chunks(output_path, _attribution_chunks(connection))


def _distribution(
    connection: sqlite3.Connection, query: str, parameters: tuple[Any, ...] = ()
) -> dict[str, int]:
    return {str(row[0]): int(row[1]) for row in connection.execute(query, parameters)}


def _corpus_summary(connection: sqlite3.Connection) -> dict[str, Any]:
    count_tables = (
        "sources",
        "crawl_runs",
        "items",
        "blobs",
        "retrievals",
        "item_blobs",
        "rights_observations",
        "derivations",
        "entities",
        "sequences",
        "sequence_frames",
        "sequence_occurrences",
        "sequence_subjects",
        "sequence_source_keys",
        "frames",
        "captions",
        "dataset_snapshots",
        "dataset_members",
        "motion_annotations",
        "archive_inventories",
        "archive_members",
        "media_observations",
        "duplicate_edges",
    )
    counts = {
        table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in count_tables
        if _table_exists(connection, table)
    }
    total_blob_bytes = int(
        connection.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM blobs").fetchone()[0]
    )
    rights_summaries = _rights_summaries(connection)
    license_states = Counter(str(summary["license_state"]) for summary in rights_summaries.values())
    observation_states = Counter(
        str(summary["observation_state"]) for summary in rights_summaries.values()
    )
    observed_license_labels: Counter[str] = Counter()
    unknown_observations = 0
    for row in connection.execute(
        """
        SELECT license_expression, license_raw
        FROM rights_observations
        ORDER BY item_id COLLATE BINARY, observed_at COLLATE BINARY, id COLLATE BINARY
        """
    ):
        label = _license_label(row)
        if label is None:
            unknown_observations += 1
        else:
            observed_license_labels[label] += 1

    source_breakdown = []
    for source in _source_records(connection):
        source_breakdown.append(
            {
                key: source[key]
                for key in (
                    "id",
                    "name",
                    "kind",
                    "item_count",
                    "retrieval_count",
                    "rights_observation_count",
                    "item_blob_occurrence_count",
                    "distinct_linked_blob_count",
                    "distinct_linked_blob_bytes",
                    "entity_count",
                    "sequence_count",
                    "sequence_frame_count",
                    "sequence_occurrence_count",
                    "sequence_subject_count",
                )
            }
        )

    datasets = []
    if _table_exists(connection, "dataset_snapshots"):
        for row in connection.execute("SELECT * FROM dataset_snapshots ORDER BY id COLLATE BINARY"):
            split_counts = _distribution(
                connection,
                """
                SELECT split, COUNT(*) FROM dataset_members
                WHERE snapshot_id=? GROUP BY split ORDER BY split COLLATE BINARY
                """,
                (row["id"],),
            )
            datasets.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "manifest_sha256": row["manifest_sha256"],
                    "parameters": _decode_json(row["parameters_json"]),
                    "code_version": row["code_version"],
                    "created_at": row["created_at"],
                    "member_count": sum(split_counts.values()),
                    "split_counts": split_counts,
                }
            )

    entity_classes: dict[str, int] = {}
    if _table_exists(connection, "entities"):
        entity_classes = _distribution(
            connection,
            """
            SELECT entity_class, COUNT(*) FROM entities
            GROUP BY entity_class ORDER BY entity_class COLLATE BINARY
            """,
        )
    normalized_actions: dict[str, int] = {}
    action_families: dict[str, int] = {}
    if _table_exists(connection, "motion_annotations"):
        normalized_actions = _distribution(
            connection,
            """
            SELECT normalized_action, COUNT(*) FROM motion_annotations
            GROUP BY normalized_action ORDER BY normalized_action COLLATE BINARY
            """,
        )
        action_families = _distribution(
            connection,
            """
            SELECT action_family, COUNT(*) FROM motion_annotations
            GROUP BY action_family ORDER BY action_family COLLATE BINARY
            """,
        )

    archives: dict[str, Any] = {}
    if _table_exists(connection, "archive_inventories"):
        archives = {
            "formats": _distribution(
                connection,
                """
                SELECT archive_format, COUNT(*) FROM archive_inventories
                GROUP BY archive_format ORDER BY archive_format COLLATE BINARY
                """,
            ),
            "member_statuses": _distribution(
                connection,
                """
                SELECT inspection_status, COUNT(*) FROM archive_members
                GROUP BY inspection_status ORDER BY inspection_status COLLATE BINARY
                """,
            ),
            "total_declared_members": int(
                connection.execute(
                    "SELECT COALESCE(SUM(member_count), 0) FROM archive_inventories"
                ).fetchone()[0]
            ),
            "total_declared_uncompressed_bytes": int(
                connection.execute(
                    """
                    SELECT COALESCE(SUM(total_uncompressed_bytes), 0)
                    FROM archive_inventories
                    """
                ).fetchone()[0]
            ),
        }

    media: dict[str, Any] = {}
    if _table_exists(connection, "media_observations"):
        media = {
            "formats": _distribution(
                connection,
                """
                SELECT media_format, COUNT(*) FROM media_observations
                GROUP BY media_format ORDER BY media_format COLLATE BINARY
                """,
            ),
            "animated": _distribution(
                connection,
                """
                SELECT CAST(is_animated AS TEXT), COUNT(*) FROM media_observations
                GROUP BY is_animated ORDER BY is_animated
                """,
            ),
            "dimensions": _distribution(
                connection,
                """
                SELECT CAST(width AS TEXT) || 'x' || CAST(height AS TEXT), COUNT(*)
                FROM media_observations GROUP BY width, height
                ORDER BY COUNT(*) DESC, width, height
                """,
            ),
            "distinct_pixel_hashes": int(
                connection.execute(
                    """
                    SELECT COUNT(DISTINCT pixel_sha256) FROM media_observations
                    WHERE pixel_sha256 IS NOT NULL
                    """
                ).fetchone()[0]
            ),
        }

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "provenance_policy": {
            "license_resolution": "none; observations are counted independently",
            "rights_scope": "item-level observations only",
            "source_license_inheritance": False,
            "unknowns_preserved": True,
        },
        "corpus": {
            "counts": counts,
            "total_blob_bytes": total_blob_bytes,
            "sources": source_breakdown,
        },
        "rights": {
            "item_license_states": dict(sorted(license_states.items())),
            "item_observation_states": dict(sorted(observation_states.items())),
            "observed_license_labels": dict(sorted(observed_license_labels.items())),
            "unknown_license_observation_count": unknown_observations,
        },
        "retrievals": {
            "status_code_counts": _distribution(
                connection,
                """
                SELECT COALESCE(CAST(status_code AS TEXT), 'none'), COUNT(*)
                FROM retrievals
                GROUP BY status_code
                ORDER BY COALESCE(CAST(status_code AS TEXT), 'none') COLLATE BINARY
                """,
            ),
            "error_count": int(
                connection.execute(
                    "SELECT COUNT(*) FROM retrievals WHERE error IS NOT NULL"
                ).fetchone()[0]
            ),
        },
        "entity_classes": entity_classes,
        "motion": {
            "normalized_actions": normalized_actions,
            "action_families": action_families,
        },
        "archives": archives,
        "media": media,
        "datasets": datasets,
    }


def export_corpus_summary_json(database_path: Path | str, output_path: Path | str) -> Path:
    with _read_connection(database_path) as connection:
        summary = _corpus_summary(connection)
    contents = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return _atomic_write_chunks(output_path, (contents,))


def export_provenance_reports(
    database_path: Path | str, output_directory: Path | str
) -> ProvenanceReportPaths:
    output_root = Path(output_directory).resolve()
    paths = ProvenanceReportPaths(
        sources_csv=output_root / "sources.csv",
        sources_jsonl=output_root / "sources.jsonl",
        inventory_jsonl=output_root / "inventory.jsonl",
        attribution_markdown=output_root / "ATTRIBUTION.md",
        corpus_summary_json=output_root / "corpus_summary.json",
        bundle_manifest_json=output_root / "bundle-manifest.json",
    )
    with _read_connection(database_path) as connection:
        _atomic_write_chunks(paths.sources_csv, (_sources_csv_text(connection),))
        _atomic_write_chunks(
            paths.sources_jsonl,
            (_json_line(record) for record in _source_records(connection)),
        )
        _atomic_write_chunks(
            paths.inventory_jsonl,
            (_json_line(record) for record in _inventory_records(connection)),
        )
        _atomic_write_chunks(paths.attribution_markdown, _attribution_chunks(connection))
        summary = (
            json.dumps(_corpus_summary(connection), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        _atomic_write_chunks(paths.corpus_summary_json, (summary,))
        bundle_files = (
            ("ATTRIBUTION.md", paths.attribution_markdown, "text/markdown"),
            ("corpus_summary.json", paths.corpus_summary_json, "application/json"),
            ("inventory.jsonl", paths.inventory_jsonl, "application/x-ndjson"),
            ("sources.csv", paths.sources_csv, "text/csv"),
            ("sources.jsonl", paths.sources_jsonl, "application/x-ndjson"),
        )
        schema_versions = tuple(
            row[0]
            for row in connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version COLLATE BINARY"
            )
        )
        bundle_manifest = {
            "artifact_kind": "spritelab_provenance_report_bundle",
            "database_schema_versions": schema_versions,
            "files": [
                {
                    "media_type": media_type,
                    "relative_path": relative_path,
                    "sha256": _sha256_path(path),
                    "size_bytes": path.stat().st_size,
                }
                for relative_path, path, media_type in bundle_files
            ],
            "read_consistency": "single_sqlite_query_only_read_transaction",
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "schema_version": 1,
        }
        bundle_payload = (
            json.dumps(bundle_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        _atomic_write_chunks(paths.bundle_manifest_json, (bundle_payload,))
    return paths
