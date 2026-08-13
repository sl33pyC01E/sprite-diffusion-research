"""Deterministic export of indexed LPC modular-layer manifests."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from spritelab.adapters.lpc import (
    LpcSheetDefinition,
    classify_lpc_path,
    parse_credits_csv,
    parse_sheet_definition,
)
from spritelab.ingest.lpc import LpcArchiveMemberFact, LpcManifestBuilder


@dataclass(frozen=True, slots=True)
class LpcManifestExport:
    archive_blob_sha256: str
    output_path: Path
    compressed_bytes: int
    compressed_sha256: str
    canonical_jsonl_sha256: str
    record_count: int
    slice_count: int
    cell_count: int
    credit_row_count: int
    definition_count: int
    geometry_counts: dict[str, int]
    credit_match_counts: dict[str, int]


def export_lpc_manifest(
    *,
    database_path: Path | str,
    archive_path: Path | str,
    archive_blob_sha256: str,
    output_path: Path | str,
    overwrite: bool = False,
) -> LpcManifestExport:
    """Stream exact indexed LPC facts into canonical deterministic JSONL.GZ.

    Every line is one ``LpcSheetManifestRecord``. The gzip header has a fixed
    timestamp and no filename, making identical inputs byte-identical. Existing
    outputs are preserved unless replacement is explicitly requested.
    """

    archive_sha256 = _sha256(archive_blob_sha256)
    archive = Path(archive_path).resolve()
    database = Path(database_path).resolve()
    output = Path(output_path).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"LPC archive does not exist: {archive}")
    if not database.is_file():
        raise FileNotFoundError(f"Index database does not exist: {database}")
    if _file_sha256(archive) != archive_sha256:
        raise ValueError("LPC archive bytes do not match archive_blob_sha256")
    if output.exists() and not overwrite:
        raise FileExistsError(f"LPC manifest already exists: {output}")

    credits, definitions, credits_document = _metadata_from_archive(archive)
    builder = LpcManifestBuilder(
        archive_blob_sha256=archive_sha256,
        credits=credits,
        definitions=definitions,
        credits_source_document=credits_document,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    canonical_digest = hashlib.sha256()
    record_count = 0
    slice_count = 0
    cell_count = 0
    geometry_counts: Counter[str] = Counter()
    credit_match_counts: Counter[str] = Counter()
    try:
        with temporary.open("wb") as raw_output:
            with (
                gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=6,
                    fileobj=raw_output,
                    mtime=0,
                ) as compressed,
                _read_connection(database) as connection,
            ):
                for record in builder.iter_records(_member_facts(connection, archive_sha256)):
                    payload = (
                        json.dumps(
                            record.as_dict(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                        + b"\n"
                    )
                    compressed.write(payload)
                    canonical_digest.update(payload)
                    record_count += 1
                    slice_count += len(record.slices)
                    cell_count += sum(len(slice_.cells) for slice_ in record.slices)
                    geometry_counts[record.geometry.status] += 1
                    credit_match_counts[record.credit.match_method] += 1
            raw_output.flush()
            os.fsync(raw_output.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return LpcManifestExport(
        archive_blob_sha256=archive_sha256,
        output_path=output,
        compressed_bytes=output.stat().st_size,
        compressed_sha256=_file_sha256(output),
        canonical_jsonl_sha256=canonical_digest.hexdigest(),
        record_count=record_count,
        slice_count=slice_count,
        cell_count=cell_count,
        credit_row_count=len(credits),
        definition_count=len(definitions),
        geometry_counts=dict(sorted(geometry_counts.items())),
        credit_match_counts=dict(sorted(credit_match_counts.items())),
    )


def _metadata_from_archive(
    archive_path: Path,
) -> tuple[tuple, tuple[LpcSheetDefinition, ...], str]:
    credits_payloads: list[tuple[str, bytes]] = []
    definitions: list[LpcSheetDefinition] = []
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            info = classify_lpc_path(member.filename)
            if info.kind == "credits":
                credits_payloads.append((info.repository_relative_path, archive.read(member)))
            elif info.kind == "sheet_definition":
                definitions.append(
                    parse_sheet_definition(
                        archive.read(member),
                        source_path=info.repository_relative_path,
                    )
                )
    if len(credits_payloads) != 1:
        raise ValueError(f"Expected exactly one LPC CREDITS.csv, found {len(credits_payloads)}")
    credits_document, credits_payload = credits_payloads[0]
    return (
        parse_credits_csv(credits_payload),
        tuple(sorted(definitions, key=lambda item: (item.source_path or "").encode("utf-8"))),
        credits_document,
    )


def _read_connection(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _member_facts(
    connection: sqlite3.Connection,
    archive_sha256: str,
):
    rows = connection.execute(
        """
        WITH ranked_media AS (
            SELECT mo.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY mo.blob_sha256
                       ORDER BY mo.inspected_at DESC, mo.inspector_version DESC
                   ) AS media_rank
            FROM media_observations AS mo
        )
        SELECT am.ordinal, am.normalized_path, am.extracted_blob_sha256,
               am.inspection_status, am.error, rm.width, rm.height,
               rm.pixel_sha256
        FROM archive_members AS am
        LEFT JOIN ranked_media AS rm
          ON rm.blob_sha256=am.extracted_blob_sha256 AND rm.media_rank=1
        WHERE am.archive_blob_sha256=?
        ORDER BY am.ordinal
        """,
        (archive_sha256,),
    )
    found = False
    for row in rows:
        found = True
        yield LpcArchiveMemberFact(
            ordinal=int(row["ordinal"]),
            member_path=str(row["normalized_path"]),
            width=int(row["width"]) if row["width"] is not None else None,
            height=int(row["height"]) if row["height"] is not None else None,
            extracted_blob_sha256=(
                str(row["extracted_blob_sha256"])
                if row["extracted_blob_sha256"] is not None
                else None
            ),
            pixel_sha256=(str(row["pixel_sha256"]) if row["pixel_sha256"] else None),
            inspection_status=str(row["inspection_status"]),
            inspection_error=str(row["error"]) if row["error"] is not None else None,
        )
    if not found:
        raise KeyError(f"No indexed archive members for {archive_sha256}")


def _sha256(value: str) -> str:
    normalized = value.casefold()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"Invalid archive SHA-256: {value!r}")
    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
