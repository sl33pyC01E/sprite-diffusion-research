from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from spritelab.adapters.github import GitHubRepositoryAdapter
from spritelab.archive import ArchiveLimits, extract_zip_to_cas, inspect_zip
from spritelab.config import GIB, load_config
from spritelab.dataset import SplitPolicy, SplitRatios
from spritelab.db import IndexDB
from spritelab.fetch import HttpFetcher
from spritelab.indexing import (
    index_zip_extraction,
    index_zip_manifest,
    inspect_and_index_media,
    inspect_media_observation,
)
from spritelab.ingest.flare import ingest_known_flare_sequences
from spritelab.ingest.freedoom import ingest_freedoom_sequences
from spritelab.ingest.opensurge import ingest_known_open_surge_sequences
from spritelab.ingest.shattered_pixel_dungeon import (
    ingest_known_shattered_pixel_dungeon_sequences,
)
from spritelab.ingest.spritecook import ingest_spritecook_sequences
from spritelab.ingest.ss14 import ingest_known_ss14_sequences
from spritelab.ingest.tmwa import ingest_known_tmwa_sequences
from spritelab.ingest.wesnoth import ingest_known_wesnoth_sequences
from spritelab.ingest.widelands import ingest_known_widelands_sequences
from spritelab.lpc_snapshot import export_lpc_manifest
from spritelab.materialize import materialize_snapshot
from spritelab.reporting import export_provenance_reports
from spritelab.snapshot import SnapshotFilters, export_snapshot
from spritelab.sources import load_source_registry, sync_source_registry
from spritelab.storage import ContentAddressedStore, DiskGuard
from spritelab.taxonomy import load_taxonomy


def build_runtime() -> tuple[ContentAddressedStore, IndexDB, DiskGuard]:
    config = load_config()
    config.storage.data_root.mkdir(parents=True, exist_ok=True)
    guard = DiskGuard(config.storage.data_root, config.storage.min_free_bytes)
    store = ContentAddressedStore(config.storage.data_root, guard)
    database = IndexDB(config.index.database)
    return store, database, guard


def command_init(_args: argparse.Namespace) -> int:
    store, database, guard = build_runtime()
    guard.require_capacity(label="project initialization")
    store.initialize()
    database.initialize()
    print(f"Initialized object store: {store.objects_root}")
    print(f"Initialized provenance index: {database.path}")
    return 0


def command_status(args: argparse.Namespace) -> int:
    store, database, guard = build_runtime()
    status = guard.status()
    payload = {
        "data_root": str(store.data_root),
        "database": str(database.path),
        "free_gib": round(status.free_bytes / GIB, 3),
        "floor_gib": round(status.floor_bytes / GIB, 3),
        "writable_budget_gib": round(status.writable_budget_bytes / GIB, 3),
        "counts": database.counts(),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def command_ingest_file(args: argparse.Namespace) -> int:
    store, database, _guard = build_runtime()
    database.initialize()
    blob = store.put_file(Path(args.path).expanduser().resolve())
    with database.connect() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO blobs(sha256, size_bytes, mime_type, storage_path, first_seen_at)
            VALUES (?, ?, NULL, ?, datetime('now'))
            """,
            (blob.sha256, blob.size_bytes, str(blob.path)),
        )
    print(json.dumps({"sha256": blob.sha256, "bytes": blob.size_bytes, "path": str(blob.path)}))
    return 0


def command_sources_sync(args: argparse.Namespace) -> int:
    _store, database, _guard = build_runtime()
    config = load_config()
    path = (
        Path(args.registry).resolve()
        if args.registry
        else config.project_root / "configs" / "sources.toml"
    )
    registry = load_source_registry(path)
    count = sync_source_registry(database, registry)
    print(f"Registered {count} sources from {path}")
    return 0


def command_sources_list(args: argparse.Namespace) -> int:
    config = load_config()
    path = (
        Path(args.registry).resolve()
        if args.registry
        else config.project_root / "configs" / "sources.toml"
    )
    registry = load_source_registry(path)
    rows = [
        {
            "id": source.id,
            "name": source.name,
            "adapter": source.adapter,
            "mode": source.acquisition_mode,
            "tier": source.quality_tier,
        }
        for source in registry.sources
    ]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            print(f"{row['id']}: {row['name']} [{row['tier']}; {row['mode']}]")
    return 0


def command_acquire_github(args: argparse.Namespace) -> int:
    store, database, guard = build_runtime()
    config = load_config()
    registry = load_source_registry(config.project_root / "configs" / "sources.toml")
    sync_source_registry(database, registry)
    source = registry.by_id(args.source_id)
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(follow_redirects=True, timeout=config.network.timeout_seconds) as client:
        fetcher = HttpFetcher(
            store,
            user_agent=config.network.user_agent,
            timeout_seconds=config.network.timeout_seconds,
            max_retries=config.network.max_retries,
            chunk_bytes=config.storage.download_chunk_bytes,
            client=client,
            extra_headers=headers,
        )
        snapshot = GitHubRepositoryAdapter(
            database=database,
            fetcher=fetcher,
            guard=guard,
        ).acquire(
            source,
            download_archive=not args.metadata_only,
            ref=args.ref,
        )
    print(
        json.dumps(
            {
                "source_id": snapshot.source_id,
                "repository": snapshot.repository,
                "commit_sha": snapshot.commit_sha,
                "archive_sha256": snapshot.archive_blob_sha256,
                "archive_path": str(snapshot.archive_path) if snapshot.archive_path else None,
                "free_gib": round(guard.status().free_bytes / GIB, 3),
            },
            indent=2,
        )
    )
    return 0


def command_archive_index(args: argparse.Namespace) -> int:
    store, database, guard = build_runtime()
    archive_sha256 = args.sha256.lower()
    archive_path = store.object_path(archive_sha256)
    if not archive_path.is_file():
        raise FileNotFoundError(f"CAS object does not exist: {archive_path}")
    limits = ArchiveLimits(
        max_members=args.max_members,
        max_member_bytes=args.max_member_gib * GIB,
        max_total_expanded_bytes=args.max_total_gib * GIB,
        max_compression_ratio=args.max_ratio,
        allow_symlink_metadata=args.allow_symlink_metadata,
    )
    guard.require_capacity(label="archive metadata indexing")
    manifest = inspect_zip(archive_path, limits=limits)
    count = index_zip_manifest(
        database,
        archive_blob_sha256=archive_sha256,
        manifest=manifest,
        limits=limits,
    )
    print(
        json.dumps(
            {
                "archive_sha256": archive_sha256,
                "inventory_sha256": manifest.inventory_sha256,
                "members": count,
                "files": manifest.regular_file_count,
                "symlinks": manifest.symlink_count,
                "uncompressed_bytes": manifest.total_uncompressed_bytes,
                "extension_counts": manifest.extension_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_archive_extract_media(args: argparse.Namespace) -> int:
    store, database, _guard = build_runtime()
    archive_sha256 = args.sha256.lower()
    archive_path = store.object_path(archive_sha256)
    if not archive_path.is_file():
        raise FileNotFoundError(f"CAS object does not exist: {archive_path}")
    limits = ArchiveLimits(allow_symlink_metadata=args.allow_symlink_metadata)
    manifest = inspect_zip(archive_path, limits=limits)
    index_zip_manifest(
        database,
        archive_blob_sha256=archive_sha256,
        manifest=manifest,
        limits=limits,
    )
    extensions = args.extension or ["png", "gif"]
    allowed = {f".{extension.casefold().lstrip('.')}" for extension in extensions}
    extraction = extract_zip_to_cas(
        archive_path,
        store,
        limits=limits,
        select=lambda member: member.extension in allowed,
        chunk_bytes=load_config().storage.download_chunk_bytes,
    )
    index_zip_extraction(
        database,
        archive_blob_sha256=archive_sha256,
        extraction=extraction,
        selected_role=args.role,
    )
    inspected = 0
    errors: list[dict[str, str]] = []
    for extracted in extraction.extracted:
        try:
            inspect_and_index_media(
                database,
                blob_sha256=extracted.blob.sha256,
                path=extracted.blob.path,
                original_name=extracted.member.normalized_name,
            )
            database.mark_archive_member_inspection(
                archive_blob_sha256=archive_sha256,
                ordinal=extracted.member.archive_index,
                status="media_inspected",
            )
            inspected += 1
        except (OSError, ValueError) as error:
            rendered_error = f"{type(error).__name__}: {error}"
            database.mark_archive_member_inspection(
                archive_blob_sha256=archive_sha256,
                ordinal=extracted.member.archive_index,
                status="media_invalid",
                error=rendered_error,
            )
            errors.append(
                {
                    "member": extracted.member.normalized_name,
                    "error": rendered_error,
                }
            )
    print(
        json.dumps(
            {
                "archive_sha256": archive_sha256,
                "selected_extensions": sorted(allowed),
                "extracted": len(extraction.extracted),
                "inspected": inspected,
                "errors": errors[: args.max_reported_errors],
                "error_count": len(errors),
            },
            indent=2,
        )
    )
    return 0 if not errors else 2


@dataclass(frozen=True)
class _MediaInspectionWork:
    ordinal: int
    blob_sha256: str
    storage_path: Path
    original_name: str


def _inspect_media_work(work: _MediaInspectionWork) -> tuple[dict[str, object] | None, str | None]:
    try:
        return (
            inspect_media_observation(
                blob_sha256=work.blob_sha256,
                path=work.storage_path,
                original_name=work.original_name,
            ),
            None,
        )
    except (OSError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


def command_archive_inspect_media(args: argparse.Namespace) -> int:
    _store, database, guard = build_runtime()
    archive_sha256 = args.sha256.lower()
    allowed = {f".{value.casefold().lstrip('.')}" for value in args.extension}
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT am.ordinal, am.normalized_path, am.extracted_blob_sha256,
                   b.storage_path
            FROM archive_members am
            JOIN blobs b ON b.sha256=am.extracted_blob_sha256
            WHERE am.archive_blob_sha256=?
              AND am.inspection_status != 'media_inspected'
            ORDER BY am.ordinal
            """,
            (archive_sha256,),
        ).fetchall()
    work = [
        _MediaInspectionWork(
            ordinal=int(row["ordinal"]),
            blob_sha256=str(row["extracted_blob_sha256"]),
            storage_path=Path(str(row["storage_path"])),
            original_name=str(row["normalized_path"]),
        )
        for row in rows
        if Path(str(row["normalized_path"])).suffix.casefold() in allowed
    ]
    processed = 0
    inspected = 0
    errors: list[dict[str, str]] = []
    observations: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal observations, statuses
        database.record_media_observations(observations)
        database.mark_archive_member_inspections(
            archive_blob_sha256=archive_sha256,
            inspections=statuses,
        )
        observations = []
        statuses = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for item, (observation, error) in zip(
            work,
            executor.map(_inspect_media_work, work),
            strict=True,
        ):
            processed += 1
            if observation is not None:
                observations.append(observation)
                statuses.append({"ordinal": item.ordinal, "status": "media_inspected"})
                inspected += 1
            else:
                statuses.append(
                    {"ordinal": item.ordinal, "status": "media_invalid", "error": error}
                )
                errors.append({"member": item.original_name, "error": error or "unknown"})
            if len(statuses) >= args.batch_size:
                guard.require_capacity(label="media inspection index batch")
                flush()
                print(f"Inspected {processed}/{len(work)} archive members", flush=True)
    flush()
    print(
        json.dumps(
            {
                "archive_sha256": archive_sha256,
                "selected_extensions": sorted(allowed),
                "candidates": len(work),
                "inspected": inspected,
                "error_count": len(errors),
                "errors": errors[: args.max_reported_errors],
            },
            indent=2,
        )
    )
    return 0 if not errors else 2


def command_reports_export(args: argparse.Namespace) -> int:
    _store, database, guard = build_runtime()
    guard.require_capacity(label="provenance report export")
    output = Path(args.output).resolve() if args.output else database.path.parent / "reports"
    paths = export_provenance_reports(database.path, output)
    print(
        json.dumps(
            {field: str(getattr(paths, field)) for field in paths.__dataclass_fields__},
            indent=2,
        )
    )
    return 0


def command_ingest_spritecook(args: argparse.Namespace) -> int:
    store, database, guard = build_runtime()
    config = load_config()
    archive_sha256 = args.sha256.lower()
    archive_path = store.object_path(archive_sha256)
    if not archive_path.is_file():
        raise FileNotFoundError(f"CAS object does not exist: {archive_path}")
    guard.require_capacity(label="SpriteCook sequence indexing")
    taxonomy = load_taxonomy(config.project_root / "configs" / "taxonomy.toml")
    result = ingest_spritecook_sequences(
        database=database,
        archive_blob_sha256=archive_sha256,
        archive_path=archive_path,
        taxonomy=taxonomy,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


def command_ingest_freedoom(args: argparse.Namespace) -> int:
    store, database, guard = build_runtime()
    config = load_config()
    archive_sha256 = args.sha256.lower()
    archive_path = store.object_path(archive_sha256)
    if not archive_path.is_file():
        raise FileNotFoundError(f"CAS object does not exist: {archive_path}")
    guard.require_capacity(label="Freedoom sequence indexing")
    taxonomy = load_taxonomy(config.project_root / "configs" / "taxonomy.toml")
    result = ingest_freedoom_sequences(
        database=database,
        archive_blob_sha256=archive_sha256,
        archive_path=archive_path,
        taxonomy=taxonomy,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


def command_dataset_export(args: argparse.Namespace) -> int:
    _store, database, guard = build_runtime()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Snapshot already exists: {output}; pass --overwrite to replace it explicitly"
        )
    guard.require_capacity(label="dataset snapshot export")
    artifact = export_snapshot(
        database.path,
        output,
        policy=SplitPolicy(
            seed=args.seed,
            ratios=SplitRatios(
                train=args.train_ratio,
                validation=args.validation_ratio,
                test=args.test_ratio,
            ),
            assignment_strategy=args.assignment_strategy,
            group_source_pack=args.group_source_pack,
        ),
        filters=SnapshotFilters(
            minimum_frame_count=args.minimum_frame_count,
            actions=tuple(args.action),
            temporal_mode=args.temporal_mode,
            include_source_ids=tuple(args.include_source),
            exclude_source_ids=tuple(args.exclude_source),
        ),
    )
    print(
        json.dumps(
            {
                "artifact_sha256": artifact.sha256,
                "coverage": asdict(artifact.coverage),
                "manifest_sha256": artifact.manifest.sha256,
                "output": str(output),
                "timing_counts": artifact.timing_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_dataset_materialize(args: argparse.Namespace) -> int:
    _store, _database, guard = build_runtime()
    buckets = tuple(args.bucket or (64, 128, 256, 512))
    result = materialize_snapshot(
        Path(args.snapshot).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        blob_root=(Path(args.blob_root).expanduser().resolve() if args.blob_root else None),
        bucket_sizes=buckets,
        anchor=args.anchor,
        padding=args.padding,
        alpha_threshold=args.alpha_threshold,
        upscale=not args.no_upscale,
        max_integer_scale=args.max_integer_scale,
        overwrite=args.overwrite,
        disk_guard=guard,
    )
    print(
        json.dumps(
            {
                "clip_count": len(result.clip_paths),
                "manifest_path": str(result.manifest_path),
                "materialization_sha256": result.sha256,
                "output_directory": str(result.output_directory),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _known_corpus_projection(
    args: argparse.Namespace,
    *,
    label: str,
    ingest: object,
) -> int:
    store, database, guard = build_runtime()
    config = load_config()
    archive_path = store.object_path(args.sha256.casefold())
    if not archive_path.is_file():
        raise FileNotFoundError(f"CAS object does not exist: {archive_path}")
    guard.require_capacity(label=f"{label} sequence indexing")
    taxonomy = load_taxonomy(config.project_root / "configs" / "taxonomy.toml")
    with database.transaction():
        result = ingest(database, archive_path, taxonomy)  # type: ignore[operator]
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


def command_ingest_shattered_pixel_dungeon(args: argparse.Namespace) -> int:
    return _known_corpus_projection(
        args,
        label="Shattered Pixel Dungeon",
        ingest=ingest_known_shattered_pixel_dungeon_sequences,
    )


def command_ingest_open_surge(args: argparse.Namespace) -> int:
    return _known_corpus_projection(
        args,
        label="Open Surge",
        ingest=ingest_known_open_surge_sequences,
    )


def command_ingest_wesnoth(args: argparse.Namespace) -> int:
    return _known_corpus_projection(
        args,
        label="Battle for Wesnoth",
        ingest=ingest_known_wesnoth_sequences,
    )


def command_ingest_flare(args: argparse.Namespace) -> int:
    return _known_corpus_projection(
        args,
        label="Flare",
        ingest=ingest_known_flare_sequences,
    )


def command_ingest_widelands(args: argparse.Namespace) -> int:
    return _known_corpus_projection(
        args,
        label="Widelands",
        ingest=ingest_known_widelands_sequences,
    )


def command_ingest_ss14(args: argparse.Namespace) -> int:
    return _known_corpus_projection(
        args,
        label="Space Station 14",
        ingest=ingest_known_ss14_sequences,
    )


def command_ingest_tmwa(args: argparse.Namespace) -> int:
    return _known_corpus_projection(
        args,
        label="The Mana World client data",
        ingest=ingest_known_tmwa_sequences,
    )


def command_export_lpc_manifest(args: argparse.Namespace) -> int:
    store, database, guard = build_runtime()
    archive_sha256 = args.sha256.casefold()
    archive_path = store.object_path(archive_sha256)
    if not archive_path.is_file():
        raise FileNotFoundError(f"CAS object does not exist: {archive_path}")
    guard.require_capacity(label="LPC layer manifest export")
    result = export_lpc_manifest(
        database_path=database.path,
        archive_path=archive_path,
        archive_blob_sha256=archive_sha256,
        output_path=Path(args.output).expanduser().resolve(),
        overwrite=args.overwrite,
    )
    payload = asdict(result)
    payload["output_path"] = str(result.output_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="spritelab")
    subcommands = root.add_subparsers(dest="command", required=True)

    init_parser = subcommands.add_parser("init", help="Initialize the data store and index")
    init_parser.set_defaults(func=command_init)

    status_parser = subcommands.add_parser("status", help="Show disk guard and index status")
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(func=command_status)

    ingest_parser = subcommands.add_parser("ingest-file", help="Put a local file in the raw store")
    ingest_parser.add_argument("path")
    ingest_parser.set_defaults(func=command_ingest_file)

    sources_parser = subcommands.add_parser("sources", help="Inspect or sync the source registry")
    sources_subcommands = sources_parser.add_subparsers(dest="sources_command", required=True)
    sources_list = sources_subcommands.add_parser("list", help="List configured sources")
    sources_list.add_argument("--registry")
    sources_list.add_argument("--json", action="store_true")
    sources_list.set_defaults(func=command_sources_list)
    sources_sync = sources_subcommands.add_parser("sync", help="Register sources in the index")
    sources_sync.add_argument("--registry")
    sources_sync.set_defaults(func=command_sources_sync)

    acquire_parser = subcommands.add_parser("acquire", help="Acquire a registered source")
    acquire_subcommands = acquire_parser.add_subparsers(dest="acquire_command", required=True)
    acquire_github = acquire_subcommands.add_parser(
        "github", help="Acquire a commit-pinned GitHub repository snapshot"
    )
    acquire_github.add_argument("source_id")
    acquire_github.add_argument("--ref")
    acquire_github.add_argument("--metadata-only", action="store_true")
    acquire_github.set_defaults(func=command_acquire_github)

    archive_parser = subcommands.add_parser("archive", help="Inspect and index archives")
    archive_subcommands = archive_parser.add_subparsers(dest="archive_command", required=True)
    archive_index = archive_subcommands.add_parser(
        "index", help="Validate and index a ZIP central directory"
    )
    archive_index.add_argument("sha256")
    archive_index.add_argument("--max-members", type=int, default=250_000)
    archive_index.add_argument("--max-member-gib", type=int, default=4)
    archive_index.add_argument("--max-total-gib", type=int, default=64)
    archive_index.add_argument("--max-ratio", type=float, default=1_000.0)
    archive_index.add_argument(
        "--allow-symlink-metadata",
        action="store_true",
        help="Index symbolic-link entries as non-extractable metadata; still reject specials",
    )
    archive_index.set_defaults(func=command_archive_index)
    archive_extract = archive_subcommands.add_parser(
        "extract-media", help="Stream selected image members to CAS and inspect them"
    )
    archive_extract.add_argument("sha256")
    archive_extract.add_argument(
        "--extension", action="append", help="Repeatable suffix (default: png, gif)"
    )
    archive_extract.add_argument("--role", default="sprite_candidate")
    archive_extract.add_argument("--max-reported-errors", type=int, default=20)
    archive_extract.add_argument(
        "--allow-symlink-metadata",
        action="store_true",
        help="Permit but never extract symbolic-link entries while validating the archive",
    )
    archive_extract.set_defaults(func=command_archive_extract_media)
    archive_inspect = archive_subcommands.add_parser(
        "inspect-media", help="Parallel-inspect already extracted archive media"
    )
    archive_inspect.add_argument("sha256")
    archive_inspect.add_argument("--extension", action="append", default=["png", "gif", "webp"])
    archive_inspect.add_argument("--workers", type=int, default=16)
    archive_inspect.add_argument("--batch-size", type=int, default=1_000)
    archive_inspect.add_argument("--max-reported-errors", type=int, default=20)
    archive_inspect.set_defaults(func=command_archive_inspect_media)

    reports_parser = subcommands.add_parser("reports", help="Export provenance reports")
    reports_subcommands = reports_parser.add_subparsers(dest="reports_command", required=True)
    reports_export = reports_subcommands.add_parser("export")
    reports_export.add_argument("--output")
    reports_export.set_defaults(func=command_reports_export)

    dataset_parser = subcommands.add_parser(
        "dataset", help="Build reproducible leakage-aware dataset artifacts"
    )
    dataset_subcommands = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    dataset_export = dataset_subcommands.add_parser(
        "export", help="Export a canonical snapshot from the provenance index"
    )
    dataset_export.add_argument("output")
    dataset_export.add_argument("--seed", default="spritelab-v1")
    dataset_export.add_argument(
        "--assignment-strategy", choices=("balanced", "hash"), default="balanced"
    )
    dataset_export.add_argument("--train-ratio", type=float, default=0.9)
    dataset_export.add_argument("--validation-ratio", type=float, default=0.05)
    dataset_export.add_argument("--test-ratio", type=float, default=0.05)
    dataset_export.add_argument("--minimum-frame-count", type=int, default=2)
    dataset_export.add_argument("--action", action="append", default=[])
    dataset_export.add_argument(
        "--include-source",
        action="append",
        default=[],
        help="Include sequences associated with this source ID; repeat for a union",
    )
    dataset_export.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="Exclude sequences associated with this source ID; repeat as needed",
    )
    dataset_export.add_argument(
        "--temporal-mode",
        choices=("known", "model_ready", "pose_only", "all"),
        default="known",
    )
    dataset_export.add_argument(
        "--group-source-pack",
        action="store_true",
        help="Place every sequence from one indexed item/archive in the same split",
    )
    dataset_export.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing snapshot path",
    )
    dataset_export.set_defaults(func=command_dataset_export)
    dataset_materialize = dataset_subcommands.add_parser(
        "materialize",
        help="Hash-verify and normalize a canonical snapshot into RGBA clip arrays",
    )
    dataset_materialize.add_argument("snapshot")
    dataset_materialize.add_argument("output")
    dataset_materialize.add_argument("--blob-root")
    dataset_materialize.add_argument(
        "--bucket",
        action="append",
        type=int,
        help="Repeatable square lossless bucket (default: 64, 128, 256, 512)",
    )
    dataset_materialize.add_argument(
        "--anchor",
        choices=(
            "top_left",
            "top_center",
            "top_right",
            "center_left",
            "center",
            "center_right",
            "bottom_left",
            "bottom_center",
            "bottom_right",
        ),
        default="bottom_center",
    )
    dataset_materialize.add_argument("--padding", type=int, default=0)
    dataset_materialize.add_argument("--alpha-threshold", type=int, default=0)
    dataset_materialize.add_argument("--max-integer-scale", type=int)
    dataset_materialize.add_argument("--no-upscale", action="store_true")
    dataset_materialize.add_argument("--overwrite", action="store_true")
    dataset_materialize.set_defaults(func=command_dataset_materialize)

    corpus_parser = subcommands.add_parser("corpus", help="Index source-specific corpus structure")
    corpus_subcommands = corpus_parser.add_subparsers(dest="corpus_command", required=True)
    corpus_spritecook = corpus_subcommands.add_parser(
        "spritecook", help="Index logical SpriteCook animation sequences"
    )
    corpus_spritecook.add_argument("sha256")
    corpus_spritecook.set_defaults(func=command_ingest_spritecook)
    corpus_freedoom = corpus_subcommands.add_parser(
        "freedoom", help="Index conservative Freedoom actor action/view sequences"
    )
    corpus_freedoom.add_argument("sha256")
    corpus_freedoom.set_defaults(func=command_ingest_freedoom)
    corpus_lpc_manifest = corpus_subcommands.add_parser(
        "lpc-manifest",
        help="Export deterministic modular-layer slices and member-level LPC credits",
    )
    corpus_lpc_manifest.add_argument("sha256")
    corpus_lpc_manifest.add_argument("output")
    corpus_lpc_manifest.add_argument("--overwrite", action="store_true")
    corpus_lpc_manifest.set_defaults(func=command_export_lpc_manifest)
    corpus_spd = corpus_subcommands.add_parser(
        "shattered-pixel-dungeon",
        help="Project exact Java-defined sprite animations from the pinned archive",
    )
    corpus_spd.add_argument("sha256")
    corpus_spd.set_defaults(func=command_ingest_shattered_pixel_dungeon)
    corpus_open_surge = corpus_subcommands.add_parser(
        "opensurge",
        help="Project exact .spr timelines and per-image credits from the pinned archive",
    )
    corpus_open_surge.add_argument("sha256")
    corpus_open_surge.set_defaults(func=command_ingest_open_surge)
    corpus_wesnoth = corpus_subcommands.add_parser(
        "wesnoth",
        help="Project conservative literal unit animations from the pinned archive",
    )
    corpus_wesnoth.add_argument("sha256")
    corpus_wesnoth.set_defaults(func=command_ingest_wesnoth)
    corpus_flare = corpus_subcommands.add_parser(
        "flare",
        help="Project exact campaign-stack entity animation tracks from the pinned archive",
    )
    corpus_flare.add_argument("sha256")
    corpus_flare.set_defaults(func=command_ingest_flare)
    corpus_widelands = corpus_subcommands.add_parser(
        "widelands",
        help="Project exact complete unmasked worker and critter animations",
    )
    corpus_widelands.add_argument("sha256")
    corpus_widelands.set_defaults(func=command_ingest_widelands)
    corpus_ss14 = corpus_subcommands.add_parser(
        "ss14",
        help="Project conservative complete-entity RSI states from Space Station 14",
    )
    corpus_ss14.add_argument("sha256")
    corpus_ss14.set_defaults(func=command_ingest_ss14)
    corpus_tmwa = corpus_subcommands.add_parser(
        "tmwa",
        help="Project conservative exact one-layer monster timelines from TMWA client data",
    )
    corpus_tmwa.add_argument("sha256")
    corpus_tmwa.set_defaults(func=command_ingest_tmwa)
    return root


def main() -> None:
    args = parser().parse_args()
    raise SystemExit(args.func(args))
