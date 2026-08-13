from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from spritelab.archive import ArchiveLimits, ZipExtraction, ZipManifest
from spritelab.db import IndexDB
from spritelab.media import extract_animation, inspect_png


def index_zip_manifest(
    database: IndexDB,
    *,
    archive_blob_sha256: str,
    manifest: ZipManifest,
    limits: ArchiveLimits,
) -> int:
    """Persist a complete, metadata-only archive member inventory."""
    database.upsert_archive_inventory(
        archive_blob_sha256=archive_blob_sha256,
        archive_format="zip",
        member_count=len(manifest.members),
        file_count=manifest.regular_file_count,
        total_uncompressed_bytes=manifest.total_uncompressed_bytes,
        total_compressed_bytes=manifest.total_compressed_bytes,
        inventory_sha256=manifest.inventory_sha256,
        policy={
            "max_members": limits.max_members,
            "max_member_bytes": limits.max_member_bytes,
            "max_total_expanded_bytes": limits.max_total_expanded_bytes,
            "max_compression_ratio": limits.max_compression_ratio,
            "allow_symlink_metadata": limits.allow_symlink_metadata,
        },
    )
    members = [
        {
            "ordinal": member.archive_index,
            "member_path": member.original_name,
            "normalized_path": member.normalized_name,
            "member_kind": (
                "directory" if member.is_directory else "symlink" if member.is_symlink else "file"
            ),
            "size_bytes": member.uncompressed_bytes,
            "compressed_bytes": member.compressed_bytes,
            "crc32": member.crc32,
            "compression_method": member.compression_method,
            "modified_at": _zip_datetime(member.modified_at),
            "inspection_status": "listed",
            "metadata": {
                "extension": member.extension,
                "compression_ratio": member.compression_ratio,
                "flag_bits": member.flag_bits,
                "unix_mode": member.unix_mode,
                "non_extractable": member.is_symlink,
                "header_offset": member.header_offset,
            },
        }
        for member in manifest.members
    ]
    return database.upsert_archive_members(
        archive_blob_sha256=archive_blob_sha256,
        members=members,
    )


def index_zip_extraction(
    database: IndexDB,
    *,
    archive_blob_sha256: str,
    extraction: ZipExtraction,
    selected_role: str,
) -> int:
    return database.register_archive_extractions(
        archive_blob_sha256=archive_blob_sha256,
        selected_role=selected_role,
        extracted=[
            {
                "ordinal": item.member.archive_index,
                "sha256": item.blob.sha256,
                "size_bytes": item.blob.size_bytes,
                "storage_path": item.blob.path,
            }
            for item in extraction.extracted
        ],
    )


def inspect_and_index_media(
    database: IndexDB,
    *,
    blob_sha256: str,
    path: Path,
    original_name: str | None = None,
    inspector_version: str = "media-v1",
) -> None:
    """Inspect an extracted PNG/GIF/APNG blob without generating derivatives."""
    observation = inspect_media_observation(
        blob_sha256=blob_sha256,
        path=path,
        original_name=original_name,
        inspector_version=inspector_version,
    )
    database.record_media_observations([observation])


def inspect_media_observation(
    *,
    blob_sha256: str,
    path: Path,
    original_name: str | None = None,
    inspector_version: str = "media-v1",
) -> dict[str, object]:
    """Return pure media facts suitable for deterministic bulk indexing."""
    suffix = Path(original_name or path.name).suffix.casefold()
    if suffix == ".png":
        png = inspect_png(path)
        total_duration: float | None = None
        if png.is_animated:
            animation = extract_animation(path)
            total_duration = animation.total_duration_ms
        return {
            "blob_sha256": blob_sha256,
            "inspector_version": inspector_version,
            "media_format": "APNG" if png.is_animated else "PNG",
            "width": png.size[0],
            "height": png.size[1],
            "mode": png.mode,
            "has_alpha": png.has_alpha,
            "is_animated": png.is_animated,
            "frame_count": png.display_frame_count,
            "loop_count": png.loop_count,
            "total_duration_ms": total_duration,
            "palette_sha256": png.palette_sha256,
            "pixel_sha256": png.first_frame_pixel_sha256,
            "metadata": {
                "bit_depth": png.bit_depth,
                "color_type": png.color_type,
                "alpha_kind": png.alpha_kind,
                "interlaced": png.interlaced,
                "chunk_types": png.chunk_types,
                "has_default_image": png.has_default_image,
            },
        }
    if suffix in {".gif", ".webp"}:
        animation = extract_animation(path)
        pixel_hashes = [
            hashlib.sha256(frame.image.tobytes()).hexdigest() for frame in animation.frames
        ]
        aggregate = hashlib.sha256("".join(pixel_hashes).encode("ascii")).hexdigest()
        return {
            "blob_sha256": blob_sha256,
            "inspector_version": inspector_version,
            "media_format": animation.format,
            "width": animation.canvas_size[0],
            "height": animation.canvas_size[1],
            "mode": animation.source_mode,
            "has_alpha": True,
            "is_animated": animation.is_animated,
            "frame_count": animation.frame_count,
            "loop_count": animation.loop_count,
            "total_duration_ms": animation.total_duration_ms,
            "pixel_sha256": aggregate,
            "metadata": {
                "source_frame_count": animation.source_frame_count,
                "durations_ms": [frame.duration_ms for frame in animation.frames],
                "frame_pixel_sha256": pixel_hashes,
            },
        }
    raise ValueError(f"Unsupported media extension for inspection: {suffix!r}")


def _zip_datetime(value: tuple[int, int, int, int, int, int]) -> str | None:
    try:
        return datetime(*value).isoformat()
    except ValueError:
        return None
