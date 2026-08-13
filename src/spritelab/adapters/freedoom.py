from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

from PIL import Image, UnidentifiedImageError

EXPECTED_FREEDOOM_ARCHIVE_SHA256 = (
    "4962902bfe9fa921c6ecb4419c55dcd40ca2b93c2d2e3b77c9fc3e89561aec78"
)
FREEDOOM_COMMIT = "d14dbbee3b6fbfb2c11cdb65eb61216e86d4ee85"
DOOM_STATE_SOURCE = (
    "https://github.com/id-Software/DOOM/blob/"
    "a77dfb96cb91780ca334d0d4cfd86957558007e0/linuxdoom-1.10/info.c"
)

_FRAME_TOKENS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_")
_ACTION_ORDER = ("idle", "run", "attack", "hurt", "death", "resurrect", "revive")

# This is not a general Doom filename rewrite.  It records one disagreement in
# the pinned Freedoom tree: the source file is VILE^0.png while buildcfg.txt asks
# DeuTex for VILE\0.  The audit only activates the hint when both sides of that
# disagreement are present and continues to preserve/report the raw name.
_PINNED_MANIFEST_ALIAS_CANDIDATES = {
    (FREEDOOM_COMMIT, "VILE^0"): r"VILE\0",
}

# These groups are a compact transcription of the sprite/frame references in
# Linux Doom 1.10's state table. They are compatibility hints, not labels inferred
# from the artwork. Overlaps are intentional and are surfaced as ambiguity.
_DOOM_ACTOR_ACTION_FRAMES: dict[str, dict[str, str]] = {
    "BOS2": {
        "idle": "AB",
        "run": "ABCD",
        "attack": "EFG",
        "hurt": "H",
        "death": "IJKLMNO",
        "resurrect": "IJKLMNO",
    },
    "BOSS": {
        "idle": "AB",
        "run": "ABCD",
        "attack": "EFG",
        "hurt": "H",
        "death": "IJKLMNO",
        "resurrect": "IJKLMNO",
    },
    "BSPI": {
        "idle": "AB",
        "run": "ABCDEF",
        "attack": "AGH",
        "hurt": "I",
        "death": "JKLMNOP",
        "resurrect": "JKLMNOP",
    },
    "CPOS": {
        "idle": "AB",
        "run": "ABCD",
        "attack": "EF",
        "hurt": "G",
        "death": "HIJKLMNOPQRST",
        "resurrect": "HIJKLMN",
    },
    "CYBR": {
        "idle": "AB",
        "run": "ABCD",
        "attack": "EF",
        "hurt": "G",
        "death": "HIJKLMNOP",
    },
    "FATT": {
        "idle": "AB",
        "run": "ABCDEF",
        "attack": "GHI",
        "hurt": "J",
        "death": "KLMNOPQRST",
        "resurrect": "KLMNOPQR",
    },
    "HEAD": {
        "idle": "A",
        "run": "A",
        "attack": "BCD",
        "hurt": "EF",
        "death": "GHIJKL",
        "resurrect": "GHIJKL",
    },
    # MT_KEEN's S_KEENSTND spawnstate and S_COMMKEEN deathstate both use frame
    # A. The death chain then runs through S_COMMKEEN12, so A is intentionally
    # shared and A-L are all retained as death artwork.
    "KEEN": {"idle": "A", "death": "ABCDEFGHIJKL", "hurt": "M"},
    "PAIN": {
        "idle": "A",
        "run": "ABC",
        "attack": "DEF",
        "hurt": "G",
        "death": "HIJKLM",
        "resurrect": "HIJKLM",
    },
    "PLAY": {
        "idle": "A",
        "run": "ABCD",
        "attack": "EF",
        "hurt": "G",
        "death": "HIJKLMNOPQRSTUVW",
    },
    "POSS": {
        "idle": "AB",
        "run": "ABCD",
        "attack": "EF",
        "hurt": "G",
        "death": "HIJKLMNOPQRSTU",
        "resurrect": "HIJK",
    },
    "SARG": {
        "idle": "AB",
        "run": "ABCD",
        "attack": "EFG",
        "hurt": "H",
        "death": "IJKLMN",
        "resurrect": "IJKLMN",
    },
    "SKEL": {
        "idle": "AB",
        "run": "ABCDEF",
        "attack": "GHIJK",
        "hurt": "L",
        "death": "LMNOPQ",
        "resurrect": "LMNOPQ",
    },
    "SKUL": {
        "idle": "AB",
        "run": "AB",
        "attack": "CD",
        "hurt": "E",
        "death": "FGHIJK",
    },
    "SPID": {
        "idle": "AB",
        "run": "ABCDEF",
        "attack": "AGH",
        "hurt": "I",
        "death": "JKLMNOPQRS",
    },
    "SPOS": {
        "idle": "AB",
        "run": "ABCD",
        "attack": "EF",
        "hurt": "G",
        "death": "HIJKLMNOPQRSTU",
        "resurrect": "HIJKL",
    },
    "SSWV": {
        "idle": "AB",
        "run": "ABCD",
        "attack": "EFG",
        "hurt": "H",
        "death": "IJKLMNOPQRSTUV",
        "resurrect": "IJKLM",
    },
    "TROO": {
        "idle": "AB",
        "run": "ABCD",
        "attack": "EFG",
        "hurt": "H",
        "death": "IJKLMNOPQRSTU",
        "resurrect": "IJKLM",
    },
    "VILE": {
        "idle": "AB",
        "run": "ABCDEF",
        "attack": "GHIJKLMNOP",
        "hurt": "Q",
        "death": "QRSTUVWXYZ",
        "revive": "[\\]",
    },
}

# BEX cast-call string identifiers have fixed actor slots in Doom II. Keeping
# this mapping explicit prevents label propagation to unrelated families.
_CC_KEY_TO_FAMILY = {
    "CC_ZOMBIE": "POSS",
    "CC_SHOTGUN": "SPOS",
    "CC_HEAVY": "CPOS",
    "CC_IMP": "TROO",
    "CC_DEMON": "SARG",
    "CC_LOST": "SKUL",
    "CC_CACO": "HEAD",
    "CC_HELL": "BOS2",
    "CC_BARON": "BOSS",
    "CC_ARACH": "BSPI",
    "CC_PAIN": "PAIN",
    "CC_REVEN": "SKEL",
    "CC_MANCU": "FATT",
    "CC_ARCH": "VILE",
    "CC_SPIDER": "SPID",
    "CC_CYBER": "CYBR",
    "CC_HERO": "PLAY",
}


class DoomSpriteNameError(ValueError):
    """Raised when a filename is not a Doom sprite-lump name."""


class FreedoomArchiveError(ValueError):
    """Raised when an archive cannot be audited as a Freedoom source tree."""


@dataclass(frozen=True)
class DoomFrameRotation:
    frame_token: str
    frame_index: int
    vanilla_frame_range_valid: bool
    rotation: int
    mirrored: bool
    canonical_transform: str
    pair_index: int


@dataclass(frozen=True)
class DoomSpriteName:
    raw_filename: str
    raw_stem: str
    extension: str
    family: str
    references: tuple[DoomFrameRotation, ...]


@dataclass(frozen=True)
class FrameAliasHint:
    archive_lump_name: str
    manifest_lump_name: str
    family: str
    archive_frame_token: str
    manifest_frame_token: str
    rotation: int
    confidence: str
    basis: str
    evidence_member_path: str
    evidence_sha256: str


@dataclass(frozen=True)
class ActionHint:
    family: str
    frame_token: str
    candidate_actions: tuple[str, ...]
    ambiguous: bool
    unknown: bool
    basis: str
    evidence_url: str
    reason: str | None
    interpreted_frame_token: str
    confidence: str
    alias_hint: FrameAliasHint | None


@dataclass(frozen=True)
class CCLabelHint:
    cc_key: str
    label: str
    family: str | None
    line_number: int
    raw_line: str
    mapping_basis: str | None


@dataclass(frozen=True)
class DehackedFramePatch:
    frame_number: int
    line_number: int
    comment_context: tuple[str, ...]
    fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SpriteImageAudit:
    member_path: str
    raw_filename: str
    raw_stem: str
    width: int
    height: int
    image_mode: str
    has_transparency: bool
    size_bytes: int
    compressed_size_bytes: int
    crc32: str
    sha256: str
    parsed_name: DoomSpriteName | None
    parse_error: str | None


@dataclass(frozen=True)
class DimensionCount:
    width: int
    height: int
    physical_file_count: int
    parsed_file_count: int
    unparsed_file_count: int


@dataclass(frozen=True)
class RotationCount:
    rotation: int
    reference_count: int
    physical_file_count: int


@dataclass(frozen=True)
class FrameSourceReference:
    frame_token: str
    raw_filename: str
    member_path: str
    rotation: int
    mirrored: bool
    canonical_transform: str
    pair_index: int


@dataclass(frozen=True)
class FrameAudit:
    frame_token: str
    frame_index: int
    rotations: tuple[int, ...]
    rotation_scheme: str
    rotation_complete: bool
    duplicate_rotation_references: tuple[int, ...]
    reference_count: int
    direct_reference_count: int
    mirrored_reference_count: int
    physical_file_count: int
    raw_filenames: tuple[str, ...]
    source_references: tuple[FrameSourceReference, ...]
    action_hint: ActionHint


@dataclass(frozen=True)
class RotationTrackAudit:
    rotation: int
    frame_tokens: tuple[str, ...]
    source_references: tuple[FrameSourceReference, ...]
    missing_frame_tokens: tuple[str, ...]
    complete_for_action_group: bool


@dataclass(frozen=True)
class ActionSequenceAudit:
    sequence_key: str
    identity_key: str
    action: str
    frame_tokens: tuple[str, ...]
    raw_filenames: tuple[str, ...]
    rotation_tracks: tuple[RotationTrackAudit, ...]
    loop_hint: bool | None
    sequence_semantics: str
    state_occurrence_order_preserved: bool
    timing_preserved: bool
    overlaps_other_action_groups: bool
    ambiguous_frame_tokens: tuple[str, ...]
    basis: str


@dataclass(frozen=True)
class FamilyAudit:
    family: str
    identity_key: str
    label_hints: tuple[CCLabelHint, ...]
    physical_file_count: int
    frame_reference_count: int
    raw_filenames: tuple[str, ...]
    frame_tokens: tuple[str, ...]
    rotations: tuple[RotationCount, ...]
    dimensions: tuple[DimensionCount, ...]
    frames: tuple[FrameAudit, ...]
    sequences: tuple[ActionSequenceAudit, ...]
    unknown_action_frame_tokens: tuple[str, ...]
    ambiguous_action_frame_tokens: tuple[str, ...]


@dataclass(frozen=True)
class BuildCommentHint:
    family: str
    raw_lump_name: str
    comment: str
    line_number: int


@dataclass(frozen=True)
class BuildManifestAudit:
    member_path: str
    sha256: str
    expected_lump_count: int
    expected_lump_names: tuple[str, ...]
    missing_from_archive: tuple[str, ...]
    extra_in_archive: tuple[str, ...]
    alias_hints: tuple[FrameAliasHint, ...]
    comment_hints: tuple[BuildCommentHint, ...]


@dataclass(frozen=True)
class EvidenceDocument:
    member_path: str
    relative_path: str
    evidence_kind: str
    scope: str
    size_bytes: int
    sha256: str
    detected_license_identifiers: tuple[str, ...]
    detection_basis: tuple[str, ...]


@dataclass(frozen=True)
class CreditRecord:
    names: tuple[str, ...]
    aliases: tuple[str, ...]
    emails: tuple[str, ...]
    websites: tuple[str, ...]
    contributions: tuple[str, ...]
    raw_lines: tuple[str, ...]

    @property
    def display_name(self) -> str:
        if self.names:
            return self.names[0]
        if self.aliases:
            return self.aliases[0]
        return "unknown contributor"


@dataclass(frozen=True)
class CreditsAudit:
    member_path: str
    sha256: str
    record_count: int
    sprite_related_record_count: int
    sprite_related_records: tuple[CreditRecord, ...]


@dataclass(frozen=True)
class AuditIssue:
    code: str
    message: str
    related_names: tuple[str, ...]


@dataclass(frozen=True)
class FreedoomArchiveCounts:
    zip_member_count: int
    sprite_png_file_count: int
    parsed_sprite_file_count: int
    unparsed_sprite_file_count: int
    dual_pair_file_count: int
    frame_reference_count: int
    family_count: int
    family_frame_group_count: int
    unique_dimension_count: int
    cc_label_hint_count: int


@dataclass(frozen=True)
class FreedoomArchiveAudit:
    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    repository_root: str
    repository_commit: str | None
    counts: FreedoomArchiveCounts
    sprite_files: tuple[SpriteImageAudit, ...]
    families: tuple[FamilyAudit, ...]
    dimensions: tuple[DimensionCount, ...]
    rotations: tuple[RotationCount, ...]
    cc_label_hints: tuple[CCLabelHint, ...]
    dehacked_member_path: str | None
    dehacked_sha256: str | None
    dehacked_frame_patches: tuple[DehackedFramePatch, ...]
    build_manifest: BuildManifestAudit | None
    evidence_documents: tuple[EvidenceDocument, ...]
    credits: CreditsAudit | None
    issues: tuple[AuditIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_doom_sprite_name(filename: str) -> DoomSpriteName:
    """Parse a six- or eight-character Doom sprite lump represented as a PNG."""
    raw_filename = filename.rsplit("/", 1)[-1]
    dot = raw_filename.rfind(".")
    if dot < 0:
        raise DoomSpriteNameError(f"Sprite filename has no extension: {raw_filename!r}")
    raw_stem = raw_filename[:dot]
    extension = raw_filename[dot:]
    if extension.lower() != ".png":
        raise DoomSpriteNameError(f"Sprite is not a PNG source file: {raw_filename!r}")
    if len(raw_stem) not in {6, 8}:
        raise DoomSpriteNameError(f"Doom sprite stem must contain 6 or 8 characters: {raw_stem!r}")
    family_raw = raw_stem[:4]
    if not all(character.isascii() and character.isalnum() for character in family_raw):
        raise DoomSpriteNameError(f"Invalid four-character sprite family: {family_raw!r}")

    references = (_parse_frame_rotation(raw_stem[4], raw_stem[5], pair_index=0),)
    if len(raw_stem) == 8:
        references += (_parse_frame_rotation(raw_stem[6], raw_stem[7], pair_index=1),)
    return DoomSpriteName(
        raw_filename=raw_filename,
        raw_stem=raw_stem,
        extension=extension,
        family=family_raw.upper(),
        references=references,
    )


def _parse_frame_rotation(
    raw_frame: str, raw_rotation: str, *, pair_index: int
) -> DoomFrameRotation:
    frame_token = raw_frame.upper()
    if frame_token not in _FRAME_TOKENS:
        raise DoomSpriteNameError(f"Invalid Doom frame token: {raw_frame!r}")
    if raw_rotation not in "012345678":
        raise DoomSpriteNameError(f"Invalid Doom rotation: {raw_rotation!r}")
    frame_index = ord(frame_token) - ord("A")
    return DoomFrameRotation(
        frame_token=frame_token,
        frame_index=frame_index,
        vanilla_frame_range_valid=0 <= frame_index < 29,
        rotation=int(raw_rotation),
        mirrored=pair_index == 1,
        canonical_transform="horizontal_flip" if pair_index == 1 else "identity",
        pair_index=pair_index,
    )


def action_hint_for_frame(family: str, frame_token: str) -> ActionHint:
    normalized_family = family.upper()
    normalized_frame = frame_token.upper()
    groups = _DOOM_ACTOR_ACTION_FRAMES.get(normalized_family)
    if groups is None:
        actions: tuple[str, ...] = ()
        reason = "family_has_no_embedded_canonical_state_mapping"
    else:
        actions = tuple(
            action for action in _ACTION_ORDER if normalized_frame in groups.get(action, "")
        )
        if not actions:
            reason = "frame_not_present_in_canonical_action_groups"
        elif len(actions) > 1:
            reason = "canonical_state_table_reuses_frame_across_action_groups"
        else:
            reason = None
    return ActionHint(
        family=normalized_family,
        frame_token=normalized_frame,
        candidate_actions=actions,
        ambiguous=len(actions) != 1,
        unknown=not actions,
        basis="canonical_doom_1.10_state_names_as_compatibility_hints",
        evidence_url=DOOM_STATE_SOURCE,
        reason=reason,
        interpreted_frame_token=normalized_frame,
        confidence="state_table_direct" if actions else "unmapped",
        alias_hint=None,
    )


def parse_dehacked_cc_labels(text: str) -> tuple[CCLabelHint, ...]:
    hints = []
    pattern = re.compile(r"^\s*(CC_[A-Z0-9_]+)\s*=\s*(.*?)\s*$")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        match = pattern.match(raw_line)
        if match is None:
            continue
        key, label = match.groups()
        family = _CC_KEY_TO_FAMILY.get(key)
        hints.append(
            CCLabelHint(
                cc_key=key,
                label=label,
                family=family,
                line_number=line_number,
                raw_line=raw_line,
                mapping_basis=("doom_ii_cast_call_identifier_to_sprite_family" if family else None),
            )
        )
    return tuple(hints)


def parse_dehacked_frame_patches(text: str) -> tuple[DehackedFramePatch, ...]:
    lines = text.splitlines()
    patches: list[DehackedFramePatch] = []
    index = 0
    pending_comments: list[str] = []
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            pending_comments.clear()
            index += 1
            continue
        if stripped.startswith("#"):
            pending_comments.append(stripped.removeprefix("#").strip())
            index += 1
            continue
        match = re.fullmatch(r"Frame\s+(\d+)", stripped, flags=re.IGNORECASE)
        if match is None:
            pending_comments.clear()
            index += 1
            continue
        line_number = index + 1
        frame_number = int(match.group(1))
        fields: list[tuple[str, str]] = []
        index += 1
        while index < len(lines):
            field_line = lines[index].strip()
            if not field_line or field_line.startswith(("#", "Frame ", "Pointer ", "[")):
                break
            if "=" in field_line:
                key, value = field_line.split("=", 1)
                fields.append((key.strip(), value.strip()))
            index += 1
        patches.append(
            DehackedFramePatch(
                frame_number=frame_number,
                line_number=line_number,
                comment_context=tuple(comment for comment in pending_comments if comment),
                fields=tuple(fields),
            )
        )
        pending_comments.clear()
    return tuple(patches)


def parse_credits(text: str) -> tuple[CreditRecord, ...]:
    blocks: list[list[str]] = []
    block: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if block:
                blocks.append(block)
                block = []
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        block.append(line)
    if block:
        blocks.append(block)

    records = []
    for raw_lines in blocks:
        fields: dict[str, list[str]] = defaultdict(list)
        for line in raw_lines:
            match = re.match(r"^([NSEWD]):\s*(.*)$", line)
            if match:
                fields[match.group(1)].append(match.group(2))
        if not fields:
            continue
        records.append(
            CreditRecord(
                names=tuple(fields["N"]),
                aliases=tuple(fields["S"]),
                emails=tuple(fields["E"]),
                websites=tuple(fields["W"]),
                contributions=tuple(fields["D"]),
                raw_lines=tuple(raw_lines),
            )
        )
    return tuple(records)


def audit_freedoom_archive(
    archive_path: Path | str,
    *,
    expected_sha256: str | None = None,
) -> FreedoomArchiveAudit:
    """Audit a Freedoom source ZIP without extracting or mutating it."""
    path = Path(archive_path).resolve()
    archive_sha256 = _hash_file(path)
    if expected_sha256 is not None and archive_sha256 != expected_sha256.lower():
        raise FreedoomArchiveError(
            f"Expected archive SHA-256 {expected_sha256.lower()}, received {archive_sha256}"
        )

    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            repository_root = _discover_repository_root(infos)
            commit_match = re.fullmatch(r"freedoom-([0-9a-f]{40})", repository_root)
            repository_commit = commit_match.group(1) if commit_match else None
            sprite_files = _audit_sprite_files(archive, infos, repository_root)
            dehacked_path = f"{repository_root}/lumps/dehacked/dehacked.txt"
            dehacked_text, dehacked_sha256 = _optional_text(archive, dehacked_path)
            cc_hints = parse_dehacked_cc_labels(dehacked_text or "")
            frame_patches = parse_dehacked_frame_patches(dehacked_text or "")
            manifest = _audit_build_manifest(
                archive,
                repository_root,
                repository_commit,
                sprite_files,
            )
            evidence = _audit_evidence_documents(archive, infos, repository_root)
            credits = _audit_credits(archive, repository_root)
    except BadZipFile as error:
        raise FreedoomArchiveError(f"Not a readable ZIP archive: {path}") from error

    families = _build_family_audits(
        sprite_files,
        cc_hints,
        manifest.alias_hints if manifest is not None else (),
    )
    dimensions = _dimension_counts(sprite_files)
    rotations = _rotation_counts(sprite_files)
    parsed_files = tuple(file for file in sprite_files if file.parsed_name is not None)
    unparsed_files = tuple(file for file in sprite_files if file.parsed_name is None)
    dual_pair_count = sum(
        len(file.parsed_name.references) == 2 for file in parsed_files if file.parsed_name
    )
    reference_count = sum(
        len(file.parsed_name.references) for file in parsed_files if file.parsed_name
    )
    issues = _build_issues(sprite_files, families, manifest)
    return FreedoomArchiveAudit(
        archive_path=str(path),
        archive_sha256=archive_sha256,
        archive_size_bytes=path.stat().st_size,
        repository_root=repository_root,
        repository_commit=repository_commit,
        counts=FreedoomArchiveCounts(
            zip_member_count=len(infos),
            sprite_png_file_count=len(sprite_files),
            parsed_sprite_file_count=len(parsed_files),
            unparsed_sprite_file_count=len(unparsed_files),
            dual_pair_file_count=dual_pair_count,
            frame_reference_count=reference_count,
            family_count=len(families),
            family_frame_group_count=sum(len(family.frames) for family in families),
            unique_dimension_count=len(dimensions),
            cc_label_hint_count=len(cc_hints),
        ),
        sprite_files=sprite_files,
        families=families,
        dimensions=dimensions,
        rotations=rotations,
        cc_label_hints=cc_hints,
        dehacked_member_path=dehacked_path if dehacked_text is not None else None,
        dehacked_sha256=dehacked_sha256,
        dehacked_frame_patches=frame_patches,
        build_manifest=manifest,
        evidence_documents=evidence,
        credits=credits,
        issues=issues,
    )


def audit_known_freedoom_archive(archive_path: Path | str) -> FreedoomArchiveAudit:
    return audit_freedoom_archive(archive_path, expected_sha256=EXPECTED_FREEDOOM_ARCHIVE_SHA256)


def _discover_repository_root(infos: list[ZipInfo]) -> str:
    roots = {
        info.filename[: -len("/sprites/README")]
        for info in infos
        if info.filename.endswith("/sprites/README")
    }
    if len(roots) != 1:
        raise FreedoomArchiveError(
            f"Expected exactly one Freedoom sprites/README root, found {sorted(roots)!r}"
        )
    return roots.pop()


def _audit_sprite_files(
    archive: ZipFile, infos: list[ZipInfo], repository_root: str
) -> tuple[SpriteImageAudit, ...]:
    prefix = f"{repository_root}/sprites/"
    sprite_infos = sorted(
        (
            info
            for info in infos
            if info.filename.startswith(prefix)
            and "/" not in info.filename[len(prefix) :]
            and info.filename.lower().endswith(".png")
        ),
        key=lambda info: info.filename.encode("utf-8"),
    )
    audits = []
    for info in sprite_infos:
        payload = archive.read(info)
        raw_filename = PurePosixPath(info.filename).name
        try:
            parsed_name = parse_doom_sprite_name(raw_filename)
            parse_error = None
        except DoomSpriteNameError as error:
            parsed_name = None
            parse_error = str(error)
        try:
            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
                mode = image.mode
                has_transparency = mode in {"RGBA", "LA"} or "transparency" in image.info
                image.verify()
        except (UnidentifiedImageError, OSError) as error:
            raise FreedoomArchiveError(f"Unreadable sprite PNG {info.filename}: {error}") from error
        audits.append(
            SpriteImageAudit(
                member_path=info.filename,
                raw_filename=raw_filename,
                raw_stem=raw_filename.rsplit(".", 1)[0],
                width=width,
                height=height,
                image_mode=mode,
                has_transparency=has_transparency,
                size_bytes=info.file_size,
                compressed_size_bytes=info.compress_size,
                crc32=f"{info.CRC:08x}",
                sha256=hashlib.sha256(payload).hexdigest(),
                parsed_name=parsed_name,
                parse_error=parse_error,
            )
        )
    return tuple(audits)


def _build_family_audits(
    sprite_files: tuple[SpriteImageAudit, ...],
    cc_hints: tuple[CCLabelHint, ...],
    alias_hints: tuple[FrameAliasHint, ...],
) -> tuple[FamilyAudit, ...]:
    grouped: dict[str, list[SpriteImageAudit]] = defaultdict(list)
    for file in sprite_files:
        if file.parsed_name is not None:
            grouped[file.parsed_name.family].append(file)
    hints_by_family: dict[str, list[CCLabelHint]] = defaultdict(list)
    for hint in cc_hints:
        if hint.family:
            hints_by_family[hint.family].append(hint)
    aliases_by_frame = {(hint.family, hint.archive_frame_token): hint for hint in alias_hints}

    families = []
    for family in sorted(grouped, key=lambda value: value.encode("utf-8")):
        files = sorted(grouped[family], key=lambda file: file.raw_filename.encode("utf-8"))
        frame_references: dict[str, list[tuple[SpriteImageAudit, DoomFrameRotation]]] = defaultdict(
            list
        )
        for file in files:
            assert file.parsed_name is not None
            for reference in file.parsed_name.references:
                frame_references[reference.frame_token].append((file, reference))

        frames = []
        for frame_token in sorted(frame_references, key=_frame_sort_key):
            references = frame_references[frame_token]
            unique_files = {file.raw_filename for file, _reference in references}
            reference_rotation_counts = Counter(
                reference.rotation for _file, reference in references
            )
            rotation_set = set(reference_rotation_counts)
            if rotation_set == {0}:
                rotation_scheme = "all_views"
            elif rotation_set == set(range(1, 9)):
                rotation_scheme = "directional_8"
            elif 0 in rotation_set:
                rotation_scheme = "mixed_all_views_and_directional"
            else:
                rotation_scheme = "incomplete_directional"
            duplicate_rotations = tuple(
                sorted(
                    rotation for rotation, count in reference_rotation_counts.items() if count > 1
                )
            )
            source_references = tuple(
                FrameSourceReference(
                    frame_token=frame_token,
                    raw_filename=file.raw_filename,
                    member_path=file.member_path,
                    rotation=reference.rotation,
                    mirrored=reference.mirrored,
                    canonical_transform=reference.canonical_transform,
                    pair_index=reference.pair_index,
                )
                for file, reference in sorted(
                    references,
                    key=lambda pair: (
                        pair[1].rotation,
                        pair[1].pair_index,
                        pair[0].raw_filename.encode("utf-8"),
                    ),
                )
            )
            frames.append(
                FrameAudit(
                    frame_token=frame_token,
                    frame_index=ord(frame_token) - ord("A"),
                    rotations=tuple(sorted(rotation_set)),
                    rotation_scheme=rotation_scheme,
                    rotation_complete=(
                        rotation_scheme in {"all_views", "directional_8"}
                        and not duplicate_rotations
                    ),
                    duplicate_rotation_references=duplicate_rotations,
                    reference_count=len(references),
                    direct_reference_count=sum(
                        not reference.mirrored for _, reference in references
                    ),
                    mirrored_reference_count=sum(reference.mirrored for _, reference in references),
                    physical_file_count=len(unique_files),
                    raw_filenames=tuple(
                        sorted(unique_files, key=lambda value: value.encode("utf-8"))
                    ),
                    source_references=source_references,
                    action_hint=_action_hint_for_audited_frame(
                        family,
                        frame_token,
                        aliases_by_frame.get((family, frame_token)),
                    ),
                )
            )
        frames_tuple = tuple(frames)
        identity_key = f"doom_sprite_family:{family}"
        sequences = _build_action_sequences(identity_key, frames_tuple)
        families.append(
            FamilyAudit(
                family=family,
                identity_key=identity_key,
                label_hints=tuple(hints_by_family.get(family, [])),
                physical_file_count=len(files),
                frame_reference_count=sum(frame.reference_count for frame in frames_tuple),
                raw_filenames=tuple(file.raw_filename for file in files),
                frame_tokens=tuple(frame.frame_token for frame in frames_tuple),
                rotations=_rotation_counts(tuple(files)),
                dimensions=_dimension_counts(tuple(files)),
                frames=frames_tuple,
                sequences=sequences,
                unknown_action_frame_tokens=tuple(
                    frame.frame_token for frame in frames_tuple if frame.action_hint.unknown
                ),
                ambiguous_action_frame_tokens=tuple(
                    frame.frame_token for frame in frames_tuple if frame.action_hint.ambiguous
                ),
            )
        )
    return tuple(families)


def _action_hint_for_audited_frame(
    family: str,
    frame_token: str,
    alias_hint: FrameAliasHint | None,
) -> ActionHint:
    direct = action_hint_for_frame(family, frame_token)
    if not direct.unknown or alias_hint is None:
        return direct

    aliased = action_hint_for_frame(family, alias_hint.manifest_frame_token)
    if aliased.unknown:
        return direct
    return ActionHint(
        family=direct.family,
        frame_token=direct.frame_token,
        candidate_actions=aliased.candidate_actions,
        ambiguous=True,
        unknown=False,
        basis=("probable_pinned_freedoom_manifest_alias_then_canonical_doom_1.10_state_names"),
        evidence_url=DOOM_STATE_SOURCE,
        reason="probable_manifest_alias_retained_without_rewriting_raw_frame_token",
        interpreted_frame_token=alias_hint.manifest_frame_token,
        confidence=alias_hint.confidence,
        alias_hint=alias_hint,
    )


def _build_action_sequences(
    identity_key: str, frames: tuple[FrameAudit, ...]
) -> tuple[ActionSequenceAudit, ...]:
    sequences = []
    for action in (*_ACTION_ORDER, "unknown"):
        selected = tuple(
            frame
            for frame in frames
            if (
                frame.action_hint.unknown
                if action == "unknown"
                else action in frame.action_hint.candidate_actions
            )
        )
        if not selected:
            continue
        raw_filenames = sorted(
            {name for frame in selected for name in frame.raw_filenames},
            key=lambda value: value.encode("utf-8"),
        )
        ambiguous_frames = tuple(
            frame.frame_token for frame in selected if frame.action_hint.ambiguous
        )
        rotation_tracks = _build_rotation_tracks(selected)
        sequences.append(
            ActionSequenceAudit(
                sequence_key=f"{identity_key}:action:{action}",
                identity_key=identity_key,
                action=action,
                frame_tokens=tuple(frame.frame_token for frame in selected),
                raw_filenames=tuple(raw_filenames),
                rotation_tracks=rotation_tracks,
                loop_hint=None,
                sequence_semantics="ordered_unique_artwork_projection_not_state_cycle",
                state_occurrence_order_preserved=False,
                timing_preserved=False,
                overlaps_other_action_groups=any(
                    len(frame.action_hint.candidate_actions) > 1 for frame in selected
                ),
                ambiguous_frame_tokens=ambiguous_frames,
                basis=(
                    "no_canonical_action_group_for_frame"
                    if action == "unknown"
                    else (
                        "canonical_doom_1.10_state_names_plus_probable_manifest_alias"
                        if any(frame.action_hint.alias_hint is not None for frame in selected)
                        else "canonical_doom_1.10_state_names_as_compatibility_hints"
                    )
                ),
            )
        )
    return tuple(sequences)


def _build_rotation_tracks(frames: tuple[FrameAudit, ...]) -> tuple[RotationTrackAudit, ...]:
    rotations = sorted(
        {reference.rotation for frame in frames for reference in frame.source_references}
    )
    tracks = []
    all_frame_tokens = tuple(frame.frame_token for frame in frames)
    for rotation in rotations:
        references = tuple(
            reference
            for frame in frames
            for reference in frame.source_references
            if reference.rotation == rotation
        )
        present = {
            frame.frame_token
            for frame in frames
            if any(reference.rotation == rotation for reference in frame.source_references)
        }
        frame_tokens = tuple(frame for frame in all_frame_tokens if frame in present)
        missing = tuple(frame for frame in all_frame_tokens if frame not in present)
        tracks.append(
            RotationTrackAudit(
                rotation=rotation,
                frame_tokens=frame_tokens,
                source_references=references,
                missing_frame_tokens=missing,
                complete_for_action_group=not missing,
            )
        )
    return tuple(tracks)


def _dimension_counts(
    sprite_files: tuple[SpriteImageAudit, ...],
) -> tuple[DimensionCount, ...]:
    grouped: dict[tuple[int, int], list[SpriteImageAudit]] = defaultdict(list)
    for file in sprite_files:
        grouped[(file.width, file.height)].append(file)
    return tuple(
        DimensionCount(
            width=width,
            height=height,
            physical_file_count=len(files),
            parsed_file_count=sum(file.parsed_name is not None for file in files),
            unparsed_file_count=sum(file.parsed_name is None for file in files),
        )
        for (width, height), files in sorted(grouped.items())
    )


def _rotation_counts(
    sprite_files: tuple[SpriteImageAudit, ...],
) -> tuple[RotationCount, ...]:
    reference_counts: Counter[int] = Counter()
    physical_files: dict[int, set[str]] = defaultdict(set)
    for file in sprite_files:
        if file.parsed_name is None:
            continue
        for reference in file.parsed_name.references:
            reference_counts[reference.rotation] += 1
            physical_files[reference.rotation].add(file.raw_filename)
    return tuple(
        RotationCount(
            rotation=rotation,
            reference_count=reference_counts[rotation],
            physical_file_count=len(physical_files[rotation]),
        )
        for rotation in sorted(reference_counts)
    )


def _audit_build_manifest(
    archive: ZipFile,
    repository_root: str,
    repository_commit: str | None,
    sprite_files: tuple[SpriteImageAudit, ...],
) -> BuildManifestAudit | None:
    member_path = f"{repository_root}/buildcfg.txt"
    text, digest = _optional_text(archive, member_path)
    if text is None or digest is None:
        return None
    lines = text.splitlines()
    in_sprites = False
    expected: list[str] = []
    comment_hints: list[BuildCommentHint] = []
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if stripped.lower() == "[sprites]":
            in_sprites = True
            continue
        if in_sprites and stripped.startswith("[") and stripped.endswith("]"):
            break
        if not in_sprites or not stripped or stripped.startswith(("#", ";")):
            continue
        code, separator, comment = raw_line.partition(";")
        tokens = code.split()
        if not tokens:
            continue
        raw_lump = tokens[0]
        try:
            parsed = parse_doom_sprite_name(f"{raw_lump}.png")
        except DoomSpriteNameError:
            continue
        normalized_lump = parsed.raw_stem.upper()
        expected.append(normalized_lump)
        if separator and comment.strip():
            comment_hints.append(
                BuildCommentHint(
                    family=parsed.family,
                    raw_lump_name=raw_lump,
                    comment=comment.strip(),
                    line_number=line_number,
                )
            )
    actual = {file.raw_stem.upper() for file in sprite_files}
    expected_set = set(expected)
    missing = tuple(sorted(expected_set - actual, key=_utf8_key))
    extra = tuple(sorted(actual - expected_set, key=_utf8_key))
    alias_hints = _manifest_alias_hints(
        repository_commit=repository_commit,
        manifest_member_path=member_path,
        manifest_sha256=digest,
        missing_lump_names=missing,
        extra_lump_names=extra,
    )
    return BuildManifestAudit(
        member_path=member_path,
        sha256=digest,
        expected_lump_count=len(expected),
        expected_lump_names=tuple(expected),
        missing_from_archive=missing,
        extra_in_archive=extra,
        alias_hints=alias_hints,
        comment_hints=tuple(comment_hints),
    )


def _manifest_alias_hints(
    *,
    repository_commit: str | None,
    manifest_member_path: str,
    manifest_sha256: str,
    missing_lump_names: tuple[str, ...],
    extra_lump_names: tuple[str, ...],
) -> tuple[FrameAliasHint, ...]:
    missing = set(missing_lump_names)
    extra = set(extra_lump_names)
    hints = []
    for (commit, archive_lump), manifest_lump in sorted(
        _PINNED_MANIFEST_ALIAS_CANDIDATES.items(), key=lambda item: item[0]
    ):
        if repository_commit != commit or archive_lump not in extra or manifest_lump not in missing:
            continue
        archive_name = parse_doom_sprite_name(f"{archive_lump}.png")
        manifest_name = parse_doom_sprite_name(f"{manifest_lump}.png")
        archive_reference = archive_name.references[0]
        manifest_reference = manifest_name.references[0]
        if (
            len(archive_name.references) != 1
            or len(manifest_name.references) != 1
            or archive_name.family != manifest_name.family
            or archive_reference.rotation != manifest_reference.rotation
        ):
            continue
        hints.append(
            FrameAliasHint(
                archive_lump_name=archive_lump,
                manifest_lump_name=manifest_lump,
                family=archive_name.family,
                archive_frame_token=archive_reference.frame_token,
                manifest_frame_token=manifest_reference.frame_token,
                rotation=archive_reference.rotation,
                confidence="probable_commit_scoped_manifest_alias",
                basis=(
                    "pinned_source_filename_is_the_unique_known_counterpart_of_"
                    "a_missing_build_manifest_lump"
                ),
                evidence_member_path=manifest_member_path,
                evidence_sha256=manifest_sha256,
            )
        )
    return tuple(hints)


def _audit_evidence_documents(
    archive: ZipFile, infos: list[ZipInfo], repository_root: str
) -> tuple[EvidenceDocument, ...]:
    documents = []
    prefix = f"{repository_root}/"
    for info in sorted(infos, key=lambda value: value.filename.encode("utf-8")):
        if info.is_dir() or not info.filename.startswith(prefix):
            continue
        relative_path = info.filename[len(prefix) :]
        basename = PurePosixPath(relative_path).name.lower()
        is_credit = basename in {"credits", "credits-levels", "credits-music", "credit.txt"}
        is_license = any(token in basename for token in ("copying", "license"))
        if not (is_credit or is_license):
            continue
        payload = archive.read(info)
        text = payload.decode("utf-8", "replace")
        identifiers, bases = _detect_license_identifiers(text)
        if relative_path == "COPYING.adoc":
            scope = "project_root_license_evidence"
        elif relative_path == "CREDITS":
            scope = "project_root_contributor_index"
        elif "/" in relative_path:
            scope = "subdirectory_evidence_not_inherited_by_sprites"
        else:
            scope = "project_root_supporting_evidence"
        documents.append(
            EvidenceDocument(
                member_path=info.filename,
                relative_path=relative_path,
                evidence_kind="credits" if is_credit else "license",
                scope=scope,
                size_bytes=info.file_size,
                sha256=hashlib.sha256(payload).hexdigest(),
                detected_license_identifiers=identifiers,
                detection_basis=bases,
            )
        )
    return tuple(documents)


def _detect_license_identifiers(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    identifiers: set[str] = set()
    bases: list[str] = []
    for identifier in re.findall(r"SPDX-License-Identifier:\s*([^\s]+)", text):
        identifiers.add(identifier)
        bases.append("explicit_spdx_identifier")
    if "CC0 1.0 Universal" in text:
        identifiers.add("CC0-1.0")
        bases.append("full_text_title_cc0_1_0_universal")
    if (
        "Redistribution and use in source and binary forms" in text
        and "Neither the name of the Freedoom project" in text
    ):
        identifiers.add("BSD-3-Clause")
        bases.append("freedoom_root_three_clause_license_text")
    if "GNU GENERAL PUBLIC LICENSE" in text and "Version 2, June 1991" in text:
        identifiers.add("GPL-2.0-only")
        bases.append("full_text_title_gpl_version_2")
    return (
        tuple(sorted(identifiers, key=_utf8_key)),
        tuple(dict.fromkeys(bases)),
    )


def _audit_credits(archive: ZipFile, repository_root: str) -> CreditsAudit | None:
    member_path = f"{repository_root}/CREDITS"
    text, digest = _optional_text(archive, member_path)
    if text is None or digest is None:
        return None
    records = parse_credits(text)
    sprite_records = tuple(
        record
        for record in records
        if any("sprite" in contribution.lower() for contribution in record.contributions)
    )
    return CreditsAudit(
        member_path=member_path,
        sha256=digest,
        record_count=len(records),
        sprite_related_record_count=len(sprite_records),
        sprite_related_records=sprite_records,
    )


def _build_issues(
    sprite_files: tuple[SpriteImageAudit, ...],
    families: tuple[FamilyAudit, ...],
    manifest: BuildManifestAudit | None,
) -> tuple[AuditIssue, ...]:
    issues = []
    unparsed = tuple(file.raw_filename for file in sprite_files if file.parsed_name is None)
    if unparsed:
        issues.append(
            AuditIssue(
                code="unparsed_sprite_names",
                message="PNG files in sprites/ do not follow the Doom lump naming convention.",
                related_names=unparsed,
            )
        )
    out_of_range = tuple(
        f"{file.raw_filename}:{reference.frame_token}"
        for file in sprite_files
        if file.parsed_name is not None
        for reference in file.parsed_name.references
        if not reference.vanilla_frame_range_valid
    )
    if out_of_range:
        issues.append(
            AuditIssue(
                code="frames_outside_vanilla_range",
                message=(
                    "Parsed frame tokens fall outside Linux Doom's 0-28 frame range; "
                    "the raw names are retained but not normalized away."
                ),
                related_names=out_of_range,
            )
        )
    if manifest and manifest.missing_from_archive:
        issues.append(
            AuditIssue(
                code="manifest_names_missing_from_archive",
                message="Build-manifest sprite names have no case-insensitive PNG stem match.",
                related_names=manifest.missing_from_archive,
            )
        )
    if manifest and manifest.extra_in_archive:
        issues.append(
            AuditIssue(
                code="archive_names_absent_from_manifest",
                message="Sprite PNG stems are not listed by the build manifest.",
                related_names=manifest.extra_in_archive,
            )
        )
    if manifest and manifest.alias_hints:
        issues.append(
            AuditIssue(
                code="probable_manifest_frame_aliases",
                message=(
                    "Pinned source and build-manifest names have a probable frame-token "
                    "alias. Raw names and both mismatch observations remain preserved."
                ),
                related_names=tuple(
                    f"{hint.archive_lump_name}->{hint.manifest_lump_name}"
                    for hint in manifest.alias_hints
                ),
            )
        )
    incomplete_rotations = tuple(
        f"{family.family}:{frame.frame_token}:{frame.rotation_scheme}"
        for family in families
        for frame in family.frames
        if not frame.rotation_complete
    )
    if incomplete_rotations:
        issues.append(
            AuditIssue(
                code="incomplete_or_conflicting_rotation_sets",
                message=(
                    "Frame rotations are neither one rotation-0 all-view reference nor "
                    "one reference for each directional rotation 1-8."
                ),
                related_names=incomplete_rotations,
            )
        )
    unknown = tuple(
        f"{family.family}:{frame}"
        for family in families
        for frame in family.unknown_action_frame_tokens
    )
    if unknown:
        issues.append(
            AuditIssue(
                code="unknown_action_frames",
                message=(
                    "Frames lack an embedded canonical Doom actor-state action mapping; "
                    "no action was guessed."
                ),
                related_names=unknown,
            )
        )
    ambiguous = tuple(
        f"{family.family}:{frame.frame_token}"
        for family in families
        for frame in family.frames
        if len(frame.action_hint.candidate_actions) > 1
    )
    if ambiguous:
        issues.append(
            AuditIssue(
                code="overlapping_action_frames",
                message=(
                    "Canonical Doom states reuse these frames across multiple action groups; "
                    "all candidates are retained."
                ),
                related_names=ambiguous,
            )
        )
    return tuple(issues)


def _optional_text(archive: ZipFile, member_path: str) -> tuple[str | None, str | None]:
    try:
        payload = archive.read(member_path)
    except KeyError:
        return None, None
    return payload.decode("utf-8", "replace"), hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _frame_sort_key(frame_token: str) -> tuple[int, bytes]:
    return ord(frame_token) - ord("A"), frame_token.encode("utf-8")


def _utf8_key(value: str) -> bytes:
    return value.encode("utf-8")
