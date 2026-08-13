from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from spritelab.adapters.lpc import (
    LPC_ACTION_SPECS,
    LpcAnimationCue,
    LpcCredit,
    LpcParseError,
    LpcSheetDefinition,
    classify_lpc_path,
    credit_filename_candidates,
    sheet_animation_cues,
)

LpcGeometryStatus = Literal[
    "canonical",
    "oversize",
    "noncanonical",
    "malformed",
    "layout_join_required",
    "uninspected",
]
LpcCreditStatus = Literal["resolved", "unresolved"]


@dataclass(frozen=True)
class LpcArchiveMemberFact:
    """Exact, read-only archive/media facts needed for one manifest record.

    The manifest builder deliberately accepts facts rather than a database or a
    file path. Callers can stream rows from SQLite, JSONL, or another index
    without coupling this module to storage or loading any sprite pixels.
    """

    ordinal: int
    member_path: str
    width: int | None = None
    height: int | None = None
    extracted_blob_sha256: str | None = None
    pixel_sha256: str | None = None
    inspection_status: str = "unknown"
    inspection_error: str | None = None


@dataclass(frozen=True)
class LpcCellRectangle:
    """One lossless crop address in an action-split sheet."""

    frame_index: int
    source_grid_index: int
    column_index: int
    row_index: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class LpcDirectionSliceSpec:
    """A direction-specific animation view expressed only as source rectangles."""

    stable_id: str
    source_action: str
    normalized_action: str
    view: str
    direction: str
    source_row_index: int
    frame_size: int
    frame_count: int
    expected_frame_count: int
    loopable: bool
    frame_indices: tuple[int, ...]
    cells: tuple[LpcCellRectangle, ...]


@dataclass(frozen=True)
class LpcGeometryFinding:
    """Geometry result retained even when a sheet cannot yet be sliced."""

    status: LpcGeometryStatus
    source_width: int | None
    source_height: int | None
    direction_count: int | None
    expected_frame_size: int | None
    expected_frame_count: int | None
    actual_frame_size: int | None
    actual_frame_count: int | None
    frame_size_matches_canonical: bool | None
    frame_count_matches_canonical: bool | None
    detail: str


@dataclass(frozen=True)
class LpcCreditClaimEvidence:
    """One source-authored attribution/license claim and its evidence document."""

    source_document: str
    filename: str
    notes: str
    authors: tuple[str, ...]
    licenses: tuple[str, ...]
    urls: tuple[str, ...]


@dataclass(frozen=True)
class LpcCreditResolution:
    """Deterministic member-level credit resolution, including failures."""

    status: LpcCreditStatus
    match_method: str
    confidence: float
    candidate_filenames: tuple[str, ...]
    matched_reference: str | None
    definition_sources_considered: tuple[str, ...]
    claims: tuple[LpcCreditClaimEvidence, ...]
    authors: tuple[str, ...]
    license_tokens: tuple[str, ...]
    source_urls: tuple[str, ...]


@dataclass(frozen=True)
class LpcSheetManifestRecord:
    """One source PNG represented as a compositing layer, never a full entity."""

    schema_version: str
    stable_sheet_id: str
    archive_occurrence_id: str
    source_id: str
    archive_blob_sha256: str
    archive_member_ordinal: int
    archive_member_path: str
    repository_relative_path: str
    content_path: str
    extracted_blob_sha256: str | None
    pixel_sha256: str | None
    record_kind: str
    is_complete_entity: bool
    composition_required: bool
    layer_identity: str | None
    category: str | None
    entity_family_cue: str | None
    body_type: str | None
    plane: str | None
    palette: str | None
    source_action: str | None
    normalized_action: str | None
    inspection_status: str
    inspection_error: str | None
    geometry: LpcGeometryFinding
    slices: tuple[LpcDirectionSliceSpec, ...]
    credit: LpcCreditResolution

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record suitable for JSONL manifests."""

        return asdict(self)


@dataclass(frozen=True)
class _DefinitionBinding:
    layer_path: str
    definition_source: str
    layer_index: int
    custom_animation: str | None
    credits: tuple[LpcCredit, ...]


class LpcManifestBuilder:
    """Build compact LPC sheet records from a stream of indexed member facts.

    Credit rows and definitions are indexed once. ``iter_records`` then keeps
    memory proportional to a single sheet record, apart from those small
    attribution indexes. It never decodes, crops, or writes image pixels.
    """

    def __init__(
        self,
        *,
        archive_blob_sha256: str,
        credits: Iterable[LpcCredit] = (),
        definitions: Iterable[LpcSheetDefinition] = (),
        credits_source_document: str = "CREDITS.csv",
    ) -> None:
        self.archive_blob_sha256 = _sha256(archive_blob_sha256, "archive_blob_sha256")
        self.credits_source_document = credits_source_document
        grouped: defaultdict[str, list[LpcCredit]] = defaultdict(list)
        for credit in credits:
            grouped[credit.filename].append(credit)
        self._credits = {filename: tuple(rows) for filename, rows in sorted(grouped.items())}
        self._definitions = _index_definition_bindings(definitions)

    def iter_records(
        self,
        members: Iterable[LpcArchiveMemberFact],
    ) -> Iterator[LpcSheetManifestRecord]:
        """Yield sheet records in input order, filtering non-sheet members."""

        for member in members:
            record = self.build_record(member)
            if record is not None:
                yield record

    def build_record(self, member: LpcArchiveMemberFact) -> LpcSheetManifestRecord | None:
        """Build one manifest record, or ``None`` for a non-sheet member."""

        if isinstance(member.ordinal, bool) or not isinstance(member.ordinal, int):
            raise TypeError("member ordinal must be an integer")
        if member.ordinal < 0:
            raise ValueError("member ordinal must be non-negative")
        info = classify_lpc_path(member.member_path)
        if not info.is_sheet_candidate or info.content_path is None:
            return None

        extracted_sha256 = _optional_sha256(
            member.extracted_blob_sha256,
            "extracted_blob_sha256",
        )
        pixel_sha256 = _optional_sha256(member.pixel_sha256, "pixel_sha256")
        stable_sheet_id = _sheet_stable_id(info.repository_relative_path)
        geometry, slices = _geometry_and_slices(
            stable_sheet_id=stable_sheet_id,
            member=member,
            source_action=info.source_action,
            normalized_action=info.normalized_action,
        )
        credit = self._resolve_credit(member.member_path, info.content_path)
        return LpcSheetManifestRecord(
            schema_version="lpc-sheet-manifest-v1",
            stable_sheet_id=stable_sheet_id,
            archive_occurrence_id=(f"sha256:{self.archive_blob_sha256}:member:{member.ordinal}"),
            source_id="universal_lpc",
            archive_blob_sha256=self.archive_blob_sha256,
            archive_member_ordinal=member.ordinal,
            archive_member_path=info.archive_path,
            repository_relative_path=info.repository_relative_path,
            content_path=info.content_path,
            extracted_blob_sha256=extracted_sha256,
            pixel_sha256=pixel_sha256,
            record_kind="modular_compositing_layer_sheet",
            is_complete_entity=False,
            composition_required=True,
            layer_identity=info.layer_identity,
            category=info.category,
            entity_family_cue=info.entity_family,
            body_type=info.body_type,
            plane=info.plane,
            palette=info.palette,
            source_action=info.source_action,
            normalized_action=info.normalized_action,
            inspection_status=member.inspection_status,
            inspection_error=member.inspection_error,
            geometry=geometry,
            slices=slices,
            credit=credit,
        )

    def _resolve_credit(self, member_path: str, content_path: str) -> LpcCreditResolution:
        bindings = _longest_definition_bindings(content_path, self._definitions)
        custom_animations = tuple(
            dict.fromkeys(
                binding.custom_animation
                for binding in bindings
                if binding.custom_animation is not None
            )
        )
        candidates: list[str] = list(credit_filename_candidates(member_path))
        for custom_animation in custom_animations:
            candidates.extend(
                credit_filename_candidates(
                    member_path,
                    custom_animation=custom_animation,
                )
            )
        candidates = list(dict.fromkeys(candidates))
        definition_sources = tuple(dict.fromkeys(binding.definition_source for binding in bindings))

        for candidate_index, candidate in enumerate(candidates):
            rows = self._credits.get(candidate)
            if not rows:
                continue
            method = (
                "credits_csv_exact_filename"
                if candidate_index == 0 and candidate == content_path
                else "credits_csv_deterministic_path_candidate"
            )
            confidence = 1.0 if method == "credits_csv_exact_filename" else 0.97
            claims = tuple(_claim_evidence(row, self.credits_source_document) for row in rows)
            return _credit_resolution(
                status="resolved",
                match_method=method,
                confidence=confidence,
                candidates=tuple(candidates),
                matched_reference=candidate,
                definition_sources=definition_sources,
                claims=claims,
            )

        definition_claims = tuple(
            _claim_evidence(credit, binding.definition_source)
            for binding in bindings
            for credit in binding.credits
        )
        definition_claims = _unique_claims(definition_claims)
        if definition_claims:
            matched_paths = tuple(dict.fromkeys(binding.layer_path for binding in bindings))
            return _credit_resolution(
                status="resolved",
                match_method="sheet_definition_layer_prefix",
                confidence=0.85,
                candidates=tuple(candidates),
                matched_reference=" | ".join(matched_paths),
                definition_sources=definition_sources,
                claims=definition_claims,
            )

        return _credit_resolution(
            status="unresolved",
            match_method="none",
            confidence=0.0,
            candidates=tuple(candidates),
            matched_reference=None,
            definition_sources=definition_sources,
            claims=(),
        )


def iter_lpc_manifest_records(
    *,
    archive_blob_sha256: str,
    members: Iterable[LpcArchiveMemberFact],
    credits: Iterable[LpcCredit] = (),
    definitions: Iterable[LpcSheetDefinition] = (),
    credits_source_document: str = "CREDITS.csv",
) -> Iterator[LpcSheetManifestRecord]:
    """Functional streaming entry point for callers that do not retain a builder."""

    builder = LpcManifestBuilder(
        archive_blob_sha256=archive_blob_sha256,
        credits=credits,
        definitions=definitions,
        credits_source_document=credits_source_document,
    )
    yield from builder.iter_records(members)


def _geometry_and_slices(
    *,
    stable_sheet_id: str,
    member: LpcArchiveMemberFact,
    source_action: str | None,
    normalized_action: str | None,
) -> tuple[LpcGeometryFinding, tuple[LpcDirectionSliceSpec, ...]]:
    if source_action is None:
        return (
            LpcGeometryFinding(
                status="layout_join_required",
                source_width=member.width,
                source_height=member.height,
                direction_count=None,
                expected_frame_size=None,
                expected_frame_count=None,
                actual_frame_size=None,
                actual_frame_count=None,
                frame_size_matches_canonical=None,
                frame_count_matches_canonical=None,
                detail=(
                    "sheet path has no action token; a definition or custom layout join is required"
                ),
            ),
            (),
        )

    spec = LPC_ACTION_SPECS[source_action]
    if member.width is None or member.height is None:
        detail = "media dimensions are unavailable"
        if member.inspection_error:
            detail = f"{detail}: {member.inspection_error}"
        return (
            LpcGeometryFinding(
                status="uninspected",
                source_width=member.width,
                source_height=member.height,
                direction_count=len(spec.directions),
                expected_frame_size=spec.canonical_frame_size,
                expected_frame_count=spec.canonical_frames,
                actual_frame_size=None,
                actual_frame_count=None,
                frame_size_matches_canonical=None,
                frame_count_matches_canonical=None,
                detail=detail,
            ),
            (),
        )
    if not _is_positive_int(member.width) or not _is_positive_int(member.height):
        return (
            LpcGeometryFinding(
                status="malformed",
                source_width=member.width,
                source_height=member.height,
                direction_count=len(spec.directions),
                expected_frame_size=spec.canonical_frame_size,
                expected_frame_count=spec.canonical_frames,
                actual_frame_size=None,
                actual_frame_count=None,
                frame_size_matches_canonical=None,
                frame_count_matches_canonical=None,
                detail="media dimensions must be positive integers",
            ),
            (),
        )

    direction_count = len(spec.directions)
    inferred_frame_size = (
        member.height // direction_count if member.height % direction_count == 0 else None
    )
    inferred_frame_count = (
        member.width // inferred_frame_size
        if inferred_frame_size and member.width % inferred_frame_size == 0
        else None
    )
    try:
        cues = sheet_animation_cues(
            member.member_path,
            width=member.width,
            height=member.height,
            strict=True,
        )
    except LpcParseError as exc:
        return (
            LpcGeometryFinding(
                status="malformed",
                source_width=member.width,
                source_height=member.height,
                direction_count=direction_count,
                expected_frame_size=spec.canonical_frame_size,
                expected_frame_count=spec.canonical_frames,
                actual_frame_size=inferred_frame_size,
                actual_frame_count=inferred_frame_count,
                frame_size_matches_canonical=(
                    inferred_frame_size == spec.canonical_frame_size
                    if inferred_frame_size is not None
                    else None
                ),
                frame_count_matches_canonical=(
                    inferred_frame_count == spec.canonical_frames
                    if inferred_frame_count is not None
                    else None
                ),
                detail=str(exc),
            ),
            (),
        )

    if not cues:
        raise AssertionError("an action-addressed valid sheet must emit animation cues")
    frame_size = cues[0].frame_size
    frame_count = cues[0].frame_count
    if all(cue.canonical_geometry for cue in cues):
        status: LpcGeometryStatus = "canonical"
        detail = "geometry matches the canonical action sheet"
    elif frame_size > spec.canonical_frame_size:
        status = "oversize"
        detail = (
            f"valid native oversize grid uses {frame_size}-pixel cells; "
            "rectangles preserve source resolution"
        )
    else:
        status = "noncanonical"
        detail = "valid rectangular grid differs from the canonical action geometry"

    slices = tuple(
        _slice_spec(
            stable_sheet_id=stable_sheet_id,
            cue=cue,
            row_index=row_index,
            expected_frame_count=spec.canonical_frames,
        )
        for row_index, cue in enumerate(cues)
    )
    return (
        LpcGeometryFinding(
            status=status,
            source_width=member.width,
            source_height=member.height,
            direction_count=direction_count,
            expected_frame_size=spec.canonical_frame_size,
            expected_frame_count=spec.canonical_frames,
            actual_frame_size=frame_size,
            actual_frame_count=frame_count,
            frame_size_matches_canonical=frame_size == spec.canonical_frame_size,
            frame_count_matches_canonical=frame_count == spec.canonical_frames,
            detail=detail,
        ),
        slices,
    )


def _slice_spec(
    *,
    stable_sheet_id: str,
    cue: LpcAnimationCue,
    row_index: int,
    expected_frame_count: int,
) -> LpcDirectionSliceSpec:
    cells = tuple(
        LpcCellRectangle(
            frame_index=frame_index,
            source_grid_index=row_index * cue.frame_count + frame_index,
            column_index=frame_index,
            row_index=row_index,
            x=frame_index * cue.frame_size,
            y=row_index * cue.frame_size,
            width=cue.frame_size,
            height=cue.frame_size,
        )
        for frame_index in range(cue.frame_count)
    )
    return LpcDirectionSliceSpec(
        stable_id=f"{stable_sheet_id}|{cue.source_action}|{cue.direction}",
        source_action=cue.source_action,
        normalized_action=cue.normalized_action,
        view=cue.view,
        direction=cue.direction,
        source_row_index=row_index,
        frame_size=cue.frame_size,
        frame_count=cue.frame_count,
        expected_frame_count=expected_frame_count,
        loopable=cue.loopable,
        frame_indices=tuple(range(cue.frame_count)),
        cells=cells,
    )


def _index_definition_bindings(
    definitions: Iterable[LpcSheetDefinition],
) -> dict[str, tuple[_DefinitionBinding, ...]]:
    indexed: defaultdict[str, list[_DefinitionBinding]] = defaultdict(list)
    for definition in definitions:
        source = (
            definition.source_path or f"sheet_definition:{definition.type_name}:{definition.name}"
        )
        for layer in definition.layers:
            for _body_type, layer_path in layer.body_paths:
                indexed[layer_path].append(
                    _DefinitionBinding(
                        layer_path=layer_path,
                        definition_source=source,
                        layer_index=layer.index,
                        custom_animation=layer.custom_animation,
                        credits=definition.credits,
                    )
                )
    return {
        layer_path: tuple(
            sorted(
                bindings,
                key=lambda item: (
                    item.definition_source,
                    item.layer_index,
                    item.custom_animation or "",
                ),
            )
        )
        for layer_path, bindings in sorted(indexed.items())
    }


def _longest_definition_bindings(
    content_path: str,
    bindings: dict[str, tuple[_DefinitionBinding, ...]],
) -> tuple[_DefinitionBinding, ...]:
    parts = content_path.split("/")
    for end in range(len(parts) - 1, 0, -1):
        candidate = "/".join(parts[:end])
        matches = bindings.get(candidate)
        if matches:
            return matches
    return ()


def _claim_evidence(credit: LpcCredit, source_document: str) -> LpcCreditClaimEvidence:
    return LpcCreditClaimEvidence(
        source_document=source_document,
        filename=credit.filename,
        notes=credit.notes,
        authors=credit.authors,
        licenses=credit.licenses,
        urls=credit.urls,
    )


def _unique_claims(
    claims: Sequence[LpcCreditClaimEvidence],
) -> tuple[LpcCreditClaimEvidence, ...]:
    unique: dict[tuple[Any, ...], LpcCreditClaimEvidence] = {}
    for claim in claims:
        key = (
            claim.source_document,
            claim.filename,
            claim.notes,
            claim.authors,
            claim.licenses,
            claim.urls,
        )
        unique.setdefault(key, claim)
    return tuple(unique.values())


def _credit_resolution(
    *,
    status: LpcCreditStatus,
    match_method: str,
    confidence: float,
    candidates: tuple[str, ...],
    matched_reference: str | None,
    definition_sources: tuple[str, ...],
    claims: tuple[LpcCreditClaimEvidence, ...],
) -> LpcCreditResolution:
    authors = tuple(dict.fromkeys(author for claim in claims for author in claim.authors))
    licenses = tuple(dict.fromkeys(license_ for claim in claims for license_ in claim.licenses))
    urls = tuple(dict.fromkeys(url for claim in claims for url in claim.urls))
    return LpcCreditResolution(
        status=status,
        match_method=match_method,
        confidence=confidence,
        candidate_filenames=candidates,
        matched_reference=matched_reference,
        definition_sources_considered=definition_sources,
        claims=claims,
        authors=authors,
        license_tokens=licenses,
        source_urls=urls,
    )


def _sheet_stable_id(repository_relative_path: str) -> str:
    digest = hashlib.sha256(
        f"universal_lpc:sheet:v1:{repository_relative_path}".encode()
    ).hexdigest()
    return f"ulpc:sheet:v1:{digest}"


def _sha256(value: str, label: str) -> str:
    normalized = value.casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest")
    return normalized


def _optional_sha256(value: str | None, label: str) -> str | None:
    return None if value is None else _sha256(value, label)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
