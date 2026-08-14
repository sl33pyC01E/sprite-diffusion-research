"""Normalize compatible M.U.G.E.N six-action materializations without copying pixels."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MugenMaterializationView:
    """One verified in-memory view over a finalized materialization manifest."""

    characters: tuple[dict[str, Any], ...]
    projection_contract: str


def normalize_mugen_materialization(
    manifest: dict[str, Any], *, manifest_sha256: str
) -> MugenMaterializationView:
    """Return the current streamed schema or a strict legacy-v2 zero-copy view.

    Legacy schema-v2 artifacts already used one shared world transform across all
    selected actions. They enter this view only when every selected clip is present
    and repeats that exact character transform. Reported clipping stays literal so
    the downstream broad/dense quality policy can exclude the affected character
    without hiding it or rejecting unrelated safe rows. The function changes
    metadata in memory only; source arrays remain under their immutable root.
    """

    _digest(manifest_sha256, "manifest_sha256")
    if manifest.get("projection_version") == 2:
        rows = _rows(manifest.get("characters"), "streamed characters")
        return MugenMaterializationView(tuple(rows), "streamed_projection_v2")
    if not (
        manifest.get("artifact_kind") == "mugen_fixed_schema_core_training_view"
        and manifest.get("schema_version") == 2
    ):
        raise ValueError("unsupported MUGEN materialization projection contract")
    characters = _rows(manifest.get("characters"), "legacy characters")
    clips = _rows(manifest.get("clips"), "legacy clips")
    clips_by_identity: dict[str, list[dict[str, Any]]] = {}
    for clip in clips:
        identity_id = _text(clip, "identity_id")
        clips_by_identity.setdefault(identity_id, []).append(clip)
    output = []
    seen_variants = set()
    seen_identities = set()
    for character in characters:
        identity_id = _text(character, "identity_id")
        if identity_id in seen_identities:
            raise ValueError(f"legacy materialization duplicates identity: {identity_id}")
        seen_identities.add(identity_id)
        selected = clips_by_identity.pop(identity_id, None)
        if not selected:
            raise ValueError(f"legacy character has no selected clips: {identity_id}")
        expected_ids = character.get("slot_record_ids")
        if not isinstance(expected_ids, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in expected_ids.items()
        ):
            raise ValueError(f"legacy slot record IDs are invalid: {identity_id}")
        actual_by_id = {_text(clip, "record_id"): clip for clip in selected}
        if len(actual_by_id) != len(selected) or set(actual_by_id) != set(expected_ids.values()):
            raise ValueError(f"legacy selected clips differ from slot records: {identity_id}")
        world_view = _object(character.get("world_view_transform"), "legacy world view")
        normalized_clips = []
        for slot, record_id in sorted(expected_ids.items(), key=lambda item: item[0].encode()):
            clip = actual_by_id[record_id]
            if _text(clip, "slot") != slot:
                raise ValueError(f"legacy clip slot differs: {identity_id}/{slot}")
            clipping = clip.get("clipped_visible_pixels")
            if isinstance(clipping, bool) or not isinstance(clipping, int) or clipping < 0:
                raise ValueError(f"legacy clip clipping count is invalid: {identity_id}/{slot}")
            if clip.get("world_view_transform") != world_view:
                raise ValueError(f"legacy clip world transform differs: {identity_id}/{slot}")
            normalized_clips.append({**clip, "source_action_index": -1})
        source = _normalized_source(character, identity_id=identity_id)
        variant_id = _legacy_variant_id(
            character,
            manifest_sha256=manifest_sha256,
            identity_id=identity_id,
        )
        if variant_id in seen_variants:
            raise ValueError(f"legacy materialization duplicates variant: {variant_id}")
        seen_variants.add(variant_id)
        definitions = character.get("definitions")
        if definitions is None:
            definitions = []
        if not isinstance(definitions, list) or any(
            not isinstance(definition, dict) for definition in definitions
        ):
            raise ValueError(f"legacy definitions are invalid: {identity_id}")
        output.append(
            {
                **character,
                "clips": normalized_clips,
                "definitions": definitions,
                "identity_label_provenance_only": _identity_label(character, definitions),
                "source": source,
                "variant_id": variant_id,
            }
        )
    if clips_by_identity:
        raise ValueError(f"legacy clips reference unknown identities: {len(clips_by_identity)}")
    return MugenMaterializationView(
        tuple(sorted(output, key=lambda row: row["variant_id"].encode())),
        "legacy_fixed_schema_v2_zero_copy",
    )


def materialization_identity_labels(character: dict[str, Any]) -> tuple[str, ...]:
    """Return literal provenance labels suitable for leakage grouping, not prompts."""

    values = []
    explicit = character.get("identity_label_provenance_only")
    if isinstance(explicit, str) and explicit.strip():
        values.append(explicit.strip())
    definitions = character.get("definitions")
    if isinstance(definitions, list):
        for definition in definitions:
            if not isinstance(definition, dict):
                continue
            for key in ("display_name", "name"):
                value = definition.get(key)
                if isinstance(value, str) and value.strip():
                    values.append(value.strip())
    return tuple(dict.fromkeys(values))


def _normalized_source(character: dict[str, Any], *, identity_id: str) -> dict[str, Any]:
    source = dict(_object(character.get("source"), "legacy character source"))
    sff = source.get("sff")
    if not isinstance(sff, dict):
        sha256 = source.get("sff_sha256")
        _digest(sha256, "legacy sff_sha256")
        sff = {"sha256": sha256}
        member = source.get("sff_member")
        if isinstance(member, str) and member:
            sff["member_path"] = member
        source["sff"] = sff
    else:
        _digest(sff.get("sha256"), "legacy source SFF sha256")
    if not isinstance(source.get("air"), dict):
        member = source.get("air_member")
        if isinstance(member, str) and member:
            source["air"] = {"member_path": member}
    source["legacy_identity_id"] = identity_id
    return source


def _legacy_variant_id(character: dict[str, Any], *, manifest_sha256: str, identity_id: str) -> str:
    source = _object(character.get("source"), "legacy character source")
    catalog_variant = source.get("catalog_variant_id")
    if isinstance(catalog_variant, str) and catalog_variant.strip():
        return catalog_variant.strip()
    digest = hashlib.sha256(
        f"mugen_legacy_fixed_schema_v2\0{manifest_sha256}\0{identity_id}".encode()
    ).hexdigest()
    return "mugen_legacy_variant_" + digest[:32]


def _identity_label(character: dict[str, Any], definitions: list[dict[str, Any]]) -> str:
    for definition in definitions:
        for key in ("display_name", "name"):
            value = definition.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    resource = character.get("resource")
    if isinstance(resource, dict):
        title = resource.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return _text(character, "identity_id")


def _rows(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{label} are invalid")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be non-empty text")
    return result


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value
