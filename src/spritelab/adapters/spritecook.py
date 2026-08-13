"""Pure path and metadata interpretation for SpriteCook's free asset archive.

The adapter deliberately does not open archives, inspect pixels, or write to the
index.  It turns source evidence into conservative hints that an acquisition or
normalization layer can persist without losing the source's original wording.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePath, PurePosixPath
from typing import Any

_REPOSITORY_ROOT = re.compile(r"^spritecook-free-game-assets(?:-[0-9a-f]{7,40})?$", re.IGNORECASE)
_MEDIA_EXTENSIONS = frozenset({".png", ".webp", ".gif", ".apng", ".jpg", ".jpeg"})
_CANONICAL_ACTIONS = frozenset({"attack", "death", "hurt", "idle", "jump", "run", "walk"})
_SOURCE_STATE_TOKENS = frozenset({"closed", "damaged", "intact", "open"})

# These folders contain one identity and duplicate media also present beneath
# examples/pixel-art-characters/<identity>/.  Keeping the list explicit avoids
# treating an arbitrary future ``pixel-art-*`` collection as one identity.
_SINGLE_IDENTITY_PIXEL_EXAMPLES = frozenset(
    {
        "pixel-art-astronaut-squirrel",
        "pixel-art-chunky-hamster",
        "pixel-art-chunky-parrot",
        "pixel-art-clockwork-robot",
        "pixel-art-cute-penguin",
        "pixel-art-friendly-brown-dog",
        "pixel-art-gilded-knight",
        "pixel-art-ginger-cat",
        "pixel-art-happy-monk",
        "pixel-art-hero-sheep",
        "pixel-art-paladin-knight",
        "pixel-art-red-cloaked-rogue",
        "pixel-art-royal-goblin",
        "pixel-art-tiny-pirate",
        "pixel-art-warrior-cat",
    }
)

_ANIMAL_TOKENS = frozenset(
    {
        "bat",
        "cat",
        "crab",
        "dog",
        "frog",
        "hamster",
        "mouse",
        "owl",
        "parrot",
        "penguin",
        "sheep",
        "squirrel",
        "turtle",
        "wolf",
    }
)
_HUMANOID_TOKENS = frozenset(
    {
        "archer",
        "astronaut",
        "barbarian",
        "druid",
        "dwarf",
        "gnome",
        "goblin",
        "hero",
        "hunter",
        "knight",
        "merchant",
        "monk",
        "paladin",
        "pirate",
        "rogue",
        "samurai",
        "vampire",
        "warrior",
        "warriro",  # Source filename typo: ragdoll_warriro.png.
        "wiz",
        "wizard",
    }
)
_MONSTER_TOKENS = frozenset({"goblin", "monster", "skeleton", "vampire"})
_CREATURE_TOKENS = frozenset({"mushroom", "slime"})
_ROBOT_TOKENS = frozenset({"automaton", "clockwork", "robot"})


@dataclass(frozen=True)
class SpriteCookSetting:
    """One settings row, preserving the source label and value verbatim."""

    label: str
    value: str


@dataclass(frozen=True)
class SpriteCookExample:
    """One example record from ``index.json``."""

    slug: str
    title: str | None
    category: str | None
    prompt: str | None
    settings: tuple[SpriteCookSetting, ...]
    animation_count: int | None
    preview_path: str | None
    folder: str | None
    source_url: str | None


@dataclass(frozen=True)
class SpriteCookIndex:
    """Collection-level fields and ordered examples from ``index.json``."""

    generated_at: str | None
    source_url: str | None
    license_expression: str | None
    examples: tuple[SpriteCookExample, ...]

    def example(self, slug: str | None) -> SpriteCookExample | None:
        """Return an example by normalized slug without mutating the index."""

        if slug is None:
            return None
        wanted = _slug(slug)
        return next((example for example in self.examples if _slug(example.slug) == wanted), None)


@dataclass(frozen=True)
class SpriteCookProvenance:
    """Source claims that can be attached to one archive-member occurrence."""

    original_archive_member: str
    archive_member: str
    repository_commit: str | None
    example_slug: str | None
    collection_url: str | None
    example_url: str | None
    generated_at: str | None
    license_expression: str | None
    license_evidence_members: tuple[str, ...]
    title: str | None
    source_category: str | None
    prompt: str | None
    prompt_scope: str | None
    settings: tuple[SpriteCookSetting, ...]
    declared_animation_count: int | None
    preview_path: str | None
    declared_folder: str | None


@dataclass(frozen=True)
class SpriteCookMemberHint:
    """Conservative classification derived from one member path and metadata.

    Candidate tuples are intentionally allowed to be empty or contain multiple
    values.  A caller should not collapse those cases to a guessed label.
    """

    archive_member: str
    example_slug: str | None
    identity_key: str | None
    raw_entity_hint: str | None
    normalized_entity_hint: str | None
    normalized_entity_class_candidates: tuple[str, ...]
    entity_basis: str
    raw_action_hint: str | None
    normalized_action_candidates: tuple[str, ...]
    action_basis: str
    media_role_candidates: tuple[str, ...]
    provenance: SpriteCookProvenance


def parse_index_metadata(
    payload: bytes | str | Mapping[str, Any],
) -> SpriteCookIndex:
    """Parse SpriteCook ``index.json`` content without performing I/O."""

    raw: Any
    if isinstance(payload, bytes):
        raw = json.loads(payload.decode("utf-8"))
    elif isinstance(payload, str):
        raw = json.loads(payload)
    else:
        raw = payload
    if not isinstance(raw, Mapping):
        raise ValueError("SpriteCook index must be a JSON object")

    raw_examples = raw.get("examples", ())
    if not isinstance(raw_examples, (list, tuple)):
        raise ValueError("SpriteCook index examples must be an array")

    examples: list[SpriteCookExample] = []
    seen_slugs: set[str] = set()
    for position, value in enumerate(raw_examples):
        if not isinstance(value, Mapping):
            raise ValueError(f"SpriteCook example {position} must be an object")
        slug = _required_string(value, "slug", context=f"example {position}")
        normalized_slug = _slug(slug)
        if normalized_slug in seen_slugs:
            raise ValueError(f"Duplicate SpriteCook example slug: {slug}")
        seen_slugs.add(normalized_slug)

        raw_settings = value.get("settings", ())
        if not isinstance(raw_settings, (list, tuple)):
            raise ValueError(f"SpriteCook settings for {slug!r} must be an array")
        settings: list[SpriteCookSetting] = []
        for setting_position, setting in enumerate(raw_settings):
            if not isinstance(setting, Mapping):
                raise ValueError(
                    f"SpriteCook setting {setting_position} for {slug!r} must be an object"
                )
            settings.append(
                SpriteCookSetting(
                    label=_required_string(
                        setting,
                        "label",
                        context=f"setting {setting_position} for {slug!r}",
                    ),
                    value=_required_string(
                        setting,
                        "value",
                        context=f"setting {setting_position} for {slug!r}",
                    ),
                )
            )

        animation_count = value.get("animationCount")
        if animation_count is not None and (
            isinstance(animation_count, bool) or not isinstance(animation_count, int)
        ):
            raise ValueError(f"SpriteCook animationCount for {slug!r} must be an integer or null")

        examples.append(
            SpriteCookExample(
                slug=slug,
                title=_optional_string(value, "title"),
                category=_optional_string(value, "category"),
                prompt=_optional_string(value, "prompt"),
                settings=tuple(settings),
                animation_count=animation_count,
                preview_path=_optional_string(value, "previewPath"),
                folder=_optional_string(value, "folder"),
                source_url=_optional_string(value, "sourceUrl"),
            )
        )

    return SpriteCookIndex(
        generated_at=_optional_string(raw, "generatedAt"),
        source_url=_optional_string(raw, "source"),
        license_expression=_optional_string(raw, "license"),
        examples=tuple(examples),
    )


def normalize_member_path(member_path: str | PurePath) -> str:
    """Remove the commit-varying ZIP root and normalize separators.

    The result remains an archive-relative path.  Absolute and traversal paths
    are rejected rather than silently converted into plausible source members.
    """

    raw = str(member_path).replace("\\", "/")
    path = PurePosixPath(raw)
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if path.is_absolute() or not parts or ".." in parts or ":" in parts[0]:
        raise ValueError(f"Unsafe or empty SpriteCook member path: {member_path!s}")

    if "examples" in parts:
        parts = parts[parts.index("examples") :]
    elif len(parts) > 1 and _REPOSITORY_ROOT.fullmatch(parts[0]):
        parts = parts[1:]
    return PurePosixPath(*parts).as_posix()


def classify_member(
    member_path: str | PurePath,
    *,
    index: SpriteCookIndex | None = None,
) -> SpriteCookMemberHint:
    """Classify one member path using path evidence and optional index metadata."""

    normalized_path = normalize_member_path(member_path)
    path = PurePosixPath(normalized_path)
    example_slug = _example_slug(path)
    example = index.example(example_slug) if index else None
    provenance = provenance_for_member(member_path, index=index)
    roles = candidate_media_roles(normalized_path)

    if path.suffix.lower() not in _MEDIA_EXTENSIONS:
        return SpriteCookMemberHint(
            archive_member=normalized_path,
            example_slug=example_slug,
            identity_key=None,
            raw_entity_hint=None,
            normalized_entity_hint=None,
            normalized_entity_class_candidates=(),
            entity_basis="not_media",
            raw_action_hint=None,
            normalized_action_candidates=(),
            action_basis="not_media",
            media_role_candidates=roles,
            provenance=provenance,
        )

    stem = _strip_media_role_suffix(path.stem)
    raw_action, normalized_actions, action_basis = _action_hint(example_slug, stem)
    raw_entity, normalized_entity, namespace = _identity_hint(
        example_slug,
        path,
        stem,
        raw_action,
    )
    identity_key = (
        f"spritecook:{namespace}:{normalized_entity}" if namespace and normalized_entity else None
    )
    entity_candidates, entity_basis = _entity_class_candidates(
        example_slug,
        normalized_entity,
        source_category=example.category if example else None,
    )
    return SpriteCookMemberHint(
        archive_member=normalized_path,
        example_slug=example_slug,
        identity_key=identity_key,
        raw_entity_hint=raw_entity,
        normalized_entity_hint=normalized_entity,
        normalized_entity_class_candidates=entity_candidates,
        entity_basis=entity_basis,
        raw_action_hint=raw_action,
        normalized_action_candidates=normalized_actions,
        action_basis=action_basis,
        media_role_candidates=roles,
        provenance=provenance,
    )


def candidate_media_roles(member_path: str | PurePath) -> tuple[str, ...]:
    """Return ordered role candidates based only on source naming and location."""

    normalized_path = normalize_member_path(member_path)
    path = PurePosixPath(normalized_path)
    name = path.name.lower()
    example_slug = _example_slug(path)

    if name in {"license", "license.txt"}:
        return ("license_evidence",)
    if name == "index.json":
        return ("collection_index", "descriptive_metadata")
    if name == "readme.md":
        return ("descriptive_metadata",)
    if path.suffix.lower() not in _MEDIA_EXTENSIONS:
        return ()
    if path.stem.lower().endswith(("_sheet", "_spritesheet")):
        return ("sprite_sheet", "horizontal_animation_frames_candidate")
    if path.suffix.lower() in {".webp", ".gif", ".apng"}:
        return ("animation_container", "preview")

    if example_slug == "spell-icon-set":
        return ("spell_icon", "effect_or_object")
    if example_slug == "inventory-icons":
        return ("inventory_icon", "object_sprite")
    if example_slug == "stylized-seamless-textures":
        return ("seamless_texture", "environment_asset")
    if example_slug == "logo-names-art":
        return ("logo_art",)
    if example_slug == "detailed-splash-art":
        return ("splash_art",)
    if example_slug == "isometric-buildings":
        return ("environment_sprite", "static_image")
    if example_slug == "detailed-characters-anime":
        return ("static_character_art", "static_image")
    if example_slug == "game-asset-pack":
        return ("static_sprite", "object_or_entity")
    if example_slug in (
        _SINGLE_IDENTITY_PIXEL_EXAMPLES | {"pixel-art-characters", "tiny-pixel-art"}
    ):
        return ("static_sprite", "reference_image")
    return ("static_image",)


def provenance_for_member(
    member_path: str | PurePath,
    *,
    index: SpriteCookIndex | None = None,
) -> SpriteCookProvenance:
    """Join a member to collection/example claims without inventing missing fields."""

    original_path, repository_commit = _original_member_and_commit(member_path)
    normalized_path = normalize_member_path(original_path)
    path = PurePosixPath(normalized_path)
    example_slug = _example_slug(path)
    example = index.example(example_slug) if index else None

    evidence = ["LICENSE", "README.md", "index.json"]
    if example_slug:
        evidence[0:0] = [
            f"examples/{example_slug}/LICENSE.txt",
            f"examples/{example_slug}/README.md",
        ]

    prompt_scope: str | None = None
    if example and example.prompt is not None:
        prompt_scope = (
            "identity" if example_slug in _SINGLE_IDENTITY_PIXEL_EXAMPLES else "collection"
        )

    return SpriteCookProvenance(
        original_archive_member=original_path,
        archive_member=normalized_path,
        repository_commit=repository_commit,
        example_slug=example_slug,
        collection_url=index.source_url if index else None,
        example_url=example.source_url if example else None,
        generated_at=index.generated_at if index else None,
        license_expression=index.license_expression if index else None,
        license_evidence_members=tuple(evidence),
        title=example.title if example else None,
        source_category=example.category if example else None,
        prompt=example.prompt if example else None,
        prompt_scope=prompt_scope,
        settings=example.settings if example else (),
        declared_animation_count=example.animation_count if example else None,
        preview_path=example.preview_path if example else None,
        declared_folder=example.folder if example else None,
    )


def repository_commit_hint(member_path: str | PurePath) -> str | None:
    """Return the commit embedded in a GitHub-generated archive root, if present."""

    _original_path, commit = _original_member_and_commit(member_path)
    return commit


def _example_slug(path: PurePosixPath) -> str | None:
    parts = path.parts
    if len(parts) >= 2 and parts[0] == "examples":
        return _slug(parts[1])
    return None


def _original_member_and_commit(member_path: str | PurePath) -> tuple[str, str | None]:
    raw = str(member_path).replace("\\", "/")
    path = PurePosixPath(raw)
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if path.is_absolute() or not parts or ".." in parts or ":" in parts[0]:
        raise ValueError(f"Unsafe or empty SpriteCook member path: {member_path!s}")

    commit: str | None = None
    match = re.fullmatch(
        r"spritecook-free-game-assets-([0-9a-f]{40})",
        parts[0],
        flags=re.IGNORECASE,
    )
    if match:
        commit = match.group(1).lower()
    return PurePosixPath(*parts).as_posix(), commit


def _strip_media_role_suffix(stem: str) -> str:
    lowered = stem.lower()
    for suffix in ("_spritesheet", "_sheet"):
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _action_hint(
    example_slug: str | None,
    stem: str,
) -> tuple[str | None, tuple[str, ...], str]:
    lowered_stem = _slug(stem)

    if example_slug in _SINGLE_IDENTITY_PIXEL_EXAMPLES or example_slug == "pixel-art-characters":
        if lowered_stem in _CANONICAL_ACTIONS:
            return stem, (lowered_stem,), "filename_exact"
        return None, (), "unresolved"

    if example_slug == "tiny-pixel-art":
        terminal = lowered_stem.rsplit("-", 1)[-1]
        if terminal in _CANONICAL_ACTIONS:
            raw_terminal = re.split(r"[-_]", stem)[-1]
            return raw_terminal, (terminal,), "filename_terminal_token"
        return None, (), "unresolved"

    if example_slug == "game-asset-pack":
        terminal = lowered_stem.rsplit("-", 1)[-1]
        if terminal in _SOURCE_STATE_TOKENS:
            raw_terminal = re.split(r"[-_]", stem)[-1]
            return raw_terminal, (), "filename_state_token_unmapped"

    # Do not infer a semantic action from a container extension or from words
    # such as "animated".  They establish a media role, not what the subject does.
    return None, (), "unresolved"


def _identity_hint(
    example_slug: str | None,
    path: PurePosixPath,
    stem: str,
    raw_action: str | None,
) -> tuple[str | None, str | None, str | None]:
    parts = path.parts
    if example_slug == "pixel-art-characters" and len(parts) >= 4:
        raw = parts[2]
        return raw, _slug(raw), "pixel-character"

    if example_slug in _SINGLE_IDENTITY_PIXEL_EXAMPLES:
        raw = example_slug.removeprefix("pixel-art-")
        return raw, _slug(raw), "pixel-character"

    if example_slug == "tiny-pixel-art":
        raw = stem
        if raw_action:
            raw = re.sub(
                rf"[-_]{re.escape(raw_action)}$",
                "",
                raw,
                flags=re.IGNORECASE,
            )
        normalized = _slug(raw)
        if normalized == "tiny-wiz":
            normalized = "tiny-wizard"
        return raw, normalized, "tiny-pixel-art"

    if not example_slug:
        return None, None, None

    raw = stem
    if example_slug == "game-asset-pack" and raw_action:
        raw = re.sub(
            rf"[-_]{re.escape(raw_action)}$",
            "",
            raw,
            flags=re.IGNORECASE,
        )
    normalized = _slug(raw)
    namespace = {
        "detailed-characters-anime": "detailed-character",
        "isometric-buildings": "isometric-building",
    }.get(example_slug, example_slug)
    return raw, normalized, namespace


def _entity_class_candidates(
    example_slug: str | None,
    normalized_entity: str | None,
    *,
    source_category: str | None,
) -> tuple[tuple[str, ...], str]:
    if not normalized_entity:
        return (), "unresolved"
    if example_slug == "isometric-buildings":
        return ("environment", "object"), "source_collection_role"
    if example_slug == "inventory-icons":
        return ("object",), "source_collection_role"
    if example_slug == "spell-icon-set":
        return ("effect", "object"), "source_collection_role_ambiguous"
    if example_slug == "stylized-seamless-textures":
        return ("environment",), "source_collection_role"
    if example_slug in {"detailed-splash-art", "logo-names-art"}:
        return (), "non_entity_artwork"

    tokens = frozenset(normalized_entity.split("-"))
    candidates: list[str] = []
    if tokens & _ROBOT_TOKENS:
        candidates.append("robot")
    if tokens & _ANIMAL_TOKENS:
        candidates.append("animal")
    if tokens & _MONSTER_TOKENS:
        candidates.append("monster")
    if tokens & _CREATURE_TOKENS:
        candidates.append("creature")
    if tokens & _HUMANOID_TOKENS:
        candidates.append("humanoid")

    if candidates:
        return tuple(dict.fromkeys(candidates)), "source_identity_tokens"
    if example_slug == "game-asset-pack":
        return ("object",), "source_collection_role_fallback"
    if source_category == "character":
        return ("unknown",), "source_category_character_unresolved"
    return (), "unresolved"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _required_string(value: Mapping[str, Any], key: str, *, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"SpriteCook {context} field {key!r} must be a non-empty string")
    return result


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    result = value.get(key)
    if result is None:
        return None
    if not isinstance(result, str):
        raise ValueError(f"SpriteCook field {key!r} must be a string or null")
    return result
