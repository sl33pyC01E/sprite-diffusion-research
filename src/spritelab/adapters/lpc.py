from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any


class LpcParseError(ValueError):
    """Raised when an LPC path or metadata document is malformed."""


@dataclass(frozen=True)
class LpcActionSpec:
    source_action: str
    normalized_action: str
    directions: tuple[str, ...]
    canonical_frames: int
    canonical_frame_size: int = 64
    loopable: bool = False


@dataclass(frozen=True)
class LpcPathInfo:
    archive_path: str
    repository_relative_path: str
    content_path: str | None
    kind: str
    category: str | None = None
    layer_identity: str | None = None
    source_action: str | None = None
    normalized_action: str | None = None
    body_type: str | None = None
    palette: str | None = None
    plane: str | None = None
    entity_family: str | None = None

    @property
    def is_sheet_candidate(self) -> bool:
        return self.kind == "sheet"


@dataclass(frozen=True)
class LpcCredit:
    filename: str
    notes: str
    authors: tuple[str, ...]
    licenses: tuple[str, ...]
    urls: tuple[str, ...]


@dataclass(frozen=True)
class LpcLayerDefinition:
    index: int
    z_position: int | None
    body_paths: tuple[tuple[str, str], ...]
    custom_animation: str | None = None
    is_mask: bool = False

    @property
    def body_types(self) -> tuple[str, ...]:
        return tuple(body_type for body_type, _ in self.body_paths)

    def path_for(self, body_type: str) -> str | None:
        return dict(self.body_paths).get(body_type)


@dataclass(frozen=True)
class LpcRecolorRule:
    channel: str
    material: str | None
    base: str | None
    palettes: tuple[str, ...]
    type_name: str | None = None
    label: str | None = None


@dataclass(frozen=True)
class LpcSheetDefinition:
    source_path: str | None
    name: str
    type_name: str
    layers: tuple[LpcLayerDefinition, ...]
    animations: tuple[str, ...]
    variants: tuple[str, ...]
    tags: tuple[str, ...]
    required_tags: tuple[str, ...]
    credits: tuple[LpcCredit, ...]
    recolor_rules: tuple[LpcRecolorRule, ...]
    aliases: tuple[tuple[str, str], ...]
    priority: int | None
    match_body_color: bool

    @property
    def body_types(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(body_type for layer in self.layers for body_type, _ in layer.body_paths)
        )


@dataclass(frozen=True)
class LpcDirectionLayout:
    source_action: str
    direction: str
    row_index: int
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class LpcAnimationLayout:
    frame_width: int
    frame_height: int
    columns: int
    rows: int
    direction_layouts: tuple[LpcDirectionLayout, ...]

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.source_action for item in self.direction_layouts))


@dataclass(frozen=True)
class LpcAnimationCue:
    stable_id: str
    layer_identity: str
    source_action: str
    normalized_action: str
    view: str
    direction: str
    frame_size: int
    frame_count: int
    loopable: bool
    canonical_geometry: bool


@dataclass(frozen=True)
class LpcPalette:
    name: str
    colors: tuple[str, ...]


@dataclass(frozen=True)
class LpcPaletteDefinition:
    source_path: str
    material: str
    scheme: str
    palettes: tuple[LpcPalette, ...]


_CARDINAL_DIRECTIONS = ("north", "west", "south", "east")
_PLANE_TOKENS = frozenset({"fg", "bg", "mg", "foreground", "background", "behind", "mask"})
_BODY_TYPES = frozenset(
    {
        "adult",
        "child",
        "elderly",
        "female",
        "male",
        "muscular",
        "pregnant",
        "teen",
        "thin",
        "universal",
    }
)
_ACTION_SPECS = (
    LpcActionSpec("spellcast", "cast", _CARDINAL_DIRECTIONS, 7),
    LpcActionSpec("cast", "cast", _CARDINAL_DIRECTIONS, 7),
    LpcActionSpec("thrust", "attack", _CARDINAL_DIRECTIONS, 8),
    LpcActionSpec("walk", "walk", _CARDINAL_DIRECTIONS, 9, loopable=True),
    LpcActionSpec("slash", "attack", _CARDINAL_DIRECTIONS, 6),
    LpcActionSpec("shoot", "shoot", _CARDINAL_DIRECTIONS, 13),
    LpcActionSpec("hurt", "hurt", ("south",), 6),
    LpcActionSpec("climb", "climb", ("north",), 6, loopable=True),
    LpcActionSpec("idle", "idle", _CARDINAL_DIRECTIONS, 2, loopable=True),
    LpcActionSpec("jump", "jump", _CARDINAL_DIRECTIONS, 5),
    LpcActionSpec("sit", "sit", _CARDINAL_DIRECTIONS, 3, loopable=True),
    LpcActionSpec("emote", "emote", _CARDINAL_DIRECTIONS, 3),
    LpcActionSpec("run", "run", _CARDINAL_DIRECTIONS, 8, loopable=True),
    LpcActionSpec("combat_idle", "idle", _CARDINAL_DIRECTIONS, 2, loopable=True),
    LpcActionSpec("backslash", "attack", _CARDINAL_DIRECTIONS, 13),
    LpcActionSpec("halfslash", "attack", _CARDINAL_DIRECTIONS, 6),
    LpcActionSpec("attack_slash", "attack", _CARDINAL_DIRECTIONS, 6),
    LpcActionSpec("attack_slash_reverse", "attack", _CARDINAL_DIRECTIONS, 6),
    LpcActionSpec("attack_backslash", "attack", _CARDINAL_DIRECTIONS, 13),
    LpcActionSpec("attack_halfslash", "attack", _CARDINAL_DIRECTIONS, 6),
    LpcActionSpec("attack_thrust", "attack", _CARDINAL_DIRECTIONS, 8),
)
LPC_ACTION_SPECS: Mapping[str, LpcActionSpec] = MappingProxyType(
    {spec.source_action: spec for spec in _ACTION_SPECS}
)

_CREDIT_ACTION_ALIASES: Mapping[str, tuple[str, ...]] = {
    "combat_idle": ("combat",),
    "backslash": ("1h_backslash",),
    "halfslash": ("1h_halfslash",),
    "walk": ("walk_128",),
    "slash": ("slash_128", "slash_oversize", "slash_reverse_oversize"),
    "thrust": ("thrust_oversize",),
    "attack_slash": (
        "slash_128",
        "slash_oversize",
        "slash_reverse_oversize",
        "tool_axe",
        "tool_hammer",
        "tool_whip",
    ),
    "attack_slash_reverse": ("slash_reverse_oversize", "slash_oversize"),
    "attack_backslash": ("backslash_128",),
    "attack_halfslash": ("halfslash_128",),
    "attack_thrust": ("thrust_oversize", "tool_rod"),
}

_KNOWN_COLOR_NAMES = frozenset(
    {
        "aegean",
        "amber",
        "amethyst",
        "apple",
        "apricot",
        "ash",
        "ash_brown",
        "azure",
        "beige",
        "black",
        "blonde",
        "blue",
        "blue_violet",
        "bluegray",
        "brass",
        "bright_green",
        "bronze",
        "brown",
        "carrot",
        "ceramic",
        "cerise",
        "cerulean",
        "charcoal",
        "chestnut",
        "chocolate",
        "coffee",
        "copper",
        "coral",
        "cornflower",
        "cyan",
        "dark_brown",
        "dark_gray",
        "dark_green",
        "denim",
        "dove",
        "emerald",
        "fern",
        "forest",
        "fur_black",
        "fur_brown",
        "fur_copper",
        "fur_gold",
        "fur_grey",
        "fur_tan",
        "fur_white",
        "garnet",
        "ginger",
        "gold",
        "gray",
        "green",
        "hazel",
        "heather",
        "honey",
        "ice",
        "indigo",
        "iron",
        "ivory",
        "lavender",
        "leather",
        "lemon",
        "light_brown",
        "linen",
        "mahogany",
        "maple",
        "maroon",
        "mauve",
        "midnight",
        "mint",
        "mustard",
        "navy",
        "neptune",
        "oak",
        "ochre",
        "olive",
        "olivine",
        "orange",
        "pale_green",
        "peach",
        "pearl",
        "periwinkle",
        "pink",
        "platinum",
        "plum",
        "porcelain",
        "powder",
        "purple",
        "raven",
        "red",
        "red_orange",
        "redhead",
        "rose",
        "royal",
        "ruby",
        "salmon",
        "sandy",
        "sepia",
        "shadow",
        "silver",
        "sky",
        "slate",
        "smoke",
        "soot",
        "spring",
        "steel",
        "strawberry",
        "swamp",
        "tan",
        "taupe",
        "tawny",
        "teal",
        "tumeric",
        "umber",
        "violet",
        "walnut",
        "white",
        "wine",
        "yellow",
        "zombie",
        "zombie_green",
    }
)

_ANIMAL_SPECIES = frozenset(
    {"boarman", "cat", "mouse", "pig", "rabbit", "rat", "sheep", "wartotaur", "wolf"}
)
_FANTASY_SPECIES = frozenset({"goblin", "minotaur", "orc", "troll"})
_REPTILE_SPECIES = frozenset({"alien", "lizard"})
_UNDEAD_SPECIES = frozenset({"frankenstein", "jack", "skeleton", "vampire", "zombie"})
_REPOSITORY_ANCHORS = frozenset(
    {
        "credits.csv",
        "palette_definitions",
        "public",
        "readme-images",
        "scripts",
        "sheet_definitions",
        "sources",
        "spritesheets",
        "styles",
        "tests",
        "tools",
        "vite",
    }
)
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def classify_lpc_path(path: str) -> LpcPathInfo:
    """Classify an archive path and derive palette-independent sheet cues."""

    archive_path = _normalize_relative_path(path)
    repository_relative = _repository_relative_path(archive_path)
    parts = repository_relative.split("/")
    first = parts[0].casefold()
    suffix = PurePosixPath(repository_relative).suffix.casefold()
    basename = parts[-1].casefold()

    if basename == "credits.csv":
        return LpcPathInfo(archive_path, repository_relative, None, "credits")
    if first == "spritesheets":
        if suffix != ".png":
            kind = "sheet_directory" if not suffix else "sheet_auxiliary"
            return LpcPathInfo(archive_path, repository_relative, None, kind)
        return _classify_sheet_path(archive_path, repository_relative)
    if first == "sheet_definitions" and suffix == ".json":
        kind = "definition_meta" if basename.startswith("meta_") else "sheet_definition"
        return LpcPathInfo(archive_path, repository_relative, None, kind)
    if first == "palette_definitions" and suffix == ".json":
        kind = "palette_meta" if basename.startswith("meta_") else "palette_definition"
        return LpcPathInfo(archive_path, repository_relative, None, kind)
    if first == "readme-images":
        return LpcPathInfo(archive_path, repository_relative, None, "documentation_asset")
    if suffix in {".md", ".txt"} or basename.startswith(("readme", "license")):
        return LpcPathInfo(archive_path, repository_relative, None, "documentation")
    if first == "tools":
        is_layout = parts[1:2] == ["layout"] and suffix == ".json"
        kind = "layout_definition" if is_layout else "tool_asset"
        return LpcPathInfo(archive_path, repository_relative, None, kind)
    if first in {"sources", "public", "vite", "styles"}:
        return LpcPathInfo(archive_path, repository_relative, None, "ui_or_source")
    if suffix in {".js", ".ts", ".cjs", ".css", ".scss", ".html"}:
        return LpcPathInfo(archive_path, repository_relative, None, "source_code")
    return LpcPathInfo(archive_path, repository_relative, None, "other")


def parse_sheet_definition(
    payload: str | bytes | Mapping[str, Any],
    *,
    source_path: str | None = None,
) -> LpcSheetDefinition:
    """Parse one non-meta ``sheet_definitions`` JSON object."""

    data = _json_object(payload, label="sheet definition")
    name = _required_string(data, "name")
    type_name = _required_string(data, "type_name")

    layers: list[LpcLayerDefinition] = []
    for key, raw_layer in data.items():
        match = re.fullmatch(r"layer_(\d+)", key)
        if not match:
            continue
        if not isinstance(raw_layer, Mapping):
            raise LpcParseError(f"{key} must be an object")
        z_position = raw_layer.get("zPos")
        if z_position is not None and (
            isinstance(z_position, bool) or not isinstance(z_position, int)
        ):
            raise LpcParseError(f"{key}.zPos must be an integer")
        custom_animation = _optional_string(
            raw_layer.get("custom_animation"), f"{key}.custom_animation"
        )
        is_mask = raw_layer.get("is_mask", False)
        if not isinstance(is_mask, bool):
            raise LpcParseError(f"{key}.is_mask must be a boolean")

        body_paths: list[tuple[str, str]] = []
        for body_type, raw_path in raw_layer.items():
            if body_type in {"zPos", "custom_animation", "is_mask"}:
                continue
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise LpcParseError(f"{key}.{body_type} must be a non-empty path")
            body_paths.append((body_type, _normalize_relative_path(raw_path)))
        layers.append(
            LpcLayerDefinition(
                index=int(match.group(1)),
                z_position=z_position,
                body_paths=tuple(sorted(body_paths)),
                custom_animation=custom_animation,
                is_mask=is_mask,
            )
        )
    if not layers:
        raise LpcParseError("sheet definition has no layer_N object")

    credits_raw = data.get("credits", [])
    if not isinstance(credits_raw, Sequence) or isinstance(credits_raw, (str, bytes)):
        raise LpcParseError("credits must be an array")
    credits = tuple(parse_credit_row(item) for item in credits_raw)

    aliases_raw = data.get("aliases", {})
    if not isinstance(aliases_raw, Mapping):
        raise LpcParseError("aliases must be an object")
    aliases: list[tuple[str, str]] = []
    for alias, target in aliases_raw.items():
        if not isinstance(alias, str) or not isinstance(target, str):
            raise LpcParseError("aliases must map strings to strings")
        aliases.append((alias, target))

    priority = data.get("priority")
    if priority is not None and (isinstance(priority, bool) or not isinstance(priority, int)):
        raise LpcParseError("priority must be an integer")
    match_body_color = data.get("match_body_color", False)
    if not isinstance(match_body_color, bool):
        raise LpcParseError("match_body_color must be a boolean")

    return LpcSheetDefinition(
        source_path=_normalize_relative_path(source_path) if source_path else None,
        name=name,
        type_name=type_name,
        layers=tuple(sorted(layers, key=lambda layer: layer.index)),
        animations=_string_sequence(data.get("animations", ()), "animations"),
        variants=_string_sequence(data.get("variants", ()), "variants"),
        tags=_string_sequence(data.get("tags", ()), "tags"),
        required_tags=_string_sequence(data.get("required_tags", ()), "required_tags"),
        credits=credits,
        recolor_rules=_parse_recolor_rules(data.get("recolors")),
        aliases=tuple(sorted(aliases)),
        priority=priority,
        match_body_color=match_body_color,
    )


def parse_credit_row(row: Mapping[str, Any]) -> LpcCredit:
    """Parse either an embedded definition credit or one CREDITS.csv row."""

    if not isinstance(row, Mapping):
        raise LpcParseError("credit row must be an object")
    raw_filename = row.get("filename", row.get("file"))
    if not isinstance(raw_filename, str) or not raw_filename.strip():
        raise LpcParseError("credit row requires filename or file")
    filename = _normalize_relative_path(raw_filename.strip())
    notes = row.get("notes", "")
    if notes is None:
        notes = ""
    if not isinstance(notes, str):
        raise LpcParseError("credit notes must be text")
    return LpcCredit(
        filename=filename,
        notes=notes.strip(),
        authors=_credit_values(row.get("authors"), "authors"),
        licenses=_credit_values(row.get("licenses"), "licenses"),
        urls=_credit_values(row.get("urls"), "urls"),
    )


def parse_credits_csv(payload: str | bytes) -> tuple[LpcCredit, ...]:
    """Parse the generated LPC credit index, including UTF-8 BOM input."""

    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise LpcParseError("CREDITS.csv is not UTF-8") from exc
    elif isinstance(payload, str):
        text = payload.removeprefix("\ufeff")
    else:
        raise TypeError("payload must be str or bytes")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    required = {"filename", "notes", "authors", "licenses", "urls"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise LpcParseError("CREDITS.csv is missing required columns")
    return tuple(parse_credit_row(row) for row in reader)


def group_credits_by_filename(
    rows: Sequence[LpcCredit],
) -> dict[str, tuple[LpcCredit, ...]]:
    """Retain multiple provenance claims for filenames with duplicate rows."""

    grouped: defaultdict[str, list[LpcCredit]] = defaultdict(list)
    for row in rows:
        grouped[row.filename].append(row)
    return {filename: tuple(items) for filename, items in sorted(grouped.items())}


def credit_filename_candidates(
    path: str,
    *,
    custom_animation: str | None = None,
) -> tuple[str, ...]:
    """Return deterministic CREDITS.csv keys that may cover a generated sheet.

    Generated recolors add a palette filename and some split layers put
    foreground/background after the action. The credit index instead records an
    uncolored path with the plane before the action filename.
    """

    info = classify_lpc_path(path)
    if not info.is_sheet_candidate or info.content_path is None:
        return ()
    candidates = [info.content_path]
    tokens = _sheet_tokens(info.content_path)
    action_index = _rightmost_action_index(tokens)
    if action_index is None:
        if custom_animation:
            stemless = tokens[:-1]
            plane = tuple(token for token in stemless if token in _PLANE_TOKENS)
            prefix = tuple(token for token in stemless if token not in _PLANE_TOKENS)
            candidates.append("/".join((*prefix, *plane, f"{custom_animation}.png")))
        return tuple(dict.fromkeys(candidates))

    action = tokens[action_index]
    prefix = list(tokens[:action_index])
    tail = tokens[action_index + 1 :]
    planes = [token for token in tail if token in _PLANE_TOKENS]
    if action.startswith("attack_"):
        prefix.append(action)
    action_names = (action, *_CREDIT_ACTION_ALIASES.get(action, ()))
    if custom_animation:
        action_names = (*action_names, custom_animation)
    for action_name in action_names:
        candidates.append("/".join((*prefix, *planes, f"{action_name}.png")))
    return tuple(dict.fromkeys(candidates))


def parse_animation_layout(payload: str | bytes | Mapping[str, Any]) -> LpcAnimationLayout:
    """Parse ``tools/layout/universal-expanded.json``-style row metadata."""

    data = _json_object(payload, label="animation layout")
    frame_size = _integer_pair(data.get("frame_size"), "frame_size")
    grid_size = _integer_pair(data.get("size"), "size")
    rows = data.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise LpcParseError("rows must be an array")
    if len(rows) != grid_size[1]:
        raise LpcParseError("layout row count does not match size")

    parsed_rows: list[LpcDirectionLayout] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise LpcParseError(f"rows[{row_index}] must be an array")
        if len(row) != grid_size[0]:
            raise LpcParseError(f"rows[{row_index}] width does not match size")
        populated = [cell for cell in row if cell is not None]
        if not populated:
            continue
        if not all(isinstance(cell, Mapping) for cell in populated):
            raise LpcParseError(f"rows[{row_index}] cells must be objects or null")
        actions = {_required_string(cell, "name") for cell in populated}
        directions = {_required_string(cell, "direction") for cell in populated}
        if len(actions) != 1 or len(directions) != 1:
            raise LpcParseError(f"rows[{row_index}] mixes actions or directions")
        frame_indices: list[int] = []
        for cell in populated:
            frame = cell.get("frame")
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                raise LpcParseError(f"rows[{row_index}] has an invalid frame index")
            frame_indices.append(frame)
        parsed_rows.append(
            LpcDirectionLayout(
                source_action=actions.pop(),
                direction=_normalize_direction(directions.pop()),
                row_index=row_index,
                frame_indices=tuple(frame_indices),
            )
        )
    return LpcAnimationLayout(
        frame_width=frame_size[0],
        frame_height=frame_size[1],
        columns=grid_size[0],
        rows=grid_size[1],
        direction_layouts=tuple(parsed_rows),
    )


def sheet_animation_cues(
    path: str,
    *,
    width: int,
    height: int,
    strict: bool = True,
) -> tuple[LpcAnimationCue, ...]:
    """Infer per-direction animation cues from an action-split LPC PNG geometry."""

    info = classify_lpc_path(path)
    if not info.is_sheet_candidate or info.source_action is None or info.layer_identity is None:
        return ()
    spec = LPC_ACTION_SPECS[info.source_action]
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    direction_count = len(spec.directions)
    if height % direction_count:
        if strict:
            raise LpcParseError(
                f"{info.content_path!r} height {height} is not divisible by "
                f"{direction_count} directions"
            )
        return ()
    frame_size = height // direction_count
    if frame_size <= 0 or width % frame_size:
        if strict:
            raise LpcParseError(
                f"{info.content_path!r} width {width} is not divisible by frame size {frame_size}"
            )
        return ()
    frame_count = width // frame_size
    canonical = frame_size == spec.canonical_frame_size and frame_count == spec.canonical_frames
    return tuple(
        LpcAnimationCue(
            stable_id=f"{info.layer_identity}|{info.source_action}|{direction}",
            layer_identity=info.layer_identity,
            source_action=info.source_action,
            normalized_action=spec.normalized_action,
            view="top_down_cardinal",
            direction=direction,
            frame_size=frame_size,
            frame_count=frame_count,
            loopable=spec.loopable,
            canonical_geometry=canonical,
        )
        for direction in spec.directions
    )


def parse_palette_definition(
    payload: str | bytes | Mapping[str, Any],
    *,
    source_path: str,
) -> LpcPaletteDefinition:
    """Parse a non-meta palette definition and retain every ordered color ramp."""

    normalized_source = _normalize_relative_path(source_path)
    info = classify_lpc_path(normalized_source)
    if info.kind != "palette_definition":
        raise LpcParseError("source_path is not a non-meta palette definition")
    data = _json_object(payload, label="palette definition")
    path = PurePosixPath(info.repository_relative_path)
    if len(path.parts) < 3:
        raise LpcParseError("palette definition path lacks a material directory")
    material = path.parts[1]
    stem = path.stem
    scheme = stem.removeprefix(f"{material}_")
    palettes: list[LpcPalette] = []
    for name, raw_colors in data.items():
        if not isinstance(name, str) or not name:
            raise LpcParseError("palette names must be non-empty strings")
        colors = _string_sequence(raw_colors, f"palette {name}")
        if not colors or any(not _HEX_COLOR.fullmatch(color) for color in colors):
            raise LpcParseError(f"palette {name!r} must contain #RRGGBB colors")
        palettes.append(LpcPalette(name=name, colors=colors))
    return LpcPaletteDefinition(
        source_path=info.repository_relative_path,
        material=material,
        scheme=scheme,
        palettes=tuple(palettes),
    )


def _classify_sheet_path(archive_path: str, repository_relative: str) -> LpcPathInfo:
    content_path = repository_relative.removeprefix("spritesheets/")
    tokens = _sheet_tokens(content_path)
    action_index = _rightmost_action_index(tokens)
    source_action = tokens[action_index] if action_index is not None else None
    spec = LPC_ACTION_SPECS.get(source_action) if source_action else None
    stem = tokens[-1]

    if action_index is None:
        identity_tokens = list(tokens)
        palette = stem if _looks_like_palette(stem) else None
        if palette:
            identity_tokens.pop()
    else:
        prefix = list(tokens[:action_index])
        tail = list(tokens[action_index + 1 :])
        if source_action.startswith("attack_"):
            prefix.append(source_action)
        planes = [token for token in tail if token in _PLANE_TOKENS]
        non_planes = [token for token in tail if token not in _PLANE_TOKENS]
        palette = non_planes[-1] if non_planes and _looks_like_palette(non_planes[-1]) else None
        identity_tail = non_planes[:-1] if palette else non_planes
        identity_tail = [token for token in identity_tail if token not in prefix]
        identity_tokens = [*prefix, *planes, *identity_tail]

    plane_tokens = [token for token in tokens if token in _PLANE_TOKENS]
    body_tokens = [token for token in tokens if token in _BODY_TYPES]
    category = tokens[0] if tokens else None
    identity_path = "/".join(identity_tokens)
    return LpcPathInfo(
        archive_path=archive_path,
        repository_relative_path=repository_relative,
        content_path=content_path,
        kind="sheet",
        category=category,
        layer_identity=f"ulpc:{identity_path}" if identity_path else None,
        source_action=source_action,
        normalized_action=spec.normalized_action if spec else None,
        body_type=body_tokens[-1] if body_tokens else None,
        palette=palette,
        plane=plane_tokens[-1] if plane_tokens else None,
        entity_family=_infer_entity_family(tokens),
    )


def _sheet_tokens(content_path: str) -> tuple[str, ...]:
    parts = content_path.split("/")
    parts[-1] = PurePosixPath(parts[-1]).stem
    return tuple(parts)


def _rightmost_action_index(tokens: Sequence[str]) -> int | None:
    return next(
        (index for index in range(len(tokens) - 1, -1, -1) if tokens[index] in LPC_ACTION_SPECS),
        None,
    )


def _looks_like_palette(token: str) -> bool:
    if token in {"base", "original"} or token in _KNOWN_COLOR_NAMES:
        return True
    components = token.split("_")
    return len(components) > 1 and all(component in _KNOWN_COLOR_NAMES for component in components)


def _infer_entity_family(tokens: Sequence[str]) -> str:
    token_set = set(tokens)
    if token_set & _ANIMAL_SPECIES:
        return "anthropomorphic_animal"
    if token_set & _FANTASY_SPECIES:
        return "fantasy_humanoid"
    if token_set & _REPTILE_SPECIES:
        return "reptilian_humanoid"
    if token_set & _UNDEAD_SPECIES:
        return "undead_humanoid"
    return "humanoid_layer"


def _repository_relative_path(path: str) -> str:
    parts = path.split("/")
    for index, part in enumerate(parts):
        if part.casefold() in _REPOSITORY_ANCHORS:
            return "/".join(parts[index:])
    if len(parts) > 1 and parts[0].casefold().startswith("universal-lpc"):
        return "/".join(parts[1:])
    return path


def _normalize_relative_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise LpcParseError("path must be non-empty text")
    portable = unicodedata.normalize("NFC", path.strip().replace("\\", "/"))
    if portable.startswith("/") or re.match(r"^[A-Za-z]:", portable):
        raise LpcParseError(f"absolute LPC path is not accepted: {path!r}")
    parts: list[str] = []
    for part in portable.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            raise LpcParseError(f"traversal in LPC path: {path!r}")
        parts.append(part)
    if not parts:
        raise LpcParseError("path has no normalized components")
    return "/".join(parts)


def _json_object(payload: str | bytes | Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        data = payload
    else:
        if isinstance(payload, bytes):
            try:
                text = payload.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise LpcParseError(f"{label} is not UTF-8") from exc
        elif isinstance(payload, str):
            text = payload
        else:
            raise TypeError("JSON payload must be str, bytes, or mapping")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LpcParseError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise LpcParseError(f"{label} root must be an object")
    return data


def _parse_recolor_rules(raw: Any) -> tuple[LpcRecolorRule, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise LpcParseError("recolors must be an object")
    channels: list[tuple[str, Mapping[str, Any]]] = []
    if "material" in raw or "palettes" in raw:
        channels.append(("default", raw))
    for key, value in raw.items():
        if key.startswith("color_"):
            if not isinstance(value, Mapping):
                raise LpcParseError(f"recolors.{key} must be an object")
            channels.append((key, value))
    rules: list[LpcRecolorRule] = []
    for channel, value in channels:
        rules.append(
            LpcRecolorRule(
                channel=channel,
                material=_optional_string(value.get("material"), f"recolors.{channel}.material"),
                base=_optional_string(value.get("base"), f"recolors.{channel}.base"),
                palettes=_string_sequence(
                    value.get("palettes", ()), f"recolors.{channel}.palettes"
                ),
                type_name=_optional_string(value.get("type_name"), f"recolors.{channel}.type_name"),
                label=_optional_string(value.get("label"), f"recolors.{channel}.label"),
            )
        )
    return tuple(rules)


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LpcParseError(f"{key} must be non-empty text")
    return value.strip()


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LpcParseError(f"{label} must be text or null")
    return value.strip() or None


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LpcParseError(f"{label} must be an array of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise LpcParseError(f"{label} must contain non-empty strings")
        result.append(item.strip())
    return tuple(result)


def _credit_values(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _string_sequence(value, label)
    raise LpcParseError(f"credit {label} must be text or an array of strings")


def _integer_pair(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise LpcParseError(f"{label} must contain two positive integers")
    return value[0], value[1]


def _normalize_direction(direction: str) -> str:
    aliases = {
        "n": "north",
        "north": "north",
        "w": "west",
        "west": "west",
        "s": "south",
        "south": "south",
        "e": "east",
        "east": "east",
    }
    normalized = aliases.get(direction.casefold())
    if normalized is None:
        raise LpcParseError(f"unknown LPC direction: {direction!r}")
    return normalized
