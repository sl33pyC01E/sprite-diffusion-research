"""Torch-free configuration and input contracts for sprite generation models."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

RGBA_CHANNELS = 4

DEFAULT_ENTITY_CLASSES = (
    "humanoid",
    "animal",
    "creature",
    "monster",
    "robot",
    "vehicle",
    "object",
    "prop",
    "effect",
    "projectile",
    "environment",
    "other",
    "unknown",
)
DEFAULT_ACTION_CLASSES = (
    "idle",
    "crouch",
    "sleep",
    "walk",
    "run",
    "sprint",
    "crawl",
    "jump",
    "fall",
    "land",
    "climb",
    "swim",
    "fly",
    "hover",
    "burrow",
    "attack",
    "attack_melee",
    "attack_ranged",
    "shoot",
    "cast",
    "defend",
    "dodge",
    "hurt",
    "death",
    "interact",
    "use",
    "work",
    "carry",
    "push",
    "pull",
    "eat",
    "drink",
    "talk",
    "emote",
    "dance",
    "celebrate",
    "laugh",
    "cry",
    "spawn",
    "despawn",
    "transform",
    "sit",
    "action",
    "custom",
    "other",
    "unknown",
)
DEFAULT_VIEW_CLASSES = (
    "front",
    "three_quarter",
    "side",
    "back",
    "top_down",
    "isometric",
    "other",
    "unknown",
)
DEFAULT_DIRECTION_CLASSES = (
    "none",
    "unknown",
    "left",
    "right",
    "up",
    "down",
    "north",
    "south",
    "east",
    "west",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
)
DEFAULT_LOOP_MODES = ("loop", "one_shot", "ping_pong", "unknown")
DEFAULT_CONDITION_TOKENS = (
    "identity",
    "entity",
    "action",
    "view",
    "direction",
    "loop",
)

_REQUIRED_ENTITY_CLASSES = frozenset(
    {
        "humanoid",
        "animal",
        "creature",
        "monster",
        "robot",
        "vehicle",
        "object",
        "effect",
        "other",
        "unknown",
    }
)
_REQUIRED_ACTION_CLASSES = frozenset({"idle", "run", "emote", "action", "unknown"})
_REQUIRED_VIEW_CLASSES = frozenset({"front", "side", "back", "top_down", "isometric", "unknown"})
_REQUIRED_DIRECTION_CLASSES = frozenset({"none", "left", "right", "unknown"})
_REQUIRED_LOOP_MODES = frozenset({"loop", "one_shot", "ping_pong", "unknown"})
_REQUIRED_CONDITION_TOKENS = frozenset(DEFAULT_CONDITION_TOKENS)
_SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _validate_vocabulary(
    name: str,
    values: tuple[str, ...],
    *,
    required: frozenset[str],
) -> None:
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicate labels")
    invalid = [value for value in values if not _SLUG_PATTERN.fullmatch(value)]
    if invalid:
        raise ValueError(f"{name} labels must be lowercase slugs; invalid: {invalid!r}")
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"{name} is missing required labels: {missing!r}")


@dataclass(frozen=True, slots=True)
class ConditioningSchema:
    """Vocabulary and grouping contract for structured sprite conditioning.

    Text embeddings can add arbitrary detail, while these controlled labels make
    actions and rendering orientation measurable and steerable. ``identity`` is
    deliberately separate from ``entity`` so one character can share appearance
    across several action clips.
    """

    entity_classes: tuple[str, ...] = DEFAULT_ENTITY_CLASSES
    action_classes: tuple[str, ...] = DEFAULT_ACTION_CLASSES
    view_classes: tuple[str, ...] = DEFAULT_VIEW_CLASSES
    direction_classes: tuple[str, ...] = DEFAULT_DIRECTION_CLASSES
    loop_modes: tuple[str, ...] = DEFAULT_LOOP_MODES
    condition_tokens: tuple[str, ...] = DEFAULT_CONDITION_TOKENS
    phase_bins: int = 8
    min_actions_per_identity: int = 2

    def __post_init__(self) -> None:
        _validate_vocabulary(
            "entity_classes", self.entity_classes, required=_REQUIRED_ENTITY_CLASSES
        )
        _validate_vocabulary(
            "action_classes", self.action_classes, required=_REQUIRED_ACTION_CLASSES
        )
        _validate_vocabulary("view_classes", self.view_classes, required=_REQUIRED_VIEW_CLASSES)
        _validate_vocabulary(
            "direction_classes",
            self.direction_classes,
            required=_REQUIRED_DIRECTION_CLASSES,
        )
        _validate_vocabulary("loop_modes", self.loop_modes, required=_REQUIRED_LOOP_MODES)
        _validate_vocabulary(
            "condition_tokens",
            self.condition_tokens,
            required=_REQUIRED_CONDITION_TOKENS,
        )
        _require_positive_int("phase_bins", self.phase_bins)
        _require_positive_int("min_actions_per_identity", self.min_actions_per_identity)
        if self.min_actions_per_identity < 2:
            raise ValueError("min_actions_per_identity must be at least 2")

    @property
    def minimum_condition_tokens(self) -> int:
        return len(self.condition_tokens)

    def validate_clip(self, clip: SpriteClipCondition, *, expected_frames: int) -> None:
        """Validate one normalized clip against the controlled vocabulary."""

        _require_positive_int("expected_frames", expected_frames)
        if not clip.identity_id.strip():
            raise ValueError("identity_id cannot be empty")
        if clip.entity_class not in self.entity_classes:
            raise ValueError(f"unknown entity_class: {clip.entity_class!r}")
        if clip.action not in self.action_classes:
            raise ValueError(f"unknown action: {clip.action!r}")
        if clip.view not in self.view_classes:
            raise ValueError(f"unknown view: {clip.view!r}")
        if clip.direction not in self.direction_classes:
            raise ValueError(f"unknown direction: {clip.direction!r}")
        if clip.loop_mode not in self.loop_modes:
            raise ValueError(f"unknown loop_mode: {clip.loop_mode!r}")
        if len(clip.frame_phases) != expected_frames:
            raise ValueError(
                "frame_phases length must match expected_frames; "
                f"got {len(clip.frame_phases)} and {expected_frames}"
            )
        for index, phase in enumerate(clip.frame_phases):
            if not math.isfinite(phase):
                raise ValueError(f"frame_phases[{index}] must be finite")
            endpoint_mode = clip.loop_mode in {"one_shot", "unknown"}
            upper_bound = 1.0 if endpoint_mode else math.nextafter(1.0, 0.0)
            if phase < 0.0 or phase > upper_bound:
                interval = "[0, 1]" if endpoint_mode else "[0, 1)"
                raise ValueError(f"frame_phases[{index}] must be in {interval}; got {phase}")
        if clip.loop_mode == "one_shot" and any(
            later < earlier
            for earlier, later in zip(clip.frame_phases, clip.frame_phases[1:], strict=False)
        ):
            raise ValueError("one_shot frame phases must be nondecreasing")


@dataclass(frozen=True, slots=True)
class SpriteClipCondition:
    """Torch-free labels attached to one ordered, normalized animation clip."""

    identity_id: str
    entity_class: str
    action: str
    view: str
    direction: str
    loop_mode: str
    frame_phases: tuple[float, ...]
    sequence_id: str | None = None


def validate_identity_action_groups(
    clips: Iterable[SpriteClipCondition],
    schema: ConditioningSchema,
) -> dict[str, tuple[str, ...]]:
    """Require each identity group to expose multiple distinct actions.

    The returned index is stable and suitable for constructing grouped samplers.
    Validation is intentionally independent of PyTorch and dataset storage.
    """

    grouped: dict[str, set[str]] = {}
    for clip in clips:
        identity_id = clip.identity_id.strip()
        if not identity_id:
            raise ValueError("identity_id cannot be empty")
        if clip.action not in schema.action_classes:
            raise ValueError(f"unknown action for identity {identity_id!r}: {clip.action!r}")
        grouped.setdefault(identity_id, set()).add(clip.action)

    if not grouped:
        raise ValueError("at least one identity/action clip is required")

    insufficient = {
        identity_id: sorted(actions)
        for identity_id, actions in grouped.items()
        if len(actions) < schema.min_actions_per_identity
    }
    if insufficient:
        raise ValueError(
            "identities must have at least "
            f"{schema.min_actions_per_identity} distinct actions: {insufficient!r}"
        )
    return {identity_id: tuple(sorted(actions)) for identity_id, actions in sorted(grouped.items())}


@dataclass(frozen=True, slots=True)
class PatchGrid:
    """Derived patch geometry for a fixed-size sprite clip."""

    frames: int
    rows: int
    columns: int
    patch_size: int
    channels: int

    @property
    def tokens_per_frame(self) -> int:
        return self.rows * self.columns

    @property
    def total_tokens(self) -> int:
        return self.frames * self.tokens_per_frame

    @property
    def pixels_per_patch(self) -> int:
        return self.patch_size**2

    @property
    def patch_vector_width(self) -> int:
        return self.channels * self.pixels_per_patch


@dataclass(frozen=True, slots=True)
class ConditioningShape:
    """Normalized shape of a batch of context tokens."""

    batch_size: int
    tokens: int
    width: int


@dataclass(frozen=True, slots=True)
class PixelDiTConfig:
    """Configuration for a fixed 64x64x8 native-RGBA PixelDiT.

    Smaller geometry may be supplied for unit tests and scaling experiments, but
    all dimensions remain explicit and are checked before the torch model runs.
    """

    height: int = 64
    width: int = 64
    num_frames: int = 8
    channels: int = RGBA_CHANNELS
    patch_size: int = 2
    model_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    condition_dim: int = 768
    phase_harmonics: int = 4
    conditioning: ConditioningSchema = field(default_factory=ConditioningSchema)

    def __post_init__(self) -> None:
        for name in (
            "height",
            "width",
            "num_frames",
            "channels",
            "patch_size",
            "model_dim",
            "depth",
            "num_heads",
            "condition_dim",
            "phase_harmonics",
        ):
            _require_positive_int(name, getattr(self, name))
        if self.channels != RGBA_CHANNELS:
            raise ValueError(
                f"native sprite diffusion requires exactly {RGBA_CHANNELS} RGBA channels"
            )
        if self.height % self.patch_size or self.width % self.patch_size:
            raise ValueError(
                "height and width must both be divisible by patch_size; "
                f"got {(self.height, self.width)} and {self.patch_size}"
            )
        if self.model_dim % self.num_heads:
            raise ValueError(
                "model_dim must be divisible by num_heads; "
                f"got {self.model_dim} and {self.num_heads}"
            )
        if not math.isfinite(self.mlp_ratio) or self.mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be finite and positive; got {self.mlp_ratio!r}")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout!r}")
        if self.conditioning.phase_bins != self.num_frames:
            raise ValueError(
                "conditioning.phase_bins must equal num_frames for the fixed-phase scaffold; "
                f"got {self.conditioning.phase_bins} and {self.num_frames}"
            )

    @property
    def patch_grid(self) -> PatchGrid:
        return PatchGrid(
            frames=self.num_frames,
            rows=self.height // self.patch_size,
            columns=self.width // self.patch_size,
            patch_size=self.patch_size,
            channels=self.channels,
        )

    def expected_video_shape(self, batch_size: int) -> tuple[int, int, int, int, int]:
        _require_positive_int("batch_size", batch_size)
        return (batch_size, self.num_frames, self.channels, self.height, self.width)


def _normalize_shape(name: str, shape: Sequence[int], *, rank: int) -> tuple[int, ...]:
    normalized = tuple(shape)
    if len(normalized) != rank:
        raise ValueError(f"{name} must have rank {rank}; got shape {normalized!r}")
    for index, dimension in enumerate(normalized):
        _require_positive_int(f"{name}[{index}]", dimension)
    return normalized


def validate_video_shape(shape: Sequence[int], config: PixelDiTConfig) -> PatchGrid:
    """Validate the public ``[B, T, C, H, W]`` video tensor contract."""

    normalized = _normalize_shape("video", shape, rank=5)
    expected_tail = (config.num_frames, config.channels, config.height, config.width)
    if normalized[1:] != expected_tail:
        raise ValueError(
            f"video must use [B, T, C, H, W] with tail {expected_tail!r}; got {normalized!r}"
        )
    return config.patch_grid


def validate_conditioning_shape(
    shape: Sequence[int],
    config: PixelDiTConfig,
    *,
    batch_size: int,
) -> ConditioningShape:
    """Validate pooled ``[B, D]`` or token ``[B, L, D]`` conditioning."""

    _require_positive_int("batch_size", batch_size)
    normalized = tuple(shape)
    if len(normalized) not in {2, 3}:
        raise ValueError(f"conditioning must have shape [B, D] or [B, L, D]; got {normalized!r}")
    for index, dimension in enumerate(normalized):
        _require_positive_int(f"conditioning[{index}]", dimension)
    if normalized[0] != batch_size:
        raise ValueError(f"conditioning batch must be {batch_size}; got {normalized[0]}")
    width = normalized[-1]
    if width != config.condition_dim:
        raise ValueError(f"conditioning width must be {config.condition_dim}; got {width}")
    tokens = 1 if len(normalized) == 2 else normalized[1]
    return ConditioningShape(batch_size=batch_size, tokens=tokens, width=width)


def validate_phase_shape(
    shape: Sequence[int],
    config: PixelDiTConfig,
    *,
    batch_size: int,
) -> tuple[int, int]:
    """Validate explicit normalized phase inputs with shape ``[B, T]``."""

    _require_positive_int("batch_size", batch_size)
    normalized = _normalize_shape("frame_phase", shape, rank=2)
    expected = (batch_size, config.num_frames)
    if normalized != expected:
        raise ValueError(f"frame_phase must have shape {expected!r}; got {normalized!r}")
    return normalized
