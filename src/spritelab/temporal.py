"""Explicit, interpolation-free temporal selection for sprite clips."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

TemporalLoopMode = Literal["loop", "one_shot", "ping_pong"]


@dataclass(frozen=True, slots=True)
class TemporalSelection:
    """Recorded mapping from canonical output phases to authored frame ordinals."""

    source_frame_count: int
    target_frame_count: int
    loop_mode: TemporalLoopMode
    source_phases: tuple[float, ...]
    target_phases: tuple[float, ...]
    source_ordinals: tuple[int, ...]
    phase_offset: float
    selection_method: str = "nearest_authored_phase_no_interpolation_v1"

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def select_temporal_frames(
    source_frame_count: int,
    target_frame_count: int,
    *,
    loop_mode: TemporalLoopMode,
    source_phases: Sequence[float] | None = None,
    phase_offset: float = 0.0,
) -> TemporalSelection:
    """Select whole source frames at canonical phases without pixel interpolation.

    Loops use phases in ``[0, 1)`` and permit a cyclic ``phase_offset``. One-shot
    traversals include both endpoints. Ping-pong clips describe one complete
    forward-and-back cycle in ``[0, 1)``; the target phase is folded onto the
    authored forward traversal before nearest-frame selection.
    """

    _positive_integer("source_frame_count", source_frame_count)
    _positive_integer("target_frame_count", target_frame_count)
    if loop_mode not in {"loop", "one_shot", "ping_pong"}:
        raise ValueError(f"loop_mode must be 'loop', 'one_shot', or 'ping_pong'; got {loop_mode!r}")
    if not isinstance(phase_offset, int | float) or isinstance(phase_offset, bool):
        raise ValueError("phase_offset must be a finite number")
    phase_offset = float(phase_offset)
    if not math.isfinite(phase_offset):
        raise ValueError("phase_offset must be a finite number")
    if loop_mode == "one_shot" and phase_offset != 0.0:
        raise ValueError("phase_offset must be zero for one_shot clips")
    if loop_mode != "one_shot" and not 0.0 <= phase_offset < 1.0:
        raise ValueError("phase_offset must be in [0, 1) for cyclic clips")

    phases = _source_phases(
        source_frame_count,
        loop_mode=loop_mode,
        supplied=source_phases,
    )
    target_phases = _target_phases(
        target_frame_count,
        loop_mode=loop_mode,
        phase_offset=phase_offset,
    )
    ordinals: list[int] = []
    for target_phase in target_phases:
        if loop_mode == "loop":
            ordinal = min(
                range(source_frame_count),
                key=lambda index: (_cyclic_distance(phases[index], target_phase), index),
            )
        else:
            traversal_phase = (
                _ping_pong_traversal_phase(target_phase)
                if loop_mode == "ping_pong"
                else target_phase
            )
            ordinal = min(
                range(source_frame_count),
                key=lambda index: (abs(phases[index] - traversal_phase), index),
            )
        ordinals.append(ordinal)
    return TemporalSelection(
        source_frame_count=source_frame_count,
        target_frame_count=target_frame_count,
        loop_mode=loop_mode,
        source_phases=phases,
        target_phases=target_phases,
        source_ordinals=tuple(ordinals),
        phase_offset=phase_offset,
    )


def apply_temporal_selection[T](
    frames: Sequence[T],
    selection: TemporalSelection,
) -> tuple[T, ...]:
    """Apply a previously recorded mapping to an ordered frame sequence."""

    if len(frames) != selection.source_frame_count:
        raise ValueError(
            "frames length must match selection.source_frame_count; "
            f"got {len(frames)} and {selection.source_frame_count}"
        )
    return tuple(frames[index] for index in selection.source_ordinals)


def _source_phases(
    frame_count: int,
    *,
    loop_mode: TemporalLoopMode,
    supplied: Sequence[float] | None,
) -> tuple[float, ...]:
    if supplied is None:
        if frame_count == 1:
            return (0.0,)
        denominator = frame_count if loop_mode == "loop" else frame_count - 1
        return tuple(index / denominator for index in range(frame_count))
    phases = tuple(float(value) for value in supplied)
    if len(phases) != frame_count:
        raise ValueError(
            "source_phases length must match source_frame_count; "
            f"got {len(phases)} and {frame_count}"
        )
    previous = -math.inf
    for index, phase in enumerate(phases):
        if not math.isfinite(phase):
            raise ValueError(f"source_phases[{index}] must be finite")
        upper_ok = phase < 1.0 if loop_mode == "loop" else phase <= 1.0
        if phase < 0.0 or not upper_ok:
            interval = "[0, 1)" if loop_mode == "loop" else "[0, 1]"
            raise ValueError(f"source_phases[{index}] must be in {interval}; got {phase}")
        if phase < previous:
            raise ValueError("source_phases must be nondecreasing")
        previous = phase
    if len(set(phases)) != len(phases):
        raise ValueError("source_phases must not contain duplicates")
    return phases


def _target_phases(
    frame_count: int,
    *,
    loop_mode: TemporalLoopMode,
    phase_offset: float,
) -> tuple[float, ...]:
    if loop_mode == "one_shot":
        if frame_count == 1:
            return (0.0,)
        return tuple(index / (frame_count - 1) for index in range(frame_count))
    return tuple((phase_offset + index / frame_count) % 1.0 for index in range(frame_count))


def _ping_pong_traversal_phase(cycle_phase: float) -> float:
    return 2.0 * cycle_phase if cycle_phase <= 0.5 else 2.0 * (1.0 - cycle_phase)


def _cyclic_distance(left: float, right: float) -> float:
    distance = abs(left - right)
    return min(distance, 1.0 - distance)


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
