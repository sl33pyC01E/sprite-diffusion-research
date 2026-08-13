from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Split = Literal["train", "validation", "test"]
AssignmentStrategy = Literal["hash", "balanced"]


@dataclass(frozen=True, slots=True)
class SequenceSample:
    sequence_id: str
    identity_id: str
    source_id: str
    source_pack_id: str
    entity_class: str
    action: str
    view: str
    direction: str
    loop_mode: str
    frame_count: int
    source_blob_sha256: tuple[str, ...]
    duplicate_group_ids: tuple[str, ...] = ()
    quality_tier: str = "unknown"
    sample_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "sequence_id",
            "identity_id",
            "source_id",
            "source_pack_id",
            "entity_class",
            "action",
            "view",
            "direction",
            "loop_mode",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if not math.isfinite(self.sample_weight) or self.sample_weight <= 0:
            raise ValueError("sample_weight must be finite and positive")
        for digest in self.source_blob_sha256:
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"Invalid source blob SHA-256: {digest!r}")


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float = 0.9
    validation: float = 0.05
    test: float = 0.05

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("split ratios must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("split ratios must sum to 1")


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    seed: str
    ratios: SplitRatios = field(default_factory=SplitRatios)
    assignment_strategy: AssignmentStrategy = "hash"
    group_identity: bool = True
    group_source_pack: bool = True
    group_exact_blobs: bool = True
    group_duplicate_ids: bool = True

    def __post_init__(self) -> None:
        if not self.seed:
            raise ValueError("split seed cannot be empty")
        if self.assignment_strategy not in {"hash", "balanced"}:
            raise ValueError(f"Unknown assignment strategy: {self.assignment_strategy!r}")
        if not any(
            (
                self.group_identity,
                self.group_source_pack,
                self.group_exact_blobs,
                self.group_duplicate_ids,
            )
        ):
            raise ValueError("at least one leakage grouping axis must be enabled")


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    sequence_id: str
    split: Split
    component_id: str


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: int
    policy: SplitPolicy
    samples: tuple[SequenceSample, ...]
    assignments: tuple[SplitAssignment, ...]

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            asdict(self),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CoverageReport:
    sample_count: int
    identity_count: int
    multi_action_identity_count: int
    split_counts: dict[str, int]
    entity_counts: dict[str, int]
    action_counts: dict[str, int]
    entity_action_counts: dict[str, int]
    temporal_sequence_count: int
    single_frame_sequence_count: int


class _UnionFind:
    def __init__(self, keys: Iterable[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root), key=lambda value: value.encode("utf-8"))
        self.parent[high] = low


def build_dataset_manifest(
    samples: Iterable[SequenceSample],
    policy: SplitPolicy,
) -> DatasetManifest:
    ordered = tuple(sorted(samples, key=lambda sample: sample.sequence_id.encode("utf-8")))
    if not ordered:
        raise ValueError("at least one sequence sample is required")
    ids = [sample.sequence_id for sample in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("sequence IDs must be unique")

    components = leakage_components(ordered, policy)
    assignments: list[SplitAssignment] = []
    balanced_splits = (
        _balanced_component_splits(components, policy)
        if policy.assignment_strategy == "balanced"
        else {}
    )
    for component_id, members in sorted(components.items()):
        split = (
            balanced_splits[component_id]
            if policy.assignment_strategy == "balanced"
            else _hash_split(policy.seed, component_id, policy.ratios)
        )
        assignments.extend(
            SplitAssignment(
                sequence_id=sequence_id,
                split=split,
                component_id=component_id,
            )
            for sequence_id in sorted(members, key=lambda value: value.encode("utf-8"))
        )
    assignments.sort(key=lambda assignment: assignment.sequence_id.encode("utf-8"))
    manifest = DatasetManifest(
        schema_version=1,
        policy=policy,
        samples=ordered,
        assignments=tuple(assignments),
    )
    validate_no_leakage(manifest)
    return manifest


def leakage_components(
    samples: tuple[SequenceSample, ...],
    policy: SplitPolicy,
) -> dict[str, tuple[str, ...]]:
    union = _UnionFind(sample.sequence_id for sample in samples)
    indexes: list[dict[str, list[str]]] = []
    if policy.group_identity:
        indexes.append(_index_values(samples, lambda sample: (sample.identity_id,)))
    if policy.group_source_pack:
        indexes.append(_index_values(samples, lambda sample: (sample.source_pack_id,)))
    if policy.group_exact_blobs:
        indexes.append(_index_values(samples, lambda sample: sample.source_blob_sha256))
    if policy.group_duplicate_ids:
        indexes.append(_index_values(samples, lambda sample: sample.duplicate_group_ids))
    for index in indexes:
        for members in index.values():
            anchor = members[0]
            for member in members[1:]:
                union.union(anchor, member)

    grouped: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        grouped[union.find(sample.sequence_id)].append(sample.sequence_id)
    result: dict[str, tuple[str, ...]] = {}
    for members in grouped.values():
        ordered = tuple(sorted(members, key=lambda value: value.encode("utf-8")))
        digest = hashlib.sha256("\0".join(ordered).encode("utf-8")).hexdigest()
        result[f"component:{digest}"] = ordered
    return result


def validate_no_leakage(manifest: DatasetManifest) -> None:
    by_sequence = {assignment.sequence_id: assignment for assignment in manifest.assignments}
    if set(by_sequence) != {sample.sequence_id for sample in manifest.samples}:
        raise ValueError("split assignments do not cover exactly the manifest samples")
    components: dict[str, set[Split]] = defaultdict(set)
    for assignment in manifest.assignments:
        components[assignment.component_id].add(assignment.split)
    leaked = {key: values for key, values in components.items() if len(values) > 1}
    if leaked:
        raise ValueError(f"leakage components cross splits: {leaked}")


def multi_action_identity_groups(
    samples: Iterable[SequenceSample],
    *,
    minimum_actions: int = 2,
) -> dict[str, tuple[str, ...]]:
    if minimum_actions < 1:
        raise ValueError("minimum_actions must be positive")
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        grouped[sample.identity_id][sample.action].append(sample.sequence_id)
    return {
        identity: tuple(
            sequence_id
            for action in sorted(actions, key=lambda value: value.encode("utf-8"))
            for sequence_id in sorted(actions[action], key=lambda value: value.encode("utf-8"))
        )
        for identity, actions in sorted(grouped.items())
        if len(actions) >= minimum_actions
    }


def coverage_report(manifest: DatasetManifest) -> CoverageReport:
    splits = Counter(assignment.split for assignment in manifest.assignments)
    entities = Counter(sample.entity_class for sample in manifest.samples)
    actions = Counter(sample.action for sample in manifest.samples)
    pairs = Counter(f"{sample.entity_class}:{sample.action}" for sample in manifest.samples)
    identities = {sample.identity_id for sample in manifest.samples}
    multi_action = multi_action_identity_groups(manifest.samples)
    temporal = sum(sample.frame_count > 1 for sample in manifest.samples)
    return CoverageReport(
        sample_count=len(manifest.samples),
        identity_count=len(identities),
        multi_action_identity_count=len(multi_action),
        split_counts=dict(sorted(splits.items())),
        entity_counts=dict(sorted(entities.items())),
        action_counts=dict(sorted(actions.items())),
        entity_action_counts=dict(sorted(pairs.items())),
        temporal_sequence_count=temporal,
        single_frame_sequence_count=len(manifest.samples) - temporal,
    )


def canonical_frame_phases(frame_count: int, loop_mode: str) -> tuple[float, ...]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frame_count == 1:
        return (0.0,)
    if loop_mode == "loop":
        return tuple(index / frame_count for index in range(frame_count))
    if loop_mode in {"one_shot", "unknown"}:
        return tuple(index / (frame_count - 1) for index in range(frame_count))
    if loop_mode == "ping_pong":
        return tuple(index / (frame_count - 1) for index in range(frame_count))
    raise ValueError(f"Unknown loop mode: {loop_mode!r}")


def _index_values(
    samples: tuple[SequenceSample, ...],
    values: Any,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        for value in values(sample):
            if value:
                result[str(value)].append(sample.sequence_id)
    return result


def _hash_split(seed: str, component_id: str, ratios: SplitRatios) -> Split:
    digest = hashlib.sha256(f"{seed}\0{component_id}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < ratios.train:
        return "train"
    if unit < ratios.train + ratios.validation:
        return "validation"
    return "test"


def _balanced_component_splits(
    components: dict[str, tuple[str, ...]],
    policy: SplitPolicy,
) -> dict[str, Split]:
    """Greedily balance whole leakage components against requested sample ratios.

    Components are placed largest-first. The squared normalized target error is
    deterministic and sample-count aware; a seeded hash is used only to break
    exact ties. When enough components exist, every non-zero-ratio split is
    guaranteed at least one component. This avoids an evaluation-empty split on
    small corpora while never breaking a leakage component.
    """

    split_ratios: dict[Split, float] = {
        "train": policy.ratios.train,
        "validation": policy.ratios.validation,
        "test": policy.ratios.test,
    }
    active_splits = tuple(split for split, ratio in split_ratios.items() if ratio > 0)
    total_samples = sum(len(members) for members in components.values())
    targets = {split: total_samples * split_ratios[split] for split in active_splits}
    counts = {split: 0 for split in active_splits}
    component_counts = {split: 0 for split in active_splits}
    ordered = sorted(
        components.items(),
        key=lambda item: (
            -len(item[1]),
            _seeded_digest(policy.seed, item[0]),
            item[0].encode("utf-8"),
        ),
    )
    result: dict[str, Split] = {}
    for index, (component_id, members) in enumerate(ordered):
        remaining_including_current = len(ordered) - index
        empty_splits = tuple(split for split in active_splits if component_counts[split] == 0)
        candidates = (
            empty_splits
            if empty_splits and remaining_including_current == len(empty_splits)
            else active_splits
        )
        size = len(members)

        def candidate_key(
            split: Split,
            component_size: int = size,
            candidate_component: str = component_id,
        ) -> tuple[float, bytes, bytes]:
            projected = dict(counts)
            projected[split] += component_size
            error = sum(
                ((projected[name] - targets[name]) ** 2) / max(targets[name], 1.0)
                for name in active_splits
            )
            return (
                error,
                _seeded_digest(policy.seed, f"{candidate_component}\0{split}"),
                split.encode("utf-8"),
            )

        selected = min(candidates, key=candidate_key)
        result[component_id] = selected
        counts[selected] += size
        component_counts[selected] += 1
    return result


def _seeded_digest(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()
