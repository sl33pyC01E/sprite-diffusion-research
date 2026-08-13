# Reproducible dataset snapshots

`spritelab.snapshot` turns the provenance SQLite index into a timestamp-free,
leakage-aware `DatasetManifest`. It opens the database with SQLite `mode=ro`, enables
`query_only`, and holds one read transaction, so snapshot generation never registers a
dataset or otherwise changes the source index.

## Temporal eligibility

The default `SnapshotFilters()` selects only sequences with at least two frames and
genuine timing evidence. A sequence is timing-known only when:

1. its declared frame count is greater than one;
2. one frame table contains exactly ordinals `0..frame_count-1`;
3. every one of those rows has a positive duration; and
4. neither sequence metadata nor motion conditioning explicitly marks timing, engine
   timing, or state occurrence order as unknown.

This deliberately excludes multi-frame Doom-style unique-pose projections from temporal
training. They remain available for spatial/action-conditioned training with
`temporal_mode="pose_only"` or `temporal_mode="all"`. The exported `timing_counts`
distinguishes actual timing-known samples from pose-only samples. The inherited
`CoverageReport.temporal_sequence_count` is structural (`frame_count > 1`), so consult
`timing_counts` when exporting a non-default mixed snapshot.

Timing-known is intentionally narrower than “several ordered images,” but it is
not a claim that loop semantics or normalized frame phases are known. For example,
an engine may provide exact per-frame delays while its asset metadata does not say
whether playback repeats. Such a sequence remains valid evidence and can be exactly
materialized, but the current fixed-phase model loader rejects it when `loop_mode`
and phases are unknown. Do not convert that rejection into an inferred loop. Build a
model-ready source stratum with explicit loop/phase evidence, or define and document
a separate conditioning policy first.

Use `temporal_mode="model_ready"` (CLI: `--temporal-mode model_ready`) for the
stricter current fixed-phase loader boundary. It additionally requires phases in
`sequence_frames`, an explicit `loop`, `one_shot`, or `ping_pong` mode with valid
nonduplicated phase bounds, or a verified phase-null prefix followed by a contiguous
phase-zero `intro_then_loop` tail. Legacy timing rows without phases and unknown loop
modes remain in `known` snapshots but are excluded from `model_ready`. This filter
does not synthesize phases or infer repeat behavior.

## Source selection

`SnapshotFilters.include_source_ids` and `exclude_source_ids` select projected corpora
before splitting. Values are stripped, deduplicated, and sorted by UTF-8 bytes; source IDs
remain case-sensitive. A sequence is associated with the union of its indexed item source,
its `sequence_source_keys` source, and every subject entity source. Consequently, a
repository projection that has no item or subject is still selectable through its
source-scoped sequence key.

An empty include tuple means all sources. A non-empty include tuple keeps a sequence when
at least one resolved association is included. A non-empty exclude tuple drops a sequence
when any resolved association is excluded, so exclusion takes precedence for a sequence
associated with multiple sources. The include and exclude tuples themselves may not
overlap. Sequences with no resolved source association do not satisfy a non-empty include
filter, but are retained by an exclude-only filter.

The command-line flags are repeatable and use union semantics within each list:

```console
spritelab dataset export data/index/snapshots/repository-actors-v1.json \
  --include-source open_surge \
  --include-source shattered_pixel_dungeon \
  --exclude-source freedoom
```

When both source filters are omitted, their empty fields are not added to the canonical
JSON. Existing default snapshot bytes and hashes therefore remain unchanged.

## Preserved identity and provenance

Each `SequenceSample` keeps the exact sequence ID, primary entity ID, source ID, item ID
as `source_pack_id`, normalized action, quality tier, and every source/frame blob SHA-256.
Its metadata also records all subject/entity IDs and roles, source-scoped sequence keys,
raw source action, source and item records, frame coordinates and durations, archive
member occurrences, CAS blob records, retrieval IDs, rights-observation IDs,
item-blob-occurrence IDs, duplicate-edge IDs, and the evidence used for temporal
eligibility.

Leakage components are formed by the existing `SplitPolicy` over:

- the selected primary identity;
- every subject identity (represented as a stable leakage-group token);
- exact shared frame/source blobs;
- transitive connected components of indexed duplicate edges; and
- source packs when `group_source_pack=True`.

An item is the source-pack boundary when a sequence has an item. Without one, the first
archive occurrence is used, then the source ID. For large repository archives,
`group_source_pack=True` can intentionally place the whole archive in one split; pass
`group_source_pack=False` only when identity/blob/duplicate grouping is the desired
evaluation boundary.

`SplitPolicy.assignment_strategy="hash"` retains the simple stable hash assignment. On a
small corpus, independent hashing can legitimately leave a low-ratio evaluation split empty.
Use `assignment_strategy="balanced"` for experiment snapshots: whole leakage components are
placed largest-first to minimize sample-count error against the requested ratios, with the seed
used for deterministic tie-breaking. When there are at least as many components as non-zero
splits, this strategy guarantees each such split receives a component. It never breaks a
component to improve the ratio.

## Integration API

```python
from pathlib import Path

from spritelab.dataset import SplitPolicy
from spritelab.snapshot import SnapshotFilters, export_snapshot

artifact = export_snapshot(
    Path("data/index/spritelab.sqlite3"),
    Path("data/index/snapshots/temporal-v1.json"),
    policy=SplitPolicy(
        seed="temporal-v1",
        assignment_strategy="balanced",
        group_source_pack=False,
    ),
    filters=SnapshotFilters(
        minimum_frame_count=2,
        actions=("idle", "walk", "run", "attack"),
        temporal_mode="known",
        include_source_ids=("open_surge", "shattered_pixel_dungeon"),
        exclude_source_ids=("freedoom",),
    ),
)

print(artifact.manifest.sha256)  # hash of the canonical DatasetManifest
print(artifact.sha256)  # hash of the complete exported artifact payload
print(artifact.coverage)
print(artifact.timing_counts)
```

Use `load_sequence_samples(...)` when a caller needs the filtered records before
splitting, `build_snapshot_from_index(...)` to build without writing, and
`write_snapshot(...)` to atomically write an already-built artifact. Identical index
facts, filters, and split policy produce identical UTF-8 bytes and hashes; filesystem
paths, run time, and export time are not included.
