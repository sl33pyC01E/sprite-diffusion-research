# Provenance-preserving clip materialization

`spritelab.materialize` converts a canonical dataset snapshot into deterministic,
training-ready RGBA clip arrays. It is deliberately stricter than a general sprite
importer: the materializer only executes frame selections that the provenance index has
already made explicit.

## Exact reconstruction contract

For every sequence, the materializer requires `metadata.frame_provenance` to contain
exactly the ordinals `0..frame_count-1`. Each row must identify a carrier with
`source_blob_sha256` and a playback frame with `source_frame_index`. The implementation:

- validates the snapshot's embedded `manifest_sha256` before touching outputs;
- resolves every `metadata.blob_records[].storage_path`, verifies its declared byte size,
  and streams the file through SHA-256;
- decodes PNG/APNG, GIF, and lossless or lossy WebP through
  `media.animation.extract_animation`;
- addresses decoded frames by their true source index, including APNGs whose poster image
  is not part of playback;
- supports repeated source indices and sequences assembled from multiple carriers;
- preserves ordinal order, recorded durations, and recorded phases without temporal
  interpolation or deduplication; and
- records both native selected-frame pixel hashes and the normalized output-frame hashes.

Static-image cell coordinates are never guessed. A complete audited
`metadata.frame_rect` with integer `left/top/right/bottom` bounds and either
`coordinate_space="source_sheet"` or `coordinate_space="source_image"` is executed
directly after bounds and redundant width/height validation. These two literals have
the same narrow meaning here: exact pixel coordinates in the decoded, single-frame
source carrier. The crop, carrier size, logical source index, and reconstruction
method are retained in output provenance. Other crop/bounding-box/UV forms raise
`UnsupportedSheetCoordinatesError`. Likewise, requesting source frame 3 from a static
PNG does not mean "cell 3" without an exact rectangle and is rejected.

Output frame provenance retains the validated input literal as
`source_rect_coordinate_space` and records the decoded single-frame carrier dimensions
as `source_carrier_size`. Schema-v1 also retains the older `source_sheet_size` and
`audited_source_sheet_rectangle_v1` fields for compatibility even when the precise
input literal was `source_image`; consumers should prefer the new neutral fields.

### Audited pixel-transform contract

Pixel rewrites are opt-in per frame. The materializer executes no source-specific
heuristic and does not infer a color key from filenames, palette entries, or pixel
frequency. A frame may supply `metadata.pixel_transforms`, where every operation
must have exactly these fields:

- `schema="spritelab.pixel_transform.v1"`;
- `op="exact_uint8_rgb_to_rgba_zero"`;
- `rgb`, exactly three uint8 integers;
- non-empty `evidence`, whose rows have a safe archive member path, lowercase member
  SHA-256, sorted positive source line numbers, scope, and claim; and
- `transform_sha256`, the verified canonical hash of the preceding fields.

The operation compares the three uint8 RGB channels for exact equality and writes
RGBA `[0, 0, 0, 0]` only at matching pixels. Alpha does not weaken the comparison.
There is no fuzzy threshold: for Open Surge, `[255, 0, 255]` matches while
`[254, 0, 255]` does not. Frames with no transform metadata are byte-preserved at
this stage. An unknown schema/operation, malformed RGB/evidence, unsafe evidence
path, duplicate evidence or transform, or digest mismatch raises
`UnsupportedPixelTransformError` before that clip is published.

Each frame's manifest provenance retains the complete validated transform, execution
result and matched-pixel count, as well as `pre_transform_pixel_sha256` and
`post_transform_pixel_sha256`. `source_frame_pixel_sha256` remains the pre-transform
hash of the selected source/crop. The sequence-level `pixel_transform` record repeats
the unique transforms and per-frame pre/post hashes and totals all exact matches.

## Lossless spatial buckets

The default target buckets are square 64, 128, 256, and 512 pixel canvases. Callers may
provide square integers or `(width, height)` pairs. Buckets are deduplicated and ordered by
area, then dimensions, so the selected bucket is deterministic regardless of input order.

`normalization.normalize_sprite_sequence` first aligns the clip to one shared alpha-content
union. The smallest bucket that contains that union is selected. The only allowed spatial
resampling is positive-integer nearest-neighbor upscale; an integer scale of one performs no
resampling. If no bucket fits, `NoLosslessBucketError` is raised. There is no implicit
downscale, crop-to-fit, bilinear filtering, or per-frame recentering.

## Output layout and manifest

Each sequence is written to a path derived only from the SHA-256 of its sequence ID:

```text
<output>/
  clips/
    train|validation|test/
      <sha256(sequence_id)>.npy
  materialization.json
```

The NumPy file is a C-order `uint8` tensor shaped `(frames, height, width, 4)`. It contains
RGBA rather than RGB so transparency remains a first-class training target. NumPy headers
contain no timestamp, and the manifest is canonical sorted-key JSON with no runtime timestamp
or absolute output path.

Every sequence record includes:

- sequence, identity, source, source-pack, entity, action, view, direction, quality, and
  assigned split identifiers;
- source blob hashes, sizes, MIME types, source-scoped sequence keys, retrieval IDs,
  rights-observation IDs, item-blob IDs, and archive occurrences;
- exact per-frame carrier hash/index, native dimensions, source pixel hash, recorded duration,
  phase, decoded carrier duration, audited pixel transform, and pre/post-transform hashes;
- the generated caption and its `description_basis`;
- the complete normalization transform, transform SHA-256, target bucket, and normalized
  frame pixel hashes; and
- the `.npy` relative path, shape, dtype, file SHA-256, array-content SHA-256, and byte size.

The top-level record keeps the canonical source snapshot SHA-256 and its embedded dataset
manifest SHA-256. This lets an experiment trace a tensor back through the immutable snapshot
to every source carrier and citation record.

## Atomic and append-safe behavior

Clips are written to same-directory temporary files, flushed and synced, then published
atomically. The manifest is published last. By default, any pre-existing target clip or
manifest causes `ExistingOutputError` before materialization begins; existing artifacts are
never silently replaced or deleted. Pass `overwrite=True` only for an intentional atomic
replacement.

Successful clips are durable immediately. If a later source is corrupt or too large for all
buckets, earlier valid clips remain and the final manifest is absent. The materializer removes
only its unpublished temporary file on failure; it does not roll back or clean up valid output.
A retry with the default policy will therefore surface the partial artifacts for an explicit
operator decision instead of concealing or destroying them.

An optional `DiskGuard` can enforce the project's free-space floor before each temporary clip
and manifest write.

## Python API

```python
from pathlib import Path

from spritelab.materialize import materialize_snapshot
from spritelab.storage import DiskGuard

output = materialize_snapshot(
    Path("data/index/snapshots/temporal-v1.json"),
    Path("data/processed/temporal-v1"),
    # Only needed when blob_records contain relative storage paths.
    blob_root=Path("data"),
    bucket_sizes=(64, 128, 256, 512),
    disk_guard=DiskGuard(Path("C:/"), 100 * 1024**3),
)

print(output.manifest_path)
print(output.sha256)
```

This module intentionally has no CLI wrapper yet. The Python boundary makes the first
training snapshot auditable before adding batch orchestration or source-specific sheet
materialization.

## First materialized research snapshot

The balanced SpriteCook temporal snapshot was materialized under
`data/processed/temporal-v2-balanced/` without overwriting any prior artifact. Its
canonical materialization-manifest SHA-256 is
`7799f8631465384f8e4aeb16d3303bb2b7e0e6d879796d115f6a456a90b066a9`.
All 49 source sequences reconstructed successfully. The lossless bucket distribution
is five at 64x64, 33 at 128x128, nine at 256x256, and two at 512x512. Forty-four
clips contain eight authored frames and five contain twelve; no temporal resampling
was performed in the stored materialization.

`spritelab.training_data` is the verified training boundary for this artifact. It
checks each `.npy` byte digest, typed-array shape, and array-content digest before use;
converts straight `uint8` RGBA to premultiplied channel-first float32 in `[-1, 1]`;
and can apply the explicit interpolation-free phase selection recorded by
`spritelab.temporal` when a fixed model frame count is required. The reverse
conversion unpremultiplies display output and forces invisible RGB to zero.

After exact source-sheet projection was added, the first three-source temporal
artifact was materialized at `data/processed/temporal-v5-opensurge-r2/`. Its
manifest SHA-256 is
`ba2c9abba301dc0501c08601846c784d36b3bced73a02aa3d6abd63fbabab9e4`.
It contains 850 timing-known clips: 440 Shattered Pixel Dungeon, 361 Open Surge,
and 49 SpriteCook. Lossless buckets contain 566 clips at 32x32, 199 at 64x64,
61 at 128x128, 19 at 256x256, and five at 512x512. A stricter action-labeled
subset contains 553 clips under `data/processed/temporal-v5-action-known/`, with
manifest SHA-256
`5a9ec37364e07cc95a696e850736c90d622771599ea32505e1e153f44c5d8567`.

Those two `temporal-v5` manifests predate the audited Open Surge color-key contract.
They faithfully cropped source PNG rectangles but left exact opaque magenta mask
pixels opaque, so they are retained as diagnostic/provenance artifacts and must not
be treated as corrected Open Surge training inputs. A replacement materialization
must be produced only after projection v2 metadata is present; this code change does
not overwrite either existing artifact.

The first corrected action-labeled replacement is
`data/processed/temporal-v6-corrected-core/`. Its source snapshot canonical SHA-256
is `f58274c2802c13eb0baa9d087c0752914075ea112249cc1f9fbc71a345275cf2`,
snapshot manifest SHA-256 is
`dc40ebcbf85b654095aa3b231109c80bc36ae2adbf51bfe9a87e27b8e185d194`,
and materialization SHA-256 is
`7befdcddcd504c5766e17f3d715efd6679df53a20dddc40faf373d8e98bb55e6`.
It deliberately source-filters the same three complete-entity corpora and contains
the same 553 action-known sequences (431 Shattered Pixel Dungeon, 77 Open Surge,
45 SpriteCook) for a matched correction comparison. All 666 Open Surge frames
declare the v2 transform; 1,143,264 exact source-crop pixels were zeroed before
normalization. The independent pixel-quality audit SHA-256 is
`383e4fab91534b26a931c69d0693adbdcfc957d398b471d009f70c6b435bda27`:
it verified all 553 arrays and found zero exact opaque-magenta pixels in every
source. New model experiments should use this or a later versioned materialization.

For an exact controlled rerun of the invalid 48-clip broad experiment, the
`temporal-v6-corrected-core-v5split` snapshot reuses the original split seed and
preserves all 553 sequence IDs, identity components, and train/validation/test
assignments. Its snapshot canonical SHA-256 is
`21c30231dfc71ab106ff617b17f0989cb02ff755f3e57bf7fa459272a16e7000` and its
embedded dataset-manifest SHA-256 is
`55c6067bbd3202761975a4ab729539243b105958d3ff725ffbf487860f7fe22c`.
The ordinary corrected materialization has canonical SHA-256
`52351c116de6b996fbf51039561c8f7f34403f8b55c9bc2ee875b29bc19963a6`.

Removing the color-key background changes the smallest lossless bucket for four
sequences, so a second controlled artifact at
`data/processed/temporal-v6-corrected-core-v5split-min64/` uses 64 as its smallest
bucket. Its canonical materialization SHA-256 is
`169f50abb30c82f00ba1b4be8f582e46775e846918ed559625b10aea845814cb`.
All 48 sequence IDs from the old 64-pixel experiment remain train examples and are
64x64 in this artifact. Of those, 31 Open Surge clips change and 17 clips are
byte-identical; the corrected transform removes 399,403 opaque-magenta pixels from
the exact fixed-eight-frame targets. Mean premultiplied-RGBA distance between the
old and corrected target batches is 0.20215, and mean visible occupancy falls from
0.41561 to 0.19817. This makes the next run a split-, identity-, sequence-, geometry-,
and seed-matched data-correction comparison rather than a new sample selection.

The unsuffixed `temporal-v5-opensurge/` directory is retained as a two-clip partial
attempt that exposed an overly broad coordinate-metadata detector. It has no final
manifest and is not a dataset artifact. The corrected `-r2` output was written to a
new directory instead of deleting or overwriting that diagnostic evidence.
