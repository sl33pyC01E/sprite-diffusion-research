# Materialized RGBA pixel-quality audits

`spritelab.pixel_quality` performs a read-only audit of the native RGBA arrays
named by a materialization manifest. It is intended to surface measurable source
facts such as an opaque color-key background without silently deciding how those
facts should be repaired.

`build_materialized_pixel_quality_audit()` first uses the canonical training-data
loader to verify the complete manifest: schema version, declared sequence count,
semantic fields, file sizes, file SHA-256 digests, and array-content SHA-256
digests. It then re-reads and re-hashes each selected native stored `.npy` while
scanning it. Source filtering therefore changes the reported scan, not the scope
of manifest verification. Intro/loop or fixed-frame training projections are not
used for this audit.

The default `PixelQualityDetectionConfig` detects exact opaque magenta
`RGB=(255,0,255), alpha=255`. Callers can supply any explicit tuple of exact RGB
sentinels. A sentinel match is detection-only: it does not imply transparency,
does not label the art defective, does not match alpha below 255, and never changes
a pixel. Near colors such as `(254,0,255)` do not match unless explicitly listed.

Reports contain:

- exact transparent (`alpha=0`), opaque (`alpha=255`), partial-alpha, and visible
  (`alpha>0`) pixel counts and fractions;
- fully opaque and fully transparent frame and clip flags;
- exact opaque-sentinel affected clip, frame, and pixel counts;
- visible occupancy of the unique one-pixel border and unique corner coordinates;
- per-clip records plus aggregate summaries by source, split, and source/split;
- the exact materialization-manifest digest and a canonical detection-config digest.

Every fraction includes its integer numerator (`count`) and denominator. This
makes the report auditable without reconstructing totals from rounded percentages.
The JSON export is canonical, atomic, guarded by `DiskGuard` when supplied, and
refuses to overwrite an existing path.

Example:

```python
from pathlib import Path

from spritelab.pixel_quality import (
    PixelQualityDetectionConfig,
    export_materialized_pixel_quality_audit,
)

result = export_materialized_pixel_quality_audit(
    Path("data/processed/temporal-v5-action-known/materialization.json"),
    Path("data/index/reports/action-known-pixel-quality.json"),
    config=PixelQualityDetectionConfig(
        opaque_rgb_sentinels=((255, 0, 255),),
    ),
    source_ids=("open_surge",),
)
print(result.artifact_sha256)
```

The `sources`, `splits`, and `source_splits` sections use only selected clips;
`verification.verified_clip_count` still describes the complete verified manifest.
Keep any later color-key correction as a separate, explicit, hash-recorded
normalization transform so this evidence remains an account of the stored input.

## Historical pre-fix audits

Two immutable reports quantify the previously materialized Open Surge color-key
error. They describe stored arrays and do not modify them.

- `temporal-v5-action-known-pixel-quality-v1.json` has SHA-256
  `acc396205802e50bf86096a4c2db304aef785a2be94677ba9bb5f2a62005e2ef`.
  It verified 553 clips. Among its 77 Open Surge clips, 58 clips and 450 of
  666 frames contain 1,147,497 exact opaque-magenta pixels (7.52% of the Open
  Surge source pixels). Shattered Pixel Dungeon and SpriteCook have zero matches.
- `temporal-v5-opensurge-r2-pixel-quality-v1.json` has SHA-256
  `d9398ea1ab9fb481398b5a1fcecc8348798e2f47987434d8ea58fafe68a103c4`.
  It verified 850 clips. Among its 361 Open Surge clips, 232 clips and 1,706 of
  2,967 frames contain 3,349,768 exact opaque-magenta pixels (12.09% of the Open
  Surge source pixels). The other two sources again have zero matches.

These counts are native materialization counts, before any fixed-frame training
selection. Corrected versioned materializations should be audited with the same
configuration and must report zero exact opaque-magenta pixels for Open Surge.

That gate passes for `temporal-v6-corrected-core`. Its canonical audit SHA-256 is
`383e4fab91534b26a931c69d0693adbdcfc957d398b471d009f70c6b435bda27`.
All 553 clips were re-hash-verified; Open Surge, Shattered Pixel Dungeon, and
SpriteCook each report zero affected clips, frames, and exact opaque-magenta pixels.

The split-matched minimum-64 variant used for the controlled 48-clip model rerun
passes the same gate. Its report
`temporal-v6-corrected-core-v5split-min64-pixel-quality-v1.json` has SHA-256
`db56b3ebef6e4a826c03d9605f5475fe127b65d7745dfc2cd9f4dedd60b560ce`;
all 553 stored arrays were verified and every source again has zero exact opaque
magenta. This report audits the full versioned materialization, not merely the 48
clips selected by the experiment.
