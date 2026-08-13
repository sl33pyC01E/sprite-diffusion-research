# Generated RGBA decoding

The trained model emits continuous RGBA values. Raw uint8 sample arrays remain the
canonical model artifacts and are always evaluated directly.

`spritelab.decode` provides an optional, explicit hard-alpha derivative for pixel-art
inspection. A configured threshold maps alpha to exactly zero or 255, zeros RGB
under pixels made transparent, and leaves visible RGB unchanged. It performs no
spatial filtering, connected-component removal, resizing, palette quantization, or
frame interpolation.

Each decoded `.npy` has a JSON sidecar containing the threshold, exact operation,
source and output paths, source/output file hashes, and source/output array-content
hashes. A decoded preview must be labeled as derived; it is not raw model output and
must not replace the canonical sample in evaluation.

## Threshold calibration

`spritelab.decode_calibration.export_hard_alpha_threshold_calibration` evaluates an
explicit threshold list against ordered, exactly matched source/target `.npy` arrays.
Both sides must be uint8 `[T, H, W, 4]` RGBA arrays with unique sample IDs in the same
order and identical per-pair shapes. The utility does not resize, reorder, or infer
pairings from filenames.

For each threshold it reports per-sample and unweighted macro-mean premultiplied RGBA
MAE, alpha IoU, and alpha MAE. Selection is deterministic and lexicographic: minimize
premultiplied RGBA MAE, maximize alpha IoU, minimize alpha MAE, then prefer the lower
numeric threshold. This objective and its tie-break are recorded in the artifact.

The caller must label the data as either `training_target_estimate` or
`held_out_validation`. A training-target sweep is an optimistic in-sample diagnostic;
it must not be reported as validation. The canonical JSON records that distinction,
the exact source and target paths, file and array-content SHA-256 hashes, shapes,
per-threshold results, and selection rationale. Publication is atomic and refuses to
replace any existing artifact.

## Clip-global palette derivative

Hard alpha fixes translucent silhouettes but leaves the continuous model's thousands
of near-duplicate visible RGB values intact. `global_palette_decode_rgba` first applies
the explicit hard-alpha operation, then fits one adaptive median-cut palette to all
visible generated pixels across the entire clip. Every frame shares that palette and
dithering is disabled. The API accepts no target or reference palette.

`GlobalPaletteDecodeConfig` makes both the alpha threshold and maximum palette size
explicit. `export_global_palette_decode` publishes a no-clobber `.npy` plus a canonical
sidecar recording source/output hashes, Pillow version, quantizer, palette scope,
parameters, and visible color counts before and after. This remains a derived display
artifact; raw model output is still canonical.

The choice is backed by a matched in-sample diagnostic rather than aesthetics alone.
For the alpha-weighted 1,500-step Fetid Rat continuation, hard-alpha-192 clips averaged
3,160 visible RGB values. A generated-only 32-color palette reduced premultiplied RGBA
MAE from `0.024544` to `0.024062`, composite-black MAE from `0.026062` to `0.025420`,
and temporal-delta MAE from `0.025100` to `0.023463`. The canonical safe-checkpoint
replay evaluation is
`data/inference/fetid-rat-alpha4-step1500-endpoint-palette32-v1/evaluation.json`
(SHA-256 `245c86a25a28edd94f4514f98d177cd482736940681042ac3081066356d55ef1`).
The palette sweep is
`data/experiments/fetid-rat-palette-diagnostic-v1/diagnostic.json` (SHA-256
`7464c05abbf1545ad2666ae184ae9f101c1c53f89806193f6c1243dd370fa032`).

Those numbers use training targets and are optimistic. Thirty-two colors was the best
tested matched setting for this checkpoint; it is not a universal default for future
models or held-out sprite styles. Palette count and alpha threshold should be
recalibrated on held-out data once a valid evaluation split is available.

## Palette-size calibration

`export_global_palette_size_calibration` performs that comparison without allowing
the target art to leak into palette fitting. It first fixes an explicit hard-alpha
threshold, then fits each candidate median-cut palette only to all visible generated
pixels within each source clip. Targets are used solely for paired measurement. The
file-backed source/target IDs, ordering, shapes, file hashes, and array-content hashes
use the same strict contract as threshold calibration.

Selection is deterministic and lexicographic: minimize premultiplied RGBA MAE, then
composite-black MAE, then temporal-delta MAE, and finally prefer the smaller palette.
The canonical no-clobber JSON records every per-sample result, aggregate result,
candidate size, fixed threshold, Pillow version, quantizer, and the fact that neither
reference nor target colors were used to fit the generated palette. As with threshold
selection, a training-target estimate must not be reported as held-out validation.

## Atomic derived-preview bundles

`spritelab.decode_bundle.export_decode_preview_bundle` packages several decoded
clips for inspection without changing the status of the raw samples. Every input is
an ordered `DecodeBundleClipRef` with a safe unique ID, resolved `.npy` path, expected
file SHA-256, explicit per-frame durations, and an explicit `loop`, `one_shot`, or
`ping_pong` mode. Inputs must be uint8 `[T, H, W, 4]` arrays. The exporter neither
resizes nor realigns clips and refuses an incorrect hash, duplicate ID/path, implicit
timing, unsafe filename, or existing output directory.

The decode settings are also explicit: one hard-alpha threshold and one or more
clip-global palette sizes. Hard-alpha arrays are produced by
`hard_alpha_decode_rgba`; palette arrays are produced independently from each raw
generated clip by `global_palette_decode_rgba`. The palette API receives no target
array or reference palette. Each variant gets a derived `.npy`, a raw-linked decode
sidecar, nearest-neighbor APNG, contact sheet, and preview sidecar. The raw arrays are
read and re-hashed but never opened for writing, and every sidecar labels its output
as display-only and non-canonical.

The caller supplies hash-pinned source-report, hard-alpha-calibration, and
palette-calibration artifact references. Calibration JSON kinds are checked, the
requested alpha threshold and palette sizes must be covered by the linked sweeps, and
palette calibrations must use the same alpha threshold. These links are evidence, not
a claim that every displayed variant was the selected winner; training-target
calibrations remain optimistic in-sample estimates.

Publication is bundle-atomic. All files are written no-clobber into a sibling staging
directory under `DiskGuard`, external inputs are re-hashed, and the completed staging
directory is promoted to a previously absent destination. A canonical
`bundle-index.json` records the caller order, raw paths/file and array hashes,
timing/loop metadata, explicit parameters, provenance artifact paths/hashes, and the
relative path, byte size, media type, role, and SHA-256 of every payload. The index
does not list itself because a file cannot contain its own digest; its canonical
SHA-256 is returned as `DecodePreviewBundleResult.index_sha256`.
