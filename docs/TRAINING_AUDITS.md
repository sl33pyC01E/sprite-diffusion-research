# Training-readiness audits

`spritelab.training_audit` turns a materialization manifest into a canonical,
hash-verified model-readiness report. It loads every declared `.npy`, verifies the
file and array digests, applies the requested fixed-frame projection, and reports
coverage by split, source, pack, resolution bucket, entity class, action, view,
direction, and loop mode.

The audit separately checks identity, source-blob, exact-array, fixed-training-tensor,
and source-pack overlap between splits. Source-pack overlap is reported even when
identity and pixels are disjoint: it is a style/domain overlap, not necessarily a
subject leak, and should be interpreted explicitly in evaluation claims.

It also enumerates causal action-contrast eligibility inside each split and spatial
bucket. A contrast requires exact agreement on identity, description, entity class,
view, direction, loop mode, and fixed frame phases; action is the only changing
condition. Byte-identical same-action duplicates use one conceptual representative,
while same-action/different-target conflicts are counted as exclusions. Actions that
resolve to the same exact fixed target are not causal steering contrasts: the audit
keeps at most one action representative per target digest and requires at least two
distinct digests. It separately reports cross-action aliases, rows left without a
target-distinct counterpart, conflicts, same-action duplicates, and the total endpoint
exclusion count. None of these endpoint exclusions removes a row from ordinary training.

Artifacts are canonical JSON, are published atomically, and never overwrite an
existing path. The manifest digest and source-snapshot identity are embedded so
later experiment reports can cite the exact readiness state they used.

The current 553-clip action-known materialization has a fixed-eight-frame audit
SHA-256 of `b8e8359e55ce09df01658188ad68c6bf62504cbd4295143c79598fc2b47fbfb9`.
It contains 135 identities, 107 with multiple actions, and has no identity,
source-blob, exact-array, or fixed-target-array overlap between train, validation,
and test. All three source packs do cross splits, so this snapshot supports
identity-disjoint in-domain diagnostics but not a pack-held-out domain-generalization
claim. A separate pack-held-out snapshot should be exported after more independent
corpora have been projected.

The corrected 553-clip replacement `temporal-v6-corrected-core` has fixed-eight-frame
training-audit SHA-256
`22d09cb18352db08f98f6b8a1ef0776b8d3e30339400b6264b1d3349e2529794`.
It preserves the same 135 identities, 107 multi-action identities, action/entity
coverage, and identity/blob/array leakage boundaries, while changing Open Surge
pixels through the documented engine color-key transform. The three source packs
still cross splits; the same in-domain-only evaluation limitation applies.

The controlled minimum-64, original-split variant has fixed-eight-frame audit
SHA-256
`33a2a2fe54e03eb0bac9643e15fe94c485afee138d2e228725b593e8eca240e4`.
It verifies the same 553 sequence IDs and the same 497/28/28
train/validation/test assignments as `temporal-v5-action-known`, while its source
snapshot and materialization hashes point only to the corrected Open Surge v2
projection. This is the readiness artifact to cite for the matched replacement of
the invalid 48-clip 64-pixel run.

The Widelands model-ready materialization has source-manifest SHA-256
`8503d46d9305890393df7e31421e2f4261d9e8ae620b688a4112fa24d1973616`.
Its target-distinct schema-v2 audit has SHA-256
`4bf003409b5d514c8dec7a2a94f38f3be44a7abb5098503b03f057451bcd3af0`.
The 166-row training split contains 42 nominal same-identity action pairs, but 36
pairs share the same exact fixed target due to authored Widelands aliases. The
corrected endpoint plan therefore selects 6 groups and 12 rows, explicitly excludes
72 alias/no-distinct-counterpart rows from endpoint supervision, and retains all 166
rows for the base objective. See `WIDELANDS_CAUSAL12_EXPERIMENT.md` for the exact
provenance evidence and bounded baseline.
