# Freedoom sprite corpus audit

This audit is tied to the immutable CAS object
`4962902bfe9fa921c6ecb4419c55dcd40ca2b93c2d2e3b77c9fc3e89561aec78`, a
ZIP of Freedoom commit `d14dbbee3b6fbfb2c11cdb65eb61216e86d4ee85`. The adapter reads the ZIP
directly and does not extract or modify it.

## Naming and grouping

Doom sprite lumps use a four-character family, one frame character, and one
rotation digit. A second frame/rotation pair means the same physical image is
also installed horizontally flipped. Rotation `0` supplies every view; rotations
`1` through `8` are view-specific. This behavior is documented and implemented
in the pinned [Linux Doom sprite loader](https://github.com/id-Software/DOOM/blob/a77dfb96cb91780ca334d0d4cfd86957558007e0/linuxdoom-1.10/r_things.c#L143-L213).

The parser preserves the source filename and both frame/rotation references. It
marks the first pair's canonical transform as `identity` and the second pair's
as `horizontal_flip`; consumers must apply that transform when materializing a
canonical view. It groups identity only at the four-character family level. It
does not merge families merely because their artwork or names look related.

The archive contains:

- 1,328 physical PNG files in `sprites/`;
- 1,325 parseable Doom lump names and three placeholders (`blank.png`,
  `dummy.png`, and `nomonst.png`);
- 398 files containing two frame/rotation pairs;
- 1,723 logical frame/rotation references;
- 140 four-character families and 652 family/frame groups;
- 730 physical-image dimension buckets, of which 728 occur among parsed lumps.

Logical rotation-reference counts are 499 for rotation `0` and 153 each for
rotations `1` through `8`. Every parsed family/frame group is structurally
complete: it has either one rotation-`0` all-view reference or exactly one
reference for every rotation `1` through `8`.

## Labels and actions

Seventeen item labels come directly from `CC_*` values in
`lumps/dehacked/dehacked.txt`. The adapter maps only the fixed Doom II cast-call
slots to their corresponding sprite family. For example, `CC_ZOMBIE = zombie`
maps to `POSS`, and `CC_CYBER = assault tripod` maps to `CYBR`. These labels are
not propagated to other families.

Action groups are compatibility hints transcribed from the pinned
[Linux Doom state table](https://github.com/id-Software/DOOM/blob/a77dfb96cb91780ca334d0d4cfd86957558007e0/linuxdoom-1.10/info.c#L128).
The audit retains overlap instead of forcing a label. For example, `POSS` frames
`A` and `B` occur in both standing and running states, while frames `H` through
`K` occur in both death and reverse resurrection states. `KEEN` frame `A` is
both its spawn/idle art and the first frame of the canonical death chain; frames
`A` through `L` are therefore retained as death art. Nineteen actor families
have at least one state-derived action group. Across the corpus, 314
family/frame groups remain action-unknown, 114 directly mapped groups have
multiple candidate actions, and one additional group has a probable filename
alias interpretation.

The emitted action `frame_tokens` are ordered sets of unique artwork, not
reconstructed engine state cycles. Repeated state occurrences and per-state
tics are not present. Consequently every action group reports unknown loop
status, `state_occurrence_order_preserved = false`, and
`timing_preserved = false`. These sets are suitable for finding candidate art,
but a training pipeline must derive timing separately before treating them as
animation loops.

The pinned source contains `VILE^0.png`, while its build manifest expects
`VILE\0`. Those names are the unique known extra/missing counterparts at this
commit, so the audit records a **probable**, commit-scoped alias and exposes
`revive` as a candidate action by interpreting `^` as the manifest's `\` frame.
It does not rewrite the raw filename or frame token: `^` still decodes to frame
index 29, outside Linux Doom's accepted 0-28 range, and both the manifest-missing
and archive-extra observations remain in the output. The manifest also omits the
three placeholder PNGs.

## Rights and credits evidence

Evidence is indexed by archive member path, byte size, and SHA-256. Scope is
recorded without inheritance:

- root `COPYING.adoc` contains the Freedoom three-clause BSD text and is detected
  as project-root `BSD-3-Clause` evidence;
- root `CREDITS` contains 251 contributor records, 66 of which explicitly mention
  sprite work;
- `CREDITS-LEVELS`, `CREDITS-MUSIC`, and the in-game credit text are retained as
  supporting evidence;
- `dist/COPYING.CC0` is recorded as subdirectory-scoped CC0 evidence and is not
  treated as a sprite license;
- GPL v2 texts under `lumps/colormap/` and `lumps/playpal/` are likewise retained
  only at their observed subdirectory scope.

The root BSD evidence is the relevant project-wide observation in this snapshot,
but the audit keeps all narrower and potentially confusing evidence documents so
later provenance reports can cite what was actually present.
