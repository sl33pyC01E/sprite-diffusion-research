# /tg/station DMI corpus audit

This adapter is a pure, read-only audit of the mob sprites in an exact
[/tg/station](https://github.com/tgstation/tgstation) repository snapshot. It
does not extract the ZIP, execute repository code, or write to the live corpus
database. Its first-pass scope is every `icons/mob/**/*.dmi` file, with
conservative selection of whole animals, humanoids, nonhuman players,
silicon/robots, monsters, creatures, and animated objects.

## Immutable source identity

- Repository: `https://github.com/tgstation/tgstation`
- Commit: [`ca8b5ffda1ec44e0fa04a32801dc5f6e3b77f0e9`](https://github.com/tgstation/tgstation/tree/ca8b5ffda1ec44e0fa04a32801dc5f6e3b77f0e9)
- Commit timestamp reported by GitHub: `2026-08-12T08:04:57Z`
- Archive URL:
  `https://codeload.github.com/tgstation/tgstation/zip/ca8b5ffda1ec44e0fa04a32801dc5f6e3b77f0e9`
- Archive response ETag:
  `"ade05c6567faddf6417104e00879fa697ec07189d70933a903142c59c71f9de1"`
- Archive size: `193,871,729` bytes
- Archive SHA-256:
  `6f37531d28b8e48ca9399daccdbeef3683e9561eca0cf2272c4bad11c5a2a07c`
- Validated root: `tgstation-ca8b5ffda1ec44e0fa04a32801dc5f6e3b77f0e9`
- Canonical ZIP inventory SHA-256:
  `ad3e1356ccc701577b3a1c612b2487f4a12544082f55cdc27bcce5e93636d678`
- Canonical audit-record SHA-256:
  `22cb2cc6bc828c082728287d2c47702834cd23667f01246ec5fc50af60fe2249`
- CAS object:
  `data/raw/objects/sha256/6f/37/6f37531d28b8e48ca9399daccdbeef3683e9561eca0cf2272c4bad11c5a2a07c`

Acquisition used the project's guarded resumable HTTP fetcher and immutable
SHA-256 content-addressed store. The 100 GiB free-space floor remained in
force. The archive is not copied to a friendly filename and is not unpacked.

## Exact DMI semantics

A DMI is a PNG atlas with state declarations in a compressed
`zTXt Description` chunk. All 401 audited files have exactly one such chunk,
with keyword `Description`, PNG compression method `0`, and DMI version `4.0`.
The audit retains the decoded Description verbatim as well as decoded-text,
raw-chunk, CRC, full-file, frame-pixel, untimed-sequence, and timed-sequence
hashes.

Atlas behaviour is grounded in the pinned repository's own
[`tools/dmi/__init__.py`](https://github.com/tgstation/tgstation/blob/ca8b5ffda1ec44e0fa04a32801dc5f6e3b77f0e9/tools/dmi/__init__.py), whose exact
payload is 7,522 bytes with SHA-256
`ee64b893a87dd08a55942900c422053c906cbf72e5b4bcbd293a7d0e0dbe9d63`.
The adapter reproduces these facts without importing or executing that file:

- default cells are 32×32 unless `width` and `height` override them;
- direction order is south, north, east, west, southeast, southwest,
  northeast, northwest;
- source cells advance by state declaration, temporal frame, then direction;
- the resulting cell stream is packed row-major in the PNG;
- absent delays default to one decisecond;
- declared delays, loop count, rewind flag, movement flag, and every hotspot
  declaration remain explicit.

Durations are represented both as their literal DMI decisecond strings and as
exact integer milliseconds. The decisecond interpretation also follows the
[official BYOND icon/runtime reference](https://www.byond.com/docs/ref/info.html#/icon).
A delay list whose length disagrees with the temporal frame count is not
repeated, truncated, or otherwise repaired.

Repeated state names are intentionally retained. A still state and a
`movement = 1` state with the same name form an evidence-backed idle/walk pair;
two declarations with the same name and the same movement flag are ambiguous
runtime keys and are quarantined rather than assigned guessed precedence.

## Exact snapshot totals

The safely validated archive contains 19,584 members: 17,796 regular files,
1,788 directories, and no symlinks. Its members total 276,585,074 uncompressed
bytes and 188,324,505 compressed bytes.

| Audit measure | Exact count |
| --- | ---: |
| Mob DMI files found / parsed / malformed | 401 / 401 / 0 |
| DMI state declarations | 11,862 |
| Declared source atlas cells | 60,051 |
| Temporally animated states | 1,314 |
| Directional states | 9,338 |
| Movement variants | 107 |
| States with declared delays | 1,308 |
| Rewind states | 127 |
| Finite-loop states | 174 |
| Conservative complete-entity candidates | 1,686 |
| Complete-entity sequences passing quarantine | 1,680 |
| Passing sequences with an action label | 1,209 |
| Passing animated action sequences | 260 |
| Complete identity/action groups | 1,150 |
| Groups with at least two distinct actions | 283 |
| Multi-action groups with an animated action | 140 |

The pack-level separation is 65 complete-entity candidate packs, 280 modular
component packs, 24 effect packs, 30 icon/UI packs, and 2 ambiguous packs.
State-level roles are 1,686 whole-entity candidates, 8,675 modular components,
610 effects/overlays, 804 icons/UI states, and 87 ambiguous states.

Passing complete-entity sequences span all requested broad entity families:

| Entity class | Sequences |
| --- | ---: |
| Animal | 480 |
| Creature | 146 |
| Humanoid | 30 |
| Monster | 550 |
| Animated object | 16 |
| Robot / silicon / rideable | 458 |

These are source-evidence classes, not a claim that every state is visually
independent. Runtime-composited humanoid body parts are deliberately excluded;
the 30 humanoid sequences are from conservative whole-entity packs such as the
simple-human/tourist family.

Source action cues among passing states are:

| Action cue | Sequences |
| --- | ---: |
| idle | 712 |
| death | 325 |
| walk | 104 |
| transform | 14 |
| attack | 13 |
| sleep | 12 |
| stun | 11 |
| emote | 5 |
| jump | 4 |
| spawn | 5 |
| fly | 2 |
| run | 2 |

These labels come only from explicit movement flags, exact still/movement name
pairs, explicit state-name tokens, or conservative directional base-state
evidence. A later projector must map them onto the shared conditioning
taxonomy; the adapter does not silently rewrite unsupported source actions.

## Geometry and duplicate evidence

All 401 images decode as PNG: 253 are RGBA and 148 are indexed-color images
with transparency. Frame geometries include 32×32, 64×64, 96×96, rectangular
32×48/32×64/32×96 cells, and eleven other exact dimensions. Every source cell
stores its absolute atlas index, state-relative index, temporal index,
direction, rectangle, duration, and normalized RGBA SHA-256.

Of the atlases, 127 have exact declared capacity. The other 274 have 1,457
trailing grid cells not claimed by DMI metadata. Those cells remain inventory
evidence and are never converted into frames.

There is one byte-identical DMI group:

- `icons/mob/cows.dmi`
- `icons/mob/simple/cows.dmi`
- shared SHA-256:
  `c32afa45c7f2399a6ad42ffcdd58c543725d4f3258b2d5d7432cf61dc5b59cb2`

Across passing whole-entity records, 45 timed-sequence duplicate groups contain
48 excess copies. Both path identity and hashes are retained so a dataset split
can prevent exact-payload leakage without erasing upstream lineage.

## Quarantine policy

The exact snapshot has no wholly malformed mob DMI, but it does contain
state-level contradictions or ambiguities:

- 23 states have delay-list lengths that disagree with frame counts;
- 32 states contain a hotspot position outside their declared temporal range;
- 13 declarations are excess name-plus-movement runtime-key duplicates;
- non-whole-entity packs and explicit layer/effect states stay outside the
  complete-entity selection regardless of whether they decode cleanly.

The parser still records exact source geometry and hashes for these records.
It emits no guessed duration, precedence rule, composite, or whole-entity
label. Synthetic tests additionally prove that unsafe ZIP traversal and
unsupported DMI metadata become hard validation errors or hash-addressed
malformed records, as appropriate.

## Rights, credit, and citation scope

The pinned README states that all assets, including icons, are licensed under
Creative Commons Attribution-ShareAlike 3.0 unless otherwise indicated. Its
exact evidence record is:

- `README.md`, 4,749 bytes, SHA-256
  `c785d87bb165d1d7d29d78a6e285dbf7875ffa4efd589d9b1d7256135f264420`.

The current code-license evidence is `LICENSE`, 34,520 bytes, SHA-256
`76a97c878c9c7a8321bb395c2b44d3fe2f8d81314d219b20138ed0e2dddd5182`;
the historical GPLv3 evidence is `GPLv3.txt`, 35,147 bytes, SHA-256
`8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903`.
Those code grants are recorded separately from the CC BY-SA 3.0 asset grant.

No path-local author, credit, copyright, or license document exists under the
pinned `icons/mob` tree. Consequently, the adapter does not invent a per-file
artist. Every pack instead retains:

- repository and exact commit;
- exact archive member and logical path;
- DMI SHA-256 and immutable commit blob URL;
- a commit-scoped history URL for contributor attribution;
- the README asset-license evidence and its “unless otherwise indicated”
  caveat.

If a later source-history audit discovers a narrower per-file exception, that
evidence must take precedence over the README default. Suggested citation:

> /tg/station contributors. `tgstation/tgstation`, commit
> `ca8b5ffda1ec44e0fa04a32801dc5f6e3b77f0e9`, exact `icons/mob/...dmi`
> path, CC BY-SA 3.0 for assets unless otherwise indicated.

## Deliberate boundary before projection

No live SQLite row, source registry entry, CLI command, extracted member, or
training snapshot was created by this work. A later projector should:

1. register the already acquired archive blob and retrieval evidence;
2. select only exact DMI members required by passing states;
3. extract selected members through the guarded archive/CAS path rather than
   unpacking the repository;
4. crop atlas rectangles and split direction tracks while retaining DMI state
   and declaration identity;
5. attach README rights evidence and exact commit/path history to every
   projected sequence;
6. run exact-array and timed-sequence leakage checks before a snapshot is
   called model-ready.

The pure adapter is `src/spritelab/adapters/tgstation.py`; deterministic
synthetic, safety, and exact-CAS regressions are in
`tests/test_tgstation_adapter.py`.
