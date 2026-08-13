# Widelands exact-snapshot worker/critter corpus audit

This adapter audits one immutable Widelands repository ZIP directly in the
content-addressed store. It does not execute Lua, extract the repository, or
write to the live corpus database. The audited surface is deliberately narrow:
canonical worker and critter manifests, every PNG beneath those entity trees,
the engine files needed to establish animation semantics, and pinned
license/credit evidence.

## Snapshot and acquisition identity

| Field | Exact value |
| --- | --- |
| Repository | `https://github.com/widelands/widelands` |
| Commit | `fbe33f1b96e877ebe7352c29ad3bd06770bd5e0a` |
| Commit citation | `https://github.com/widelands/widelands/tree/fbe33f1b96e877ebe7352c29ad3bd06770bd5e0a` |
| Commit time reported by GitHub | `2026-08-11T15:58:00Z` |
| Archive URL | `https://codeload.github.com/widelands/widelands/zip/fbe33f1b96e877ebe7352c29ad3bd06770bd5e0a` |
| Archive SHA-256 | `51093e130b2f4098716bc442ce3d8566d94b4590cfa233ba1c74e5d173a48186` |
| Archive size | 497,242,680 bytes |
| CAS path | `data/raw/objects/sha256/51/09/51093e130b2f4098716bc442ce3d8566d94b4590cfa233ba1c74e5d173a48186` |
| ZIP root | `widelands-fbe33f1b96e877ebe7352c29ad3bd06770bd5e0a` |
| Canonical audit-record SHA-256 | `2208e5ef94bbe6adbe80e2c668336ef04bcd79997e5794d9b81df5ded9ad9a86` |

GitHub's repository metadata reported 3,255,223 KiB before acquisition. The
commit archive was therefore fetched with the project's resumable
`ContentAddressedStore`/`DiskGuard`, checking the 100-GiB free-space floor for
every chunk. The compressed commit snapshot was much smaller than the
repository estimate. Acquisition did not touch SQLite.

`audit_known_widelands_archive()` hashes the archive and rejects any other
digest or ZIP root. Safe-path validation also rejects traversal, absolute paths,
encrypted members, non-regular entries, duplicate names, and case collisions.
The frozen audit records serialize as canonical, sorted-key JSON; their record
hash excludes only the hash field itself.

## Audited surface and exact inventory

The pinned ZIP contains 26,840 members: 25,166 files and 1,674 directory
entries. The adapter selects exactly these canonical manifests:

- 161 `data/tribes/workers/<tribe>/<entity>/init.lua` files;
- 16 `data/world/critters/<entity>/init.lua` files; and
- 177 actual `new_*_type` constructor calls after Lua comments are removed.

The manifest files total 267,991 bytes. Constructor roles are:

| Constructor role | Entities |
| --- | ---: |
| Worker | 141 |
| Critter | 16 |
| Carrier | 10 |
| Soldier | 5 |
| Ferry | 5 |

The two entity trees contain 10,741 PNGs. Their exact, mutually exclusive
partition is:

| Media role | PNGs | Treatment |
| --- | ---: | --- |
| Complete-entity animation source | 5,335 | Track source; 2,074 are canonical scale-1 sources |
| Player-color mask layer | 5,160 | Retained separately; never silently composited |
| Menu/UI icon | 177 | Auxiliary evidence, not a body frame |
| Equipment or level/status icon | 69 | Auxiliary evidence, not a body frame |
| Unreferenced layer/effect | 0 | Would be quarantined as auxiliary if present |

This partition accounts for all 10,613 worker-tree PNGs and all 128
critter-tree PNGs. The worker PNG payload is 85,972,003 uncompressed bytes and
the critter payload is 667,692 bytes. Complete-entity sources occupy 80,026,250
bytes; separate player-color masks occupy 6,163,580 bytes; menu and status
evidence occupies 449,865 bytes.

The 5,335 complete-entity sources break down by engine scale as 1,087 at 0.5x,
2,074 at 1x, 1,087 at 2x, and 1,087 at 4x. The audit retains every available
scale but emits frame rectangles against the mandatory neutral 1x source, so
resolution variants do not become false identity/action examples.

## Literal Lua support and refusal boundary

The parser is a comment-aware tokenizer and literal-table reader, not a Lua
runtime. It supports all syntax actually used by the 177 pinned manifests:

- inline `animations` and `spritesheets` tables;
- references to earlier literal tables, used by soldier definitions;
- literal strings, integers, booleans, nested tables, and negative hotspots;
- `animation_directory = dirname` and `path.dirname(__file__)`; and
- per-animation directories of the form `dirname .. "subdirectory"`, used by
  Frisian soldier appearance variants.

Function calls, program execution, loops, arbitrary table mutation, and other
dynamic Lua are not evaluated. Relevant unresolved structure raises a parse
error rather than inventing frames. Long documentation blocks and commented
constructor examples are removed by the tokenizer before discovery, so they do
not inflate entity or animation counts.

The parser retains entity ID, tribe, constructor role, conservative entity
class, manifest/member path, source line, declared animation name, basename,
appearance prefix (`variant_hint`), directory, direction, hotspot, timing,
loop flag, every source scale, player-color-mask link, source hash, source
dimensions, and every scale-1 frame rectangle.

## Engine-derived frame semantics

Five same-commit engine documents are hash-bound into the audit:

| Evidence member | SHA-256 |
| --- | --- |
| `src/logic/map_objects/map_object.cc` | `0606381b07a41799adfdf5e1fcb4283d1d7766120a8fdb4a5753a8e19ffd0c85` |
| `src/graphic/animation/animation.cc` | `cc7f03a33c1a42124a6f227142f900d708e1767ccb2ab8e87a17b8e228e4e217` |
| `src/graphic/animation/nonpacked_animation.cc` | `02cc51464457f7bff7f6a6e9259252023948ad5379db80f53bef60cd889b4ccb` |
| `src/graphic/animation/spritesheet_animation.cc` | `f2a9771d39e149c69d96a79f1d439cd6d94d5f0c4969bcf0eaa73bcb4f6fb8e0` |
| `src/io/filesystem/filesystem.cc` | `29d5996d87770315070ff74a45bee449a93ab83fca5ab664d437041701e76f1c` |

Those files establish the following behavior reproduced by the adapter:

- a directional declaration expands in this exact engine order:
  northeast, east, southeast, southwest, west, northwest;
- the direction suffix is appended to both the logical animation name and its
  basename before source lookup;
- supported source suffixes are `_0.5`, `_1`, `_2`, and `_4`, with an
  unsuffixed fallback only when mandatory scale 1 is absent;
- packed-sheet frame `i` is at column `i % columns`, row `i // columns`;
- sheet dimensions must divide evenly by the declared grid;
- capacity must contain all declared frames and may not contain a wholly unused
  final row; the audit records 2,074 harmless partial-row surplus cells;
- numbered files use the engine's exact search precedence: unnumbered, then
  `_0`-style, `_00`-style, and `_000`-style contiguous sequences;
- `fps` becomes `1000 // fps` milliseconds per frame; absent `fps` is the
  engine's 250-ms default; and
- `play_once = true` holds the final frame, while the default loops.

All 2,275 audited tracks have exact scale-1 sources, valid source geometry, and
exact frame order. No track is quarantined. They comprise 107 numbered-file
tracks and 2,168 packed-sheet tracks, producing 34,414 frame occurrences. The
smallest canonical frame is 7x10 pixels and the largest is 83x49 pixels.

None of the pinned declarations sets `play_once = true`; all 2,275 animation
objects therefore loop at the engine layer. Worker programs can stop a looping
animation after a requested duration, but this adapter does not execute or
flatten worker programs. In particular, it does not relabel death as one-shot
merely because that would be convenient for training.

## Steering labels and entity breadth

There are 700 source animation declarations before six-way direction expansion
and 2,275 logical action/direction tracks afterward. Every entity exposes at
least two normalized action classes. Counts below are direction-expanded:

| Normalized action | Tracks | Frame occurrences | Literal basis |
| --- | ---: | ---: | --- |
| Walk | 1,074 | 15,888 | `walk*` declarations |
| Carry | 810 | 11,970 | `walkload*` / loaded-walk declarations |
| Idle | 179 | 2,868 | `idle*` declarations |
| Work | 90 | 1,991 | Explicit work, farming, gathering, fishing, cutting, release, water, and related worker-action names |
| Attack | 55 | 670 | `atk*` / attack declarations |
| Dodge | 32 | 460 | `eva*` / evade declarations |
| Death | 20 | 300 | `die*` / death declarations |
| Eat | 15 | 267 | `eat*` declarations |

No pinned action remains unlabeled by that explicit vocabulary. Raw declared
names and basenames remain authoritative; normalized labels are additions, not
replacements.

Entity classification is intentionally narrow:

| Entity class | Entities | Basis |
| --- | ---: | --- |
| Humanoid | 151 | Worker/carrier/soldier constructor family, excluding five explicit pack-animal IDs |
| Animal | 21 | 16 critter constructors plus tapir, horse, ox, donkey, and reindeer carrier IDs |
| Vehicle | 5 | Ferry constructors |

The animals include badger, bear, bunny, chamois, deer, duck, fox, lynx,
marten, moose, reindeer, sheep, stag, wild boar, wisent, wolf, and five
pack-animal workers. The five tribes contribute 26 Amazon, 31 Atlantean, 33
Barbarian, 34 Empire, and 37 Frisian worker-family entities; the world tree adds
16 critters.

There are 315 directional declarations, each yielding six tracks (1,890
direction tracks), plus 385 nondirectional tracks. Appearance variants remain
attached to one entity identity. For example, the Frisian soldier's rookie,
helmet, sword, and hero directories are variants of `frisians_soldier`, not
four unrelated identities. `basename` aliases are also explicit: a critter's
`eating` action may legally reuse its `idle` pixels without being deduplicated
away as a labeling error.

Thirteen primary-image hashes occur in duplicate groups, representing 15 excess
byte-identical files. Downstream splitting must remain entity/provenance aware;
global hash deduplication must not silently sever an action alias or collapse
identity structure.

## Rights and attribution evidence

The repository's in-game licensing page states that the game is licensed under
GPL version 2.0 or, at the recipient's option, a later version. The audit records
`GPL-2.0-or-later` and binds all evidence bytes:

| Evidence | SHA-256 |
| --- | --- |
| `COPYING` | `8177f97513213526df2cf6184d8ff986c675afb514d4e68a404010521b880643` |
| `data/txts/LICENSE.lua` | `081c8efe81ea36b1280f2ddfcb8f21642bdfcdf8473493fd1052b5bcb7a67fa5` |
| `CREDITS` | `41750043cf62ade184b219cd044ef25882c41896c815cd3d9aace9a676e4802d` |
| `data/txts/developers.json` | `2bad4608cc9f9e335a650766cb13526eca84ef54ce0bd4684ea9e9f7b7d99a39` |

The audited worker/critter directories contain no per-entity `LICENSE`,
`CREDITS`, `AUTHORS`, or `README` files. Project-level GPL and contributor
evidence must therefore travel with every projected record, but it must not be
misrepresented as file-level authorship. Appropriate documentation should cite
the Widelands Development Team, repository, immutable commit, license evidence,
and the project contributor list. If source sprites or derived frame crops are
redistributed, GPL source/notice obligations need a separate compliance review.
This audit records evidence; it is not a legal conclusion about model training
or generated weights.

## Projection boundary

The current work is audit-only. Nothing from this adapter has been inserted into
the live SQLite index. A later projector should enforce these gates:

1. accept only `complete_entity` tracks with `exact_source_sequence = true`;
2. use scale-1 neutral sources as canonical frame pixels;
3. retain other scales and player-color masks as linked variants/layers;
4. preserve entity ID across every action, direction, and appearance variant;
5. keep menu/status/equipment media out of body-frame sequences;
6. attach all four rights documents and all five engine-semantics documents;
7. preserve the archive, manifest, image, and audit hashes; and
8. run identity/blob leakage audits before training snapshot inclusion.

The pure entry points are `audit_widelands_archive(path)`,
`audit_known_widelands_archive(path)`, and
`known_widelands_cas_path(raw_root)`. Synthetic tests lock parsing, aliases,
direction order, row-major crops, integer timing, numbered-file precedence,
player-color separation, auxiliary roles, unsafe ZIP rejection, and canonical
serialization. The exact-CAS regression locks every headline count and the
canonical audit hash above.

## Conservative database projection

The follow-on projector is now implemented in
`src/spritelab/ingest/widelands.py`. Its immutable projection-manifest SHA-256
is
`6e9566274dbd1e2b7159814da7b2ef60532b5888a885b5373253fa7d89ca3fab`.
Planning is pure and deterministically partitions all 2,275 audited tracks and
all 34,414 frame occurrences; planning itself does not touch SQLite.

The admission rule is deliberately stricter than the audit rule. A track is a
database sequence only when its mandatory 1x source already contains the exact
complete runtime pixels. A player-color mask is not an optional annotation:
Widelands combines it with its body using an engine blend operation and a
runtime-selected player color. The projector neither chooses that parameter nor
claims to reproduce the engine blend. It therefore records each body, mask,
frame pairing, and required `player_color` parameter in the immutable modular
exclusion ledger, but creates no entity, sequence, or frame row from that body.
Masks are never promoted to independent entities or training sequences.

| Projection partition | Tracks | Frames | Entities represented |
| --- | ---: | ---: | ---: |
| Exact unmasked complete-entity sequences | 193 | 3,272 | 22 |
| Modular body + required player-color mask, unresolved | 2,082 | 31,142 | 156 |
| Total, matching the source audit | 2,275 | 34,414 | 177 source entities |

`frisians_diker` contributes both admissible unmasked tracks and masked modular
tracks, so the entity counts in the first two rows overlap by one. The 22
admitted identities comprise 20 animals and two humanoids: all 16 critters,
four animal carriers, `atlanteans_spiderbreeder`, and `frisians_diker`.
Admitted action/direction tracks are:

| Canonical action | Sequences |
| --- | ---: |
| Walk | 126 |
| Carry | 30 |
| Idle | 22 |
| Eat | 15 |

There are 183 animated and 10 static admitted sequences. All 193 are literal
engine loops; no loop policy is guessed. The direction mapping is explicit:
northeast/east/southeast/southwest/west/northwest becomes
up-right/right/down-right/down-left/left/up-left, while a nondirectional source
remains recorded as nondirectional and is stored as taxonomy `unknown` rather
than inventing a facing. Every database frame retains the exact source member
digest, source-frame index, native rectangle, integer duration, phase, source
direction, and body-layer index. Stable source keys distinguish action aliases;
separate content-deduplication keys reveal byte- and timeline-identical aliases
without erasing their steering labels.

### Evidence closure and deduplication

The plan requires 4,180 archive members before a write can begin:

- 3,994 source-layer members: 154 admitted body members, 1,920 quarantined
  body members, and their 1,920 required mask members;
- all 177 entity manifests;
- all four collection rights/credits documents; and
- all five engine-semantics documents.

The 3,994 layer paths resolve to 3,548 distinct payload hashes. There are 207
duplicate-hash groups and 446 path-level excess occurrences; these are retained
as provenance occurrences rather than treated as extra pixels. Including the
four rights and five engine files gives 4,003 explicit member/hash requirements
and 3,557 distinct evidence payload hashes. Manifest identity is bound by the
exact archive digest, member path/ordinal, and canonical source-audit digest.
If projected, the 193 admitted sequences create 2,123 provenance occurrence
links: their body sources, one manifest, four rights documents, and five engine
documents per sequence.

Preflight refuses a write unless the source registration, source item, archive
inventory, every required member, every extracted layer blob, and every audited
layer hash agree. `check_widelands_projection_readiness()` opens the database in
SQLite read-only mode and enables `PRAGMA query_only`; it can safely inspect a
live index without projecting. During implementation, the live query found the
Widelands source registered but no source item or archive inventory; no live write
was attempted then. Projection tests create and mutate only a temporary fixture
database and lock idempotency, exact ordering/timing, direction conversion,
occurrence evidence, and mask exclusion behavior. The later explicit integration
is recorded below.

The projector carries the project-level `GPL-2.0-or-later` conclusion and exact
`COPYING`, in-game license, `CREDITS`, and `developers.json` evidence on every
admitted record. It adds no file-level creator claim and no rights observation,
because the evidence does not establish per-file authorship. The redistribution
and legal-review caveat above remains unchanged: this mechanism preserves the
available evidence and citation chain; it does not turn collection-level credit
into per-sprite attribution or make a legal conclusion about training.

## Live integration result (2026-08-12)

The exact commit archive was registered as source item
`widelands/widelands` and pinned by SHA-256
`51093e130b2f4098716bc442ce3d8566d94b4590cfa233ba1c74e5d173a48186`.
The ZIP inventory contains 26,840 members and 25,166 regular files, with
inventory SHA-256
`a26375b3b0946b9cc491abded029746c1b4d2eec415a7fa1ec5fd2001ff7bbc3`.
Only the plan's 4,180 required evidence members were extracted into CAS. All
3,994 selected PNG source-layer occurrences decoded successfully under strict
media inspection, and the query-only readiness check matched every expected
source hash before projection.

The projection then ran inside one explicit `IndexDB.transaction()` and produced:

- 193 sequences, including 183 animated and 10 static loops;
- 3,272 ordered frame references and 22 complete-entity identities;
- 2,123 sequence-to-archive evidence links; and
- zero new `rights_observations`, because collection-level evidence was already
  attached to the indexed source item.

The live result reproduced projection-manifest SHA-256
`6e9566274dbd1e2b7159814da7b2ef60532b5888a885b5373253fa7d89ca3fab`.
The 2,082 unresolved body/player-color-mask tracks and their 31,142 frames remain
in the immutable exclusion ledger; the integration did not turn them into
independent entities or silently choose a player color.

## Verified training materialization (2026-08-12)

The source-scoped temporal snapshot
`data/index/snapshots/widelands-temporal-action-v1.json` contains the 183
multi-frame sequences and 21 identities that satisfy the snapshot's known-timing
contract. Its canonical artifact SHA-256 is
`b7c72b8f412d155b4d4884750e9ef0685eceda0d7635ab0f35fd9139f66964d4` and
its embedded dataset-manifest SHA-256 is
`e5e7b426f138f6c79f3df25a37c4d3e54367f27b3e710698f344e24023de0fe1`.
The deterministic split is 166 train, eight validation, and nine test clips;
all 19 identities with more than one admitted action remain grouped against
identity leakage.

Exact `source_image` rectangles were materialized without inferred cells,
downscaling, or non-nearest resampling. The materialization contains 183 native
RGBA clips in the 64-by-64 bucket. The first artifact had SHA-256
`00d3e6f34592fee86e7d55ad1c5f64711c4d9fa49029f8c491de1828c4dc82d9`.
It remains immutable, but its schema-v1 output provenance used only legacy
sheet-named fields. The additive, no-clobber v2 artifact at
`data/processed/widelands-temporal-action-v2` has materialization SHA-256
`74ea49b0c4b8c0dcb01994c07522a47079ef5a3337b8a7298427d4eed9a4fbe0`
and manifest-file SHA-256
`e8dace1f934c126ef915d94923445170813bfb803019b7a50d31727cb5e18b50`.
Every frame now explicitly retains `source_rect_coordinate_space="source_image"`
and neutral `source_carrier_size`; all 183 NPY file and typed-array hashes are
byte-identical to v1.
The independent training loader reverified every file and typed-array hash and
successfully retimed every source clip to eight frames while preserving each
loop's authored duration weights. Native lengths remain heterogeneous: six
clips have two frames, two have seven, one has eight, 25 have ten, two have 19,
and 147 have 20.

The generic fixed-phase eligibility gate was also exercised across both newly
integrated sources. Snapshot
`data/index/snapshots/widelands-ss14-model-ready-v1.json` used
`--temporal-mode model_ready` and selected exactly these 183 Widelands loops,
while excluding all 13 SS14 timelines whose durations are exact but loop mode
and phases are unknown. Its artifact SHA-256 is
`3259822d9cd8fc3e7ee5030c9b0c8c483b5f1f727ea28f44464eac5be2083e96`
and embedded manifest SHA-256 is
`5fdacef77286f748991572ed95bff2bf428d511a01a19b943c8b26eedbe6f148`.
The corresponding no-clobber materialization SHA-256 is
`625c81f4581b2edc0c183df80d775d60a23a78f426e71a4c65d68556d3499b2b`;
all 183 clips load and retime to `[8,64,64,4]`.

The current no-clobber pixel-quality audit at
`data/index/reports/widelands-temporal-action-v2-pixel-quality.json` has file
SHA-256 `691632bf1dc6ba05998da81a5f0a557c9b69c7d5a674a0e7501550f51c2938d8`.
(The preserved v1 audit, over byte-identical arrays, has SHA-256
`b1f9d6d0f566e63a8b37b71df6e0f858408365aa9c356a598bc015ef3ccc55e8`.)
It scanned all 3,262 native frames and 13,361,152 pixels after independently
verifying the complete manifest. No clip or frame is wholly opaque or wholly
transparent, all frames contain visible pixels, and there are zero opaque exact
magenta (`#ff00ff`) pixels. Partial alpha appears in 3,242 frames, reflecting
the source artwork rather than a transparency inference. No corner is occupied;
1.136% of unique border pixels are visible across the corpus, a retained source
geometry fact rather than an automatic crop or repair.
