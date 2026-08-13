# TMWA legacy client-data corpus

This adapter treats The Mana World client-data snapshot as immutable provenance
evidence, not as a folder of images to repair. It reads one exact ZIP from the
content-addressed store, audits the declarations in memory, and constructs a
deterministic projection plan. It does not extract, recolor, composite, crop, or
write during audit or planning.

The resulting safe subset is suitable for private sprite-animation research,
but that purpose does not relax provenance or licensing requirements. This
document is an engineering record, not a legal conclusion.

## Immutable snapshot

| Fact | Exact value |
| --- | --- |
| Source | `themanaworld/tmwa-client-data` |
| Provenance source ID | `tmwa_client_data` |
| Repository commit | `3e63a6f033b6406fe855dba14dbead3db28671fd` |
| ZIP root | `tmwa-client-data-3e63a6f033b6406fe855dba14dbead3db28671fd` |
| CAS object | `data/raw/objects/sha256/7b/7a/7b7a8c61eca284b35ddedaa1af4210a1694933e36ebdc4299bd73395a6532152` |
| Archive SHA-256 | `7b7a8c61eca284b35ddedaa1af4210a1694933e36ebdc4299bd73395a6532152` |
| Archive bytes | 65,557,370 |
| Central-directory inventory SHA-256 | `53bdb96c7165b55e6da670b1234228bd21797fecb2ef049b91a227ef2b9aa4c7` |
| Projection-plan SHA-256 | `b0962aafa56673c294b6a81a1748430097d8170f5813d4d2f7daf36bc3dfbe6d` |

The exact-archive entry point rejects any byte length, SHA-256, root, commit,
inventory hash, or pinned count that differs. ZIP paths are normalized and
checked for absolute paths, drive prefixes, traversal, duplicate paths,
encryption, excessive size, and members outside the one root. The one symlink,
`tools -> ../tools/client/`, is inventoried as a symlink and never followed.

## Inventory and full audit

The central directory contains 5,082 members: 169 directories, 4,912 regular
files, and one symlink. There are 4,913 non-directory entries. Declared expanded
bytes are 193,431,213 and aggregate compressed member bytes are 64,110,574.

The existing live inventory marks 4,169 relevant records as extracted:

- 2,521 XML documents;
- 1,636 PNG images;
- eight `.txt` files;
- two `.md` files; and
- `COPYING` plus `license-missing`, which have no suffix.

Every relevant record is read and hashed. All 1,636 PNGs are opened and fully
decoded with Pillow, not merely identified from their headers. They are all
RGBA PNGs. They span 396 distinct dimensions. No PNG failed decoding.

There are exactly 756 XML files under `graphics/sprites/**/*.xml`; every one has
a `<sprite>` root and parses successfully. Their physical families are:

| Family | XML files |
| --- | ---: |
| equipment | 393 |
| monsters | 176 |
| hairstyles | 88 |
| npcs | 54 |
| icons | 31 |
| model | 10 |
| root of `graphics/sprites` | 3 |
| races | 1 |

The physical declarations contain 762 imagesets, 209 includes, 3,501 actions,
12,744 authored animations, 31,395 `<frame>` commands, 1,601 `<sequence>`
commands, 7,062 `<end>` commands, six `<jump>` commands, three `<label>`
commands, and five `<goto>` commands. Loading every definition as a runtime
root, including inherited actions, produces 20,718 effective authored tracks
and 65,366 resolved frame occurrences at runtime variant zero.

The adapter retains all 587 XML comments across the repository verbatim with
their member, logical path, and line. Of these, 117 comments occur in 61 sprite
definition files. Comments are evidence claims, not automatically parsed into
authors or grants.

## Pinned ManaPlus behavior

Runtime interpretation is pinned to ManaPlus commit
[`986a3bff49af01f6abd13c1d3b9d41cf50c557ce`](https://github.com/ManaPlus/ManaPlus/tree/986a3bff49af01f6abd13c1d3b9d41cf50c557ce).
Only small immutable source responses were examined; no second source archive
was indexed or added to the corpus.

| Immutable source | Git blob SHA-1 | Bytes | SHA-256 | Semantics used |
| --- | --- | ---: | --- | --- |
| [`spritedef.cpp`](https://github.com/ManaPlus/ManaPlus/blob/986a3bff49af01f6abd13c1d3b9d41cf50c557ce/src/resources/sprite/spritedef.cpp) | `2ccabd5aa477c74940d0c1e4f8f693a20c78ae2b` | 22,433 | `ea9852b6f17f8d2c30ecee333f7493d4945dcea7f5daeb2b80c750f427dd3407` | include order, variants, imagesets, commands, offsets |
| [`imageset.cpp`](https://github.com/ManaPlus/ManaPlus/blob/986a3bff49af01f6abd13c1d3b9d41cf50c557ce/src/resources/imageset.cpp) | `bb7f1bc621959f2eba2a8f80ec351fe44f7beb48` | 2,318 | `efc8d3d7e7f5ac118ef6b97d05eacb628bc910b8ff8ccef18a7d893104175dac` | complete row-major cells |
| [`dye.cpp`](https://github.com/ManaPlus/ManaPlus/blob/986a3bff49af01f6abd13c1d3b9d41cf50c557ce/src/resources/dye/dye.cpp) | `37a8d5950f149e34c67bed87a9d43d17cdb560e1` | 8,237 | `88868ec94f682bb52e91ae2cad4c284f3aefad9712f8751c8da9ed51bc0403dc` | palette placeholders and channel transforms |
| [`animation.cpp`](https://github.com/ManaPlus/ManaPlus/blob/986a3bff49af01f6abd13c1d3b9d41cf50c557ce/src/resources/animation/animation.cpp) | `f438744ccc74d2738f3004dc4636b2b1c3450a9c` | 3,011 | `defecd8def93ee2ae35004b0738b9933309b44fd063b5fc19176c4325781a085` | stored frame and command fields |
| [`animatedsprite.cpp`](https://github.com/ManaPlus/ManaPlus/blob/986a3bff49af01f6abd13c1d3b9d41cf50c557ce/src/resources/sprite/animatedsprite.cpp) | `a34aba041d74c98ae71022b839c102c281d3ad07` | 12,925 | `2742c23d2490d6c9bbf77354e532d48734adc330bbfb1bede516f23deb4cc5b9` | delay, wrap, end, and control flow |
| [`action.cpp`](https://github.com/ManaPlus/ManaPlus/blob/986a3bff49af01f6abd13c1d3b9d41cf50c557ce/src/resources/action.cpp) | `b2909b442b2f1bdaa1bbbc4aae042d42fa76b436` | 2,969 | `b2f8ebddb1105bf31a9b7bf7e3e85cb4bac9af9dc2b9585c2f6669ba699fe8d1` | direction lookup and fallback |
| [`defaults.cpp`](https://github.com/ManaPlus/ManaPlus/blob/986a3bff49af01f6abd13c1d3b9d41cf50c557ce/src/defaults.cpp) | `5fa3a6b7b48f5abb381da49b806ab725e72488b1` | 27,184 | `0fee23b274d52e2e6bc7eaa5277963f83b599f4b355b31d8bef363393cf8fea7` | `fixDeadAnimation` default true |
| [`client.cpp`](https://github.com/ManaPlus/ManaPlus/blob/986a3bff49af01f6abd13c1d3b9d41cf50c557ce/src/progs/manaplus/client.cpp) | `a632e6ccf307df2e90db4249193b339c18dbf30e` | 64,668 | `708f06757b113aa827c5a502602a01d98bf5799570bae8717ab7797909639d77` | server feature assignment |

The reproduced semantics are intentionally narrow:

- `<include file="…">` resolves from `graphics/sprites/`, not relative to the
  including XML. A global processed-file set suppresses repeat/cycle loading.
  All 209 archive include targets resolve; the graph has no cycle and no
  duplicate visit requiring suppression. Maximum closure size is three files.
- Children are processed in document order. Imageset names are
  first-definition-wins across the include closure; action names overwrite.
  Runtime action aliases and direction fallbacks are not synthesized as authored
  training tracks.
- Variant offsets belong to the document that declares the frame command. The
  conservative projection uses runtime variant zero and retains `variants`,
  `variant_offset`, shifted, and unshifted indices.
- `<sequence start="a" end="b">` is inclusive and may ascend or descend.
  Per-command delays and offsets are expanded exactly. The current snapshot does
  not use the engine's `repeat` or `value` sequence forms.
- Engine draw offsets are retained alongside XML offsets:
  `x = xmlX + imagesetX - cellWidth/2 + 16` and
  `y = xmlY + imagesetY - cellHeight + 32`.
- A positive delay advances automatically. A zero-delay frame holds. Ordinary
  tracks wrap; a terminal `<end>` returns to `stand`. Jump/label/goto and random
  gates remain unresolved control flow and are excluded.
- ManaPlus defaults `fixDeadAnimation` to true and then consults the server
  feature database. This snapshot's `features.xml` has no override, so the
  effective value is true. The audit retains the declared delay and records the
  exact runtime adjustment that forces the last authored `dead` frame to zero.
  A multi-frame dead track therefore advances to a permanent hold and is
  conservatively excluded; a one-frame dead track is an exact hold.

## Sheets, geometry, and palette operations

The 762 imagesets use 587 distinct source literals and 586 distinct base image
paths. There are 357 declarations with a `|…` dye expression. Every source
literal, base path, channel expression, grid size, imageset offset, member hash,
and decoded PNG dimension is retained.

Four imageset declarations point at unavailable base PNGs:

- `graphics/sprites/equipment/shields/steel-dyable.png`;
- `graphics/sprites/equipment/legs/pants-male.png`; and
- two declarations of `graphics/sprites/equipment/head/knighthelm-dyable.png`.

Five imagesets have pixels outside a complete cell grid. ManaPlus admits only
complete cells, so the adapter records the x/y remainders rather than padding,
trimming, or changing the declared grid. Two locally referenced cases expose an
invalid source index: Halifax has no complete declared cell and
`portal_32x64-01.xml` references cell 12 when complete cells are indexed 0–11.
The effective audit reports 18 tracks with an out-of-bounds frame after include
reuse.

ManaPlus dye syntax uses `R,G,Y,B,M,C,W,S,A` channels and can combine explicit
palettes with one-letter placeholders supplied by a semantic binding. The audit
records internal imageset and external entity suffixes separately. No dye is
instantiated and no pixel is recolored. Any track needing a palette transform is
excluded even when its geometry is otherwise known.

## Semantic roots and layer roles

Root documents are expanded through their own root-relative `name` includes.
The audit distinguishes a complete one-layer entity from an explicit runtime
composite and from modular body, race, hair, and equipment layers. These are not
silently flattened.

| Corpus | Documents | Entities | Sprite refs | Unique XML paths | Zero layer | Single layer | Multi-layer | Palette refs | Unresolved refs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| monsters | 1 | 233 | 442 | 224 | 0 | 167 | 66 | 241 | 0 |
| NPCs | 259 | 260 | 908 | 209 | 18 | 93 | 149 | 654 | 0 |
| items | 1,171 | 1,159 | 1,401 | 459 | 490 | 57 | 612 | 1,009 | 4 |
| avatars | 1 | 2 | 8 | 7 | 0 | 0 | 2 | 7 | 0 |
| pets | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| horses | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mercenaries | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| homunculuses | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| elementals | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| effects | 1 | 179 | 0 | 0 | 179 | 0 | 0 | 0 | 0 |
| emotes | 1 | 43 | 43 | 1 | 0 | 43 | 0 | 0 | 0 |

`monsters.xml` has one declared include of `mods/monsters.xml`. The repository's
`.gitmodules` marks `mods` as a submodule, but the acquired source archive does
not contain it. The include is retained as
`included_document_unavailable`; counts above cover only the acquired root file.

The four unresolved item sprite paths are the two Knight's Helmet gender paths,
Sweet Tooth Staff's `equipment/weapons/staff.xml`, and Black Steel Shield's
`equipment/shields/steel-dyable.xml`. They are not redirected to similarly named
files.

Entity breadth is conservative. The source corpus role provides the base
monster/effect/NPC/object distinction. A small reviewed token list separates
animal, fantasy humanoid, undead monster, animated object/plant, and other
creature cases while preserving names and literals. A quadruped cue is emitted
only for reviewed cat, crocodile, frog, Mouboo/Moubi, and Tortuga tokens; it is
not presented as a visual or anatomical proof. In the projected semantic set,
54 unique monster identities comprise 28 monsters, 15 animals, seven objects,
and four humanoids; six have the conservative quadruped cue.

Effects are kept separate: 179 `effects.xml` definitions point to particle/audio
resources rather than sprite timelines, while 43 emotes bind a variant of the
shared `graphics/sprites/emote.xml`. Neither is admitted by the monster-only
safe projection.

## Rights evidence and quarantine

Three repository documents are stored verbatim with independent scope:

- `license.md`, including its warning that the information may be incomplete or
  incorrect;
- `license-missing`, as explicit negative/missing evidence; and
- `COPYING`, as repository GPLv2 text, not an inferred per-file license.

Markdown rows are split at the first and last table delimiter so attribution
cells containing embedded `|` characters stay verbatim. The result is 1,534
table claims covering 1,532 unique paths. Two paths have duplicate rows; one of
those duplicates disagrees. `license-missing` has 309 unique path claims. Four
paths occur in both documents. Unknown-contributor markers remain explicit.

Across all 1,636 PNGs, the conservative status counts are:

| Status | PNGs |
| --- | ---: |
| documented path claim | 968 |
| explicitly license-missing | 304 |
| no asset-path claim | 231 |
| unresolved contributor or license literal | 128 |
| contradictory, including a disagreeing duplicate | 5 |

Only `documented_path_claim` images can enter the safe projection. Even then,
the complete raw claim and repository warning are retained; the projector does
not create a `rights_observations` row or claim an independently verified
chain-of-title.

## Deterministic projection

Planning is pure: the full plan is canonical JSON with sorted keys and compact
separators, and its SHA-256 covers archive/member facts, all semantic bindings,
all XML comments, rights documents and claims, engine evidence, admitted
records, and full excluded tracks. Local archive location is omitted from the
canonical identity. The exact current hash is
`b0962aafa56673c294b6a81a1748430097d8170f5813d4d2f7daf36bc3dfbe6d`.
The projector contract is `tmwa_exact_provenance_v3`: engine-native `hold` is
represented as shared `loop` for its one-frame repeated pose, and
`one_shot_return_to_stand` is represented as shared `one_shot`. Both original
engine labels remain in sequence, motion, and frame metadata. This prevents
database rows from carrying loop literals rejected by the training loader while
preserving the exact return-to-stand fact. Each source cell is also emitted as
an executable `metadata.frame_rect` in `source_image` coordinates, alongside
the original TMWA `source_png` coordinate-space literal and `[x,y,width,height]`
cell tuple. No crop is created during projection.
The primary, leakage-safe identity remains the physical sprite-definition
resource. Its class is the unanimous class of its admitted semantic bindings,
and its human label is the path-stable definition filename stem; all source
semantic entity names and IDs remain linked as secondary subjects. This keeps
one appearance resource in one split without collapsing animal, monster,
humanoid, and object conditioning to `unknown`.

A track is admitted only when all of the following hold:

- it has at least one resolved, single-layer, palette-free `monsters.xml`
  binding with no unresolved sprite attributes;
- every declared frame resolves, in order, to an in-bounds cell on one source
  PNG and retains exact duration and offsets;
- no internal or external palette operation is required;
- the loop/hold/terminal-end behavior is deterministic and the track has no
  jump/label/goto, random gate, or unsupported command;
- multi-frame durations are positive; and
- the source PNG has one non-contradictory, non-missing path claim without an
  unknown contributor marker.

This admits 853 sequences, 4,153 frames, 53 definition roots, 53 source PNGs,
and 54 semantic monster identities. There are 737 records with a one-document
closure and 116 with a two-document closure. Admitted loop modes are 280 loops,
305 zero-delay single-frame holds, and 268 terminal-end sequences. Direction
counts are 156 each for down, up, and right; 154 left; 51 each for the four
diagonals; and 27 authored default-direction tracks.

Source actions remain literal. Reviewed mappings include `stand -> idle`,
`dead -> death`, direct walk/attack/spawn/cast/hurt mappings, specific melee
weapon actions, bow/distance actions, and magic/wand actions. Unmapped literals,
such as `attack_splash`, remain `unknown` in the shared taxonomy rather than
being guessed.

There are 19,865 excluded tracks. Reasons overlap:

| Exclusion reason | Tracks |
| --- | ---: |
| no safe complete one-layer monster binding | 19,350 |
| unresolved imageset palette transform | 10,774 |
| source image explicitly license-missing | 7,333 |
| source image has no asset-path claim | 1,070 |
| unresolved contributor/license literal | 754 |
| contradictory source-image claims | 169 |
| loop/end behavior unresolved | 207 |
| zero-delay hold inside a multi-frame track | 200 |
| multi-frame nonpositive duration | 200 |
| no resolved frames | 144 |
| declared/resolved frame counts differ | 113 |
| missing source image | 95 |
| frame index outside complete grid | 18 |
| runtime control flow | 7 |

Sequence metadata retains exact archive/member/image hashes, logical and member
paths, definition include closures, action/direction literals, source line
locations, commands, loop/end meaning, entity bindings, rights claims and
caveat, cell rectangles, row-major indices, per-frame duration and phase, XML
offset, engine offset, and source PNG dimensions. Sequence frames reference the
full immutable PNG blob. No derivative frame blob or crop is materialized.

## Database safety and validation

Readiness opens SQLite through a `file:` URI with `mode=ro`, then enables and
verifies `PRAGMA query_only = ON`. It requires:

- the exact archive blob;
- the exact ZIP inventory counts, expanded/compressed bytes, and inventory hash;
- exactly one `tmwa_client_data` item linked to that archive;
- all 4,169 relevant member rows; and
- the expected extracted SHA-256 plus registered blob row for every one of
  those members.

The projector performs this strict preflight before its first write. It does not
initialize the schema and does not call `IndexDB.transaction()` internally.
Callers may wrap it externally. It upserts stable resource and semantic entity
keys, stable sequence source keys, subjects, motion facts, source occurrences,
and sequence-frame provenance. It does not touch rights observations, crops,
materializations, artifacts, or the acquired ZIP.

On 2026-08-12, the live index passed this query-only readiness check with all
4,169 required members present, extracted, hash-matched, and registered. No live
projection was run. The exact plan was projected twice only into a temporary
database under caller-owned transactions: the first pass created 853 sequences
and the second reused all 853; final rows were 107 entities, 853 sequences, 853
source keys, 853 motion annotations, 4,153 sequence frames, and zero rights
observations.

Focused verification commands are:

```powershell
.venv\Scripts\python.exe -m ruff check src/spritelab/adapters/tmwa.py src/spritelab/ingest/tmwa.py tests/test_tmwa_adapter.py tests/test_tmwa_ingest.py
.venv\Scripts\python.exe -m pytest tests/test_tmwa_adapter.py tests/test_tmwa_ingest.py -q
```

At the final audit checkpoint, drive `C:` had 222,754,373,632 free bytes
(207.456 GiB). This is an environment observation, not part of the projection
identity.

## Live v3 integration and model-ready derivative

After independent review, the v3 plan passed query-only readiness against the
live index with all 5,082 archive rows exact and all 4,169 required members
extracted, hash-matched, and blob-registered. The pre-write SQLite file was
2,313,555,968 bytes with SHA-256
`ac2d1f1b25681b3140fca034cfd18418076927858803c949b09df6daf2d8aa18`;
`PRAGMA quick_check` returned `ok`. One caller-owned transaction created all 853
sequences, 4,153 frame rows, 53 physical resources, 54 semantic entities, and
5,234 archive-occurrence links. A subsequent v3 pass reused all 853 stable
sequence keys while updating the shared loop/cell and primary-class facts. No
rights observation was added.

The source-filtered, action-known model-ready snapshot is
`data/index/snapshots/tmwa-model-ready-action-v2.json`, SHA-256
`07b34cfb8d667cd967c0452cb1fd0865d632368c0178e6e6a27693e6091c3e78`;
its dataset-manifest SHA-256 is
`29e9b7a55a4f98f380c5d712b593bb544333e4d807509063b770791a79a774a4`.
It contains 540 multi-frame sequences across 48 physical identities: 206 animal,
276 monster, 45 humanoid, and 13 object clips. Actions are attack 238, walk 213,
idle 63, spawn 17, cast 8, and hurt 1. Split assignment is identity-aware
(485/28/27 train/validation/test) with no identity, exact-array, fixed-target,
or source-blob overlap. The single archive item necessarily remains a recorded
source-pack/style overlap across splits. The earlier `tmwa-model-ready-action-v1`
snapshot is preserved as a superseded diagnostic because its physical primary
entities were still classed `unknown`.

Exact cell materialization is
`data/processed/tmwa-model-ready-action-v1/materialization.json`, with
materialization SHA-256
`c536bf7375d3549549ef9697522d92d11a09ae2b0d4a6d5c38701e614313ad81`.
All 540 clips load natively and retime to eight frames without interpolation;
326 are 64x64, 181 are 128x128, 26 are 256x256, and 7 are 512x512. The exact
training-readiness audit SHA-256 is
`6859b015891be29de02a70d7bbfa4de4ea857e6a6c76afbe812eaecf9a3fca0c`.
It identifies 53 target-distinct action-only groups in the training split (36 at
64px, 9 at 128px, and 8 at 256px), with 106 selected representatives and no
conflicts or aliases in those groups. A separate validation group is correctly
rejected because its two action labels have one byte-identical fixed target.

The pixel-quality audit SHA-256 is
`14595634474fb5cb95049088f7a05a7ebac2c6a9fbcb6eb7cb2939d1ed37f743`.
Across 3,652 native frames and 60,743,680 pixels, every frame has visible pixels,
no clip or frame is fully opaque/transparent, and no opaque exact magenta sentinel
occurs. Partial alpha occurs in 2,550 frames. Two corner pixels and 19,045 border
pixels are visibly occupied; these are reported rather than cropped or repaired.

## Known limitations

- The absent `mods` submodule is not fetched or inferred, so the root monster
  inventory is knowingly incomplete relative to a possible submodule checkout.
- Runtime action substitution and direction fallback are documented but not
  emitted as authored training evidence.
- Only variant zero is projected. Nonzero variants are audited through their
  declarations but are not projected without an exact semantic binding choice.
- Dyes, multi-layer characters, equipment, hair, race/body layers, emotes,
  particles, and effects remain separate evidence. There is no recolor or
  composite renderer here.
- Zero-delay commands can represent deliberate holds; a zero delay inside a
  multi-frame track is excluded because a finite exported timeline would be
  misleading.
- Path-table licensing is preserved as a repository claim with its warning. It
  is not equivalent to independent legal verification.
- Morphology labels are conservative source-name/corpus cues, not pixel-derived
  anatomical annotations.
