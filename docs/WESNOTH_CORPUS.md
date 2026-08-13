# Battle for Wesnoth exact-snapshot corpus audit

This adapter audits one immutable Battle for Wesnoth repository ZIP directly in the
content-addressed store. It does not extract the archive, render image-path functions, invoke the
WML preprocessor, or write to the live corpus database. Every accepted frame retains its source
WML location, literal expression, exact archive member, member-content SHA-256, dimensions,
timing, action basis, loop basis, direction filters, and rendering context.

## Snapshot and reproducibility identity

| Field | Exact value |
| --- | --- |
| Repository | `https://github.com/wesnoth/wesnoth` |
| Commit | `52858e8fa4ae3c0427f5ad12ec11cfdf22fe2b2b` |
| Commit citation | `https://github.com/wesnoth/wesnoth/tree/52858e8fa4ae3c0427f5ad12ec11cfdf22fe2b2b` |
| Archive SHA-256 | `fd10c38abfe3406fbc1e4dfdbc03762c576e5c9376173a7f09120040cbccba3e` |
| Archive size | 730,992,710 bytes |
| ZIP root | `wesnoth-52858e8fa4ae3c0427f5ad12ec11cfdf22fe2b2b` |
| Canonical audit-record SHA-256 | `21c4d48184cb99495609de0c238a107d23e9c19d4000009df4f6893447b1e9e8` |

`audit_known_wesnoth_archive()` hashes the archive before parsing and rejects a different digest
or root. The audit dataclasses are frozen and canonical JSON serialization sorts mapping keys and
all corpus-derived records. The canonical audit hash covers the counts, complete entity and frame
records, rights evidence, issues, and projection policy; it intentionally does not include its own
hash field.

## Archive and media inventory

The pinned ZIP has 30,482 members: 29,038 files and 1,444 directory entries. It contains 2,618
`.cfg` files, 620 of them under a `units` tree, 20,264 PNG files, and 21,257 raster images across
PNG, WebP, JPEG, and JPG.

The path-based image-role audit partitions all 21,257 images without overlap:

| Source role | Images |
| --- | ---: |
| Unit sprite tree (`images/units`) | 9,068 |
| Portraits | 592 |
| Effects and projectiles | 1,401 |
| Terrain | 6,489 |
| UI, icons, themes, and buttons | 330 |
| Maps and story art | 309 |
| Other image paths | 3,068 |

Only literal `[unit_type]` records and their unit-body `[frame] image=...` declarations are
candidates for an entity animation sequence. Portraits, terrain, UI, maps, story images,
projectiles, haloes, and auxiliary layers remain indexed evidence; they are not silently promoted
to playable body sprites.

The adapter distinguishes four frame declaration roles:

| Render role | Declarations |
| --- | ---: |
| Primary unit body | 3,171 |
| Projectile | 356 |
| Auxiliary layer | 320 |
| Effect overlay | 207 |

## What is exact and what is deliberately unresolved

The adapter reads the pinned runtime implementation as evidence, especially:

- `src/units/animation.cpp`, which maps WML animation tags to runtime events and forces every
  `standing_anim` and its subparticles to cycle;
- `src/units/frame.cpp`, which establishes explicit duration/begin/end handling and the default
  horizontal/vertical flip behavior; and
- `src/units/frame_private.hpp`, which implements progressive image expansion and residual time
  allocation.

The literal progressive-image subset is reproduced exactly:

- `[1~4]` expands inclusively without invented zero padding;
- `[01~04]` preserves declared zero padding;
- `[1*3]` repeats a value three times;
- multiple square groups expand positionally, not as a Cartesian product;
- one duration applies to every expanded image;
- a duration list must match the expanded image cardinality;
- an outer frame duration allocates `(outer - total inline durations) / total image count` to each
  otherwise unspecified image, integer-truncated and clamped to 1 ms, matching the engine; and
- without a duration, the engine's progressive image default is 1 ms.

The frame record retains all raw frame attributes and their source lines, plus the complete
animation-to-frame context. This preserves `image_mod`, `halo_mod`, `halo_x`, `halo_y`, `offset`,
`x`, `y`, `directional_x`, `directional_y`, `layer`, `primary`, blending, highlighting, flip flags,
and future attributes even when the adapter does not interpret them.

Wesnoth image path functions such as `~RC(...)`, `~BLIT(...)`, `~FL(...)`, and separate
`image_mod` values are compositions, not source-pixel files. The adapter splits the immutable
base member from the modifier string, but marks the frame non-lossless. It does not rasterize or
pretend that the base PNG is the composited runtime result.

WML macro definitions are skipped and macro invocations remain source evidence. The adapter does
not claim that raw, unpreprocessed WML equals the campaign-dependent runtime configuration.
Likewise, `[if]` and `[else]` frame branches are retained independently with their direction
filters and branch paths, not concatenated into an invented timeline.

## Unit identity and morphology breadth

The literal audit finds 600 `[unit_type]` declarations representing 573 unique unit IDs. There
are 21 duplicate-ID groups with 27 excess declarations, so physical identity must include the
source config path and declaration line rather than unit ID alone. Sixty-four records declare
`[base_unit]` inheritance, which is retained but not expanded. Nested variations are preserved;
163 animation records have a non-empty variation path.

Entity class is derived only from explicit race values, explicit unit-tree placement, or a narrow
vehicle/object term. It is not guessed from pixels or fantasy names.

| Conservative entity class | Unit declarations | Safest direct animations |
| --- | ---: | ---: |
| Humanoid | 217 | 230 |
| Animal | 18 | 30 |
| Creature | 78 | 12 |
| Monster | 76 | 99 |
| Undead | 50 | 74 |
| Construct/mechanical | 18 | 2 |
| Vehicle | 11 | 25 |
| Unknown | 132 | 132 |

The last column counts animations, not entities; one unit may contribute several actions. Effects
are represented separately by the 1,401 effect/projectile image paths and 563 projectile/effect
frame declarations. The conservative taxonomy leaves 132 unit declarations unknown instead of
turning a filename into a biological assertion.

## Actions, steering, timing, and facing

The adapter maps only source tags and literal `apply_to` values. Attack filters retain exact WML
attack names and ranges, so multiple weapon-specific animations remain distinct. The complete and
safest-direct counts are:

| Normalized action | Literal records | Safest direct records |
| --- | ---: | ---: |
| Attack | 931 | 85 |
| Idle/standing | 366 | 228 |
| Defend | 105 | 58 |
| Move | 98 | 88 |
| Death | 70 | 60 |
| Emote/lead/recruit/victory | 46 | 42 |
| Heal | 26 | 21 |
| Spawn | 13 | 12 |
| Teleport | 6 | 4 |
| Move transition | 4 | 0 |
| Unknown generic event | 12 | 6 |

There are 1,677 literal animation records. The pinned engine forces all 301 standing animations
to loop. The remaining 1,376 records default to one-shot because neither the tag-specific engine
path nor their literal attributes enable cycling. This is runtime-derived loop metadata, not a
filename guess. Idle animations are therefore not automatically labeled loops: the engine may
schedule them as discrete idle events, while standing animations cycle continuously.

Direction constraints are retained at animation, conditional branch, and frame context levels.
Examples include `s,se,sw`, `n,ne,nw`, individual north/south tracks, and diagonal groups. The
pinned frame renderer defaults `auto_hflip` to true, so the immutable source track may be mirrored
at runtime for west-facing directions; primary frames default `auto_vflip` false, while auxiliary
particles default it true. The adapter records both literal overrides and effective defaults. It
does not invent eight physical directions from one mirrored source image.

## Lossless projection gate

Of the 1,677 literal animations, 1,604 contain at least one literal primary-unit frame. They expand
to 8,640 ordered primary frame occurrences. Of those occurrences, 8,597 resolve to immutable
archive members and 43 remain unresolved: 31 name missing files in this snapshot and 12 arise from
unresolved or invalid literal expressions.

The intentionally strict direct-projection subset is:

- 604 animations;
- 2,526 ordered frame occurrences; and
- 1,275 unique immutable image members.

Across every resolved primary declaration—not only the safe subset—4,968 unique image members are
referenced. Resolution basis remains explicit: 6,612 occurrences select the core binary path,
1,820 select the current campaign's binary path, 161 have a unique repository-wide image suffix,
4 use a direct repository-relative path, and 43 are unresolved.

An animation is directly projectable only when all of the following hold:

1. It belongs to a literal, nonempty unit ID.
2. Every primary frame is a literal, resolved archive image member.
3. Frame order and durations are exact.
4. The primary track has no conditional WML branch.
5. The animation contains no unexpanded macro invocation.
6. No primary frame has an inline image path function or separate `image_mod`.
7. The record does not require inherited `[base_unit]` content to define its current timeline.

The main quarantine surfaces are 1,004 macro-affected animations, 140 conditional primary tracks,
73 animations without literal primary frames, 41 transformed primary frame occurrences, 15
primary declarations with invalid image/duration cardinality, and 43 unresolved image
occurrences. These categories overlap; their sum is not an animation total. The conservative gate
keeps 1,000 of the 1,604 primary-bearing animations out of the first lossless projection rather
than silently approximating them.

## Rights and attribution evidence

Rights are collection-scoped, not promoted to individual sprite files:

| Evidence | Scope | Finding |
| --- | --- | --- |
| `README.md` | Repository and art collection | Source is GPL-2.0; most art/music is GPL-2.0-or-later, while newer contributions are CC-BY-SA-4.0. |
| `COPYING` | Repository | Full GNU GPL version 2 text. |
| `data/COPYING.txt` | Data tree | Full GNU GPL version 2 text under `data/`. |
| `copyrights.csv` | Listed exception files only | 396 file-specific license/author rows. None names an image file. |

The audit therefore stores `GPL-2.0-or-later` as repository-license evidence and the mixed
GPL-2.0-or-later / CC-BY-SA-4.0 art statement at collection scope. Every sprite's `per_asset_license`
and `per_asset_attribution` remain null unless a later evidence pass finds a file-specific source.
Absence from `copyrights.csv` is not interpreted as an individual GPL or artist assertion.

Documentation should cite the repository, exact commit URL, archive SHA-256, and the README/data
rights evidence. If a frame is later published or redistributed, preserve its exact member path
and payload hash and review its file history or contributor evidence before asserting a named
artist or one branch of the mixed art-license statement.

## Recommended database projection

The safest first projection is one sequence per exact `(config path, unit declaration line,
variation path, animation line)` record that passes `safe_primary_source_sequence`:

- entity identity: config path plus declaration line, retaining unit ID as source metadata;
- action: normalized tag/apply-to label plus source tag and attack filters;
- order and timing: expanded frame order and exact millisecond duration;
- loop: `loop` only for the engine-forced standing tracks or a literal cycles attribute;
- facing: source direction filters plus effective flip behavior, never synthetic directions;
- frame identity: selected archive member path, SHA-256, dimensions, and original expression;
- geometry: full source image (Wesnoth animations are image sequences, not inferred sheet cells);
- rendering context: raw/context WML attributes, offsets, layers, and modifiers; and
- rights: repository evidence references only, with null per-asset license/attribution.

Auxiliary, projectile, and effect records should remain linked metadata until a separate
composition-aware renderer exists. Macro-affected, conditional, transformed, inherited, missing,
and timing-ambiguous records should remain quarantined with their exact evidence. A future fuller
adapter can run Wesnoth's pinned WML preprocessor in a hermetic campaign context and render IPFs,
but that would be a new derived corpus with its own toolchain hashes and must not overwrite this
literal source audit.

## Verification

`tests/test_wesnoth_adapter.py` covers progressive image order, parallel expansion, zero padding,
repeat syntax, inline and outer timing, image modifiers, synthetic WML unit/variation parsing,
actions, loops, directions, layers, runtime flip defaults, macro/conditional quarantine, immutable
records, rights scope, unsafe ZIP rejection, deterministic serialization, CAS sharding, and the
exact pinned archive counts/hash when the CAS object is available.

## Implemented deterministic database projection

`src/spritelab/ingest/wesnoth.py` implements the recommended projection as three deliberately
separate operations:

1. `plan_wesnoth_projection` creates an immutable, serializable plan without opening SQLite.
2. `check_wesnoth_projection_readiness` opens an existing index with SQLite `mode=ro` and
   `PRAGMA query_only=ON`; it does not initialize or migrate the database.
3. `project_wesnoth_audit` writes only when a caller explicitly supplies an `IndexDB`, after a
   complete member/hash preflight. Repeating it reuses source-keyed sequences and upserts their
   facts, subjects, motion annotations, occurrences, and frames.

For the pinned archive, projection version `wesnoth_literal_primary_wml_projection_v1` has
manifest SHA-256
`1712326c432e1f143857e8d41ef03889dd91d0ad0566a6e43c040b9aaf8d1da8`. It projects exactly:

| Projection fact | Exact count |
| --- | ---: |
| Source entities | 248 |
| Safe direct sequences | 604 |
| Ordered frame occurrences | 2,526 |
| Unique primary image members/blobs | 1,275 |
| Looping sequences | 224 |
| One-shot sequences | 380 |
| Adapter-labeled actions | 598 |
| Unknown generic actions | 6 |
| Exact singleton-facing hints | 6 |
| Retained auxiliary frame declarations | 370 |
| Provenance occurrence links | 6,066 |

The occurrence links comprise the defining config, each distinct primary image used by that
sequence, both pinned engine-semantics files, and all four collection-rights evidence files. The
readiness set contains 1,526 members: 245 defining configs, 1,275 primary images, two engine
files, and four rights files. The current local index passed the exact query-only dry run with all
1,526 members present, all 1,275 source images extracted and registered under the audited payload
hash, one matching `wesnoth` source item, and no missing or mismatched prerequisite.

Each projected frame points directly to its standalone PNG blob with source frame index zero and
full-image geometry. Sequence canvas dimensions are the maximum source width and height; the
metadata explicitly reports whether all occurrence dimensions agree, because ten pinned safe
animations change source dimensions during the track. Exact duration, temporal order, loop mode,
source direction groups, canonical direction when a single literal facing exists, effective
horizontal/vertical flips, offsets, layer, coordinates, raw attributes, context attributes,
source expression, member path, payload hash, and dimensions are retained. No clipping,
repacking, interpolation, mirroring, or image path function is applied.

The adapter's conservative actions remain losslessly available as `adapter_normalized_action`.
The shared database taxonomy recognizes `idle`, `attack`, `death`, `defend`, `emote`, and `spawn`
directly. Its narrower current vocabulary maps Wesnoth's generic `move`, `heal`, and `teleport`
labels to database `unknown` rather than asserting a more specific walk/cast/transform action;
the source label and basis remain conditioning metadata for a later vocabulary revision.

All 1,073 excluded animation records and their 6,114 primary frame occurrences remain in the
plan's explicit quarantine ledger, including 41 transformed primary occurrences, 1,004
macro-affected records, and 140 conditional tracks. The projection never repairs those records.
Safe primary tracks can still reference 370 auxiliary projectile/effect declarations; these are
serialized verbatim as uncomposited evidence and `runtime_composite_complete` is false for those
sequences. They are not presented as final rendered composites.

Rights metadata is repeated only as collection/repository evidence with the scope caveat. Asset
license and creator fields remain null, and the projection creates no per-asset rights
observation. `tests/test_wesnoth_ingest.py` verifies the quarantine boundary, deterministic plan
hash, query-only readiness, registered-blob/hash preflight, idempotent projection, WML context,
timing, phases, loops, facing, geometry, auxiliary non-composition, and null per-asset rights on a
synthetic fixture, plus the exact pinned counts and live readiness when the local CAS/index are
available. No pinned sequences were written to the live database during this implementation.
