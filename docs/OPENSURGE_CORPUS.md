# Open Surge pinned-corpus audit

This audit covers the exact GitHub source archive for
[`alemart/opensurge`](https://github.com/alemart/opensurge) at commit
[`bcb3466e10913f2d5f34dec848e0c2f3ee944883`](https://github.com/alemart/opensurge/tree/bcb3466e10913f2d5f34dec848e0c2f3ee944883).

- Archive SHA-256: `1448d51c3e8bc8122ae893aad96e73f9c9ef1ffb74457109fa8cfa0ef10d0206`
- CAS path: `data/raw/objects/sha256/14/48/1448d51c3e8bc8122ae893aad96e73f9c9ef1ffb74457109fa8cfa0ef10d0206`
- Archive root: `opensurge-bcb3466e10913f2d5f34dec848e0c2f3ee944883`
- Adapter: `src/spritelab/adapters/opensurge.py`
- Exact regression audit: `tests/test_opensurge_adapter.py`

The known-archive entry point verifies both the full archive digest and the
40-character commit in the ZIP root. It does not accept a floating branch archive
or a similarly named payload.

## Verified inventory

| Measure | Exact count |
|---|---:|
| ZIP members | 1,197 |
| File members | 1,066 |
| PNG files anywhere in the archive | 164 |
| `.spr` files | 116 |
| Sprite definitions | 357 |
| Regular numeric animations | 893 |
| Transitions | 24 |
| Transition endpoints missing a referenced animation | 0 |
| Total timelines | 917 |
| Ordered `data` frame occurrences | 3,540 |
| Repeated occurrences beyond unique frame IDs per timeline | 1,493 |
| `repeat TRUE` declarations | 679 |
| `repeat FALSE` declarations | 238 |
| Explicit `repeat_from` declarations | 9 |
| Comment-labeled timelines | 708 |
| Conservatively normalized action timelines | 236 |
| Unresolved/unlabeled action timelines | 681 |
| `source_file` references | 357 |
| Unique referenced source sheets | 96 |
| Missing referenced sheets | 0 |

The 3,540 count is a timeline-occurrence count. It deliberately includes source
repetitions such as `0 0 0 1`, because those repeated cells encode holds and
cadence at the declared FPS. It is not a count of unique sheet cells.

## What the adapter preserves

For every sprite definition, the adapter records:

- the exact source identity and `.spr` evidence path/line;
- `source_file`, raw `source_rect`, `frame_size`, default `hot_spot`, and default
  `action_spot`;
- source sheet dimensions, format, mode, transparency, hashes, CRC, and credit row;
- every numeric animation ID;
- every ordered `data` index occurrence, including repeats;
- each occurrence's row-major source cell and whether that cell is inside the PNG;
- source FPS and its original token;
- source `repeat`, raw `repeat_from`, and engine-effective loop tail;
- per-animation `hot_spot` and `action_spot` overrides;
- optional programmatic `play` names;
- transition source and target IDs, transition declaration order, and frames;
- raw preceding/inline comments, the selected source label, and any conservative
  action projection;
- source-file header author/license comments and sheet-level asset credit.

The geometry follows the pinned engine implementation in `src/core/sprite.c`:
frames advance left-to-right within `source_rect`, then row-to-row. Timing and loop
semantics follow `src/core/animation.c`: each occurrence lasts `1 / fps` seconds;
`repeat_from` splits a one-time prefix from a repeating suffix; transitions are
forced non-repeating by the engine even if a source declaration says otherwise.
Both raw declarations and engine-effective values are retained.

Pixel transparency has an additional engine semantic that cannot be inferred from
the PNG alpha channel alone. At pinned `src/core/color.c:190`,
`color_is_transparent` returns true for alpha zero **or exact uint8 RGB
`(255, 0, 255)`**. The default fragment shader defines that exact magenta mask at
`src/core/shader.c:111` and, at line 116, multiplies all sampled components by zero
when RGB matches it. Therefore projection v2 records an audited pixel transform:
`exact_uint8_rgb_to_rgba_zero`, RGB `[255, 0, 255]`, output RGBA `[0, 0, 0, 0]`.
There is no distance, tolerance, or fuzzy near-magenta rule.

## Action semantics and steerability

Comments are evidence, not an invitation to infer. The adapter maps only an
explicit vocabulary such as `idle`, `walking`, `running`, `jumping`, `getting hit`,
`dead`, and the structured labels `animal N: running`. Labels such as `warming up`,
`action!`, `charging`, `default`, character names, directional state descriptions,
and unlabeled numeric IDs remain unresolved. Transition comments are preserved but
are not projected as standalone actions because their exact source/target IDs are
already recorded.

| Normalized action | Timelines | Frame occurrences |
|---|---:|---:|
| celebrate | 4 | 35 |
| crouch | 5 | 14 |
| death | 9 | 17 |
| despawn | 11 | 17 |
| emote | 3 | 28 |
| fall | 4 | 18 |
| fly | 6 | 18 |
| hover | 2 | 8 |
| hurt | 8 | 32 |
| idle | 96 | 364 |
| jump | 11 | 37 |
| land | 1 | 3 |
| push | 4 | 16 |
| run | 34 | 80 |
| shoot | 1 | 5 |
| spawn | 30 | 94 |
| walk | 7 | 56 |

This is a particularly useful source for action-conditioned animation because the
timelines have exact cadence and loop behavior rather than filename-only action
hints. The unresolved records remain available for later source-code joins or
manual review without contaminating the current conditioning labels.

## Character and creature coverage

The conservative character audit identifies 22 standalone character/collection
sprite definitions:

- 3 primary bosses and 1 separate defeated-boss variant;
- 11 enemy character or enemy-variant definitions;
- 1 friend character;
- 4 player characters;
- 2 multi-subject animal collections (modern and legacy).

Primary broad class counts are 17 animal and 5 creature definitions. Animal is the
primary class for anthropomorphic characters; `humanoid` is retained as a secondary
candidate where appropriate rather than double-counting the same definition. The
two animal collections are source sprite definitions, not claims that each sheet is
one identity.

High-value animal/creature subjects include:

| Source subject | Primary/candidate classes | Evidence-preserving tags |
|---|---|---|
| Giant Wolf | animal | boss, quadruped, wolf |
| Hydra | creature | boss, multiheaded, serpentine |
| Salamander Boss | animal / creature | boss, salamander |
| Crococopter | creature / robot | enemy, flying, hybrid |
| Fish | animal | enemy, aquatic, fish |
| SwoopHarrier | animal / creature | enemy, flying, bird-like |
| Jumping Fish | animal | enemy, aquatic, multi-variant sheet |
| Lady Bugsy | creature / animal | enemy, insect-like |
| GreenMarmot, RedMarmot | animal | enemy, quadruped, marmot |
| Mosquito | animal | enemy, flying, insect |
| RulerSalamander | animal / creature | enemy, salamander |
| Springfling | creature | enemy |
| Wolfey | animal | enemy, quadruped, wolf |
| Skaterbug | creature / animal | friend, insect-like |
| Surge | animal / humanoid | anthropomorphic, biped, rabbit |
| Neon | animal / humanoid | anthropomorphic, biped, squirrel |
| Charge | animal / humanoid | anthropomorphic, biped, badger |
| Tux | animal / humanoid | biped, penguin |

Four definitions have explicit quadruped evidence: Giant Wolf, GreenMarmot,
RedMarmot, and Wolfey. Salamander identities are not automatically tagged as
quadrupeds because the source name alone does not establish sprite morphology.
Boss limbs, masks, projectiles, orbs, and similar pieces are marked as components
or effects rather than being counted as independent complete entities.

The pinned README identifies Surge as a rabbit. Neon and Charge species are
corroborated by an [official Open Surge project forum character-design
discussion](https://forum.opensurge2d.org/viewtopic.php?id=286), and the official
[level specification](https://wiki.opensurge2d.org/Level_specification) identifies
Tux as a penguin. Those external statements support classification only; all pixel,
timing, geometry, commit, and rights counts above come from the pinned archive.

## Rights and attribution

The archive contains three distinct rights scopes that must not be collapsed:

1. The root `LICENSE` is the GPLv3 text for the repository/project.
2. Eighty-two modern `.spr` files have source-local `License: MIT` comments, plus
   source-local `Author:` and often `art by` comments.
3. `src/misc/copyright_data.csv` supplies the per-file license, author, website, and
   notes for image assets. This is the sheet-level authority used by the adapter.

The manifest has 285 total rows, including 135 image rows. Every one of the 96
unique source sheets referenced by `.spr` files has an exact image row; there are no
uncredited or path-fuzzily-matched sheets.

| Sheet license from `copyright_data.csv` | Unique referenced sheets | Sprite definitions referencing them |
|---|---:|---:|
| CC-BY-3.0 | 54 | 157 |
| CC-BY-4.0 | 25 | 131 |
| CC0-1.0 | 13 | 63 |
| CC-BY-SA-3.0 | 1 | 3 |
| CC-BY-SA-4.0 | 1 | 1 |
| Giftware | 1 | 1 |
| MIT | 1 | 1 |

Important evidence hashes:

| Evidence member | SHA-256 | Scope |
|---|---|---|
| `src/misc/copyright_data.csv` | `e606587ee6f597532bd34cab5f3d8df455f4a99f0f4bf849febb6a9b016556c8` | Per-asset rights and attribution |
| `LICENSE` | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` | Repository GPLv3 text |
| `README.md` | `46603d7981c9317ac8cf21c60c30c753ea97b14782068773aae8c6aa25d935ca` | Project description/license statement |
| `licenses/MIT-license.txt` | `a406579cd136771c705c521db86ca7d60a6f3de7c9b5460e6193a2df27861bde` | Bundled generic MIT template |
| `src/core/sprite.c` | `a6ec7f5d36d9273565c190cb63517c8e4f060b2ac8fd570cdb723fc8d997671d` | Parser and frame-geometry behavior |
| `src/core/animation.c` | `4d7345b05fee850ea165d794d271e049ac5b388dc7ae660f2f1dbb149d46c27c` | Timing and loop behavior |
| `src/core/color.c` | `036b39ec0ba2fa42ef3b6dd5e16bb789e77e9684c008d3ea4147ebd51dbd19a4` | Alpha-zero or exact-magenta transparency predicate (line 190) |
| `src/core/shader.c` | `98c2cda978c67bc85f97b33b8692ab143d0fbb4aa113a0d3558cf5cba9d37dfa` | Exact magenta mask and premultiplied RGBA zeroing (lines 111 and 116) |

The bundled MIT document contains placeholder copyright fields. It is preserved as
evidence but is not used to replace the named authors or licenses in the asset CSV.

## Geometry findings and extraction gates

All 3,540 declared frame occurrences are within their floor-divided raw sprite
grids. The adapter found no invalid `data` index. Raw sheet geometry still contains
known source inconsistencies:

- 8 sprite definitions declare a `source_rect` extending beyond the current PNG:
  `Bridge Element`, `Bridge Corner Left`, `Bridge Corner Right`, `Power Pluggy
  Clockwise`, `Power Pluggy Counterclockwise`, `SD_VERTICALDANGER`, `Charge`, and
  `Tux`.
- The modern `Animal` definition declares a 180×275 rectangle with 30×30 frames,
  so its height is not a frame-size multiple. Its referenced indices remain within
  the valid floor-divided rows.
- Three referenced cells themselves extend past the current PNG: animation 0 frame
  0 for each Power Pluggy direction, and animation 0 frame 0 for
  `SD_VERTICALDANGER`.
- The other five oversized rectangles reference only cells that are currently
  within the image.

Open Surge contains runtime rectangle-adjustment code, but normalization must not
silently rewrite raw declarations or pretend out-of-image pixels exist. The three
bad referenced cells should be excluded from extraction until reconciled; the raw
records remain indexed for reproducibility.

## Dataset-use recommendation

For a first temporal dataset snapshot:

1. Keep each complete sprite identity, its sheet hash, script path, and asset-credit
   row in one leakage group.
2. Emit source occurrences in exact `data` order; never deduplicate repeated IDs.
3. Use source FPS directly and retain the one-time prefix before `repeat_from`.
4. Keep transitions as transition records with source/target IDs, not generic action
   examples.
5. Admit only explicitly normalized action labels to action-conditioned training;
   retain all other raw labels for future resolution.
6. Exclude component/effect records from complete-entity training unless the model
   has an explicit component role.
7. Gate the three out-of-image frame occurrences before pixel slicing.
8. Export the exact `copyright_data.csv` row with every derived sequence.

This preserves what makes the corpus unusually strong—exact motion timing, repeated
holds, loop tails, and transitions—without upgrading comments, filenames, or engine
repair behavior into facts they do not establish.

## Provenance-database projection

`src/spritelab/ingest/opensurge.py` provides a pure plan, a query-only readiness
check, and an idempotent database projection for the pinned audit. The projection
does not extract, clip, or rewrite pixels. Projection version
`opensurge_spr_projection_v2` attaches the evidence-backed exact-magenta operation
to sequence and frame metadata so a later strict materializer can execute it. A
`sequence_frame` points to the exact source-sheet CAS blob and retains the absolute
cell rectangle, source frame index, ordered `data` occurrence, FPS-derived duration,
anchors, and loop-tail position.

Admission is deliberately occurrence-safe:

- the source PNG must exist and have an exact `copyright_data.csv` image row;
- the declared source rectangle must form an integral frame grid;
- every emitted occurrence must be inside both the declared grid and source PNG;
- FPS, `data`, repeat, and loop-tail declarations must be internally consistent;
- no inferred crop repair or clipping is allowed.

An oversized declared source rectangle is retained as an explicit caveat when all
cells actually referenced by a timeline are within the image. This admits 45 safe
timelines belonging to Bridge Element/Corner, Charge, and Tux while preserving
`source_rect_within_image=false`. It excludes the 36 modern `Animal` timelines
because the 180-by-275 source rectangle is not an integral 30-by-30 grid, and
excludes the three one-frame Power Pluggy/`SD_VERTICALDANGER` timelines whose
referenced cells exceed the PNG. The exclusions are records in the deterministic
plan, not silent drops.

Exact pinned plan counts:

| Projection fact | Count |
|---|---:|
| Projected sprite entities | 353 |
| Projected timelines | 878 |
| Regular animations | 854 |
| Transitions | 24 |
| Ordered frame occurrences | 3,484 |
| Comment-supported normalized actions | 199 |
| Unknown/unresolved actions | 679 |
| Plain loops | 631 |
| Intro-then-loop timelines | 9 |
| One-shot timelines | 238 |
| Timelines with explicit comment-derived direction | 26 |
| Safe timelines retaining oversized-rectangle caveat | 45 |
| Excluded candidate timelines | 39 |
| Excluded candidate occurrences | 56 |
| Actually out-of-image excluded occurrences | 3 |

Every normalized action comes from the adapter's explicit comment mapping. Numeric
animation IDs, filenames, and unresolved comments never become action labels; they
are stored as `unknown`. Direction is emitted only for the 26 timelines with an
explicit source-comment direction hint and is otherwise `unknown`. View is
`unknown` for every timeline because the `.spr` declarations do not encode view.
Transitions remain transitions with exact source/target IDs and an unknown action.

For an intro-then-loop sequence, frame `phase` is null in the one-time prefix and
is normalized over only the repeating tail. Per-frame metadata separately records
`in_intro_prefix`, `in_loop_tail`, loop-tail ordinal, loop-tail length, and effective
`repeat_from`. One-shot phases span 0 through 1; ordinary loop phases exclude the
duplicate endpoint. All occurrences, including intentional repeated frame IDs, are
materialized in source order with duration `1000 / fps` milliseconds.

The stable projection manifest SHA-256 is
`7ec1fabd908bb195cae4ae2a50374c08336f3a9d861a24fd209f2773e8d53f43`.
It hashes the canonical plan records and exclusions together with projection
version, archive hash, repository commit, and the audited pixel-transform record.
The transform's own canonical SHA-256 is
`d0860d86c815d0a4b6f7c116f7a2f31faedaaac4b7d9877f028e5545364fa306`.
Sequence and entity source keys also include the immutable archive/commit identity
and exact source declaration identity, so reruns find and update the same rows.

Per-image license, author, website, notes, manifest path, and manifest line are
copied to entity, sequence, frame, subject-link, and relevant occurrence metadata.
Each sequence links to five archive occurrences: its source PNG, its `.spr`
definition, `src/misc/copyright_data.csv`, `src/core/color.c`, and
`src/core/shader.c`. The two engine-code edges include the evidence hash, relevant
line numbers, scope, claim, transform schema/operation/RGB, and transform hash.
Projection deliberately adds zero
`rights_observations` rows because that table is append-only; adding an observation
on every rerun would not be idempotent. The indexed manifest row remains the exact
asset-level rights authority.

A query-only dry run against `data/index/spritelab.sqlite3` found the pinned archive
inventory, one matching `open_surge` source item, and all 214 required evidence
members. Every referenced PNG had a registered extracted blob with the exact hash
audited from the ZIP; there were zero missing members, missing sheet blobs, or hash
mismatches. No live database rows were changed by this readiness check.
