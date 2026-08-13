# Shattered Pixel Dungeon corpus audit

This document records the exact Shattered Pixel Dungeon snapshot audited for the
sprite-animation research pool. Counts below are produced by the pure adapter in
`src/spritelab/adapters/shattered_pixel_dungeon.py`; they are not estimates from filenames.

## Pinned source

- Repository: [00-Evan/shattered-pixel-dungeon](https://github.com/00-Evan/shattered-pixel-dungeon)
- Commit: [`7b8b845a76fe76c6b7c031ae9e570852411f56db`](https://github.com/00-Evan/shattered-pixel-dungeon/tree/7b8b845a76fe76c6b7c031ae9e570852411f56db)
- Archive SHA-256: `deed31527c24b2a500f6c79e1cf2502c4c9d8b4391529033da7015cd73c97544`
- CAS object: `data/raw/objects/sha256/de/ed/deed31527c24b2a500f6c79e1cf2502c4c9d8b4391529033da7015cd73c97544`
- Archive root: `shattered-pixel-dungeon-7b8b845a76fe76c6b7c031ae9e570852411f56db`

The known-snapshot entry point verifies both the archive digest and the commit encoded in the
archive root before parsing anything.

## Verified inventory

| Measure | Count |
|---|---:|
| ZIP members | 2,111 |
| Java files | 1,289 |
| Java class declarations parsed | 1,792 |
| PNG files anywhere in the archive | 202 |
| `Assets.Sprites` mappings | 75 |
| PNGs in `core/src/main/assets/sprites/` | 75 |
| Mapped sprite PNGs present | 75 |
| Missing mapped sprite PNGs | 0 |
| Unmapped PNGs in the sprite directory | 0 |
| Sprite-definition classes with materializable animation evidence | 115 |
| Concrete sprite-definition classes | 109 |
| Abstract sprite-definition classes | 6 |
| Relevant Java `.frames(...)` call sites | 348 |
| Relevant Java animation-clone assignment sites | 46 |
| Concrete class/action slots | 506 |
| Materialized runtime sequence variants | 659 |
| Ordered frame occurrences across those variants | 2,577 |
| Unresolved animation frame orders | 0 |
| Frame orders outside their derived sheet grids | 0 |

Seventy of the 75 mapped sprite sheets are reached by a materialized entity animation class.
The five mapped sheets without such a class are `amulet.png`, `avatars.png`, `item_icons.png`,
`items.png`, and the currently unreferenced `pet.png`. They remain indexed source assets but are
not misrepresented as animation sequences.

“Materialized runtime sequence variant” is deliberately not called a unique animation. It is a
class/action/runtime-layout alternative after inheritance and branch expansion. Several classes
share the same sheet cells, and Java `clone()` assignments intentionally reuse animation data.
Leakage grouping and duplicate linkage must preserve that relationship downstream.

## What the Java definitions provide

The source is stronger supervision than a guessed sprite-sheet slicer:

- `Assets.Sprites` resolves each symbolic texture key to an exact repository-relative PNG.
- `TextureFilm(texture, width, height)` defines a row-major cell grid. The adapter follows the
  audited engine implementation: columns and rows use integer division, so power-of-two padding
  is not interpreted as additional partial frames.
- `new Animation(fps, looped)` supplies source FPS and the exact loop flag.
- `.frames(film, ...)` supplies the exact ordered frame-index list. Repeated indices are retained;
  they are deliberate frame holds, not duplicates to remove.
- `.frames(texture.uvRect(...))` is retained as an exact pixel rectangle for the few non-grid
  animations, including ward tiers and the red sentry.
- `.clone()` is resolved using the engine’s clone semantics, which copy FPS, loop state, and
  ordered frames.

The adapter records both derived values and raw expressions. For example, DM-300 retains
`c = enraged ? 10 : 0` and `enraged ? 15 : 10`, rather than pretending the two visual states have
one frame strip or one FPS. All animation loop flags are statically resolved. Of 608 materialized
animation records, 605 have one exact FPS and three have two source-declared runtime FPS
candidates. Ten records use source FPS `0`; those are static pose states and should not be treated
as moving temporal loops.

The source FPS values observed are `0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24`.

### Inheritance and runtime variants

The audit retains rather than flattens the important ambiguities:

- Six abstract sprite families materialize through concrete nested subclasses: crystal guardian,
  crystal spire, crystal wisp, elemental, Yog fist, and shaman.
- `texOffset()` is evaluated at the concrete subclass, while each animation records the base class
  and source line where its frame order was defined.
- DM-300 has normal/enraged offsets and FPS candidates.
- Gnoll geomancer has normal/statue alternatives.
- Rat King retains four source offsets: ordinary, April Fools, winter holiday, and Ratmogrify.
- Statue and guardian-trap definitions retain six armor-tier offsets.
- Crystal-spire frame changes remain separate source call-site variants.
- Hero-derived sprites retain six possible class sheets rather than being assigned one arbitrarily.

Hero sheets require special handling. The source constructs a `TextureFilm` for a runtime armor-tier
patch, then a 12×15 grid inside that patch. All six hero PNGs are 256×128, giving 21 complete cells
per tier row and eight physically possible 15-pixel rows in the source image. The adapter records
both the local patch capacity (21) and whole-sheet grid capacity (168), but does not invent the set
of runtime-valid armor-tier values.

## Steerable action coverage

The most useful normalized conditioning groups are:

| Normalized action | Concrete classes | Source actions included | Sequence variants | Frame occurrences |
|---|---:|---|---:|---:|
| `idle` | 109 | `idle`, `activeIdle`, six `tierIdles[n]` slots | 145 | 843 |
| `run` | 108 | `run` | 153 | 577 |
| `attack` | 105 | `attack` | 150 | 436 |
| `attack_ranged` | 41 | `zap` | 69 | 150 |
| `attack_melee` | 5 | `kick`, `pumpAttack`, `slam`, `stab` | 6 | 22 |
| `death` | 108 | `die` | 112 | 465 |
| `cast` | 1 | `cast` | 1 | 3 |
| `interact` | 1 | `operate`, `read` | 2 | 14 |
| `transform` | 4 | `advancedHiding`, `hiding` | 8 | 28 |
| `fly` | 1 | `fly` | 1 | 1 |
| `jump` | 1 | `leap` | 1 | 1 |

Source-specific states `charge`, `charging`, `crumple`, `prep`, and `pump` remain unmapped rather
than being forced into misleading canonical actions. They account for 11 resolved variants and 37
frame occurrences.

The sheets do not contain explicit directional tracks. They provide one art orientation; the base
`CharSprite` implementation can flip horizontally at runtime according to target position, while
some concrete classes override turning behavior. Dataset exports should therefore keep raw
direction as unknown/single-orientation. A derived left/right flip can be added as an explicitly
recorded transform, not presented as a second authored direction.

## Entity coverage

Classification is a transparent identifier-based conditioning hint, not biological ground truth.
It uses the concrete class name and resolved source-sheet filename and records the matched token as
its basis.

| Entity class | Concrete definitions |
|---|---:|
| Humanoid | 28 |
| Monster | 26 |
| Animal | 20 |
| Creature | 19 |
| Object | 10 |
| Robot | 6 |
| Unknown | 0 |

Animal coverage spans 20 concrete class definitions over 13 sheets:

- Quadrupeds: ordinary, albino, and fetid rats; Rat King; sheep; and the rat-based surface-scene
  pet definition. This is six definitions over `rat.png`, `ratking.png`, and `sheep.png`.
- Winged animals: bat, bee, swarm, and spirit hawk.
- Aquatic/serpentine animals: piranha, phantom piranha, and snake.
- Multi-legged animals: crab, hermit crab, great crab, spinner, and scorpio/acidic scorpio.
- Other invertebrate: larva.

This adds real action-conditioned animal data, but it is not a broad quadruped corpus by itself:
the quadruped morphology is mostly rats plus one sheep sheet. Additional canine, feline, ungulate,
and large-animal sources remain necessary.

## Rights and attribution evidence

Evidence is pinned to the exact archive:

| Evidence | SHA-256 | Interpretation |
|---|---|---|
| [`LICENSE.txt`](https://github.com/00-Evan/shattered-pixel-dungeon/blob/7b8b845a76fe76c6b7c031ae9e570852411f56db/LICENSE.txt) | `d0495053051967ebe76fb1facd287d79d1ed800da1be75cf501a556bc39a0472` | Repository-root GNU GPL version 3 license text |
| [`Assets.java`](https://github.com/00-Evan/shattered-pixel-dungeon/blob/7b8b845a76fe76c6b7c031ae9e570852411f56db/core/src/main/java/com/shatteredpixel/shatteredpixeldungeon/Assets.java) | `4661b1a781c8afceb18a0c02f10b6565050bd931d4b4dfaf57c14e5eff9c6ac8` | Representative Java header grants GPL version 3 or later for the program |
| [`README.md`](https://github.com/00-Evan/shattered-pixel-dungeon/blob/7b8b845a76fe76c6b7c031ae9e570852411f56db/README.md) | `1566e56f5879a254f8cedada242c95dd181b66174ac7150495fa3284d44d625f` | Names Pixel Dungeon and Watabou as the upstream source |

All 1,289 Java files carry the audited copyright-header pattern naming:

- Oleg Dolya, 2012–2015
- Evan Debenham, 2014–2026

Important scope caveat: the archive contains no sprite-specific artist/license manifest, and none
of the 75 sprite PNGs exposes embedded author, copyright, license, source, or comment metadata.
The adapter therefore records repository-level GPL evidence and upstream attribution without
claiming a PNG-by-PNG artist identity that the snapshot does not provide. This caveat must survive
any dataset or report export.

## Adapter API and validation

- `audit_known_shattered_pixel_dungeon_archive(path)` enforces the exact SHA-256 and commit.
- `audit_shattered_pixel_dungeon_archive(path)` performs the same pure read-only audit without
  requiring the known digest.
- `parse_assets_sprite_mappings(text)` is separately testable and rejects duplicate keys or paths.
- The returned frozen dataclasses retain raw evidence member paths and line numbers and expose
  `to_dict()` for deterministic report serialization.

Focused validation:

```powershell
.venv\Scripts\ruff.exe check src\spritelab\adapters\shattered_pixel_dungeon.py tests\test_shattered_pixel_dungeon_adapter.py
.venv\Scripts\pytest.exe -q tests\test_shattered_pixel_dungeon_adapter.py
```

The exact-CAS regression covers long repeated holds (Snake and sheep), runtime variants (DM-300 and
Rat King), abstract inheritance (`FistSprite.Burning`), non-looping locomotion (Tengu), dynamic hero
sheet/tier geometry, explicit UV rectangles, clone semantics, entity morphology, source hashes,
license evidence, and upstream attribution.

## Conservative provenance-DB projection

The ingest projection is intentionally narrower than the pure Java audit. It materializes only
records with one unambiguous source sheet and absolute source-sheet coordinates. This prevents a
runtime armor-tier patch coordinate from being mistaken for an absolute PNG crop.

For the exact pinned archive, the write-free plan contains:

| Projection measure | Count |
|---|---:|
| Entities with safely materializable sequences | 103 |
| Source sheets used | 64 |
| Sequences | 631 |
| Ordered frame occurrences, including deliberate repeats | 2,439 |
| Sequences with one exact positive FPS | 615 |
| Source `FPS=0` pose-only sequences | 10 |
| Sequences retaining multiple positive FPS candidates | 6 |
| Planned archive-member occurrence links | 4,447 |
| Required indexed archive evidence members | 151 |
| Excluded runtime-sheet/tier sequence candidates | 28 |
| Frame occurrences in those exclusions | 138 |

The 28 exclusions are the eight `HeroSprite` action slots plus four slots each from
`Feint.AfterImage.AfterImageSprite`, `MirrorSprite`, `PowerOfMany.LightAllySprite`,
`PrismaticSprite`, and `ShadowClone.ShadowSprite`. Each can select one of six hero-class sheets,
and its cell coordinates are relative to a runtime-selected armor-tier patch. The projection does
not form a sheet/tier/frame Cartesian product and does not guess a tier offset. These records remain
fully represented in the adapter audit and can be added later if runtime tier evidence is parsed.

The projected entity distribution is 26 monster, 22 humanoid, 20 animal, 19 creature, 10 object,
and six robot definitions. All six audited quadruped definitions remain included.

### Timing and loop semantics

- A single positive source FPS becomes the exact per-occurrence duration `1000 / FPS` milliseconds.
  Repeated source indices remain repeated `sequence_frames` rows.
- Multiple positive FPS alternatives stay together as one candidate tuple on each exact frame-order
  variant. Their duration is null and `timing_known=false`; FPS and frame alternatives are never
  multiplied into invented combinations.
- Source `FPS=0` becomes an untimed pose (`duration_ms=NULL`, `pose_only=true`). Its raw source loop
  flag remains in metadata, while semantic `loopable` and `cycle_frames` remain unknown rather than
  creating a zero-duration loop.
- Positive-FPS source loop flags map to `loop` or `one_shot`. Direction remains `unknown`, view is
  `top_down`, and the Java horizontal-flip behavior remains evidence rather than an authored second
  direction.

Every frame points to the lossless source-sheet CAS blob and retains its exact cell rectangle,
source frame index, ordinal, coordinate space, FPS candidates, and source loop flag. Each sequence
links back to its sheet, Java animation definition, `Assets.java` mapping, concrete entity definition
when inherited, and repository license/attribution evidence. The repository-level rights caveat is
copied to entity, sequence, motion, frame, and occurrence metadata. The projection deliberately adds
zero `rights_observations`: it does not upgrade repository evidence into a per-PNG license claim.

### Idempotency and API

Stable source keys contain the pinned archive digest and commit, concrete/defining class, exact Java
member and line, raw parser context and expressions, source sheet/key, full frame order/rectangles,
and variant ordinal. A regression assertion confirms 631 distinct keys for 631 planned sequences.
Reruns reuse those keys and upsert the same entity, subject, motion, frame, and occurrence rows.

- `plan_known_shattered_pixel_dungeon_projection(path)` audits the pinned archive and returns the
  pure plan without opening the DB.
- `plan_shattered_pixel_dungeon_projection(audit)` plans any already-produced audit for testing.
- `check_shattered_pixel_dungeon_projection_readiness(database_path, plan)` opens SQLite with
  `mode=ro` and `query_only`, verifies the exact archive/item, every required evidence member, and
  every source-sheet CAS hash.
- `project_shattered_pixel_dungeon_audit(database, plan, taxonomy)` performs the idempotent core-row
  projection after a complete preflight.
- `ingest_known_shattered_pixel_dungeon_sequences(database, archive_path, taxonomy)` is the guarded
  exact-archive convenience entry point.

On 2026-08-12, a read-only dry audit of `data/index/spritelab.sqlite3` found the exact archive and one
linked source item, all 151 required evidence members, all required source-sheet blobs, and zero CAS
hash mismatches. No live DB projection was run.

Focused validation:

```powershell
.venv\Scripts\ruff.exe check src\spritelab\ingest\shattered_pixel_dungeon.py tests\test_shattered_pixel_dungeon_ingest.py
.venv\Scripts\pytest.exe -q tests\test_shattered_pixel_dungeon_ingest.py
```
