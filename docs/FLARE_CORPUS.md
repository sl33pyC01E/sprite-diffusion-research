# Flare: Empyrean Campaign exact-snapshot corpus audit

This adapter audits one immutable `flare-game` repository archive directly from the
content-addressed store. It does not extract files, generate crops, register the adapter in the
shared CLI, or modify the live database.

## Snapshot identity and scope

| Field | Exact value |
| --- | --- |
| Game repository | `https://github.com/flareteam/flare-game` |
| Game commit | `af6eee6d339ac98011864bfe89da837fe7769c28` |
| Archive SHA-256 | `9c8e58bb704f55174928990a1a375c213ffb2847d7b928d6117a84db3e1215cc` |
| Archive size | 626,244,475 bytes |
| ZIP root | `flare-game-af6eee6d339ac98011864bfe89da837fe7769c28` |
| ZIP inventory | 2,727 members: 2,499 regular files, 227 directories, 1 symlink |
| Expanded member bytes | 661,089,998, including the six-byte symlink payload |
| PNG inventory | 721, all readable |
| Runtime game mods audited | `fantasycore`, then `empyrean_campaign` |
| Paired engine semantics | `flare-engine` commit `cf4d42f09442c2d8d08b6f1bdf9b6043e73a4443` |

The archive's sole symlink is retained as non-extractable metadata:
`README.md -> README`, mode `0120777`. The regular files expand to 661,089,992 bytes. The six-byte
difference from the supplied expanded-member total is exactly the symlink target payload.

The `empyrean_campaign` mod declares `requires=fantasycore:1.15`, so the animation corpus is the
active stack of those two mods: 310 physical animation definitions in `fantasycore` and 20 in
`empyrean_campaign`. The audit deliberately excludes the repository's optional `minicore`,
`alpha_demo`, `minicore_alpha`, and `devlab` definitions. These are different game/mod stacks,
not silently interchangeable variants of the campaign.

The engine commit is not embedded in the game ZIP. It is the `flare-engine` head from the same
2026-08-08 release update window (the game commit followed it by about two minutes) and is recorded
as a separate semantics dependency, not falsely represented as part of the CAS payload.

## Authoritative animation semantics

The source of truth is the archived `animations/**/*.txt` data interpreted using the paired engine
files `AnimationSet.cpp`, `Animation.cpp`, `Animation.h`, `FileParser.cpp`, `ModManager.cpp`, and
`UtilsParsing.cpp`. The current generated wiki agrees with these fields, but the adapter's exact
counts and behavior are grounded in source rather than mutable documentation.

An explicit compressed frame has:

```text
frame=index,direction,x,y,width,height,x_offset,y_offset[,image_id]
```

`index` is timeline order. Direction is either an integer or one of the engine tokens. The exact
integer mapping is:

| Integer | Token | Normalized direction |
| ---: | --- | --- |
| 0 | `SW` | southwest |
| 1 | `W` | west |
| 2 | `NW` | northwest |
| 3 | `N` | north |
| 4 | `NE` | northeast |
| 5 | `E` | east |
| 6 | `SE` | southeast |
| 7 | `S` | south |

Rectangles are `(x, y, width, height)` source-PNG crops. Offsets are the engine's render/floor
anchor, not crop coordinates. Image IDs select among root-level `image=path,id` bindings; a blank
or unknown ID uses the first image binding. All 70,897 effective explicit frame records select a
real image and lie inside that image's encoded dimensions.

The engine allocates eight direction slots per frame. When a compressed definition supplies only
direction 0, missing directions use direction 0 at first render. The audit exposes that behavior as
explicit fallback evidence rather than pretending those source rows exist. Forty-one direction
tracks are fallback-only, covering 167 slots. Missing directions with no direction-0 frame remain
missing.

The archive also has uncompressed declarations. Exact cells are derived only when `image`,
`render_size`, `position`, and `frames` are all present. In this snapshot, two template documents
(`animations/hero.txt` and `animations/avatar/default_unpacked.txt`) supply 16 inherited/parent
timelines but no image binding, so the adapter preserves their action/timing and leaves all 544
direction slots without geometry. It never infers a grid from a PNG.

### Time and loop behavior

`duration` is the duration of the complete forward action, not a per-frame delay. The adapter keeps:

- the literal source value, such as `533ms` or `1s`;
- exact integer milliseconds;
- nominal FPS, `declared_frames / duration_seconds`; and
- the paired engine's default-60 Hz tick schedule.

At runtime, millisecond durations are rounded to the nearest tick (halves upward) using the user's
configured maximum FPS. Frames are then distributed evenly where possible or with the engine's
Bresenham branch. For example, `run` with eight frames over `533ms` becomes 32 ticks at 60 Hz,
four ticks per source frame, for an effective 533.333 ms. The configured FPS can change, so the
60 Hz schedule is evidence for the engine default, not an unconditional per-frame millisecond fact.

Loop types are source-declared and remain distinct:

- `play_once`: one forward pass, then the last displayed frame is held;
- `looped`: repeated forward passes; and
- `back_forth`: repeated forward and reverse passes (ping-pong).

The audit records `active_frame` and `active_sub_frame` too. Active frames are gameplay trigger
metadata, not alternative timeline order.

## Exact animation inventory

The source has two useful count layers. Physical declarations count what is literally written in
the 330 files: 1,541 action sections and 54,869 `frame=` lines. Effective definitions follow the
engine's `INCLUDE` behavior, so tinted campaign variants retain their inherited geometry and become
1,996 effective actions with 70,897 explicit frame records.

| Measure | Exact count |
| --- | ---: |
| Physical animation definition files | 330 |
| Definitions containing an `INCLUDE` (directly or transitively) | 61 |
| Physical action declarations | 1,541 |
| Physical explicit frame records | 54,869 |
| Effective actions after includes | 1,996 |
| Actions with exact source rectangles | 1,980 |
| Timing-only actions with no derivable geometry | 16 |
| Direction tracks (`actions x 8`) | 15,968 |
| Tracks with source-explicit frames | 15,799 |
| Fallback-only tracks | 41 |
| Incomplete/unresolved tracks | 128 |
| Effective explicit frame records | 70,897 |
| Effective direction/frame slots | 71,608 |
| Direction-0 fallback slots | 167 |
| Unresolved slots | 544 |
| Actions complete in all eight effective directions | 1,980 |
| Referenced source PNGs | 296 |
| Missing referenced PNGs | 0 |
| Explicit frame rectangles outside PNG bounds | 0 |

Effective loop types are 1,474 `play_once`, 275 `looped`, and 247 `back_forth` actions.

| Source action | Effective actions |
| --- | ---: |
| `stance` | 249 |
| `run` | 230 |
| `run_alt` | 4 |
| `swing` | 229 |
| `dash_attack` | 5 |
| `shield_bash` | 1 |
| `shoot` | 227 |
| `cast` | 232 |
| `cast_alt` | 1 |
| `die` | 233 |
| `critdie` | 34 |
| `hit` | 232 |
| `block` | 229 |
| `spawn` | 9 |
| `power` | 81 |

The deliberately small normalization map follows engine/entity state meanings: `stance -> idle`,
`run* -> run`, attack variants to `attack`, `shoot -> shoot`, `cast* -> cast`, `die/critdie ->
death`, `hit -> hurt`, `block -> block`, and `spawn -> spawn`. The generic `power` action remains
unmapped because filenames and visual content do not establish whether it is a projectile, impact,
status effect, or something else.

Explicit record counts by source direction are southwest 8,883; west 8,872; northwest 8,872;
north 8,854; northeast 8,854; east 8,854; southeast 8,854; and south 8,854. The asymmetry is source
evidence, largely single-direction power/effect definitions, not a grid error.

## Identity, creatures, entities, and reuse

Definition path families are exact path-derived roles rather than guessed morphology:

| Family | Definitions |
| --- | ---: |
| Avatar parent/template | 2 |
| Avatar attachments | 196 |
| Enemy art | 36 |
| NPC art | 15 |
| Loot art | 39 |
| Power/effect art | 42 |

Enemy and NPC files resolve through engine-style `INCLUDE` chains to 177 animation bindings:
156 enemy definitions and 21 NPC definitions. Of these, 144 are concrete definitions and 33 live
under exact template/base paths. Seventy-seven effective bindings explicitly set `humanoid=true`.
This field means that transformation grants human interaction traits; it is preserved as a source
Boolean and is not broadened into a visual anatomy assertion.

The archived `categories` values provide rich, source-authored identity candidates including
antlion, goblin, skeleton, zombie, wyvern, minotaur, sentry, undead, fire, ice, and environment
groups. Display names and complete definition paths are retained. No filename-driven zoological or
monster taxonomy is asserted. The art also includes named NPCs, humanoid gear layers, loot, traps,
projectiles, spell effects, obelisks, graves, boulders, and a minecart.

There are 976 audited runtime usage references: 156 enemy, 21 NPC, 228 power, 13 effect, and 558
item-loot occurrences. One exact base reference, `animations/enemies/wyvern_adult.txt`, is absent
from the active game mods; only a half-scale `minicore` definition exists. It is reported and not
silently substituted with another wyvern.

Identity must remain many-to-many. Several concrete entities reuse one animation definition;
tinted definitions inherit the same physical frames; and multiple items/powers reuse a visual
animation. The safest art identity is `(game commit, active mod stack, logical animation path,
source action, direction, frame index)`, while the safest semantic identity is a separate entity or
usage occurrence joined to it.

## Avatar attachments and layer order

The campaign hero is assembled from synchronized animation layers. The 196 avatar attachment
definitions comprise 66 male, 65 female, and 65 `female_dark` variants. Every attachment has the
same action set and frame counts as `animations/hero.txt`; there are zero parent mismatches. The
engine sets that parent, synchronizes playback state, and renders equipped layers in the per-direction
back-to-front order declared by `engine/hero_layers.txt`.

The eight exact orders are preserved, for example:

- southwest: `main, feet, legs, hands, chest, off, head`;
- north: `feet, legs, hands, chest, off, head, main`; and
- southeast: `feet, legs, hands, main, chest, head, off`.

The item graph yields 349 equipment occurrences with `item_type` (layer slot), item ID/name, and
`gfx` attachment ID. Each occurrence has three archived body-variant candidates; the adapter keeps
all candidates instead of choosing a gender/body variant without a hero preset or save-state.
Layers are not flattened into independent character identities, and physical attachment crops are
not presented as complete figures.

## License and credit evidence

The exact snapshot contains nine relevant, hashed evidence documents:

| Evidence | SHA-256 | Scope |
| --- | --- | --- |
| `LICENSE.txt` | `3f941b3b89cf7b8370ceb83cc76d2120d471b58735d8ca60238a751a48d7f72f` | Full CC BY-SA 3.0 legal text at repository art/data scope |
| `README` | `00039b58f496f89a97434750c4659fef10455675ccd6c26d238464bfc1850b82` | Repository project claim: art/data CC BY-SA 3.0, later versions permitted; named OFL font exceptions |
| `CREDITS.txt` | `8b0985009dd04911b851553a35941e6862f1df66a6864f9b2a5c7ff330574fbc` | Repository contributor categories and external-art acknowledgements |
| `CONTRIBUTING.md` | `cba48772cbc11cc7a7dc7ee77b650ac0c7196bc09b879fb3c75266b8387b807e` | Contribution terms: art/data CC BY-SA 3.0 or later unless otherwise noted |
| `distribution/org.flarerpg.Flare.appdata.xml` | `fd702cc325e0e0e7e80c93d8f19ed5158521014c487fbded4104dfdf3bde4fad` | Distribution metadata separates engine GPL, campaign CC BY-SA, and metadata CC0 |
| `mods/fantasycore/cutscenes/credits.txt` | `0185d598d3d74785eb1bd846ad08dbf609cf227f8cd724d18ffb407a1170ac3d` | Fantasycore credit include aggregator |
| `mods/fantasycore/cutscenes/credits_fantasycore.txt` | `386300117be1efad3dff034c680e44eade33f9b5ba1658e832969cae74e0fba9` | Fantasycore contributor categories |
| `mods/empyrean_campaign/cutscenes/credits.txt` | `51a55e3e3e2e7f5a9e4f4bd36ad91ac1edba281b46044f2788f09ff28ea176a5` | Campaign credit include aggregator |
| `mods/empyrean_campaign/cutscenes/credits_empyrean.txt` | `2407283600744283872f0bcf114d867f9147aebdc7f376dd1108deeea7df0818` | Campaign contributor categories |

The README explicitly places Liberation, Bona Nova SC, Noto, and Marck Script fonts under OFL 1.1;
those exceptions matter to the full repository but are outside animation PNG projection.

The mod credit files name visual artists at mod/category scope, not per image. `CREDITS.txt` links to
a fuller online per-file listing, but that wiki is not contained in the immutable archive and may
change. The adapter therefore preserves the link and the ambiguity; it does not fabricate a
per-asset author row. All 721 PNGs were inspected. Seventy-nine contain scalar text metadata and 18
contain explicit `Software` fields. Thirty-nine comments say “Created with GIMP.” Sixteen menu PNGs
retain a Blender `File` source path plus render-scene fields; those paths are useful production
provenance but do not name the image's author or license. No PNG has an actual creator, credit,
copyright, license, or rights field.

## Public API

The adapter lives in `spritelab.adapters.flare` and is intentionally not registered in the shared
initializer or CLI.

- `parse_animation_definition(payload, ...)` parses one physical file, retaining direct includes
  without guessing unresolved content.
- `resolve_animation_definition(path, files, ...)` follows `INCLUDE` directives over an in-memory
  logical-path mapping and preserves every source location.
- `parse_duration_milliseconds(literal)` parses the source duration independently from an engine
  tick rate.
- `engine_tick_schedule(frame_count, duration, tick_rate=60)` reproduces the paired engine's
  duration rounding and frame-distribution algorithm.
- `direction_index(value)` implements the exact named/numeric direction mapping.
- `audit_flare_archive(path)` performs the structural, active-mod, animation, image, entity,
  attachment, and rights audit for a compatible ZIP.
- `audit_known_flare_archive(path)` additionally enforces the pinned archive SHA-256 and commit root.
- `FlareArchiveAudit.to_dict()` serializes the immutable result graph without changing evidence.

`audit_flare_empyrean_archive` and `audit_known_flare_empyrean_archive` are descriptive aliases.

## Safest database projection

No database projection is performed. A later ingestion change should:

1. Create a sequence only from an effective action/direction track with complete geometry. Use an
   external key containing the game commit, ordered mod stack, logical definition path, source
   action, and direction.
2. Keep physical declaration counts/source locations separate from effective include-expanded
   definitions. Preserve inherited color/alpha modulation and all source documents.
3. Store every ordered `(frame index, direction)` occurrence and exact source rectangle/offset.
   Mark direction-0 fallbacks explicitly; never write them as source-authored direction rows.
4. Preserve the duration literal, exact milliseconds, source frame count, loop type, nominal FPS,
   and engine-default tick schedule separately. Recompute a configured-FPS schedule when relevant.
5. Quarantine the 16 geometry-free template actions and 544 unresolved slots. Do not derive cells
   from image dimensions or use a nearby avatar layer as their sheet.
6. Keep art identity separate from all entity, power, effect, and item occurrences. Attach every
   occurrence many-to-many, retaining display name, categories, `humanoid`, definition path, and
   source location.
7. Treat avatar attachments as layers. Link their parent timeline and per-direction z-order; do not
   call one equipment layer a complete humanoid. Body variant remains a candidate until composition
   state chooses it.
8. Leave generic `power` action and morphology fields null unless a direct, reviewed source field
   supports normalization. Categories are source tags, not automatically a project taxonomy.
9. Quarantine the unresolved `wyvern_adult` usage instead of borrowing `minicore` or another wyvern.
10. Store license and credit documents as scoped evidence. Apply the repository/mod claims at their
    actual scope, retain the named font exceptions, and leave per-PNG authorship unknown.

## Known limitations

- Runtime behavior is audited statically. The Flare engine is not compiled or executed.
- The paired engine commit is a separately identified same-release dependency, not bytes in the
  game archive.
- Configurable engine FPS and animation speed effects can change playback ticks at runtime.
- `active_frame` trigger behavior is retained but not simulated.
- The separately distributed engine `default` mod is absent, leaving one base animation usage
  unresolved.
- Maps may instantiate enemy/NPC files indirectly; this audit resolves definition identity and
  reuse, not every map spawn occurrence.
- The external credit wiki is mutable and not accepted as immutable per-file evidence.
- Repository/mod credit categories do not establish per-PNG authorship or chain of title.

## Deterministic database projection implementation

The earlier “Safest database projection” section records the design gate that existed when the
adapter audit was first added. That gate is now implemented in the separate, deliberately
unregistered `spritelab.ingest.flare` module. Planning and readiness remain read-only. Projection
requires an already indexed archive and already extracted source PNG blobs; it creates no crops,
writes no corpus files, and adds no append-only `rights_observations` rows.

The projection unit is one complete effective `(definition, action, direction)` track. For the
pinned snapshot the deterministic plan contains:

| Projection fact | Exact count |
| --- | ---: |
| Effective definitions represented | 328 |
| Effective actions represented | 1,980 |
| Projected action/direction sequences | 15,840 |
| Projected ordered frame slots | 71,064 |
| Source-authored explicit slots | 70,897 |
| Direction-zero fallback slots | 167 |
| Referenced source PNG members/hashes | 296 |
| Resolved source entity bindings | 176 |
| Resolved animation usage occurrences | 975 |
| Exact attachment candidate edges | 1,047 |
| Distinct projected resource/subject identities | 1,650 |
| Required indexed evidence members | 934 |

The 128 incomplete direction tracks and their 544 unresolved slots remain plan exclusions. They
are exactly the 16 geometry-free actions from `animations/hero.txt` and
`animations/avatar/default_unpacked.txt`, expanded over eight directions. The one unresolved
`wyvern_adult` usage remains a separate usage exclusion. No optional-mod definition is borrowed.

All 349 equipment occurrences remain quarantined as unresolved runtime body-variant choices even
though their 1,047 three-body candidate edges are exact and retained as candidate relations. The
projection never chooses a candidate. Item 1102, “Knife of Sacrifices,” also declares layer slot
`relic`, which is absent from every one of the eight hero layer orders; its layer position stays
null rather than being mapped to a nearby slot. Candidate relations are not claims that a body
variant or equipment state was selected.

### Geometry and timing mapping

Each `sequence_frames` row retains its exact source PNG logical/member path and SHA-256, action
frame index, rectangle, render/floor-anchor offset, effective direction, source-authored direction,
source location, and explicit/fallback status. Fallback rows remain explicitly marked as inherited
from direction zero; they are never represented as source-authored rows for the effective
direction.

Flare tracks commonly vary both crop size and offset. In the pinned plan, 15,287 of 15,840 tracks
vary rectangle dimensions and 15,238 vary offsets, so a single frame rectangle cannot safely fill
the core schema's required sequence dimensions. `sequences.width` and `height` therefore contain
only a derived anchor-relative union envelope:

```text
left   = min(offset.x)
top    = min(offset.y)
right  = max(offset.x + rectangle.width)
bottom = max(offset.y + rectangle.height)
```

The envelope origin and its summary-only status are explicit metadata. It is not a uniform-canvas
or source-grid claim; per-frame rectangles and offsets remain authoritative.

The source duration literal, exact parsed milliseconds, nominal FPS, animation type, loop mode,
active-frame declarations, and full paired-engine tick schedule are retained. Per-frame
`duration_ms` is the frame's tick count at the engine default of 60 Hz. Metadata explicitly says
that engine FPS and animation-speed effects remain runtime configurable. `play_once` retains its
hold-last semantics, ordinary `looped` actions get a forward-cycle phase, and `back_forth` phases
and cycle fields remain null rather than pretending the forward declaration is the full
forward/reverse runtime cycle.

Physical and effective evidence remain separate. Sequence metadata identifies the physical root
definition and its hash, then independently preserves the ordered effective `INCLUDE` source
documents, include directives, image-binding order, inherited action declaration location, and
effective frame records. Stable sequence keys include the archive hash, game and engine commits,
ordered active mod stack, logical animation path, action ordinal/name, and direction index. Shared
PNG bytes or geometry never collapse separate source identities.

### Identities, relations, layers, and rights

A physical animation definition is projected as a non-semantic resource identity. Enemy/NPC
definitions, powers, effects, item-loot owners, and attachment items are separate subjects joined
to their exact visual sequences. Only source `humanoid=true` is promoted to the `humanoid` class;
enemy/NPC gameplay role and raw categories remain evidence instead of being guessed into animal,
monster, or species morphology. Generic `power` remains unmapped. Avatar layers keep attachment
ID, body variant, candidate item relation, synchronized parent-action evidence, and the exact
directional back-to-front layer order without being presented as complete characters.

All nine hashed license/credit documents stay in the plan with their original scopes. A sequence
links repository evidence plus only the mod evidence relevant to its physical/include/image
sources. Per-PNG creator and license fields remain null. The projector stores the scope caveat in
resource, sequence, occurrence, and frame metadata and intentionally reports
`rights_observations_added=0`, because that table is append-only and the snapshot has no immutable
per-asset manifest.

### Projection API and readiness

The new public API is intentionally not wired into shared initializers or the CLI:

- `plan_flare_projection(audit)` builds a pure deterministic plan;
- `plan_known_flare_projection(path)` enforces the archive SHA-256, game commit, paired engine
  commit, and ordered active mods;
- `check_flare_projection_readiness(database_path, plan)` opens SQLite with `mode=ro` and
  `PRAGMA query_only=ON`;
- `project_flare_audit(database, plan, taxonomy)` projects a precomputed safe plan; and
- `ingest_known_flare_sequences(database, archive_path, taxonomy)` combines the pinned plan and
  projector for a future explicitly authorized integration step.

Descriptive `flare_empyrean` plan/readiness/project aliases are also available. The readiness
check requires exactly one `flare_empyrean` source item, matching immutable inventory counts, all
934 evidence members, all 296 source PNG blobs registered in CAS, and exact agreement between each
indexed and audited PNG hash. The implementation-time live index passed that query-only check
without being mutated during adapter development. A later explicit integration run is recorded
below.

The pinned plan manifest SHA-256 is
`e0b232b3158a21546d8f7512b18d98306d2a7a66fc993b0bf986ad7618acbc9b`.
It is canonical JSON over the projection version, all four snapshot pins, physical/effective
counts, admitted records and frames, exclusions, attachment quarantines, exact layer orders, and
scoped rights evidence. Machine-local paths, database state, timestamps, and mtimes are excluded.

### Integration caveats

- The module is not imported from `spritelab.ingest.__init__`; the explicit CLI integration route
  is `spritelab corpus flare <archive-sha256>`.
- Core projection rows are idempotent through stable keys and upserts, as verified by a double-run
  fixture. Existing DB APIs still append audit `events` on repeated entity/sequence updates.
- Upserts do not delete stale higher frame ordinals, subject links, or occurrence links. A future
  projection-version migration must handle those rows explicitly; this projector never performs
  silent cleanup.
- CLI corpus projections run inside `IndexDB.transaction()`, so all nested DB helpers share one
  commit/rollback boundary. Direct library callers must opt into the same context when they need
  atomic whole-plan projection; idempotent reruns remain supported.
- No legacy `frames` rows or cropped blobs are emitted. Consumers must reconstruct from each
  `sequence_frames` rectangle and source PNG hash.
- Runtime composition, active-frame gameplay effects, configurable FPS, and map spawn instances
  remain outside this static projection.

## Live integration result (2026-08-12)

The pinned plan was projected from archive
`9c8e58bb704f55174928990a1a375c213ffb2847d7b928d6117a84db3e1215cc`.
An initial per-helper-transaction run was interrupted after 5,424 stable sequence
keys. SQLite `PRAGMA quick_check` returned `ok`; no rows were removed. The same
idempotent plan then resumed through the explicit batched transaction path, reused
all 5,424 prior sequences, created the remaining 10,416, and completed in about 43
seconds. The final result exactly matches the audited manifest:

- 15,840 projected direction sequences and 71,064 ordered frame references;
- 70,897 explicit frames plus 167 engine-defined direction fallbacks;
- 1,650 projected subject/resource identities;
- 202,240 sequence-to-archive evidence links during the complete rerun;
- 128 excluded direction tracks, 544 unresolved slots, one excluded usage, and
  349 quarantined attachment candidates; and
- zero appended `rights_observations`, because rights evidence remains scoped to
  the already indexed source item and archive members.

The completed run reproduced projection-manifest SHA-256
`e0b232b3158a21546d8f7512b18d98306d2a7a66fc993b0bf986ad7618acbc9b`.
Versioned stdout/stderr logs are retained under
`data/index/reports/flare-live-projection-v2-batched.*`; they are operational
evidence, while the canonical plan hash above is the portable corpus identity.
