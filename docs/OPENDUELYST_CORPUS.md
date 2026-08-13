# OpenDuelyst exact-snapshot corpus audit

This corpus adapter audits one immutable OpenDuelyst repository archive directly from the
content-addressed store. It does not extract files, materialize frames, or modify the live
database.

## Snapshot identity

| Field | Exact value |
| --- | --- |
| Repository | `https://github.com/open-duelyst/duelyst` |
| Commit | `2843f2400854136598631288c2e8dfb8f5173de7` |
| Archive SHA-256 | `9d907a2d299b0f1598984192e3d4832aeb770e75fa2507370ff8e66428282f8e` |
| Archive size | 1,173,967,743 bytes |
| ZIP root | `duelyst-2843f2400854136598631288c2e8dfb8f5173de7` |
| ZIP inventory | 13,364 members: 12,807 files and 557 directory entries |
| Relevant formats | 6,303 PNG, 82 GIF, 1,385 plist, 1,578 CoffeeScript |

Content hashes reduce the 6,303 PNG paths to 5,398 unique payloads: 852 duplicate-hash groups
account for 905 excess paths. The 82 GIF paths reduce to 70 unique payloads. These source-derived
counts reconcile with the separately completed media index while keeping every archive path as an
alias instead of discarding it.

`audit_known_openduelyst_archive()` hashes the archive first and rejects any digest or root
other than those values. `audit_openduelyst_archive()` is available for structurally compatible
fixtures or independent snapshots, but its returned commit fields describe the adapter's pinned
research target; it should not be used to imply that an unpinned payload is that commit.

## Why these are not grids

The authoritative animation declarations are the 5,312 entries in
`app/data/resources.js` that have all of `name`, `framePrefix`, `frameDelay`, `img`, and
`plist`. There are 7,320 top-level RSX resources in total; ordinary images and other resources
are not promoted into animations.

The runtime establishes the remaining semantics:

1. `app/common/utils/utils_resources.js:151-180` selects plist keys beginning with the declared
   prefix when the next character is a digit, dot, or the source regexp's backspace character.
   It then performs a stable ascending sort on the last numeric token in each key.
2. `app/ui/managers/package_manager.js:866-884` multiplies the declared `frameDelay` by `0.8`,
   creates a `cc.Animation` from those ordered frame keys, and caches it using the descriptor's
   `name`.
3. Callers decide whether to wrap an action in `repeatForever`; looping is not stored in the
   plist or descriptor.
4. Unit/entity nodes flip the single source track horizontally at runtime based on owner and
   target state. The source does not label a canonical facing direction or provide independent
   directional tracks.

The adapter therefore keeps both the delay literal (for example, `.08`) and its parsed value,
records the runtime multiplier separately, and exposes the effective delay as
`declared_delay * 0.8`. It preserves plist declaration order and runtime order independently.
It never infers cells from sheet dimensions.

## Exact atlas and sequence inventory

| Measure | Exact count |
| --- | ---: |
| Animation descriptors | 5,312 |
| Unique descriptor plist paths | 1,277 |
| Unique descriptor image paths | 1,277 |
| Unique `(plist, framePrefix)` pairs | 5,310 |
| TexturePacker format-2 atlases | 1,294 |
| Atlas frame records | 69,564 |
| Source-declared frame occurrences | 69,091 |
| Unique frames selected by descriptors | 69,072 |
| Descriptors resolving at least one frame | 5,304 |
| Descriptors resolving no frames | 8 |
| Unmatched frames inside referenced atlases | 167 |
| Atlases not referenced by an animation descriptor | 17 |
| Frames in those unreferenced atlases | 325 |
| Rotated atlas records | 21 |
| Trimmed atlas records | 202 |
| Non-zero (possibly fractional) offsets | 110 |

Every one of the 69,564 atlas records has the TexturePacker fields `frame`, `offset`, `rotated`,
`sourceColorRect`, and `sourceSize`. All frame rectangles are inside the encoded image bounds.
One atlas, `resources/fx/fx_f2BK.plist`, declares a 128x128 metadata size while its PNG is
128x256; its rectangles remain in bounds. Four particle configuration plists contain invalid
plist integer values such as `0.0`. They are not texture atlases and are reported without being
reinterpreted:

- `app/resources/particles/manaspring.plist`
- `app/resources/particles/petals_001.plist`
- `app/resources/particles/petals_002.plist`
- `app/resources/particles/petals_003.plist`

Descriptor categories, based only on exact plist path components, are:

| Category | Descriptor count |
| --- | ---: |
| Unit | 4,319 |
| Icon animation | 742 |
| Effect | 238 |
| Tile | 6 |
| Rune | 6 |
| Arena effect | 1 |

There are 696 unit atlases under `app/resources/units`. The corpus therefore includes far more
than one morphology: source card names and mappings cover general/unit figures, animals by name,
structures, battle pets, golems, arcanysts, vespyrs, dervishes, effects, tiles, and icons. The
adapter does **not** turn names such as “Lantern Fox” or “Onyx Bear” into an asserted zoological
class, and it does not guess “humanoid” or “monster.” It preserves exact card identity, English
display name where resolvable, card kind, faction/race expressions, resource path, and all source
roles so a later reviewed taxonomy can make those judgments.

### Empty source declarations

These eight exact prefixes select no frame in their declared plist and must be quarantined rather
than repaired by fuzzy matching:

- `iconSuperMaliceActive`
- `iconThoughtExchangeActive`
- `f2TwilightFoxHit`
- `f1ThirdGeneralCast`
- `f3DuplicatorObelyskRun`
- `f4AbominationDeath`
- `f5OrphanAspectDeath`
- `f5OrphanAspectHit`

## Entity, action, loop, direction, and identity

The 68 `app/sdk/cards/factory/**/*.coffee` files contain 1,076
`setBaseAnimResource(...)` mappings and 4,988 exact role references to 4,708 unique RSX aliases.
The adapter retains the enclosing identifier expression, resolved numeric card ID when present in
`cardsLookup.coffee`, card construction kind, raw/localized name expression, English display name,
faction/race expression, role, source line, and numeric animation fields such as `attackDelay` and
`attackReleaseDelay`.

The source mappings include 689 `Unit` construction blocks, 62 `Artifact` blocks, 6 `Tile` blocks,
and spell variants. Race expressions include Arcanyst, Golem, Structure, BattlePet, Mech, Vespyr,
and Dervish; most mappings intentionally have no race expression.

Action and loop projection is conservative:

- One exact role gives a source-backed action label.
- Multiple roles give no normalized action; all candidates remain attached. Twenty-nine aliases
  have multiple roles.
- `idle`, `breathing`, `walk`, `castLoop`, `active`, and `occupied` are loop roles at their known
  runtime call sites.
- `attack`, `damage`, `death`, `castStart`, `castEnd`, `cast`, `apply`, and `depleted` are one-shot
  roles at their known call sites.
- Mixed or unknown roles produce `role_dependent` or `unknown`, never an invented Boolean.
- Direction remains null. Metadata records the exact single-track/runtime-horizontal-flip model.

Entity-to-art identity is many-to-many. The same animation resource is reused by multiple card
mappings, and the same physical atlas frames may serve multiple roles. The safest physical identity
is the exact plist/image path plus frame key; the safest semantic identity is the full card mapping,
not a filename stem.

### Aliases, duplicates, and collisions

Nineteen physical `(plist, frame key)` pairs are selected by more than one descriptor. Sixteen are
the active/idle views of `generalspell_f2_twilightsurge`; three are the damage/hit views of
`f6_explodingwall_hit`.

Only one non-empty pair has the same plist, complete ordered timeline, and declared delay literal:
`f6ExplodingWallDamage` and `f6ExplodingWallHit`. Empty timelines are excluded from duplicate
timeline detection.

Three RSX aliases differ from their runtime cache `name`:

| RSX alias | Runtime name |
| --- | --- |
| `neutralSkywingHit` | `neutralSkywingiHit` |
| `f3BBZephyrHit` | `f3ZephyrHit` |
| `f4FallenAspectAttack` | `f2OrizuruAttack` |

The last two create runtime-name collision groups with `f3ZephyrHit` and `f2OrizuruAttack`,
respectively. Both identities are retained because collapsing on runtime name would lose evidence
and may hide cache overwrite behavior.

## GIF previews

The snapshot has 82 GIFs containing 1,119 encoded frames: 69 under
`app/resources/unit_gifs`, 12 under `app/original_resources/unit_gifs`, and the web loading GIF.
All encode loop value `0` (infinite) and transparency. Their frame-count distribution is 67x14,
8x12, 3x10, and one each of 8, 11, 16, and 20. Each file uses one uniform encoded frame duration:
46 GIFs use 60 ms, 24 use 70 ms, and 12 use 100 ms. The 12 original-resource GIFs have
byte-identical counterparts in the resources tree.

GIF frame order, per-frame duration, loop value, transparency, hash, and raw filename-derived hints
are preserved. A hint is not promoted into authoritative identity or action. GIF timing is preview
timing and never overwrites the RSX/plist runtime timeline.

## License and credit evidence

The audit found and hashes 11 relevant evidence files:

| Evidence | SHA-256 | Scoped conclusion |
| --- | --- | --- |
| `LICENSE` | `a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499` | Root CC0-1.0 repository/project claim |
| `COPYING` | `a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499` | Same CC0 text and legal caveats |
| `README.md` | `f9f69937189a23f9f7c098f90ee86b2f3ff67fc013c31fd8baf80193417c0a1d` | States OpenDuelyst is CC0-1.0 |
| `package.json` | `184cebebf7d23eeff4b19b3c4ffdc2df30d4a72049bce472060d9d6a1a3ef7a7` | Root manifest says `CC0-1.0` |
| `desktop/package.json` | `d90e9efc2197603d00ab02fb0f708dbd78b0a17c191b85b5c767bbe98739bd49` | Desktop manifest says `CC0-1.0` |
| `app/vendor/cocos2d-html5/AUTHORS.txt` | `ef09e1e8bef6b0089240e42d27db3fb0e4a2731dfbb239698b12f2bfac4f9b4c` | Vendor authors only |
| `app/vendor/cocos2d-html5/licenses/LICENSE_cocos2d-html5.txt` | `3468be9001d3ad31cd88a3300f4e2141d3607ef83e224d67839d38b5d5733110` | Vendor subtree MIT |
| `app/vendor/cocos2d-html5/licenses/LICENSE_cocos2d-x.txt` | `03c53bfd36ac9de711f82e2d9d35b1814fb2b8034d56d93a2e3795ed94a23f84` | Vendor subtree MIT |
| `app/vendor/cocos2d-html5/licenses/LICENSE_zlib.js.txt` | `131afe3f7bdce1698beb292fb8de1a968de01bce876122a60ef5db230471c866` | Vendor subtree MIT |
| `packages/backfire/LICENSE` | `fd80e7bd6f8ce7fa548e8cbbf1875db45113672dcf6b30d49bd030bd9d4f9ca7` | `packages/backfire` MIT |
| `packages/warlock/LICENSE` | `65a09a7d656baebdcf1a11b2d0f2b7b70d5e8c96dd4756237e625739b00c3b16` | `packages/warlock` MIT |

The root files are evidence of a repository/project-level CC0 claim. They are **not** a per-asset
artist, provenance, or chain-of-title manifest, and the adapter does not promote them into such a
claim. The vendor author/license files apply only to their named subtrees. The CC0 legal text also
notes that other persons' publicity, privacy, and related rights may remain relevant.

All 6,303 PNGs are readable. Of these, 2,289 contain XMP/XML metadata and 17 have comments, all 17
being “Created with GIMP.” No PNG contains an actual creator, rights, license, credit, artist, or
author attribution field. XMP schema terms or tool metadata are not treated as authorship. No
per-asset license/credit manifest was found.

All byte-identical PNG groups are emitted as `DuplicateGroup(kind="byte_identical_png")` records
with their digest and complete path list. This is byte identity only; it is not evidence that the
paths have the same semantic role or entity identity.

## Public API

The adapter is in `spritelab.adapters.openduelyst` and deliberately is not registered in the shared
CLI or adapter initializer.

- `parse_resource_descriptors(source, ...)` parses only complete RSX animation declarations and
  retains every raw field expression and line.
- `parse_texture_packer_plist(payload, ...)` parses exact frame geometry, XML declaration order,
  offsets (including decimals), rotation, trim rectangle, source canvas, metadata, duplicates, and
  bounds evidence.
- `runtime_frame_keys(keys, prefix)` reproduces the source snapshot's selection and stable numeric
  ordering.
- `resolve_animation_sequence(declaration, atlas, ...)` joins one source declaration to exact plist
  frames and attaches roles without guessing.
- `parse_card_lookup(source)` preserves `Cards.Group.Member -> integer` identity.
- `parse_entity_animation_mappings(source, ...)` parses card/entity animation role mappings and raw
  ambiguity.
- `audit_openduelyst_archive(path)` performs a general structural, read-only ZIP audit.
- `audit_known_openduelyst_archive(path)` additionally enforces the exact CAS digest and commit root.
- `OpenDuelystAudit.to_dict()` serializes the immutable dataclass graph without modifying evidence.

`audit_open_duelyst_archive` and `audit_known_open_duelyst_archive` are spelling-compatible aliases.

## Safest database projection

No database projection is performed by this adapter. A later ingestion change should use these
rules:

1. Create one candidate sequence per **non-empty RSX descriptor alias**, not per plist, filename
   stem, runtime name, entity, or guessed grid.
2. Use a stable external key containing the commit and RSX alias. Store the potentially colliding
   runtime `name` separately.
3. Store the atlas image as source media and each exact atlas rectangle as frame evidence. To
   reconstruct a display frame, honor `rotated`, `sourceColorRect`, `sourceSize`, and `offset`;
   naive rectangle crops are not equivalent for trimmed/rotated records.
4. Preserve runtime frame order exactly. Store the declared delay literal/value, multiplier `0.8`,
   effective duration, and consumer source evidence separately.
5. Keep physical identity `(commit, plist, image)` separate from all entity mappings. Attach every
   mapping as a many-to-many occurrence; never merge cards merely because they reuse art.
6. Set normalized action only for one exact unambiguous source role. Otherwise leave it null and
   retain all role candidates and raw expressions.
7. Set loop only when the exact role/call-site evidence is unambiguous. Preserve
   `role_dependent`/`unknown` rather than coercing to false.
8. Leave direction null and store the runtime horizontal-flip semantics as metadata.
9. Quarantine the eight empty descriptors and all invalid/non-atlas plists. Do not fuzzy-correct
   prefixes or synthesize missing timelines.
10. Put GIF previews in a separate external-key namespace with their exact encoded timing. Do not
    merge them into atlas sequences by filename or use them to replace runtime timing.
11. Store license documents and their scope as evidence records. A repository-level CC0 claim must
    not become fabricated per-asset author/provenance rows.

This projection retains everything needed for later reviewed normalization while keeping all
uncertain semantics reversible.

## Known limitations

- Runtime behavior is audited statically at this exact commit; the adapter does not execute the
  game engine.
- Loop evidence is summarized from exact role/call-site behavior, not encoded in source plists.
- The source does not establish canonical facing direction or morphology classes.
- English names are resolved only for literal `i18next.t(...)` keys present in the archived English
  JSON. Raw expressions remain authoritative where resolution is unavailable.
- Filename-derived GIF identity/action values are explicitly hints.
- Unreferenced atlas frames are inventoried but not grouped into guessed animations.
- License evidence establishes the scopes described above, not per-asset authorship or clearance
  of non-copyright rights.

## Deterministic database projection

`spritelab.ingest.openduelyst` implements the conservative projection described above without
registering a shared CLI command. Planning and readiness checks are read-only; projection is an
explicit caller operation against an already indexed archive and its already extracted atlas-image
blobs. The implementation never extracts pixels, writes standalone frame artifacts, or adds an
append-only rights observation.

### Exact pinned plan

The pinned archive produces this deterministic plan:

| Measure | Exact count |
| --- | ---: |
| Source RSX aliases | 5,312 |
| Safe projected sequences | 5,302 |
| Projected frame occurrences | 69,020 |
| Physical `(plist, descriptor image)` identities | 1,276 |
| Source card/entity mapping identities | 1,076 |
| Total conservative DB entities | 2,352 |
| Sequence-subject links | 10,247 |
| Sequences with one exact source action role | 4,672 |
| Sequences with ambiguous or absent source action | 630 |
| Loop sequences | 2,692 |
| One-shot sequences | 2,006 |
| Role-dependent loop sequences | 3 |
| Unknown-loop sequences | 601 |
| Required archive evidence members | 2,638 |
| Required atlas image members with indexed CAS blobs | 1,276 |
| Quarantined aliases | 10 |
| Quarantined nonempty frame occurrences | 71 |

The canonical projection-manifest SHA-256 is
`ff7411d9a1dcd4aa76bb40dc8dbc087f563983352355684d8b37ab67847b2719`.
It hashes the projection version, archive/commit identity, every admitted record and frame fact,
every exclusion and reason, scoped rights documents, runtime source-code evidence, and the exact
card/localization evidence paths. Records and exclusions are sorted by stable external keys, so
archive traversal or dictionary iteration order cannot change the result.

The live-index dry run is currently ready: the pinned archive inventory and its source item are
present, all 2,638 required members exist, and all 1,276 required atlas image paths have registered
CAS blobs whose hashes equal the bytes audited inside the ZIP. The readiness implementation opens
SQLite with `mode=ro` and `PRAGMA query_only=ON`; the test verifies the database file hash and
modification time are unchanged.

### Admission and quarantine rules

The projector still creates at most one sequence per RSX alias. It admits a nonempty alias only
when all of these statements are true:

- the resolved sequence exactly agrees with the raw RSX alias, runtime name, prefix, plist, image,
  delay value, and delay literal;
- the descriptor image is the same image named/resolved by the plist metadata;
- the atlas image exists, has an audited SHA-256 and dimensions, and is already available as an
  indexed CAS blob before projection;
- the runtime frame keys exactly equal a fresh source-rule prefix selection and stable numeric
  sort over the plist declaration order;
- every packed rectangle is positive, inside the encoded atlas, and equal in size to its
  `sourceColorRect`;
- every `sourceColorRect` is inside its positive `sourceSize`, and one exact reconstructed canvas
  size applies to the whole sequence;
- the declared delay, runtime `0.8` multiplier, effective delay, and total duration are finite,
  positive, and mutually consistent;
- the resolved frame geometry is byte-for-byte/dataclass-for-dataclass identical to its atlas
  record. No crop, prefix, image, duration, rotation, or trim fact is repaired.

The original eight empty-prefix aliases remain quarantined. Two additional nonempty aliases are
excluded by the materialization gate:

- `f3GeneralFestiveIdle` declares `resources/units/f3_general.png`, while its plist resolves to
  `resources/units/f3_zirixfestive.png`. The projector does not guess which source blob was intended.
- `fx_fluid_sphere` has 59 selected records whose packed/source rectangle dimensions disagree and
  whose `sourceColorRect` values extend outside the declared `sourceSize`. The projector preserves
  the audit but does not pretend those records define safely reconstructable frames.

The exclusions contain the stable would-be sequence key, raw identities and paths, frame count,
all exact reasons, and unsafe frame keys. Empty-prefix aliases therefore account for zero excluded
occurrences; the two nonempty exclusions account for 71.

### Identity, aliases, and many-to-many roles

Each projected sequence key contains the immutable archive digest, commit, and RSX alias. The
possibly colliding runtime cache `name` is metadata, never the external sequence key.

The projection creates a conservative physical entity for each safe `(plist, descriptor image)`
pair and links it as the primary subject. This entity explicitly claims only physical source
identity, not a character or morphology. Every card-factory mapping is a separate stable entity
key using its source file, line, identifier expression, and card ID. All applicable mappings are
then linked with role `source_entity_mapping`, preserving their complete fields, source roles,
faction/race expressions, raw/localized names, and evidence lines. No mappings are merged merely
because they reuse atlas pixels.

Physical alias evidence remains on sequences and frames:

- all RSX aliases sharing the physical plist/image pair;
- all aliases sharing a colliding runtime name;
- exact nonempty timeline aliases;
- every shared `(plist, frame key)` alias set;
- complete byte-identical PNG path groups.

These facts are evidence of reuse, not permission to collapse semantic entities or actions.

### Timing, action, loop, and direction projection

Every `sequence_frames` row uses runtime order, the atlas declaration index as its source frame
index, and `declared_delay * 0.8 * 1000` as its duration in milliseconds. The raw delay literal,
parsed delay, multiplier, effective delay, total duration, runtime keys, and source-code consumers
are retained independently.

One exact source role is passed through the taxonomy; all roles remain in metadata. A sequence with
multiple or absent roles projects core action `unknown`, not a filename-derived label. Loop and
one-shot roles project their exact modes. The three walk/death conflicts project
`role_dependent`, with nullable loopability and phase. Unmapped loops remain `unknown`. Direction
projects `unknown`; metadata retains the single-source-track/runtime-horizontal-flip behavior and
does not assert a canonical facing direction.

### TexturePacker geometry and current schema limits

The core schema has one sequence width/height and no per-frame bbox columns on
`sequence_frames`. The gate therefore admits only sequences whose frames share one exact
`sourceSize`, and uses that reconstruction canvas for the sequence dimensions. Each frame metadata
object keeps:

- physical `(plist, frame key)` identity and runtime/atlas ordinals;
- packed atlas rectangle and its raw expression;
- `rotated`;
- `sourceColorRect` and raw expression;
- `sourceSize` and raw expression;
- fractional-capable `offset` and raw expression;
- trim and bounds flags;
- whether reconstruction requires rotation and/or a trim canvas;
- exact timing and physical alias evidence.

The row's `source_blob_sha256` remains the encoded atlas image. The projection does not populate
the standalone `frames` table because no cropped/reconstructed frame blob exists. A later
materializer must rotate/untrim onto `sourceSize`, register the derived blob and derivation, and
only then add a pixel-frame row. Treating the packed rectangle as an already reconstructed frame
would be incorrect for the 21 rotated and 202 trimmed atlas records.

The schema also has only one scalar `motion_annotations.source_action`; ambiguous action candidates
therefore remain in conditioning metadata while the scalar stays null/unknown. Likewise,
`loopable` cannot express role dependence, so it stays null for `role_dependent` and `unknown`.

### Rights behavior

Entities, sequences, frame metadata, and occurrence edges carry a structured rights-scope object
with exact evidence member paths and document hashes. It says
`repository_project_claim_only_not_asset_level`, lists root CC0 identifiers, separately lists
vendor/subtree evidence, and explicitly sets per-asset manifest, asset creator, and asset license
fields to absent/null.

Projection adds zero `rights_observations`: that table is append-only, so adding a row on every
rerun would violate idempotence, and an asset-level row would overstate the evidence. The archived
repository claim remains available for later policy review without being silently upgraded.

### Projection API

- `plan_openduelyst_projection(audit)` builds the pure deterministic plan.
- `plan_known_openduelyst_projection(path)` first enforces the pinned CAS digest and commit root.
- `check_openduelyst_projection_readiness(database_path, plan)` is the query-only dry run.
- `project_openduelyst_audit(database, plan, taxonomy)` preflights all evidence and image hashes,
  then idempotently upserts core entities, sequences, source keys, subjects, motion annotations,
  occurrences, and source-atlas frame records.
- `ingest_known_openduelyst_sequences(database, archive_path, taxonomy)` combines exact-snapshot
  planning and explicit projection.

Fixture tests run projection twice: the first pass creates sequences, the second reuses their stable
source keys, all core row counts remain constant, frame order/rotation/trim/timing remain exact, and
no rights observations appear.
