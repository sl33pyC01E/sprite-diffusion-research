# Universal LPC corpus audit

This audit is pinned to the immutable CAS object
`sha256:58a80830f1ca065f40e6d6acd678cc44551dc8902690798de7be8689d468da5b`.
The ZIP contains repository snapshot
`Universal-LPC-Spritesheet-Character-Generator-b2c85f98de52624b454dfdfac329bfee75795c2d`.
All counts below were computed directly from that object without extracting or modifying it.

## Archive inventory

| Measure | Count |
|---|---:|
| ZIP bytes | 150,428,252 |
| Central-directory entries | 95,355 |
| Regular files | 89,130 |
| Directory entries | 6,225 |
| Declared compressed member bytes | 113,815,332 |
| Declared expanded member bytes | 137,117,791 |
| PNG files | 87,978 |
| PNGs under `spritesheets/` | 87,962 |
| JSON files | 931 |
| `CREDITS.csv` rows | 13,853 |

Only a PNG below `spritesheets/` is treated as a corpus sheet candidate. The other 16 PNGs
are two readme images, one UI/GitHub image, and 13 layout, mask, cutout, or palette tool
images. This prevents screenshots and authoring aids from leaking into the training target set.

The 87,962 sheet candidates are transparent modular layers, not 87,962 finished characters.
They include bodies, heads, hair, clothing, equipment, effects, and foreground/background
planes intended to be composited on the same LPC rig.

| Layer category | PNGs | Layer category | PNGs |
|---|---:|---|---:|
| hat | 21,852 | facial | 18,090 |
| body | 14,223 | torso | 11,415 |
| neck | 7,475 | eyes | 2,875 |
| hair | 2,289 | head | 2,056 |
| dress | 1,997 | weapon | 1,459 |
| shield | 1,282 | backpack | 948 |
| legs | 829 | feet | 496 |
| shoulders | 241 | beards | 196 |
| arms | 146 | cape | 38 |
| tools | 28 | shadow | 21 |
| quiver | 6 |  |  |

## Path grammar and stable grouping

The dominant generated path grammar is:

```text
spritesheets/<category>/<asset-and-variant...>/<body-type>/<plane?>/<action>.png
spritesheets/<category>/<asset-and-variant...>/<body-type>/<plane?>/<action>/<palette>.png
```

Some weapon and split-layer assets put the plane after the action:

```text
spritesheets/<asset...>/<action>/<foreground|background|fg|bg>/<palette>.png
```

The adapter searches for the rightmost known action, removes action and palette from the
layer identity, and retains body type, structural variants, and compositing plane. For example:

```text
body/tail/cat/adult/fg/halfslash/blonde.png
  layer:  ulpc:body/tail/cat/adult/fg
  action: halfslash -> attack
  body:   adult
  plane:  fg
  palette: blonde
```

This key groups the same source layer across actions and recolors while keeping foreground and
background planes distinct. It is suitable for identity-aware splitting and multi-action
conditioning. The 194 PNGs without an action token are retained as sheet candidates but need a
definition/layout join before animation slicing.

## Actions and geometry

The repository's canonical expanded layout is 13 columns by 54 rows at 64 by 64 pixels per
cell, producing an 832 by 3456 sheet. Direction rows are ordered north, west, south, east.
`hurt` has only a south row and `climb` only a north row.

| Source action | Normalized cue | Directions | Frames per direction | Loop cue |
|---|---|---:|---:|---|
| spellcast / layout `cast` | cast | 4 | 7 | no |
| thrust | attack | 4 | 8 | no |
| walk | walk | 4 | 9 | yes |
| slash | attack | 4 | 6 | no |
| shoot | shoot | 4 | 13 | no |
| hurt | hurt | south only | 6 | no |
| climb | climb | north only | 6 | yes |
| idle | idle | 4 | 2 | yes |
| jump | jump | 4 | 5 | no |
| sit | sit | 4 | 3 | yes |
| emote | emote | 4 | 3 | no |
| run | run | 4 | 8 | yes |
| combat_idle | idle | 4 | 2 | yes |
| backslash | attack | 4 | 13 | no |
| halfslash | attack | 4 | 6 | no |

The path parser recognized an action in 87,768 of 87,962 sheet candidates:

| Path action | PNGs | Path action | PNGs |
|---|---:|---|---:|
| walk | 8,132 | thrust | 7,911 |
| hurt | 7,841 | shoot | 7,692 |
| slash | 7,579 | spellcast | 7,506 |
| idle | 5,885 | jump | 4,692 |
| emote | 4,641 | sit | 4,618 |
| backslash | 4,601 | run | 4,598 |
| halfslash | 4,572 | combat_idle | 4,526 |
| climb | 2,854 | cast | 40 |
| attack_slash | 38 | attack_backslash | 18 |
| attack_halfslash | 18 | attack_thrust | 4 |
| attack_slash_reverse | 2 | no action token | 194 |

Oversize equipment sheets use the same direction semantics with 128- or 192-pixel cells. The
adapter infers cell size from height and direction count rather than resizing them as 64-pixel
sheets. Geometry validation found exactly two malformed/outlier action sheets:

- `head/heads/skeleton/adult/halfslash.png`: 384 by 254, not divisible into four rows.
- `torso/clothes/shortsleeve/shortsleeve_cardigan/male/idle.png`: 128 by 320,
  which implies an 80-pixel row but a non-integral column count.

They should be quarantined pending visual inspection. No interpolation should be used to force
them into the standard grid.

### All sheet-candidate PNG dimensions

| Width x height | PNGs | Width x height | PNGs |
|---|---:|---|---:|
| 832 x 256 | 12,294 | 512 x 256 | 12,213 |
| 384 x 256 | 12,142 | 384 x 64 | 10,695 |
| 128 x 256 | 10,420 | 192 x 256 | 9,259 |
| 576 x 256 | 7,959 | 448 x 256 | 7,546 |
| 320 x 256 | 4,692 | 1536 x 768 | 302 |
| 1664 x 512 | 188 | 832 x 2944 | 80 |
| 768 x 1344 | 56 | 768 x 512 | 42 |
| 832 x 3456 | 27 | 1152 x 768 | 25 |
| 1024 x 512 | 12 | 128 x 128 | 8 |
| 384 x 254 | 1 | 128 x 320 | 1 |

The no-action subset consists of 80 sheets at 832 x 2944, 56 at 768 x 1344, 27 at
832 x 3456, ten wheelchair layers at 128 x 256, eight authoring fragments at 128 x 128,
and 13 other custom weapon/tool sheets.

## Definitions, body forms, and entity coverage

`sheet_definitions/` contains 769 JSON files: 657 item definitions and 112 `meta_` navigation
definitions. All 657 item definitions parse with the adapter. They contain 889 layer objects and
755 embedded credit claims. Layer body mappings occur with these frequencies (a definition may
support multiple forms):

| Body mapping | Layer mappings |
|---|---:|
| female | 847 |
| male | 839 |
| pregnant | 772 |
| muscular | 758 |
| teen | 735 |
| child | 130 |

The core body definition provides male, muscular, female, pregnant, teen, and child rigs.
Separate body definitions add skeleton and zombie. Head definitions provide meaningful entity
steering while remaining on a humanoid rig:

| Head-definition family | Definitions | Examples |
|---|---:|---|
| human | 10 | adult, elderly, gaunt, plump, small, child |
| farm/animal | 10 | mouse, pig, rabbit, rat, sheep; adult and child |
| beast | 9 | boarman, minotaur, wartotaur, wolf |
| fantasy | 7 | goblin, orc, troll |
| undead | 5 | Frankenstein, jack-o-lantern, skeleton, vampire, zombie |
| reptile | 4 | alien, lizard child/female/male |

The 45 head definitions correspond to 690 generated head-layer PNGs across 20 species/name
families. Tags include 16 `animal`, ten `human`, and five `monster` definitions. These are best
described as humanoids, anthropomorphic animals, fantasy humanoids, reptilian humanoids, and
undead humanoids. This archive does **not** provide a high-volume quadruped animal corpus; that
must come from other sources.

## Palette system

There are 22 palette-definition JSONs: 12 data files and ten metadata files. The data files
contain 215 named ramps (120 unique names after cross-material overlap): 200 six-color ramps and
15 three-color eye ramps.

| Material namespace | Unique names |
|---|---:|
| all | 75 |
| hair | 32 |
| body | 29 |
| cloth | 24 |
| eye | 10 |
| metal | 8 |
| wood | 8 |

Both ULPC and LPC Revised schemes are represented for body, eye, hair, metal, and wood. Cloth
has a ULPC file; the broad `all_lpcr` file supplies LPC Revised alternatives. Palette is an
explicit conditioning cue and should not be folded into layer identity.

## Provenance and license mixtures

`CREDITS.csv` has 13,853 rows over 13,532 unique generated filenames. There are 13,361 filenames
with one row, 21 with two, and 150 with three. Multiple rows must remain separate observations;
they are not accidental duplicates. The index contains 72 distinct author strings, 167 distinct
URLs, and 40 ordered license combinations.

License-token counts overlap because a row can list several licenses or alternatives:

| License token as written | Rows | License token as written | Rows |
|---|---:|---|---:|
| OGA-BY 3.0 | 8,077 | GPL 3.0 | 7,589 |
| CC-BY-SA 3.0 | 5,323 | OGA-BY 3.0+ | 2,381 |
| CC-BY 3.0+ | 2,332 | GPL 2.0 | 1,027 |
| CC0 | 995 | CC-BY 3.0 | 752 |
| CC-BY 4.0 | 483 | CC-BY-SA 4.0 | 231 |
| OGA-BY 4.0 | 96 | GPL 3.0+ | 60 |
| CC-BY | 45 | GPL 2.0+ | 22 |
| OGA-BY-3.0 | 18 | CC-BY-SA 3.0+ | 18 |
| OGA-SA 3.0 | 13 | CC-BY 4.0+ | 4 |

The uncommon spellings are preserved verbatim. Normalization can be added as a separate field,
but must not overwrite the source assertion.

Generated recolors do not map one-to-one to CSV rows. Usually a path such as
`.../walk/blue.png` maps to `.../walk.png`; `combat_idle`, `backslash`, and `halfslash` map to
credit keys `combat`, `1h_backslash`, and `1h_halfslash`. Some foreground/background paths also
move the plane before the action in the credit index. `credit_filename_candidates()` expresses
these deterministic joins. Against this snapshot, path-only candidates resolve 79,044 of 87,962
sheet PNGs (89.86%) to at least one CSV row. The remaining 8,918 are not necessarily
unattributed: definition-level credits remain the fallback for expression layers, legacy
original/master sheets, path substitutions, and sources that the generated CSV does not name
exactly.

## Adapter API

`spritelab.adapters.lpc` is deliberately pure: it reads no files and performs no writes.

- `classify_lpc_path()` separates sheets from UI/readme/tool assets and emits stable layer,
  action, body, plane, palette, and entity-family cues.
- `parse_sheet_definition()` handles all observed `layer_1` through `layer_8` shapes,
  multi-channel recolors, aliases, variants, tags, and embedded credits.
- `parse_credits_csv()` and `group_credits_by_filename()` preserve repeated provenance claims.
- `parse_animation_layout()` parses canonical layout JSON without executing repository code.
- `sheet_animation_cues()` emits per-direction stable IDs and validates/infer sheet geometry.
- `parse_palette_definition()` retains ordered color ramps and material/scheme identity.

## Streaming sheet manifest

`spritelab.ingest.lpc` adds a storage-independent manifest layer over the adapter. Callers
construct `LpcArchiveMemberFact` values from the exact archive-member and media-observation
rows, initialize `LpcManifestBuilder` once with parsed credit rows and sheet definitions, and
stream `builder.iter_records(facts)`. The functional `iter_lpc_manifest_records()` entry point
is equivalent. Neither entry point opens the database, reads the ZIP, decodes images, extracts
frames, or writes files.

Every `LpcSheetManifestRecord` makes the corpus semantics explicit:

- `record_kind` is `modular_compositing_layer_sheet`, `is_complete_entity` is false, and
  `composition_required` is true. The `entity_family_cue` is a steering hint for the rig, not a
  claim that a body, hat, weapon, or other individual PNG is a complete character.
- `stable_sheet_id` hashes the normalized repository-relative path, so removal of the GitHub
  archive's root directory does not change logical identity. `archive_occurrence_id` separately
  binds the observation to the pinned archive SHA-256 and central-directory ordinal.
- Each action/direction `LpcDirectionSliceSpec` stores its source row, local frame indices,
  linear source-grid indices, and exact `(x, y, width, height)` cell rectangles. This is enough
  to crop lazily while keeping the manifest compact and leaving all source pixels untouched.
- `layer_identity` deliberately groups actions and recolors of the same compositing layer;
  palette, plane, body form, source action, normalized action, and directional view remain
  separate conditioning fields.

Geometry is not reduced to a pass/fail filter. Each record retains source dimensions, expected
and actual cell/frame counts, a status, and a human-readable finding. Valid oversize cells are
kept at their native size. Malformed sheets remain addressable records with no slice specs, and
actionless sheets remain `layout_join_required` rather than being guessed into the standard
four-row layout.

### Read-only exact manifest audit

The streaming builder was run against all indexed media facts for the pinned archive above,
with all 13,853 CSV credit rows and all 657 parsed non-meta sheet definitions. It processed the
95,355 archive facts in 11.6 seconds and emitted the expected 87,962 modular-layer records,
318,979 direction slice specifications, and 2,192,770 lazy cell rectangles. No pixels were
decoded or materialized by the builder.

| Geometry status | Sheet records |
|---|---:|
| canonical | 87,206 |
| oversize, retained at native cell size | 556 |
| valid rectangular but noncanonical | 4 |
| malformed, retained without slices | 2 |
| custom/actionless layout join required | 194 |

The four valid noncanonical 64-pixel sheets are retained with their observed frame counts:

- `feet/accessory/plate_toe/male/shoot.png`: 8 observed columns versus 13 canonical.
- `feet/boots/basic/male/thrust.png`: 9 observed columns versus 8 canonical.
- `weapon/magic/wand/female/slash/wand.png`: 13 observed columns versus 6 canonical.
- `weapon/magic/wand/male/slash/wand.png`: 13 observed columns versus 6 canonical.

The two malformed records are the same 384 by 254 skeleton half-slash and 128 by 320 cardigan
idle sheets identified earlier in this audit. Keeping both in the manifest makes quarantine and
future visual review reproducible.

### Member-level credit evidence

Credit resolution uses exact, inspectable joins in descending precedence. It retains every CSV
row at a matched filename as a separate claim, including its original authors, license tokens,
URLs, and notes. When no CSV key resolves, the longest exact sheet-definition layer prefix can
supply embedded definition credits. Unresolved records are emitted with every attempted
filename and any definition sources considered; they are never silently treated as licensed.

| Match method | Match confidence | Sheet records |
|---|---:|---:|
| exact `CREDITS.csv` filename | 1.00 | 8,919 |
| deterministic recolor/action/plane/custom-animation candidate | 0.97 | 70,138 |
| exact sheet-definition layer prefix with embedded credit | 0.85 | 591 |
| unresolved | 0.00 | 8,314 |

The definition-aware deterministic candidates resolve 13 more CSV joins than the path-only
79,044 count above. Match confidence describes only confidence in the filename-to-evidence join;
it is not a legal-clearance score. License spellings and combinations remain verbatim source
claims so downstream policy can reason over them without erasing attribution evidence.
