# SuperTux exact-snapshot creature corpus audit

This adapter audits one immutable [SuperTux](https://github.com/SuperTux/supertux)
repository ZIP directly in the content-addressed store. It does not extract the
repository, execute Squirrel or C++, or write to the live corpus database. Its
scope is every `.sprite` manifest and PNG beneath `data/images/creatures/`, any
PNG outside that tree referenced by those manifests, the engine files that
define animation semantics, and pinned license/credit evidence.

## Snapshot and acquisition identity

| Field | Exact value |
| --- | --- |
| Repository | `https://github.com/SuperTux/supertux` |
| Default branch at discovery | `master` |
| Commit | `958bb9873c77f4063166d382076d4b19feb8a9c8` |
| Commit citation | `https://github.com/SuperTux/supertux/tree/958bb9873c77f4063166d382076d4b19feb8a9c8` |
| Commit time reported by GitHub | `2026-08-10T23:28:59Z` |
| Archive URL | `https://codeload.github.com/SuperTux/supertux/zip/958bb9873c77f4063166d382076d4b19feb8a9c8` |
| Archive SHA-256 | `98ea15f57224ab3374fb5a3a1bfc538fa33790eecf60c5f2193d782e96b1abc5` |
| Archive size | 290,571,350 bytes |
| Archive CAS path | `data/raw/objects/sha256/98/ea/98ea15f57224ab3374fb5a3a1bfc538fa33790eecf60c5f2193d782e96b1abc5` |
| ZIP root | `supertux-958bb9873c77f4063166d382076d4b19feb8a9c8` |
| Central-directory inventory SHA-256 | `2da2740e59deeb960db9d24505171e7a97ab2cc5b3968b82d353f643927c48d2` |
| Canonical audit-record SHA-256 | `1b5fd92ffbfe2dc7fbd9ca7f53d0c7fd2b540b84f8a3da6f0fbe722f09183703` |
| Canonical audit JSON blob | SHA-256 `90b4ca847c2150c8ca049e3297cffe0b1e39188134da503620f33d448f8ca7eb`, 4,503,828 bytes |

The repository metadata reported 841,797 KiB. Acquisition therefore used the
project's resumable `HttpFetcher`, immutable `ContentAddressedStore`, and
`DiskGuard`; the 100-GiB free-space floor was checked before the transfer and
for each chunk. No SQLite row was created or changed.

The HTTP/API evidence is independently preserved in CAS:

| Evidence | URL | SHA-256 | Bytes |
| --- | --- | --- | ---: |
| Repository metadata | `https://api.github.com/repos/SuperTux/supertux` | `7d0f3b826ea40dd48352590ce30851ee6c7aaae7baae03c8e198e9db78ca1f75` | 6,267 |
| Exact commit metadata | `https://api.github.com/repos/SuperTux/supertux/commits/958bb9873c77f4063166d382076d4b19feb8a9c8` | `b4128e1bb1bbc46ed82a4abd758805e5397287b65fba6e6ce01ebe77a645cd3a` | 7,566 |
| Exact-ref license metadata | `https://api.github.com/repos/SuperTux/supertux/license?ref=958bb9873c77f4063166d382076d4b19feb8a9c8` | `54554de21bafb5a8dfaa81d359ecf1a1684f0990a2b15d232817856d439e528b` | 49,550 |

## Archive safety and exact inventory

The ZIP has 6,708 members: 6,320 regular files, 387 directories, and one
symbolic link. Its members total 288,730,956 compressed bytes and 332,545,794
uncompressed bytes. The link is preserved as metadata, never dereferenced:

`metadata/android/en-US/images/icon.png` ->
`../../../../data/images/engine/icons/supertux-256x256.png`

The normalized target remains inside the archive and exists at the pinned
commit. Validation rejects path traversal, absolute/drive paths, backslashes,
NULs, encrypted members, duplicate or case-colliding names, escaping links,
unsupported special files, oversized members, and archives outside conservative
member/expanded-size limits. The complete archive digest is authoritative; the
separate central-directory digest makes inventory drift visible before parsing.

## Native manifest and engine semantics

There are 135 UTF-8 `supertux-sprite` manifests with 1,151 action declarations.
The pinned creature files use only the declarative fields supported by the
adapter: `name`, `hitbox`, `unisolid`, `fps`, `loops`, `loop-frame`,
`family_name`, `images`, `mirror-action`, `flip-action`, and `clone-action`.
Anything else fails closed rather than being guessed or executed.

| Action source form | Declarations |
| --- | ---: |
| Ordered image list | 662 |
| Horizontal `mirror-action` | 461 |
| Vertical `flip-action` | 16 |
| `clone-action` | 12 |
| **Total** | **1,151** |

Those declarations contain 4,134 direct image occurrences and resolve to 7,687
effective frame occurrences after aliases. Frame repetition is intentional and
preserved. A sequence such as `a, b, a` stays three frames; it is not reduced to
two unique files.

Resolution follows `src/sprite/sprite_data.cpp` in declaration order:

- mirror actions clone source surfaces with a horizontal transform;
- flip actions clone source surfaces with a vertical transform;
- clone actions copy the source action wholesale, except name/family handling;
- a clone therefore replaces an FPS or hitbox written on the clone declaration,
  matching the engine rather than the likely author intent;
- custom `loops`, 1-based `loop-frame`, default FPS 10, hitbox inheritance, and
  auto-sized hitboxes are recorded as effective values; and
- a later declaration of the same action name replaces its effective state.

The audit stores declared and effective FPS/loop values separately, exact
frame duration (`1000 / effective_fps` milliseconds), source order, source byte
hash, dimensions, alpha mode, hitbox, alias chain, and accumulated transform.
The engine evidence is byte-bound:

| Engine evidence | SHA-256 | Bytes |
| --- | --- | ---: |
| `src/sprite/sprite_data.cpp` | `c3b55f3bb5c390830cb2ab0b80a9f0400ffb857d4c1f3e817bc1cd19546b82f9` | 13,313 |
| `src/sprite/sprite.cpp` | `f433c81685ea1b0917b89243bd857b14cb7a4ea0291486306f29bf605d9f6f01` | 6,150 |

## Complete entities, components, and effects

The role boundary is an explicit pinned review in the adapter, not a broad
filename rule:

| Manifest role | Manifests | Projection treatment |
| --- | ---: | --- |
| Complete entity/body | 96 | Candidate only when its effective action is exact |
| Modular component/projectile/accessory | 23 | Separate layer/component evidence |
| Glow/light/overlay effect layer | 15 | Separate effect-conditioning evidence |
| Deprecated entity manifest | 1 | Inventory-only quarantine |

The modular set includes Ghost Tree roots, Crusher roots, darts, Crystallo's
shard, Granito and mole attack components, Mr. Cherry projectiles, Skullyhop's
trap/dart, and Tux's hat. The effect set includes eye/core/ticking glows, dart
lights, the Crystallo overlay, Ghost Tree root light, and the global fire/ice
overlays. `parent_entity_hint` retains the creature-family relationship without
pretending that these layers are complete bodies.

The 96 complete manifests cover this deliberately broad entity inventory:

| Entity class | Complete manifests | Examples |
| --- | ---: | --- |
| Animal | 20 | fish variants, bat, owl, mole, hedgehog, snail, spider, tarantula, toad |
| Humanoid | 3 | Tux, Penny, Nolok |
| Plant | 13 | Ghost Tree, walking leaves, ivy, leafshots, pumpkins, tree variants |
| Elemental | 7 | Crystallo, fire/ice/ghost flames, Kugelblitz, Livefire, Willowisp |
| Construct/device | 32 | crushers, Granito variants, bombs, traps, dispenser devices, ice blocks |
| Monster/other creature | 21 | Yeti, snowball variants, Jumpy variants, Ghoul, Skullyhop, Snowman |

Ninety-five of the 96 complete identities have at least two exact actions. The
complete-body candidate surface contains 1,010 exact action tracks and 7,103
resolved frame occurrences over 1,612 source paths (1,609 unique byte hashes).
It spans 87 native image geometries, from 32 to 1,100 pixels wide and 14 to 640
pixels high. Effective FPS values range from 4 to 60. There are 274 actions
with an explicit custom loop count and 38 declarations with a non-default loop
start.

Selected normalized steerable labels are:

| Label | Exact complete tracks |
| --- | ---: |
| Walk | 158 |
| Idle/stand | 72 |
| Run | 12 |
| Jump | 43 |
| Fall | 22 |
| Swim | 28 |
| Climb | 12 |
| Slide/skid | 26 |
| Attack | 13 |
| Throw | 5 |
| Stomp | 12 |
| Emote/taunt/rage/scratch | 18 |
| Sleep / wake | 12 / 18 |
| Death/melt/shatter | 26 |
| Hurt/stunned/iced/squished states | 172 |

Declared action names remain authoritative. Normalized labels are additions,
and 128 normalized labels are retained overall so rare behavior is not forced
into a small vocabulary. Direction parsing yields 430 left, 429 right, 7 each
of down-left/down-right, 8 each of up-left/up-right, 11 cardinal vertical, and
110 nondirectional exact tracks. Resolved complete frames comprise 3,752
identity, 3,303 horizontal-flip, 32 vertical-flip, and 16 combined-transform
occurrences.

## PNG inventory, geometry, and duplicates

The creature tree contains 1,940 static PNGs totaling 33,393,422 bytes. Every
one passes strict PNG chunk, trailing-data, Pillow decode, and dimension checks;
all decode as RGBA with an alpha channel. The adapter also strictly validates
the 16 referenced PNGs outside the creature tree (water-drop/splash and editor
stalactite images).

| PNG relationship | Unique paths |
| --- | ---: |
| Referenced inside creature tree | 1,825 |
| Referenced outside creature tree | 16 |
| Unreferenced creature-tree evidence | 115 |
| **All unique referenced paths** | **1,841** |

The complete creature-tree geometry spans 124 dimension pairs: width 26-1,100
and height 14-640. The 115 unreferenced PNGs remain auxiliary records rather
than being silently attached to the nearest manifest.

Thirty source-image hashes occur in duplicate groups, representing 33 excess
byte-identical creature-tree files. The full path membership is serialized in
the audit. Downstream splitting must stay entity/provenance aware: global hash
deduplication could sever a deliberate action alias or leak one appearance
across train/evaluation identities.

## Source defects preserved as quarantine

The audit does not repair these pinned-source facts:

1. There are 25 absent direct image occurrences representing 23 unique missing
   paths. `seasonal-snowball.sprite` and `smart-seasonal-snowball.sprite` each
   point nine times at paths outside their actual asset directory. The
   deprecated Tumbleweed manifest contributes seven stale occurrences.
2. Alias expansion makes those missing files affect 11 effective actions. They
   remain ordered frame references with `exists=false`, never dummy pixels.
3. `ghoul.sprite` declares `accel2-up` twice. The second declaration clears the
   existing surfaces and then mirrors the same action name. Under the pinned
   engine's order of operations, its source is already empty, so the effective
   `accel2-up` action has zero frames. The earlier declaration remains in the
   audit as superseded evidence.

Together these produce 12 quarantined effective actions and one empty effective
action. The 1,010 exact-complete count excludes the affected complete actions;
the deprecated Tumbleweed manifest is excluded by role as well.

## Rights and attribution evidence

GitHub identifies the repository license as `GPL-3.0`, and the exact root
license is GPL version 3. The README says SuperTux is licensed under that file
and notes that most of `data/` is also CC-by-SA. That secondary note has neither
a complete per-file map nor an explicit version at that location, so the audit
does not upgrade it to a universal per-PNG license expression.

| Evidence | SHA-256 | Bytes |
| --- | --- | ---: |
| `LICENSE.txt` | `8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903` | 35,147 |
| `README.md` | `273cc2f062930f9e5b2fe0b16d9e99ae737969af4727764b4ae62f6819aaec91` | 5,294 |
| `data/AUTHORS` | `bcfb4d94d5bcdae86923e0896788251674ffdea80311954dd00cf05e1871f527` | 10,827 |
| `data/credits.stxt` | `42dbfbb591d50dfe6c1293f750139082d230e19e2266504c298085b4b47b6cae` | 24,711 |

`data/AUTHORS` credits most graphics as of 0.7 to Rustybox, Eauix,
WeLuvGoatz, FrostC, FilipOK, and Bruhmoent and says to consult history for
details. No per-PNG author/license manifest exists under the creature tree.
Every future export should retain the repository, immutable commit, archive
hash, project license, README note, AUTHORS, and credits. This is a provenance
finding, not a legal conclusion about training or generated model weights.

## Deterministic API and projection boundary

`audit_supertux_archive(path)` audits any structurally valid fixture or snapshot.
`audit_known_supertux_archive(path)` additionally enforces the pinned archive
SHA-256, root, central-directory digest, and canonical record digest.
`known_supertux_cas_path(raw_root)` derives the expected sharded CAS path.
Frozen records expose `to_dict()` and sorted-key compact `canonical_json()`;
the record digest excludes only its own hash field.

A later database projector should require all of these gates:

1. `manifest.role == "complete_entity"`;
2. `action.effective_declaration == true`;
3. `action.exact_source_sequence == true` and every frame `exists == true`;
4. source bytes still match each recorded SHA-256;
5. frame order, repetition, transform, FPS, loop count/start, direction, and
   original action name are retained;
6. identity-aware splits are assigned before byte-level deduplication; and
7. all project-level rights evidence and its file-level attribution caveat are
   attached to every projected sequence.

Effects and modular components can support a later explicitly compositional
pipeline, but must never be presented as complete-body training targets without
a separately specified, reproducible compositing rule.

## Validation

Focused validation on the implementation snapshot:

```text
python -m pytest -q tests/test_supertux_adapter.py
8 passed

ruff check src/spritelab/adapters/supertux.py tests/test_supertux_adapter.py
All checks passed
```

The exact audit runs read-only from the 290-MB CAS archive in about 1.4 seconds
on the acquisition host. It performs no extraction, GPU work, or live-index
mutation.

## Deterministic database projection (2026-08-12)

The later-projector boundary described above is now implemented as a separate,
explicitly invoked projection module. The audit remains read-only. Planning is
pure, produces frozen recursively serializable records, and partitions every
one of the 1,151 declarations and all 7,691 frame occurrences attached to
those declarations. (The declaration total includes four frames on the one
superseded Ghoul declaration; the audit's 7,687 effective-frame count does
not.)

| Projection fact | Exact pinned value |
| --- | ---: |
| Projection version | `supertux_exact_complete_entity_transform_recipe_v1` |
| Projection-manifest SHA-256 | `78680f69abc1ffe442eab73fd8e9c34e7dd9106ba7540de73af4593e2e96109d` |
| Admitted complete-entity sequences | 1,010 |
| Admitted complete-entity identities | 96 |
| Admitted frame occurrences | 7,103 |
| Animated / static admitted sequences | 657 / 353 |
| Excluded declaration records | 141 |
| Excluded frame occurrences | 588 |
| Projected source-image paths / byte hashes | 1,612 / 1,609 |
| Excluded-ledger source-image paths / byte hashes | 245 / 239 |
| Union of referenced source-image paths / byte hashes | 1,841 / 1,814 |
| Duplicate admitted timeline groups / excess tracks | 150 / 296 |
| Required archive members | 1,982 |
| Projected archive-occurrence links | 11,735 |

The 141 exclusions comprise 79 modular-component actions, 49 effect-layer
actions, three deprecated-manifest actions, and ten non-exact or superseded
actions from otherwise complete manifests. Sixteen image paths occur in both
the admitted and exclusion ledgers; path identity and occurrence provenance
are retained even when bytes repeat. The 1,982-member evidence closure is the
1,841 referenced images, all 135 manifests, four rights/credit documents, and
two engine-semantics documents. Unreferenced creature PNGs remain bound by the
canonical source-audit hash but are not falsely attached to an animation.

A path-independent track-content key hashes ordered source bytes, native
geometry, transform, duration, FPS, and exact loop values. It identifies 150
duplicate admitted timeline groups containing 296 excess tracks. This is only
a deduplication signal: stable sequence/entity keys, source paths, manifests,
and occurrence links remain distinct, so a duplicate payload never collapses
identity or attribution provenance.

### Conditioning and exact source recipes

Every projected sequence has a stable source key derived from the immutable
archive/commit, manifest path and digest, declaration ordinal and line, and
declared action name. Entity and appearance keys are likewise deterministic
and manifest-scoped. Original action/direction labels, normalization bases,
alias target and chain, hitbox, `unisolid`, family name, declared/effective
FPS, per-occurrence duration, custom loop value, 1-based loop frame, frame
order, repeated frames, and native geometry all remain serialized.

The shared training taxonomy maps exact adapter classes as follows: animal,
humanoid, and monster remain unchanged; plant, elemental, and construct map to
the conservative animate superclass `creature`, while their exact adapter
class and classification basis remain available. Rare SuperTux actions absent
from the shared vocabulary map to the conditioning value `other` rather than
`unknown`; the source label remains authoritative. This produces 381 `other`
tracks while retaining labels such as slide, throw, stomp, wake, melt, and
shatter in metadata. View is explicitly `platformer`; nondirectional tracks
use canonical direction `none` rather than an ambiguous missing value.

Each SuperTux PNG is a single static source frame (`source_frame_index=0`).
Temporal order is represented by repeated references to those immutable blobs.
Four admitted tracks vary native dimensions across time; their sequence width
and height are the maximum native envelope, while each frame keeps its exact
full-image rectangle. No alignment, clipping, padding, or repair is claimed.

The 3,303 horizontal, 32 vertical, and 16 combined flips are not already
present in the source PNG bytes. A lossless per-frame transform recipe records
the required operation and the source occurrence. There are 437 admitted
sequences requiring at least one such later materialization. They receive the
explicit pending quality tier
`P0_exact_supertux_geometric_transform_materializer_required`; the 573
identity-only sequences receive `F0_lossless_supertux_exact_source_pixels`.
The current canonical materializer supports audited color-key transforms but
not geometric flips, so the pending sequences must not enter model inputs until
that operation is implemented and verified. Compatibility flags exist at both
sequence and frame scope. In addition, current model-ready snapshot selection
admits only generic fixed-phase loop modes; SuperTux retains engine-specific
`runtime_controlled` and `engine_custom_finite` modes with null generic phase,
so all SuperTux records currently fail that executable model-ready gate. An
end-to-end temporary-index regression locks the result at zero eligible
SuperTux model-ready samples. A caller can still deliberately make a
non-model-ready snapshot, so such exports must honor the incompatibility flags
and transform recipe rather than treating the raw source pixels as canonical.

Loop semantics also remain deliberately engine-specific:

- 736 actions have no custom loop count. Their database mode is
  `runtime_controlled`, and generic loopability, cycle length, and phase are
  left null because the engine accepts caller-supplied loop policy.
- 274 actions have the exact custom engine value `loops=1`. Their mode is
  `engine_custom_finite`; the raw value and loop start are retained, but the
  projector does not reinterpret that value as a framework-independent repeat
  count or mark the track infinitely loopable.
- The pinned admitted set has no custom infinite action.

### Rights and occurrence closure

Every projected sequence links archive occurrences for each distinct source
image it uses, its `.sprite` manifest, all four project rights/credits files,
and both engine files. Sequence, entity, subject, motion, occurrence, and frame
metadata all carry the pinned commit, archive hash, audit hash, and projection
hash as appropriate. The GPL-3.0 repository declaration, README's broader but
non-file-mapped CC-by-SA note, AUTHORS/credits summary, and per-file mapping
caveat are copied as evidence. The projector adds no new rights observation
and makes no per-PNG author or license claim.

### Query-only live-index preparation status

The preparation API opens SQLite using `mode=ro` and then enables
`PRAGMA query_only`; it returns a work plan and never extracts, registers, or
writes. A live-index check on 2026-08-12 preserved the database modification
timestamp and reported:

| Readiness fact | Observed value |
| --- | ---: |
| Source registry row present | yes |
| Pinned archive source item linked | no (0) |
| Pinned archive inventory present | no |
| Required members present | 0 / 1,982 |
| Referenced images extracted and linked | 0 / 1,841 |
| Referenced image paths whose expected blobs are registered | 0 / 1,841 |
| Manifest/rights/engine evidence verified and registered | 0 / 141 |
| Immutable hash mismatches | 0 |
| Ready for projection | no |

The resulting non-mutating preparation plan is to link the already preserved
pinned archive to a SuperTux source item, index its guarded central directory,
guardedly extract the 1,841 audited referenced PNG paths and 141 manifest,
rights, credits, and engine-evidence members into CAS, register every expected
blob, and then rerun readiness. Occurrence projection refuses a stale inventory
digest or any one of the 1,982 path-to-content hash mismatches. No live database
write or archive extraction was performed as part of this projection work.

### Projection validation

Focused tests cover deterministic planning, complete-versus-modular/effect
partitioning, stable keys, shared taxonomy mappings, alias transforms, frame
order, float timing, finite/runtime-controlled loop semantics, varying native
geometry, rights/engine/archive occurrences, read-only preparation, immutable
image and non-image evidence hash mismatch refusal, inventory and taxonomy
drift, blank/not-ready database non-mutation, model-ready fail-closed selection,
atomic rollback, and two-pass temporary-database idempotence. The exact pinned
regression additionally locks all counts and the projection-manifest digest
above.

```text
python -m pytest -q tests/test_supertux_ingest.py
11 passed

ruff check src/spritelab/ingest/supertux.py tests/test_supertux_ingest.py
All checks passed
```

## Live index preparation and projection after power-loss recovery

The earlier zero-closure readiness table is a historical pre-preparation
observation. On 2026-08-12, after an unexpected host power loss, the pinned
archive, live index, and pure projection were independently revalidated before
the first write. The archive again hashed to
`98ea15f57224ab3374fb5a3a1bfc538fa33790eecf60c5f2193d782e96b1abc5`
at 290,571,350 bytes, the database passed `PRAGMA quick_check`, all 19 focused
adapter/projector tests passed, and the recovered pure plan reproduced
projection hash
`78680f69abc1ffe442eab73fd8e9c34e7dd9106ba7540de73af4593e2e96109d`.

### Immutable item and inventory registration

The existing `supertux` source registry row was preserved. One source item was
created with ID `item_14b9d4a849954333bd2bf3db09bd1399` and external ID
`SuperTux/supertux`. It binds the exact commit, archive URL/hash/size, audit
hash, projection hash, and the three already-preserved GitHub API evidence
objects. Eight idempotently checked item/blob links cover the repository,
commit, and license API responses; pinned source archive; root license; README
rights note; AUTHORS; and credits.

The generic ZIP index and the SuperTux audit intentionally use different
inventory canonicalizations. The generic digest is
`bb285af6a7f04853b62d0dc0891f3a8a59580e34e10a87d5546662f01a0addd0`;
the adapter digest additionally binds logical paths, canonical member kinds,
Unix modes, sizes, CRCs, and compression methods and remains the authoritative
stored inventory digest
`2da2740e59deeb960db9d24505171e7a97ab2cc5b3968b82d353f643927c48d2`.
Both algorithms and hashes are retained in inventory/item metadata. All 6,708
central-directory members were indexed as metadata, but only the 1,982-member
audited closure was extracted.

### Selective extraction, media inspection, and scoped rights

The exact allowlist contained 1,841 referenced PNGs and 141 manifests,
rights/credit documents, and engine-semantics files. Its 32,935,989 source
bytes were streamed into CAS. Every extraction-stream digest matched the pure
plan, and every destination CAS object was then independently rehashed. The
1,982 member occurrences resolved to 1,953 unique payloads; 29 repeated
occurrences reused immutable payloads created earlier in the same selective
run. Nothing else in the archive was extracted.

All 1,841 image occurrences, representing 1,814 unique payloads, passed strict
PNG structure, trailing-data, Pillow decode, audited geometry/mode/alpha, and
non-animated checks under inspector `supertux-pinned-png-v1`. The terminal
archive statuses are 1,841 `media_inspected`, 141 `extracted` non-image
evidence members, 4,726 metadata-only `listed` members, and zero
`media_invalid` members. The preparation path would have retained every
invalid member and its error instead of deleting or repairing it.

Two separately scoped rights observations were added before projection:

- `rights_8e51f13b05994ba2b641e267e4aaab98` records the pinned root
  `LICENSE.txt` project/repository declaration as `GPL-3.0`. It explicitly
  says that creature-PNG applicability and per-file creator/license mappings
  were not inferred.
- `rights_047b768d5cbb4f48aef168ed0f3f2415` records only the README statement
  that most of `data/` is also CC-by-SA. Its normalized license expression is
  intentionally null because the note gives no version or per-file map and is
  not promoted to a universal data/PNG claim.

The final query-only prerequisite pass was complete: 1/1 source items,
1,982/1,982 required members, 1,841/1,841 extracted and registered images, and
141/141 verified non-image evidence files, with no missing paths, pending
extractions/registrations, or image/evidence hash mismatches.

### Atomic projection and executable exclusion gate

The first projector pass committed the complete corpus in one transaction. A
second pass reused every stable sequence key and reproduced all source-scoped
counts without adding rows:

| Live projection fact | First pass | Second pass / live total |
| --- | ---: | ---: |
| Created / reused sequences | 1,010 / 0 | 0 / 1,010 |
| Complete entities | 96 | 96 |
| Sequences / motion annotations / primary subjects | 1,010 each | 1,010 each |
| Frames | 7,103 | 7,103 |
| Archive occurrence links | 11,735 | 11,735 |
| Animated / static sequences | 657 / 353 | 657 / 353 |
| Runtime-controlled / engine-custom finite loops | 736 / 274 | 736 / 274 |
| Deferred-transform / identity-only sequences | 437 / 573 | 437 / 573 |
| Projector-added rights observations | 0 | 0 |

Normalized live entity classes are 20 animal, three humanoid, 21 monster, and
52 conservative `creature` records retaining exact plant/elemental/construct
subclasses. Frame recipes retain 3,752 identity, 3,303 horizontal, 32 vertical,
and 16 combined flip occurrences.

No SuperTux sequence is currently a model input. All 1,010 sequence metadata
records and all 7,103 frame metadata records have
`model_ready_materialization_eligible=false`. The 437 transform-dependent
tracks additionally fail geometric materializer compatibility, while the 573
identity-only tracks still fail the unnormalized SuperTux engine-loop gate.
Calling the current canonical snapshot loader with
`temporal_mode="model_ready"` and `include_source_ids=("supertux",)` returns
exactly zero samples after both the create and reuse passes.

The live database immediately before projection was 2,358,185,984 bytes with
SHA-256
`3dc12fdbc854bb1b81c36d57947e86f96b4371b1e046e03872f9ad648e364701`.
Immediately after the verified idempotence pass it was 2,441,465,856 bytes with
SHA-256
`6f3802aaed83cdc1ee4e3a8a3588328433f11b8c5b708c2aa1bb7c6a6a1dca87`;
the WAL was empty at hash time and `PRAGMA quick_check` returned `ok`.
