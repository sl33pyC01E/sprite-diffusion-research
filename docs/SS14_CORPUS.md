# Space Station 14 RSI mob corpus

This adapter audits a commit-pinned subset of the Space Station 14 repository:
`Resources/Textures/Mobs/**/*.rsi`. It is an attribution-preserving research
inventory, not a statement that every audited asset is training-eligible. The
adapter is read-only and does not write the live SQLite index.

## Immutable inputs

- Repository: <https://github.com/space-wizards/space-station-14>
- Commit: `c724191e0407f2868780de6d308183477701538e`
- Commit page:
  <https://github.com/space-wizards/space-station-14/tree/c724191e0407f2868780de6d308183477701538e>
- Commit archive SHA-256:
  `125ca78d04a4f522e04597bf49d49fdb67a8cd2c2d079be13a2b3edb5591c444`
- Commit archive size: 234,732,657 bytes
- Commit archive CAS path:
  `data/raw/objects/sha256/12/5c/125ca78d04a4f522e04597bf49d49fdb67a8cd2c2d079be13a2b3edb5591c444`
- Repository root license evidence: `LICENSE.TXT`, SHA-256
  `0ac4d87483582bfec5500d39df7889a513730deeaf434c64f44a7975c3b82381`
- Repository RSI schema evidence: `.github/rsi-schema.json`, SHA-256
  `befd549dbaafb13cb720a3891ec66ae352e0c3a997096c608a6fc1927244e44c`

RSI runtime semantics are tied to the repository's exact RobustToolbox
submodule:

- Repository: <https://github.com/space-wizards/RobustToolbox>
- Commit: `15297f18f697d3a60cc1c764614fce85d234a395`
- Commit archive SHA-256:
  `eb42a1fa7e6ca3fa5e11df5c9ca89b1fc609078973278959c02a22f600c9ed82`
- Commit archive size: 4,669,821 bytes
- Commit archive CAS path:
  `data/raw/objects/sha256/eb/42/eb42a1fa7e6ca3fa5e11df5c9ca89b1fc609078973278959c02a22f600c9ed82`
- `RsiLoading.cs` SHA-256:
  `855e2f9f267a0543ec89fc698abdd6a1b096baf758d0eb9373569acbab27f687`
- `RSIResource.cs` SHA-256:
  `fd60e1ce7bc0a37638e4177b2c2d33f6b5db6c72ae4b3dc667c6be92f1657a24`
- `RsiDirection.cs` SHA-256:
  `3de8921a09851d26a3bdb1facad4c64c8d53f5ed26a81b8da6908af73f23956a`

Both archives were acquired through the project's guarded,
content-addressed store. C: retained over 219 GiB free after acquisition,
well above the 100 GiB hard floor.

## Exact audited slice

The canonical audit record SHA-256 is
`0804bd63eed162bd09c42716f7a1ef46f712e2aa702f6a75c38552ec18fac973`.
The exact snapshot contains:

| Measure | Count |
| --- | ---: |
| RSI packs | 184 |
| Named state images | 1,980 |
| Source cells declared by metadata | 9,849 |
| Source cells decoded | 9,849 |
| Engine direction/timeline occurrences after timing fold | 9,902 |
| Animated states under engine timing | 209 |
| Directional states | 1,563 |
| Directional and animated states | 134 |
| States with a conservative normalized action | 254 |
| Complete-entity sequence candidates after rights/geometry gates | 496 |
| Animated, action-known complete-entity candidates | 46 |
| Exact-capacity image grids | 1,933 |
| Image grids with unused trailing cells | 47 |
| Unused trailing cells | 112 |
| Missing, short, or undecodable images | 0 |
| Exact image-payload duplicate groups / excess members | 42 / 53 |

The two `scars.rsi` files named `*.png` contain valid WebP payloads. The
adapter detects image format from content with Pillow and records the declared
extension separately; it does not reject valid content solely because the
filename is misleading. The final format counts are 1,978 PNG and 2 WebP.

Pack categories are repository taxonomy, not learned visual labels:

| Category | Packs |
| --- | ---: |
| Aliens | 28 |
| Animals | 49 |
| Customization | 36 |
| Demons | 7 |
| Effects | 4 |
| Elemental | 4 |
| Ghosts | 2 |
| Pets | 16 |
| Silicon | 17 |
| Species | 21 |

## RSI decoding contract

The adapter mirrors the pinned RobustToolbox loader rather than guessing a
sprite-sheet grid:

1. `size.x` and `size.y` define the cell size; an image must be an exact
   multiple of it.
2. Each state has one image named `<state>.png`. The loader detects the actual
   image payload, so the adapter does too.
3. Cells are consumed row-major from the image. Metadata determines the number
   of consumed cells; unused trailing cells are retained as an audit finding
   and excluded from frame projection.
4. Direction runs are concatenated South, North, East, West, then Southeast,
   Southwest, Northeast, Northwest. Only 1, 4, and 8 directions are valid.
5. If delays are omitted, each direction receives one 1-second frame. An empty
   delay row also becomes one 1-second frame.
6. For multiple directions, RobustToolbox folds direction-specific timings to
   a common timeline using millisecond fixed point. The adapter retains both
   the source delay rows and the exact folded delays/source-cell indices.
7. RSI metadata does not encode whether the caller loops or plays a state once.
   Loop semantics therefore remain explicitly unspecified.

Ten packs set `load.srgb` false, and three set `metaAtlas` false. These flags
are retained. They do not alter source-cell geometry.

## Rights and attribution policy

The repository's MIT root license is not treated as an override for RSI art.
Every one of the 184 `meta.json` files contains its own `license` and
`copyright` fields; the adapter stores their verbatim text plus metadata
member path, SHA-256, and size.

| Per-RSI license | Packs |
| --- | ---: |
| CC-BY-SA-3.0 | 173 |
| CC-BY-SA-4.0 | 2 |
| CC-BY-4.0 | 1 |
| CC-BY-3.0 | 1 |
| CC0-1.0 | 2 |
| CC-BY-NC-SA-3.0 | 4 |
| CC-BY-NC-SA-4.0 | 1 |

The five NC packs (13 states) are kept in the citation inventory but receive
`noncommercial_asset_license` quarantine and are ineligible for default
training projection. This is deliberately mechanical: containing `NC` in the
per-pack SPDX expression is enough to quarantine, regardless of the intended
research use.

Copyright text contains 69 tgstation-family lineage links. Sixty-three packs
include at least one immutable 40-character upstream revision; six have only a
branch, repository, or otherwise mutable reference. The adapter records URL,
host, repository, reference type, revision, optional upstream asset path, and:

- a lineage key such as
  `github:tgstation/tgstation@53d1f1477d22a11a99c6c6924977cd431075761b`;
- an asset-level dedup key when both immutable revision and path are known;
- the current SS14 image byte SHA-256 for exact-payload deduplication.

Lineage does not imply byte identity: many SS14 assets say they were edited.
Downstream dedup should first collapse exact current image hashes, then use an
immutable upstream asset key as a related-origin group, not as proof that the
pixels are identical.

## Entity, layer, and action policy

RSI packs do not by themselves define complete game entities. In particular,
`Customization` and `Species` contain body parts, organs, markings, hair,
displacement maps, and other compositing layers. The adapter assigns a
conservative state role before any projection:

| State role | Count |
| --- | ---: |
| Complete-entity candidate | 507 |
| Modular component | 1,261 |
| Effect or overlay | 147 |
| Icon, in-hand, equipment, or item view | 65 |

Category-level entity class candidates are also conservative: Animals/Pets
map to animal, Silicon to robot, Demons to monster, Aliens/Elemental/Ghosts to
creature, and Species/Customization to humanoid-layer context. This does not
claim that each state is a whole member of that class.

Obvious compositing contracts are excluded mechanically. Paths named
`displacement`/`displacements` are modular; the Station AI pack is modular
because pinned prototypes compose its base and icon states as separate layers;
the cyborg chassis `_e`, `_e_r`, `_l`, `_rad`, and `_crystal` suffixes are
emissive/light/module overlays; crack packs, glow masks, outlines, screens,
construction frames, and extracts are not whole-entity candidates. These are
still retained in the inventory for a future composition-aware pipeline.
The audit binds these decisions to seven pinned prototype evidence members:
Station AI customization/player definitions, base/selectable/Xenoborg chassis
definitions, and animal/pet definitions. Their paths, byte sizes, and SHA-256
digests are part of the canonical record rather than undocumented local notes.

Action mapping uses explicit tokens in state names only. It maps forms such as
`running`, `moving`, `dead`, `attack`, `sleeping`, `rest`, `sit`, `spawn`,
`stunned`, and `crit`. It does not infer an action from animation alone or from
pack category. Only 254 of 1,980 states receive a normalized action; the other
1,726 deliberately stay unknown. Filename mapping is a cue, not authoritative
gameplay behavior.

A default animated training candidate must satisfy all of the following:

- valid decoded geometry with enough cells;
- non-NC per-pack license;
- `complete_entity_candidate` state role;
- more than one frame on the folded engine timeline;
- an explicit normalized action cue.

This gives 46 initial animated/action-known candidates. Static complete-entity
views remain valuable for appearance or direction conditioning, but they are
not presented as action loops.

## Verification

Focused verification is in `tests/test_ss14_adapter.py`. It covers path safety,
archive pinning, exact rights evidence, non-commercial quarantine, modular
layer separation, source-cell rectangles, misleading image extensions,
tgstation lineage keys, conservative action mapping, and engine-equivalent
direction-delay folding. The exact CAS regression test locks all counts and the
canonical audit hash above.

Current limitations:

- No live SQLite projection, snapshot, or materialization is performed here.
- RSI metadata does not link states to YAML entity prototypes; a future
  projection should join pinned prototype evidence before declaring semantic
  entity identity or runtime state usage.
- A state-name action is not proof that gameplay uses it as that action.
- Loop/one-shot behavior is controlled by callers and remains unknown.
- Modular humanoids need a separately evidenced composition pipeline; layers
  must not be trained as if each were a complete humanoid sprite.
- NC content stays indexed but quarantined from the default training pool.

## Deterministic SQLite projection

`src/spritelab/ingest/ss14.py` adds a separate, deterministic projection layer.
It does not broaden the audit's eligibility claims and has not been run against
the live index. The exact projection manifest SHA-256 is
`1e8bb0e67924b57ecf67ab3523a7f3c37987fbf13ad179feab1236f3370aafe4`.

The admission gate requires all of these facts at once:

- `complete_entity_candidate` role;
- no pack or state quarantine, including no NC license component;
- a decoded image whose grid capacity exactly equals the metadata-declared
  source-cell count (surplus cells are not projected);
- one unambiguous repository-category entity class that normalizes in taxonomy
  version `1.0`;
- an explicit state-name action cue that is already a canonical action in that
  taxonomy.

The last condition closes a source/taxonomy boundary. The audit intentionally
retains literal cues `move`, `rest`, `sit`, and `stun`, but these are not
canonical conditioning actions in `configs/taxonomy.toml`. The projection does
not silently rewrite them. It records them as `action:noncanonical:<cue>`
exclusions while retaining the original state name and cue evidence. Unmapped
state names are likewise explicit `action:unmapped` exclusions.

The pinned plan contains:

| Projection measure | Count |
| --- | ---: |
| Admitted source states | 189 |
| Direction-specific SQLite sequences | 246 |
| Pack-local entity candidates | 139 |
| Folded engine frame occurrences | 297 |
| Animated direction sequences | 13 |
| Static direction sequences | 233 |
| Excluded audited states | 1,791 |
| Evidence occurrence links after projection | 2,706 |

Each state is admitted once or appears once in the exclusion ledger; the 189
admitted states plus 1,791 exclusions reconcile exactly to all 1,980 audited
states. A direction-specific sequence is then emitted for every admitted source
direction, so sequence count is intentionally higher than admitted-state count.
The admitted record-level action distribution is: death 159, run 32, sleep 16,
hurt 12, idle 12, attack 7, walk 4, emote 2, and spawn 2.

Exclusion counts overlap because a state can violate more than one gate. Exact
reason totals include 1,726 unmapped actions; 30 noncanonical `move`, six
noncanonical `rest`, six noncanonical `sit`, and two noncanonical `stun` cues;
1,261 modular components; 147 effects/overlays; 65 icon/item views; 77 ambiguous
entity classes; 47 surplus grids; and 13 states under NC packs. This overlap is
retained in the manifest instead of reducing each state to a single reason.

### Timeline, geometry, and identity

One record is created per `(RSI path, exact state name, direction index)`. This
avoids collisions between numeric appearance variants that share an entity cue.
Entities use the narrower `(RSI path, entity cue)` identity so clearly related
states in one pack can share a subject without merging assets across packs.

Frames iterate `engine_source_cell_indices` rather than assuming one source cell
per occurrence. Repeated cells therefore survive unequal directional timing
folds. Every frame stores the exact engine interval in seconds and milliseconds,
the source cell's original delay, direction and direction-local index, state-local
cell index, and its native absolute rectangle in the source sheet. No crop,
clipping, mirroring, pixel conversion, timing repair, or frame interpolation is
performed.

RSI metadata does not encode playback loop policy. Every projected sequence has
`loop_mode = unknown`; motion rows keep `loopable`, `cycle_frames`, phase zero,
and per-frame phase unset. Actions are not used to invent loop behavior.

Stable sequence and entity keys include the SS14 archive hash and commit. The
source image SHA-256 is both the DB source blob and an exact-payload dedup key.
Per-pack tgstation lineage and immutable asset keys remain separate related-origin
evidence; they are never treated as proof of byte equality. The pinned plan needs
189 image members representing 187 unique image payloads. It records one duplicate
payload group with two excess members without collapsing their per-pack rights or
occurrences.

### Rights and evidence closure

Every entity, sequence, motion annotation, subject edge, state-image occurrence,
and frame carries the verbatim per-RSI license/copyright scope and upstream
references. The exact `meta.json`, root `LICENSE.TXT`, RSI schema, and seven
classification prototype members are linked as archive occurrences. Projection
does not append `rights_observations` because that table has no idempotent scoped
key; pack-scoped rights instead remain on every idempotently updated fact and
evidence edge.

The plan requires 287 archive member paths with 282 unique audited payload hashes:
189 selected images, 89 per-pack metadata documents, the root license, RSI schema,
and seven prototype documents. RobustToolbox timing evidence belongs to its own
separately pinned archive, so SS14 occurrence edges do not falsely claim those
engine files are members of the SS14 archive. The projection instead retains the
RobustToolbox commit, archive SHA-256, and immutable source URLs in its manifest
and sequence metadata.

### Read-only readiness and verification

`check_ss14_projection_readiness` opens SQLite with `mode=ro` and
`PRAGMA query_only=ON`. It reports source registration, source item/archive link,
archive inventory, every required archive member, extracted image blobs, and any
image hash mismatch. It never initializes a database or creates a journal.

As of this audit, the query-only live check correctly reports not ready: source
ID `space_station_14` is not registered, the archive is not inventoried or linked
to a source item, and none of the 287 required members is indexed. No live
projection, snapshot, or materialization was attempted while another corpus
writer owned the database.

Focused tests in `tests/test_ss14_ingest.py` project only into temporary indexes.
They verify deterministic state partitioning, repeated folded cells, native
rectangles, exact source/engine delays, canonical-action gating, NC/modular/surplus
quarantine, unknown loop semantics, rights/lineage/dedup retention, query-only
readiness, hash mismatch refusal, and idempotent reruns.

## Existing-CAS preparation after writer quiescence

The archive is already present in the immutable CAS, so preparation must adopt
that object rather than download it again. `plan_known_ss14_preparation` is
write-free: it rechecks the archive SHA-256, audits the projection, validates the
ZIP central directory, and produces an exact extraction allowlist. Its pinned
preparation manifest SHA-256 is
`9151dbcdbc7927680306d34647d28dc4801be17f497f55cc996baa2d3f7f97a1`.

The exact inventory and extraction budget is:

| Preparation measure | Count or bytes |
| --- | ---: |
| ZIP inventory SHA-256 | `39ab37b8ed29cef313b89d6488946fa3727d9b04c38fa95a9391e18d8a700c59` |
| Central-directory members | 49,472 |
| Regular files / directories / symlinks | 43,004 / 6,468 / 0 |
| Compressed / expanded archive members | 218,134,591 / 340,248,827 bytes |
| Required evidence members | 287 |
| Required evidence compressed / expanded | 414,157 / 813,186 bytes |
| Selected projected PNG occurrences | 189 |
| Selected compressed / expanded | 356,198 / 536,519 bytes |
| Unique selected image payloads | 187 |
| Maximum new immutable CAS payload | 530,803 bytes |
| All repository PNG members (do **not** extract) | 25,832 |
| All repository PNG expanded bytes | 61,014,321 bytes |

The generic APIs cover the preparation, but the generic
`archive extract-media --extension png` CLI does not: it selects all 25,832 PNG
members, including UI, tiles, effects, and unadmitted RSI states. SS14 must pass
the plan's exact 189-member `select` iterable to `extract_zip_to_cas`.

### Which index facts are required

The ordered prerequisites are:

1. **Source sync is required.** `space_station_14` exists in
   `configs/sources.toml` but was absent from the live index at audit time. The
   current registry has 48 rows; `sources sync` idempotently upserts all 48.
2. **Offline source-item adoption is required.** Register the existing archive
   blob, the repository item with external ID
   `space-wizards/space-station-14`, and one guarded `source_archive` item-blob
   edge carrying the exact codeload URL and commit filename.
3. **A fabricated retrieval is neither required nor appropriate.** This step
   performs no HTTP request. The item metadata, original URL, archive hash, and
   item-blob edge record the offline adoption. Add a `retrievals` row only when
   a real request log exists.
4. **Full metadata-only archive inventory is required.** Projection occurrence
   evidence refers to 287 archive members, so all 49,472 central-directory rows
   are indexed. The 98 non-image evidence members need inventory rows but do not
   need independent CAS extraction.
5. **Selective image extraction is required.** Extract and register exactly 189
   member occurrences (187 unique payloads) with role
   `ss14_projected_state_image`.
6. **Media inspection is a separate QA completion step.** It is not a projection
   prerequisite because the RSI audit already decoded native geometry, but every
   extracted occurrence must end in an explicit `media_inspected` or
   `media_invalid` state. No failure is silently treated as valid.

A read-only follow-up on 2026-08-12 found that a concurrent registry sync had
since populated all 48 configured source rows, including the exact
`space_station_14` registry-v1 row. That first prerequisite is currently
satisfied. The archive blob, pinned item/link, inventory, extraction, and media
rows remain absent; the sync command stays in the runbook as an idempotent
precondition check for a fresh or restored index.

`check_ss14_preparation_readiness` opens the database in read-only/query-only
mode and also hashes the small selected CAS objects. It exposes three distinct
answers:

- `projection_prerequisites_ready`: provenance, exact inventory, extraction,
  registration, and physical CAS integrity are complete;
- `media_inspection_complete`: every unique payload and member occurrence has a
  terminal media-QA outcome;
- `all_media_valid`: every selected payload passed the strict generic PNG
  inspector. `ready` means extraction plus terminal QA are complete; consult
  `all_media_valid` separately rather than erasing explicit invalid results.

### Ordered post-writer runbook

Do not execute these writes while Flare or another process owns the live index.
After confirming all writers have exited, first sync the registry:

```powershell
.venv\Scripts\spritelab sources sync
```

Then run the following API sequence. It is rerunnable: inventory, extraction,
and observations upsert; the otherwise non-unique `item_blobs` insertion is
guarded. It deliberately stops before `project_ss14_audit`.

```python
import json
from pathlib import Path

from spritelab.adapters.ss14 import SS14_ARCHIVE_URL, known_ss14_cas_path
from spritelab.archive import ArchiveLimits, extract_zip_to_cas, inspect_zip
from spritelab.config import load_config
from spritelab.db import IndexDB
from spritelab.indexing import (
    index_zip_extraction,
    index_zip_manifest,
    inspect_media_observation,
)
from spritelab.ingest.ss14 import (
    SOURCE_ID,
    SS14_ITEM_EXTERNAL_ID,
    SS14_MEDIA_INSPECTOR_VERSION,
    SS14_SELECTED_IMAGE_ROLE,
    check_ss14_preparation_readiness,
    plan_known_ss14_preparation,
)
from spritelab.sources import load_source_registry
from spritelab.storage import ContentAddressedStore, DiskGuard
from spritelab.taxonomy import load_taxonomy

config = load_config()
database = IndexDB(config.index.database)
guard = DiskGuard(config.storage.data_root, config.storage.min_free_bytes)
store = ContentAddressedStore(config.storage.data_root, guard)
archive_path = known_ss14_cas_path(config.storage.data_root / "raw")
taxonomy = load_taxonomy(config.project_root / "configs" / "taxonomy.toml")
registry = load_source_registry(config.project_root / "configs" / "sources.toml")
source = registry.by_id(SOURCE_ID)
plan = plan_known_ss14_preparation(archive_path, taxonomy)
limits = ArchiveLimits()

# The measured isolated preparation used 55,689,216 SQLite bytes. This check
# preserves at least the configured 100 GiB floor before starting index writes.
guard.require_capacity(64 * 1024**2, label="SS14 archive preparation")
database.register_blob(
    sha256=plan.archive_sha256,
    size_bytes=plan.archive_size_bytes,
    storage_path=Path(plan.archive_path),
    mime_type="application/zip",
)

with database.connect() as connection:
    existing_item = connection.execute(
        "SELECT id, canonical_url, metadata_json FROM items WHERE source_id=? AND external_id=?",
        (SOURCE_ID, SS14_ITEM_EXTERNAL_ID),
    ).fetchone()
if existing_item is None:
    item_id = database.upsert_item(
        source_id=SOURCE_ID,
        external_id=SS14_ITEM_EXTERNAL_ID,
        canonical_url=source.root_url,
        title="space-station-14",
        creator_name="space-wizards",
        creator_url="https://github.com/space-wizards",
        metadata={
            "full_name": SS14_ITEM_EXTERNAL_ID,
            "resolved_ref": plan.repository_commit,
            "commit_sha": plan.repository_commit,
            "commit_url": plan.commit_url,
            "archive_sha256": plan.archive_sha256,
            "archive_url": plan.archive_url,
            "acquisition_state": "adopted_existing_cas_archive",
        },
    )
else:
    metadata = json.loads(existing_item["metadata_json"])
    if (
        existing_item["canonical_url"] != plan.repository_url
        or metadata.get("commit_sha") != plan.repository_commit
    ):
        raise RuntimeError("Existing SS14 source item has a different provenance pin")
    item_id = str(existing_item["id"])

with database.connect() as connection:
    existing_link = connection.execute(
        "SELECT 1 FROM item_blobs WHERE item_id=? AND blob_sha256=? "
        "AND role='source_archive' LIMIT 1",
        (item_id, plan.archive_sha256),
    ).fetchone()
if existing_link is None:
    database.link_item_blob(
        item_id=item_id,
        blob_sha256=plan.archive_sha256,
        role="source_archive",
        original_url=SS14_ARCHIVE_URL,
        original_filename=f"space-station-14-{plan.repository_commit}.zip",
    )

manifest = inspect_zip(archive_path, limits=limits)
if manifest.inventory_sha256 != plan.archive_inventory_sha256:
    raise RuntimeError("SS14 ZIP inventory changed after planning")
index_zip_manifest(
    database,
    archive_blob_sha256=plan.archive_sha256,
    manifest=manifest,
    limits=limits,
)

extraction = extract_zip_to_cas(
    archive_path,
    store,
    limits=limits,
    select=plan.selected_image_member_paths,
    chunk_bytes=config.storage.download_chunk_bytes,
)
expected = {member.member_path: member.expected_sha256 for member in plan.selected_image_members}
observed = {
    extracted.member.normalized_name: extracted.blob.sha256 for extracted in extraction.extracted
}
if observed != expected:
    raise RuntimeError("SS14 selective extraction does not match its audited allowlist")
index_zip_extraction(
    database,
    archive_blob_sha256=plan.archive_sha256,
    extraction=extraction,
    selected_role=SS14_SELECTED_IMAGE_ROLE,
)

outcomes = {}
observations = []
for extracted in extraction.extracted:
    digest = extracted.blob.sha256
    if digest in outcomes:
        continue
    try:
        observations.append(
            inspect_media_observation(
                blob_sha256=digest,
                path=extracted.blob.path,
                original_name=extracted.member.normalized_name,
                inspector_version=SS14_MEDIA_INSPECTOR_VERSION,
            )
        )
        outcomes[digest] = ("media_inspected", None)
    except (OSError, ValueError) as error:
        outcomes[digest] = ("media_invalid", f"{type(error).__name__}: {error}")

database.record_media_observations(observations)
database.mark_archive_member_inspections(
    archive_blob_sha256=plan.archive_sha256,
    inspections=[
        {
            "ordinal": extracted.member.archive_index,
            "status": outcomes[extracted.blob.sha256][0],
            "error": outcomes[extracted.blob.sha256][1],
        }
        for extracted in extraction.extracted
    ],
)

readiness = check_ss14_preparation_readiness(database.path, plan)
if not readiness.projection_prerequisites_ready:
    raise RuntimeError(readiness)
if not readiness.media_inspection_complete:
    raise RuntimeError("SS14 media QA did not reach a terminal state")
print(
    {
        "preparation_manifest": readiness.preparation_manifest_sha256,
        "projection_ready": readiness.projection_prerequisites_ready,
        "all_media_valid": readiness.all_media_valid,
        "media_observations": readiness.media_observation_count,
        "media_invalid_members": readiness.media_invalid_member_count,
        "retrieval_rows": readiness.archive_retrieval_count,
    }
)
```

The full pinned rehearsal ran only against an isolated temporary SQLite index
and temporary CAS. It reached 49,472 inventory rows, 287/287 required evidence
members, 189/189 extracted occurrences, and 187/187 unique physical blobs. The
temporary SQLite file was 55,689,216 bytes and the new CAS payload was exactly
530,803 bytes; live incremental SQLite growth may vary with page reuse. The
existing 234,732,657-byte archive is reused and no network download is needed.

Strict media QA produced 176 unique PNG observations covering 178 member
occurrences. Eleven unique, one-direction, one-frame death images were marked
`media_invalid` because they carry one or two NUL bytes after the PNG `IEND`
chunk. Pillow and the pinned RSI audit decode their declared geometry, but the
strict generic chunk inspector correctly preserves the noncanonical payload
fact. All eleven are static `death` poses (nine creatures, two animals), so the
temporal set is unaffected. Keep them out of an appearance sampler that demands
canonical PNG until a separately hashed, lineage-linked normalization removes
only the trailing NUL bytes; never replace the immutable originals.

The exact trailing-byte evidence is: `barrier_dead` 2,
`barrier_naked_dead` 1, `fossilegg_dead` 1, `glider_dead` 2,
`harvester_dead` 1, `molder_dead` 1, `pouncer_dead` 1, `skitter_dead` 1,
`leviathing_dead` 2, `bear_dead` 2, and `narsian_dead` 2 bytes. No selected
failure has a hidden chunk or nonzero trailing payload.

## Snapshot and sampling policy

The raw projection is deliberately an evidence-preserving superset, not a
balanced training distribution. Of 246 direction records, 159 are `death`; 157
of those are single-frame poses. The exact split is:

| Use | Actions and records |
| --- | --- |
| Timed animation (13) | attack 7, death 2, emote 2, spawn 2 |
| Static pose (233) | death 157, run 32, sleep 16, hurt 12, idle 12, walk 4 |

Use two separate snapshots after projection. Do not pass
`--group-source-pack`, because SS14 is one repository item and that option would
force the entire corpus into one split.

```powershell
# Exact positive-duration multi-frame tracks only: 13 records.
.venv\Scripts\spritelab dataset export `
  data\index\snapshots\ss14-temporal-v1.json `
  --seed ss14-temporal-v1 `
  --include-source space_station_14 `
  --minimum-frame-count 2 `
  --temporal-mode known

# Single-frame/non-temporal records only: 233 records before QA exclusions.
.venv\Scripts\spritelab dataset export `
  data\index\snapshots\ss14-pose-v1.json `
  --seed ss14-pose-v1 `
  --include-source space_station_14 `
  --minimum-frame-count 1 `
  --temporal-mode pose_only
```

The 13 temporal records represent only nine identities, so SS14 alone is too
small for a meaningful random 90/5/5 animation benchmark. Use it as an auxiliary
training stratum combined with larger corpora, or evaluate by held-out identity.
Sample its four temporal actions uniformly or by inverse action frequency while
retaining identity/blob duplicate groups.

Static records belong in appearance, pose, direction, or first-frame
conditioning losses, not temporal losses. Never repeat/pad a static `run`,
`walk`, or `death` pose to manufacture motion. For the pose sampler:

- collapse exact image payload hashes for sampling while retaining every
  occurrence's rights and lineage;
- give each source state total weight one, divided across its directions as
  `1 / direction_count`, so four-direction expansion does not quadruple weight;
- cap `death` at no more than 25% of each epoch after direction weighting and
  rotate the admitted death states deterministically across epochs;
- preserve all rare actions, then balance entity class and identity inside each
  action stratum;
- exclude the eleven strict-media-invalid death poses until normalized derived
  PNGs exist or the consuming loader has an explicit, tested trailing-byte
  policy.

The current snapshot exporter records deterministic membership and leakage
groups but does not implement per-action caps or weights. Export the complete
pose snapshot for provenance, then apply this policy in the training sampler;
do not permanently discard surplus death occurrences from the source index.

## Live integration result (2026-08-12)

The pinned archive is now registered under source `space_station_14`, with one
source item and one exact source-archive link. The full inventory matches
`39ab37b8ed29cef313b89d6488946fa3727d9b04c38fa95a9391e18d8a700c59`.
All 287 prerequisite members are indexed; the 189 selected image occurrences
resolve to 187 unique CAS payloads and carry the explicit
`ss14_complete_entity_state_source` role.

Terminal strict media QA is complete: 176 unique valid observations cover 178
member occurrences, and the 11 unique trailing-NUL payloads above cover the 11
remaining occurrences as `media_invalid`. Thus preparation readiness is true,
while `all_media_valid` correctly remains false. No original payload was
rewritten or replaced.

The atomic projection reproduced manifest SHA-256
`1e8bb0e67924b57ecf67ab3523a7f3c37987fbf13ad179feab1236f3370aafe4`
and added 246 direction-specific sequences, 297 ordered frame occurrences, 139
pack-local entities, and 2,706 evidence links. This comprises 13 animated and
233 static sequences; all 1,791 excluded states remain represented in the plan
rather than being repaired or relabelled.

Two immutable snapshots now preserve the intended training boundary:

- `data/index/snapshots/ss14-temporal-v1.json`: 13 multi-frame records, artifact
  SHA-256 `9c5ddd12a4b0fbef2f5fb20bd43a0c332f4b34dfcc7923132d945a3f8dd3b1b5`,
  embedded manifest SHA-256
  `92a0a20d7e3bb7bf88ac52b17d6dff5c7a1ce5ca8a1d88dcc7c47e2249cabcbf`;
- `data/index/snapshots/ss14-pose-v1.json`: 233 single-frame records, artifact
  SHA-256 `242ac37567605c7450eb0a27b1265065991b916657596476eaa4bcc56d33c717`,
  embedded manifest SHA-256
  `eb5b3d555006f61ec5710f42d1b3c4195b79fd988e8334fc898f9056c7e09a4e`.

The temporal snapshot was exactly materialized to 13 RGBA clips with
materialization SHA-256
`035f648764fadf459ef56789cf2a9ca555df9c944d6c938ae31ac7b714d9fec9`.
All file and typed-array hashes verify, and there are no opaque exact-magenta
pixels. This is an exact reconstruction artifact, not yet a fixed-phase training
set: every SS14 timeline correctly retains `loop_mode=unknown` and null phases,
so `spritelab.training_data` rejects it rather than inventing repeat semantics.
