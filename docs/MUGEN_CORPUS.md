# M.U.G.E.N anime/JUS corpus track

M.U.G.E.N is a high-priority discovery source because a character pack already
contains the three facts this project otherwise has to reconstruct: SFF sprite
storage, AIR animation ordering/timing, and DEF identity/author/file metadata.
Community roster pages currently advertise anime/JUS compilations with thousands
of character directories. These claims are discovery leads, not audited corpus
counts.

## Trust boundary

Character packs are untrusted data. SpriteLab will never launch M.U.G.E.N or
IKEMEN, execute a character, import native code, or interpret CMD/CNS logic during
acquisition. ZIP paths are validated with the shared hardened archive inspector.
Executable members are quarantined. CMD/CNS files may be retained as hashed
evidence but are not executed or used as training data. RAR and 7z inputs are
listed with libarchive, path-validated, and only exact DEF/AIR/SFF/ACT members
are streamed to memory. No archive paths are unpacked to disk.

The initial adapter in `spritelab.adapters.mugen` parses only:

- DEF `[Info]` and `[Files]` identity/provenance references;
- AIR action numbers, ordered SFF group/image references, 60 Hz durations,
  offsets, exact H/V flip flags, collision declaration counts, and `Loopstart`;
- the SFF signature/version and exact payload hash.

SFF pixel decoding is deliberately a separate milestone. A header match is not
treated as decoded media, and an AIR action is not admitted until every referenced
sprite resolves exactly.

## Semantics

Elecbyte's standard action numbers are mapped conservatively: standing to idle,
20/21 to walk, jump actions to jump, guards to block, and documented lose/hit/dead
actions to hurt/death. Recommended numeric attack ranges map to attack but retain
the exact action number and source range basis. Arbitrary custom numbers remain
unmapped unless CMD/CNS analysis later supplies a deterministic, non-executing
semantic join.

A final `-1` duration is a terminal hold. A finite action with `Loopstart` is an
intro-then-loop. A finite action without `Loopstart` loops from its first element.
Original integer ticks are authoritative; seconds are derived exactly at 60 Hz.

## First immutable sample

The first format fixture is the public Eiko Magami character mirror indexed by
Simple MUGEN 2004:

- landing page: `https://mugen.justivo.com/`
- download URL: `https://mugen.justivo.com/files/eiko.zip`
- archive SHA-256:
  `fbb11d8558852ea7e53bc5e32822e2c04625d4c22b6fdc627b8c9fb1776a362f`
- archive bytes: 969,172
- server last-modified: `Wed, 30 Jun 2010 17:22:07 GMT`
- page claims: Eiko Magami, Project A-ko, author Majere
- internal DEF claims: name `Eiko Magami`, display name `Raye Heart`; its leading
  comments name Majere and the original distribution URL.

The ZIP has ten members and passes CRC verification. It contains one DEF, one AIR,
one SFF, two ACT palettes, sound, documentation, and CMD/CNS runtime logic; there
are no native executable members. Rights remain unknown/unverified. Neither the
mirror nor pack is interpreted as granting a permissive license.

The exact read-only decode currently yields 342 SFF v1 sprite occurrences, 341
group/image keys, 298 distinct RGBA payloads, 40 linked sprites, 156 AIR actions,
and 603 ordered frame references. One action (`3220`, source comment `Exploding
tank`) is quarantined because key `(3030,0)` has two different pixel payloads and
the same action references absent key `(3030,2)`. The remaining 155 actions / 600
frame occurrences materialize to axis-aligned full-canvas RGBA without resizing.
Their normalized labels are 68 attack, 50 unknown, 9 block, 7 hurt, 5 idle, 5
jump, 4 spawn, 2 walk, 2 emote, 2 death, and 1 run. Loop evidence is retained as
102 finite loops, 37 terminal holds, and 16 intro-then-loop actions.

Three no-clobber display derivatives under
`data/index/reports/mugen-eiko-format-preview-v1/` verify the end-to-end decoder:
idle action 0, forward-walk action 20, and light-kick action 200. They are exact
source renders, not generated model output. The action-200 contact-sheet SHA-256
is `8f2a55fb53b19d653678dd14ded1ab51ec2e90a2c7bc6c4c74a1f07245d9ce1d`.

## Discovery index

High-yield leads retained for staged, size-checked acquisition include:

- Mikazuki/XDeathwing roster pages: advertised 3,000- and 4,000-character anime
  compilations, including JUS/chibi variants, hosted through public Google Drive;
- MUGEN & IKEMEN Community: 262 anime/manga resource entries plus a dedicated JUS
  category at time of discovery;
- Simple MUGEN 2004: small individually attributed legacy mirrors suitable for
  parser validation;
- creator sites and MUGEN Database pages: identity/author taxonomy and links back
  to individual releases.

For every acquired pack, retain landing-page URL, direct URL, page title/category,
listed uploader and character author separately, HTTP validators, retrieval time,
archive hash, inventory hash, every member hash, internal DEF claims, and any
license/readme evidence. Compilation membership never replaces original creator
attribution. Anime franchise identity and fan distribution do not imply reuse
permission; rights evidence and training inclusion policy remain separate facts.

### Mikazuki roster-scale discovery

The public [Mikazuki MUGEN roster page](https://sites.google.com/view/mikazukithemugenitecreations/mugen-rosters)
was indexed as landing-page metadata before any additional archive acquisition.
The corrected immutable index is
`data/index/reports/mugen-mikazuki-roster-discovery-v2.json`, SHA-256
`31ac109cbe8a7766bce4b83faad65f515fae7b8ef2616daa8a1156feacb608e7`.
Its exact 710,508-byte HTML snapshot is retained in CAS with SHA-256
`e271dfc02453af6cd9e9841e25d1a6c7eb18383dca4b5e4309de29512c1f159e`.
The earlier v1 index is retained rather than overwritten; v2 corrects only the
claimed-character-count heuristic for titles such as `1100+ JUS CHARS`.

The v2 page census contains 125 linked roster sections, 120 with downloads, 185
unique archive-entry URLs, and 68 YouTube preview occurrences. Providers are 94
Google Drive, 66 MediaFire, 10 TeraBox, 6 MEGA, 5 link-shortener, and 3 Dropbox
URLs plus one alternate TeraBox domain. Twenty-four title-keyword priority
sections contribute 35 download entries for anime, manga, JUS, Touhou, Melty
Blood, Fate, Dragon Ball, and related discovery terms. Explicit page claims span
an 1,100-character anime/JUS pack, 2,000- and 3,000-character JUS/anime packs, a
4,000-character Anime Ascension pack, and broader compilations claiming as many
as 18,000 characters.

Those numbers are uploader page claims, not decoded or deduplicated identities.
The keyword flag is a discovery-priority heuristic only. Preview videos remain
metadata-only, rights remain unverified, and no newly indexed large compilation
has yet been downloaded. Acquisition must first resolve provider metadata and
declared sizes on Spark, enforce a 100-GiB free-space floor there as well as on
the workstation, and prioritize anime/JUS archives likely to add identities not
already present in the 298-archive MFFA tranche.

The bounded priority metadata pass is
`data/index/reports/mugen-mikazuki-priority-download-metadata-v1.json`, SHA-256
`4afcea23a2098ca5b03031c72b8a77fb2d86f0efaf58e7652e090eb499577ed1`.
It visits only HTML landing/view responses, rejects non-HTML or responses over 2
MiB, and never opens a direct archive body. Of 35 anime/JUS-priority link
occurrences, 16 resolve to named single files with exact provider size metadata
and 19 remain explicit folders, encrypted-provider entries, shorteners, dead
links, or unsupported providers. The 16 known sizes total 699,842,805,738 bytes
(651.78 GiB), which exceeds Spark's current free space and therefore rules out an
undifferentiated bulk pull.

The high-value size facts materially change acquisition order. Anime All Stars 3
is 3.52 GiB; the 2,000-character JUS/Chibi archive is 38.93 GiB; Anime Ascension
v3 is exactly 98,894,887,513 bytes (92.10 GiB); and the 1,100-character AX2 JUS
screenpack file is only 121.94 MiB while its three character-set folder sizes are
still unresolved. The intended Spark-only sequence is therefore smallest
anime-specific archives first, followed by the 2,000-character JUS archive and
then Anime Ascension only while the independent 100-GiB floor remains satisfied.
Every acquired archive still requires exact hashing, inventory, decoder audit,
and cross-corpus payload deduplication before it contributes training rows.

The Spark MediaFire fetcher re-resolves the indexed landing page at retrieval
time instead of trusting an expiring direct URL.  It requires the fresh filename
and binary-size declaration to match the pinned metadata report, resumes only
from an exact HTTP byte range, fsyncs bounded chunks, enforces the 100-GiB floor
before the transfer and during streaming, and retains every partial after errors.
Completion moves the exact bytes into SHA-256 CAS and publishes a no-clobber
acquisition record containing both the indexed landing evidence and the fresh
landing-page hash.  It never executes or extracts the downloaded roster.

## First collection-scale pass

The exact Simple MUGEN landing snapshot SHA-256 is
`0b4257a12e0bac936e2773f3cf99d2c95c06b83ddf61062f05324b3f0a5c0938`.
Its 17 individually listed ZIP character mirrors were acquired through the
resumable, 100-GiB-floor CAS fetcher. The acquisition index is
`data/index/reports/mugen-justivo-acquisition-v2.json`, SHA-256
`dd891a7daa4627b449b9a43f359bc80b99dd27d89fa85428879468150ddbe836`.
Together the archives are 22,199,370 bytes.

The deterministic decoder audit is
`data/index/reports/mugen-justivo-corpus-audit-v1.json`, SHA-256
`dcca5de71b21ba5045543b612a0e6344f0d5092aa90c8cc1e5c41c9fbb06c4f7`.
It verifies all 17 packs and reports:

- 8,913 SFF sprite occurrences, including 344 linked sprites and 7,738
  pack-local distinct RGBA payloads;
- 2,810 AIR actions / 16,581 authored frame occurrences;
- 2,685 admitted actions / 15,399 exactly materialized aligned RGBA frames;
- 125 quarantined actions: 102 missing sprite keys, 9 ambiguous duplicate keys,
  10 unsupported AIR transforms, 2 non-integral offsets, and 2 empty actions;
- admitted labels: attack 1,064, block 150, death 24, emote 35, hurt 119, idle
  103, jump 87, run 16, spawn 43, walk 34, and unknown/custom 1,010;
- loop evidence: 1,900 finite loops, 678 terminal holds, and 107
  intro-then-loop clips.

This tiny 22-MB validation collection already contributes more than fifteen
thousand usable action frames. It substantiates the hypothesis that large
anime/JUS rosters could change corpus scale materially, while also quantifying
why strict per-action validation is necessary.

## MFFA anime collection-scale audit

The MUGEN & IKEMEN Community anime/manga category was indexed as 270 resource
pages and 298 downloadable archive occurrences. The immutable discovery index is
`data/index/reports/mugen-mffa-anime-discovery-v2.json`, SHA-256
`1d985ef837f94cd93344bce2fe5ccd772664fb983c50e84c36001b76e0d27280`.
All 139 ZIPs and 159 RAR/7z archives were acquired through append-only,
power-loss-resumable CAS fetchers with the 100-GiB free-space floor enforced.
The ZIP acquisition index SHA-256 is
`9551af668e4d96464690cb352e025d0abc6f1d8aaca0c04c2f8cbbe925eda91e`;
the RAR/7z index SHA-256 is
`5fcce574461846102d6457c413eeb4b09294ecfa62284a2c517d8b3f6a4bb493`.
Together they bind 8,480,344,538 archive bytes.

The corrected ZIP audit is
`data/index/reports/mugen-mffa-anime-zip-corpus-audit-v3.json`, SHA-256
`b4cd7a5caab27c79d34d4223b6e42867a445f4c57f54b5ae145fc712a7cf9701`.
It decodes 93 packs (79 SFFv1 and 14 SFFv2), with 46 explicit failures retained.
It finds 112,945 sprite occurrences, 104,056 pack-local distinct RGBA payloads,
26,223 admitted actions, and 187,452 ordered materialized frame occurrences.
Known labels comprise 9,194 attack, 798 block, 177 death, 329 emote, 625 hurt,
617 idle, 476 jump, 88 run, 415 spawn, and 179 walk actions; 13,325 custom
actions remain unknown. Runtime timing evidence is 17,342 full loops, 2,599
intro-then-loop actions, and 6,282 terminal holds.

The independent RAR/7z audit is
`data/index/reports/mugen-mffa-anime-rar7z-corpus-audit-v1.json`, SHA-256
`9433bc9a4d326b14dd8ade3780ce269c9199e01ccbecdaeaa2bf3e4be5762b41`.
It decodes 120 packs (100 SFFv1 and 20 SFFv2), with 39 failures retained. It
finds 134,526 sprite occurrences, 120,056 pack-local distinct RGBA payloads,
30,883 admitted actions, and 214,484 ordered frame occurrences. Its known
labels comprise 10,180 attack, 1,053 block, 233 death, 341 emote, 794 hurt,
799 idle, 634 jump, 112 run, 449 spawn, and 234 walk actions; 16,054 custom
actions remain unknown. Runtime timing evidence is 22,109 full loops, 2,929
intro-then-loop actions, and 5,845 terminal holds.

Combined, this tranche verifies 247,471 decoded sprite occurrences, 57,106
admitted animations, and 401,936 ordered materialized frame occurrences. These
are source-corpus facts, not generated samples. The training materializer keeps
known finite loops with at least two positive-duration and two pixel-distinct
frames, caps each action class per identity, removes same-action tensor
duplicates, and selects eight duration-weighted frames. A single neutral-motion
reference extent fixes scale for every action of an identity; the MUGEN world
origin is placed at a stable bottom-center anchor with exact floor-index nearest
sampling into 128x128 RGBA. This avoids action-dependent zoom and preserves
character scale across idle, locomotion, hurt, and attack controls.

Rights remain unknown or unverified for these fan uploads. Landing-page uploader
claims, internal DEF author/name fields, archive identity, media members, and
source action numbers remain separate provenance facts. No permissive rights are
inferred, and no MUGEN character code is executed.

## Combined model-ready materialization

The ZIP and RAR/7z training projections were merged without rewriting pixel
arrays: their exact `.npy` payloads are hard-linked into
`data/processed/mugen-mffa-anime-combined-action-v1/`. The canonical
materialization manifest SHA-256 is
`8d43b387765cb1289ab3491d95f26ce52c00b47b22c14ace269f71a9c84fd7bb`.
It contains 5,906 clips at 8x128x128 RGBA: 4,408 train clips from 168
identities, 655 validation clips from 25 identities, and 843 untouched test
clips from 34 identities. Identity overlap between every split pair is exactly
zero.

The combined known-action counts are 3,436 attack, 778 idle, 600 spawn, 298
walk, 246 hurt, 195 jump, 129 defend, 110 run, 91 emote, and 23 death clips.
There are 5,856 humanoid and 50 robot clips. Raw frequency is not the training
distribution: the training sampler chooses an identity and then one of that
identity's available actions uniformly, preventing attack-heavy characters from
dominating solely through authored action count.

Text conditioning uses the immutable OpenAI CLIP ViT-B/32 description table at
`data/processed/semantic-text/mugen-mffa-openai-clip-vit-b32-c7244be-v1/`.
Its manifest SHA-256 is
`80b37151978bfcff565547cb67e61ee04a153225ef0da3e03ef6fdfcc09f9453`
and its canonical embedding-array SHA-256 is
`8f14a9c1179d2cfa7541f100e8a55d319b0b18a3105bbaa218d61e3e5a14dbab`.
All 216 distinct materialized descriptions resolve exactly. The first broad
quality run uses an 8-layer, width-256 temporal DiT, 8 frames, CLIP text plus
factorized action/entity/view/direction/loop conditioning, horizontal-flip
augmentation, EMA, rectified-flow training, and matched endpoint supervision.
Held-out evaluation remains identity-disjoint and will not be described as text
generalization unless the untouched split and prompt controls support that
claim.

## Structured combat action evidence

`spritelab.adapters.mugen_logic` adds a static, non-executing CNS evidence parser.
It reads only literal `Statedef` facts and literal `HitDef` fields; triggers,
expressions, controllers, and character code are never evaluated. Elecbyte's action
number recommendations provide broad `normal_attack`, `special_attack`, and
`super_attack` tiers plus reserved locomotion/guard/state verbs. Literal HitDef
`attr` can corroborate normal/special/super and distinguish projectile or throw;
literal `animtype` can supply light/medium/heavy. AIR comments can supply explicit
punch/kick/weapon/projectile/throw vocabulary. Conflicting claims remain unresolved
rather than letting the last claim win.

The first materialization-wide taxonomy intentionally uses only facts already in the
combined manifest: source action number and retained source meaning. It does not yet
claim light/medium/heavy or attack form, because CNS/AIR evidence has not been joined
across the full acquired collection. This is a deliberate data-format boundary, not
an inference gap hidden by a default. The canonical report lives at
`data/index/reports/mugen-mffa-action-taxonomy-v1.json`; its own hash is recorded
after generation. The later CNS/AIR join must retain source archive/member hashes,
literal line evidence, and ambiguity instead of replacing this coarse projection.

## Motion-role VLM precision audit

A deterministic 77-clip sample spans 20 broad verbs with up to four clips per
verb. Each Qwen request binds the exact reference-plus-eight-frame contact sheet,
sequence and identity IDs, expected verb, request bytes, response bytes, and the
Qwen 3.5 122B model alias. The first structured-decoding run is retained at
`data/processed/mugen-mffa-motion-role-vlm-decisions-v1/manifest.json`, SHA-256
`b96e2bd1808af6132873d3cb2af0d80b76bb635fbbc915f25ab40a2ca58c2854`, but is
invalid as a visual-curation estimate. Llama.cpp JSON-schema-constrained decoding
collapsed obvious same-subject sheets toward semantically incorrect enum values;
one answer also duplicated an array enum despite `uniqueItems`. No v1 decision is
used for admission or exclusion.

The corrected prompt-constrained JSON run removes `response_format` while placing
the same literal schema and allowed enum values in the user message. Its prompt
contract SHA-256 is
`27539f1f1e41706979e12927904829f51407e5a4b7d42c406e667cd8ddcf0aeb`; its
manifest is
`data/processed/mugen-mffa-motion-role-vlm-decisions-v2-prompt-json/manifest.json`,
SHA-256
`ba64bdd72f8e88ad485629970306aa2caacefd908df31b24132e5b3c4b1e856e`.
All 77 responses parse strictly: 72 pass the conservative same-primary-subject
motion gate and five are explicitly rejected. Most importantly, all 48 sampled
clips from the exact 12 verbs in the canonical broad latent-motion manifest pass
(backstep, block, crouch, dizzy, get-up, hurt, idle, jump, normal attack, run,
turn, and walk). The five rejections occur only in broader death, special-attack,
super-attack, and victory candidates. This bounded precision audit supports the
current canonical verb slice; it is not a collection-wide recall estimate and
does not replace the exact pixel gate.

The historical still-plan-v1 held-out gallery is also non-canonical. In
particular, `sequence_01ad78ce15d11303392281009aa154e2` pairs an orange-furred
fighter description with an effect-only blue target. The exact pixel audit marks
all eight frames failed (`pixel_gate_status=all_fail`, no passing indices), and
the canonical primary-motion manifest excludes it. Therefore
`data/inference/mugen-mffa-sd14-lora-step2500-heldout-comparison-v1/` must not be
shown as valid target/generation evidence. Its retained files document why the
subject-bearing gate was added; they do not describe the corrected corpus.

### Advanced combat-action admission

The coarse action-number taxonomy contains materially more combat evidence than
the first primary-motion model uses: 2,603 normal attacks, 710 special attacks,
123 super attacks, and 129 blocks before subject/pixel gating.  Exact all-frame
pixel gating plus one representative per identity/verb leaves 217 normal attacks,
132 special attacks, 23 super attacks, and 48 blocks.  These counts are sufficient
to investigate separate combat tokens, but an action-number tier alone does not
prove that the visible clip is a clean primary-fighter motion rather than a full
screen effect, assist, transformation, or projectile-only sequence.

`mugen-mffa-combat-motion-role-vlm-sample-v1.json` therefore selects 32 stable,
all-frame pixel-passing examples for each of `block`, `normal_attack`,
`special_attack`, and `super_attack`, stratified across the immutable identity
splits where available.  Its 128 contact sheets are intended for the pinned
Qwen-3.5-122B same-subject/action-match audit.  No special/super token is admitted
to the canonical trainer until that audit passes; rejected and ambiguous clips
remain indexed instead of being silently coerced.

## Dense standard-schema reset (2026-08-14)

The earlier broad-motion experiments were data-pipeline diagnostics, not a fair
capacity test of the intended model. They mixed sparse identities and uneven verbs,
then emphasized identity-held-out failures before training a genuinely dense MUGEN
distribution. That interpretation is superseded.

The corpus now follows MUGEN's native AIR schema first. Every parseable character is
indexed independently of model geometry, and the canonical dense view is idle,
walk, jump, block, attack A, and attack B. Jump and guard phase actions remain
separately addressable in the authoritative catalog. Run, hurt, death, spawn,
victory/emote, normal attacks, specials, and supers also remain indexed; a six-slot
view is a balanced training projection, never a reason to discard the other actions.

The exact MFFA schema catalog contains 212 decoded characters. Raw AIR availability
finds 207 complete six-slot characters; the native pixel materialization retains all
212 and resolves 193 complete six-slot characters after action-local pixel/transform
checks. The canonical leakage-safe fixed view contains 1,239 clips at
`data/processed/mugen-mffa-schema-core-b128-f8-v2/`, manifest SHA-256
`72731b487a2a4148b1945a67c34a6b41f96e178733dbd6917987038053be7f8b`.
The first v1 split is non-canonical because 15 exact-pixel duplicate groups crossed
split boundaries. The v2 splitter forms connected components over exact SFF and
rendered array hashes before deterministic 90/5/5 assignment; zero exact-array
duplicate group crosses its splits.

Anime All Stars 3 is pinned as archive SHA-256
`76556ed6959db685589ce3db15d1a527c270db7e08713dc853a9d0e0fb718299`
(3,775,696,702 bytes). Its extracted character subtree was verified against the
archive inventory at 3,337 files and 5,845,612,887 bytes with no traversal, symlink,
duplicate, case-collision, missing-file, or size-mismatch finding. The corrected
exact AIR catalog at
`data/index/reports/mugen-anime-all-stars-3-air-schema-catalog-v2.json` has SHA-256
`6e19f47a77603daaff16b30b87afb65e41e674b13e5ae446b9dbfc9d3c933c81`.
It resolves 120 unique fighters and 30,961 authored actions: all 120 have idle,
walk, jump, and attacks; 119 have block and therefore the complete six-slot view.
The remaining 19 DEF failures are storyboards, corrupt alternates, or unresolved
backup definitions and do not represent 19 missing primary fighters.

The native v2 materialization recovers both legacy SFFv2 decompressed-size-prefix
variants and action-locally quarantines corrupt SFFv1 nodes. It materializes all 120
fighters and 713 core slots, with 113 fighters retaining all six pixel-resolved
slots. Its manifest SHA-256 is
`755a2d58b3d267f09e270d0f08a55f675def7e871064d9c253a2b71e0b9ce1ae`.
The leakage-safe 8x128x128 view has the same 713 clips, zero cross-split exact-array
duplicate groups, and manifest SHA-256
`602540cfd15792c904caeec589bf3281877b79ad53817a54ac06fd70d15eb18e`.

The 2,000+ JUS/Chibi archive is exact size 41,804,753,407 bytes and SHA-256
`eb9983574ebc441f44d668693c402befde62aac6eaa604e615652b660e4a596a`.
Selective extraction retains 3,142 DEF plus 2,332 AIR files (197,178,514 bytes),
with no absolute/traversal path, duplicate, or case-collision finding. Explicit AIR
recovery skips 24 malformed element rows across 17 files without guessing values;
all 2,332 AIR files then parse. The DEF-to-AIR-to-archive-SFF join resolves 2,310
fighter variants, of which 2,202 provide the complete six-slot core. Resolved
variant coverage is idle 2,300, walk 2,277, jump 2,267, block 2,263, normal attack
2,258, special attack 2,087, and super attack 1,814. The compact authoritative AIR
catalog is
`data/index/reports/mugen-iidx-jus-chibi-air-schema-catalog-v1.json`, SHA-256
`4b4b9532dbf4df38d55210659fe7748ffa51503204c353aa031373dfb2defdab`.

The downloaded Anime Ascension file is exact size 98,894,887,513 bytes and SHA-256
`d3aa7e4ba16e7983851850ae1bb6d01f09eeeb8d19b05e793f31697c9ae3d142`.
It contains 27,158,350,425 trailing zero bytes. The losslessly trimmed meaningful
extent is 71,736,537,088 bytes, SHA-256
`0a16a93be8971843ea1822cffd95942364e2b9f6ce05a1dd921ce490f1a71294`;
appending that exact zero count reconstructs the downloaded bytes and hash. The RAR
has a pre-existing header error, so admission is fail-closed per member. Original
and trimmed containers expose byte-identical 7,944-member metadata inventories,
canonical inventory SHA-256
`7b20ac9e6e963997ff7e010d4645d390361f9a21ee93e1dc5e8da0f48834c130`,
and every selectively extracted member matches its declared size and CRC.

The Anime Ascension metadata tree contains 4,365 DEF and 3,567 AIR files. All AIR
files parse after evidence-retaining omission of 257 malformed rows across 41
files. The DEF-to-AIR-to-SFF join resolves 3,456 fighter variants, including 3,275
complete six-slot variants. Resolved coverage is idle 3,446, walk 3,408, jump
3,393, block 3,387, normal attack 3,353, special attack 3,182, and super attack
2,798. Its compact authoritative AIR catalog is
`data/index/reports/mugen-anime-ascension-air-schema-catalog-v1.json`, SHA-256
`5602a57b867b74324e2908fd19fcc316c2d73dfc0c8dbab58e69e1c51c7e7938`.
Combined with JUS/Chibi and Anime All Stars, the current pre-deduplication
pool is 5,886 resolved variants and 5,596 complete six-slot variants. No CMD/CNS/ST
or executable content is run. Large SFFs remain in their source archives and will
be streamed one member at a time; unreadable members are quarantined individually.

The original Ascension download also contains an exact 27,158,350,425-byte zero
tail beginning at byte 71,736,537,088. That verified range is marked sparse on
NTFS: the logical file remains 98,894,887,513 bytes and its full SHA-256 remains
`d3aa7e4ba16e7983851850ae1bb6d01f09eeeb8d19b05e793f31697c9ae3d142`,
while about 25.3 GiB of physical storage is reclaimed. The separate operation
record is
`data/index/reports/mugen-anime-ascension-original-sparse-trim-v1.json`, SHA-256
`f7ee3196baad17bcfd20796a113d67631966003e747caf357702abf313800dc7`.

Acquisition trimming is content-aware. The two large catalogs share 1,476 exact
AIR-SHA plus SFF-size/CRC candidate pairs; full streamed SFF SHA-256 is still
required before a pair is treated as an exact duplicate. A duplicate keeps both
source occurrences and their DEF identity/author evidence but writes only one
tensor set. Same-SFF variants with different AIR definitions remain distinct.
The fixed core worker emits at most six `8x128x128xRGBA` clips per admitted
variant (idle, walk, jump, block, and two pixel-distinct normal attacks), streams
one SFF into one isolated subprocess, and enforces the 100 GiB free-space floor.
This is a bounded training view; the complete AIR catalogs retain every special,
super, and phase action for later views.

Projection v1 is superseded and training-ineligible. It fitted the shared spatial
view without the two attack tracks and therefore clipped visible attack pixels in
560 of the first 2,001 attack clips inspected (243 `attack_a`, 317 `attack_b`).
Projection v2 fits one world-origin transform over every admitted core action and
rejects any nonzero visible-pixel clipping. It also prefers the standard guard-hold
action 130 over guard-start action 120 when both are available. The first 25 v2
occurrences produced 24 materialized characters, 19 complete six-slot characters,
135 hash-verified arrays, and zero clipped clips; extreme shared-view scales are
retained for broad coverage and will be separated by explicit quality tiers rather
than silently cropped or discarded.

Both SFF generations use explicit sprite-local corruption recovery in this streamed
view. A malformed PCX/RLE/LZ/PNG payload or a link to an already quarantined sprite
produces an indexed decode-exclusion row with archive index, group/image key, reason,
and literal error detail. Other valid sprites in the same SFF remain available, so an
unreferenced corrupt effect cannot discard an otherwise complete fighter. Header,
table, palette, and archive-member failures remain container-level failures; recovery
never invents pixels, repairs compressed bytes, or substitutes runtime logic.

The complete JUS/Chibi projection-v2 materialization is now published at
`data/processed/mugen-iidx-jus-chibi-schema-core-b128-f8-v2/materialization.json`,
SHA-256
`b0266d06ff45ef5fe9199d20339a02eacde49262bd23cd792cec6cc6a338cce8`.
It resolves 2,265 materialized characters and 13,092 core action clips. Exactly
2,022 characters contain all six requested slots; per-slot coverage is idle 2,161,
walk 2,196, jump 2,165, block 2,165, attack A 2,211, and attack B 2,194. Nineteen
archive occurrences are exact character duplicates and 26 are explicit exclusions.
The resumable first pass retained five failed variants; a sprite-local recovery retry
subsequently materialized all five without regenerating already valid characters.

Rights and authorship remain evidence, not inference. Each archive URL, archive hash,
member path, DEF author/name claim, AIR hash, SFF hash, action number, timing row, and
derived pixel hash stays separately indexable. Unknown fan-asset rights are not
converted into a permissive per-sprite claim.

The final streamed-v2 audit publishes two non-destructive views. The `broad` view
retains every unclipped character with a visible idle reference, including incomplete
and unusually scaled variants. The `dense` view requires all six core slots, no empty
output frame, at least three genuinely animated
slots, and at least four distinct action arrays. These are explicit training-quality
thresholds, not deletion rules; excluded rows and exact reasons remain in the audit.
Every NPY file byte hash, canonical array hash, shape, dtype, per-frame visible-pixel
count, and nonempty-frame hash is reverified before either view is published.

Scale is published as an evaluation variable rather than used as a hidden architecture
limit. The main dense-coverage tier admits every complete, unclipped fighter with no
empty output frame and a positive fitted scale; native scale is retained as exact
metadata and cannot exclude an otherwise valid character. Unlike the diagnostic dense
slice above, this coverage tier sets the dynamic-slot floor to zero and the distinct
array floor to one. Static authored actions and shared-frame aliases therefore remain
available to the base corpus, while the matched-action trainer separately requires
exact target-distinct pairs before treating two verbs as causal steering supervision.
The `scale >= 0.5` slice, where native art is reduced by no more than 2x, remains a
labeled high-fidelity evaluation/control slice only. Other scale bands may also be
reported, but none controls main-corpus admission. The broad codec/appearance view
likewise retains scale outliers; no source character is deleted. Results must be
stratified by scale so added coverage cannot masquerade as high-fidelity evidence.

Dataset splits are transitive components, not independent rows. Exact full-SFF,
complete action-array, and nonempty-frame hashes are grouped, and conservative DEF
identity labels are normalized with Unicode NFKC, case folding, and alphanumeric token
separation. Thus independently drawn variants with the same literal name (for example,
`M. Bison` and `m bison`) cannot cross train/validation/test boundaries. This is
stronger than exact-pixel deduplication but still does not claim franchise-aware alias
resolution for differently worded names.

The broad and dense autoencoder bridges are zero-copy: they point the verified generic
training loader at all selected source materialization roots instead of duplicating
clip tensors.
Until literal Spark captions are joined, its DEF-label descriptions are explicitly
`autoencoder_only`; conditional-generation trainers fail closed on that artifact.
Broad and dense views now share one split universe: transitive duplicate and literal
identity-label components are built over every broad-eligible character with an idle
reference, and the dense motion subset inherits those assignments. Captions are
generated once for the broad appearance set. A dense join may consume that verified
caption superset only when every selected variant's identity, split, reference-frame
index, and exact reference-frame hash agree; unused broad captions are counted and
cannot substitute for a missing dense caption.
The frozen codec likewise encodes the broad sequence set once. The dense motion join
accepts that latent-cache superset only after verifying the cache's source
materialization hash and every selected sequence's identity, split, source pixel
path/hash, latent geometry, and idle-reference frame. Extra broad latent rows are
counted but ignored, avoiding a second multi-gigabyte cache for the dense subset.

The stage-one still plan is appearance-only. It selects exactly one verified
premultiplied-RGBA temporal-medoid frame from the idle clip of each identity and uses
the literal visual caption plus a constant canonical full-body/transparent/side-view
format prompt. It does not attach walk, jump, block, or attack text and does not treat
the other seven logical idle frames as extra targets. Those frames remain useful to
the codec and stage-two reference-motion model. This preserves the intended boundary:
text describes who to draw; the separate motion DiT receives what that reference
should do.

The earlier MFFA and Anime All Stars schema-v2 materializations are now admitted
through a strict zero-copy compatibility view rather than being abandoned or copied.
The bridge verifies every character-to-clip join, selected record ID, shared world
transform, array byte/content hash, and literal clipping count. Missing source action
indices remain the explicit sentinel `-1`; MFFA resource titles are retained only as
provenance/leakage labels and never used as visual training captions. Per-character
quality gates still apply, so legacy clipping does not become acceptable merely
because the source predates the streamed projector.

The exact combined legacy audit contains 332 characters: 306 have all six slots,
168 pass the broad gate, and 110 pass the dense gate. The dense view contributes
660 action clips and 107 unique SFF identities. Splits are assigned over all 160
broad-eligible characters that have an idle reference, so the dense subset inherits
92 train, 11 validation, and 7 test characters. Exclusions remain indexed: 162 characters report visible
clipping, 67 fall below the view-scale threshold, 26 lack all six actions, 6 have
too few dynamic slots, 5 have too few distinct action arrays, and 2 contain empty
output frames/slots (reasons can overlap). The canonical quality audit is
`data/index/reports/mugen-legacy-six-action-quality-audit-v1.json`, SHA-256
`aef28c901ea550fefc2aba51e8041d7eda4e29e2f1cd3092a69fff4f97f7bb82`;
the current dense manifest is
`data/processed/mugen-legacy-six-action-dense-v2.json`, SHA-256
`4009847836d16907d1518f3ff3b994d5f9646d930900189f8faf5e5355f360a0`.
The earlier v1 manifest is superseded because it computed split components over the
dense subset alone instead of the common broad universe.

The six-slot view is a first dense denominator, not a fixed model vocabulary. The
official Elecbyte AIR standard assigns reserved animation numbers for standing,
turning, crouching, forward/backward walking, five jump phases, running/back-hopping,
standing/crouching/aerial guard phases, loss/win/intro/taunt, hit reactions, falling,
lying down, and recovery; it also recommends the ranges 200-999 for normal attacks,
1000-2999 for special attacks, and 3000-4999 for hyper attacks
(https://www.elecbyte.com/mugendocs/air.html). The architecture and manifest schemas
therefore retain literal action numbers and can add these verbs without changing the
DiT geometry.

Across the 5,766 pre-deduplication JUS plus Ascension variants, exact action-presence
counts already support a much wider follow-up view. Both walking actions 20/21 occur
in 5,670 variants; crouch actions 10/11/12 in 5,497; jump actions
40/41/42/43/47 in 5,563; run/hop actions 100/105 in 5,485; all three guard-hold
actions 130/131/132 in 5,477; all three guard-end actions 140/141/142 in 5,466;
and all three guard-hit actions 150/151/152 in 5,447. The light high/low/crouch hit
set 5000/5010/5020 occurs in 5,575 variants, while fall/ground/lie/get-up actions
5050/5100/5110/5120 occur in 5,626. At least one normal-range attack occurs in
5,627 variants, one special-range action in 5,269, one hyper-range action in 4,612,
one win action 180-189 in 5,447, intro 190 in 4,555, and taunt 195 in 3,108.
These are occurrence counts before exact cross-archive deduplication and pixel
renderability checks; they establish data availability, not final admission counts.
The next extended projector should use the engine-reserved labels literally and
retain generic range labels for attacks unless CMD/CNS evidence proves a more
specific move name or strength.
