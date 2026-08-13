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
