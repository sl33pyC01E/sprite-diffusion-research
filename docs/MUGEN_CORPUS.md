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
evidence but are not executed or used as training data. RAR and 7z packs remain
metadata-only until an equivalently hardened extractor exists.

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
