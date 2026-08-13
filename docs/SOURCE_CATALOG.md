# Sprite and Animation Source Catalog

Research snapshot: **2026-08-11**. This catalog records what each source contains,
how it can be reached, what its publisher says about rights, and how useful its
provenance is. It is deliberately not a single moral or legal allow/deny list.
Discovery, acquisition method, rights observations, technical quality, and eventual
experiment membership remain separate fields in the source index.

The project is private, noncommercial research and its weights are not intended for
release. That posture is recorded as context; it does not change a source's license,
an artist's attribution request, or a platform's API and access conditions.

## Working tiers

| Tier | Meaning | Typical handling |
|---|---|---|
| `A_lossless_open` | Official or creator-controlled, lossless files with a clear public license or item-level license evidence | Retrieve originals; retain license, credits, release, path, and hash |
| `B_mixed_open_source` | Strong animation data in an open repository, but licenses are copyleft, mixed, or contain asset-level exceptions | Retrieve into a separate pool; resolve rights per file/layer before publishing artifacts |
| `C_noncommercial_reference` | High-value proprietary or ripped sprites useful for private analysis and representation research | Preserve publisher/franchise/source facts; isolate from redistributable corpus and outputs |
| `D_ugc_discovery` | Broad user-generated galleries, mirrors, APIs, or social feeds where each item has independent provenance and rights | Use official discovery interfaces where available; retain creator/source chain and per-item evidence |
| `E_video_discovery` | Preview, tutorial, showcase, and process video useful for finding packs or labeling motion | Index metadata, channel, source links, and timestamps; obtain reusable source media separately |

Rate values in [`configs/sources.toml`](../configs/sources.toml) are conservative
starting points, not assertions of permission. Adapters must still honor published
API quotas, `Retry-After`, robots instructions where applicable, and operator contact
requests.

## A — strong lossless/open sources

### Kenney

- [Asset library](https://kenney.nl/assets); [license/support](https://kenney.nl/support).
- Official packs provide original PNGs and sheets. Pixel examples include Platformer
  Art Pixel (900 files), Pixel Platformer (200), Input Prompts Pixel (800), Pixel Pack
  (98), and Pixel Shmup (128). Animation is usually represented by poses or sheet
  frames rather than a consistent animation manifest.
- Kenney states that asset-page content is CC0. Credit is optional; the Kenney logo is
  not part of the CC0 asset grant.
- Prefer official ZIPs over previews and store the pack page, archive hash, internal
  path, and release information.

### OpenGameArt

- [Advanced catalog](https://opengameart.org/art-search-advanced), [license FAQ](https://opengameart.org/content/faq), and [submission rules](https://opengameart.org/content/art-submission-guidelines).
- The catalog showed 35,733 submissions at the research snapshot, across all media.
  Sprite content includes static PNGs, sheets, frame sequences, and animated GIFs.
- License and attribution are item-specific: CC0, CC-BY, CC-BY-SA, OGA-BY, GPL, or
  combinations. The downloadable item and its Copyright/Attribution Notice are the
  evidence; a preview image is not automatically licensed with the download.
- There is no supported bulk API. An [official forum response](https://opengameart.org/forumtopic/opengameart-api)
  warns that service-impacting scraping may be blocked. Crawl slowly, cache, and
  contact the operator before sustained bulk acquisition.

### Liberated Pixel Cup / Universal LPC

- [Universal LPC generator repository](https://github.com/liberatedpixelcup/Universal-LPC-Spritesheet-Character-Generator),
  [LPC collection](https://opengameart.org/content/liberated-pixel-cup-0), and
  [original contest rules](https://lpc.opengameart.org/content/lpc-rules).
- Especially valuable structured character motion: walk, slash, thrust, spellcast,
  shoot, hurt, run, jump, and climb, with composable equipment/body layers.
- Original LPC work was dual CC-BY-SA 3.0/GPL3. The current generator is mixed CC0,
  CC-BY, CC-BY-SA, OGA-BY, and GPL. `CREDITS.csv` maps layers to author, license, and
  origin, and the generator can export credits for a selected composite.
- Pin the Git commit and keep the complete and selected credit manifests with every
  generated sheet.

### OpenDuelyst

- [Official open repository](https://github.com/open-duelyst/duelyst).
- Dense pixel unit and effects atlases plus XML/plist animation metadata. This is one
  of the strongest sources for aligned action/effect animation; no official sprite
  count is stated.
- Game resources are declared CC0 in the repository. Record the exact commit and
  retain the root license; classify logos and trademarks separately.

### Tiny Speck / Glitch

- [Items](https://github.com/tinyspeck/glitch-items), [avatars](https://github.com/tinyspeck/glitch-avatars),
  and [locations](https://github.com/tinyspeck/glitch-locations).
- Large creator-released asset collections. Avatars include animation and Flash
  source. The visual style is vector/Flash rather than strict pixel art, but the
  motion and layered character structure are useful augmentation material.
- Repositories use CC0 with documented exceptions; Glitch logo/trademark material is
  excluded, and the avatar repository identifies a ColorMatrix code exception.

### Freedoom

- [Official repository](https://github.com/freedoom/freedoom).
- Complete Doom-compatible sprite sets with rotations, movement, attack, pain, and
  death states. The project enforces original/non-ripped contributions.
- BSD 3-Clause. Preserve `COPYING`, credits, and commit with extracted assets.

### Screaming Brain Studios

- [Official downloads](https://screamingbrainstudios.com/downloads/).
- Pixel packs, tiles, kits, and an Animated section. The
  [Animated Flags Pack](https://screamingbrainstudios.itch.io/animated-flags-pack),
  for example, contains 229 shaded and 229 flat flags at 16 frames each.
- The publisher states its packs are CC0; credit is appreciated but not required.
  Retain the individual pack page and archive rather than relying only on the hub.

### Game Assets for the People

- [Official catalog](https://gameassets.joshmoody.org/).
- A small clean seed set: 56 officially stated assets spanning sprites, models,
  music, and sound; animation coverage is limited.
- Publisher states all assets are CC0 and need no attribution.

### Wikimedia Commons

- [Animated pixel art](https://commons.wikimedia.org/wiki/Category:Animated_pixel_art),
  [sprite category](https://commons.wikimedia.org/wiki/Category:Sprites_(computer_graphics)),
  [reuse guide](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia),
  and [MediaWiki API](https://commons.wikimedia.org/wiki/Commons:API/MediaWiki).
- At snapshot time the animated-pixel category had 19 direct files and the sprite
  category 32 direct files; counts are dynamic. GIF/APNG and static sprite material
  are present, including fan/game subjects.
- Every file page has independent license and attribution evidence. Query
  `categorymembers` and `imageinfo` with `extmetadata`, preserve the file-page
  revision, and screen subject/trademark facts separately from the copyright tag.
- Follow Wikimedia's [API usage](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines)
  and [descriptive User-Agent](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy)
  policies.

### FreeGameSprites

- [Catalog/about](https://freegamesprites.com/en/about).
- The site reports 20,054 sprite records and 18,258 pixel-art-tagged records, with
  PNG download and both static and animated examples. Some aggregate counters and
  promotional scope statements are inconsistent.
- The operator labels the material CC0 and requires no credit, but creator/upstream
  provenance is comparatively thin and there is no documented bulk API. Keep every
  item page and seek an operator-approved bulk route before a sustained crawl.

### SpriteCook free examples

- [Free asset repository](https://github.com/SpriteCook/spritecook-free-game-assets).
- Eight small AI-generated example groups with prompts/settings and a CC0 claim.
- Keep in a labeled synthetic-source stratum; record generator/version and any
  recoverable model/workflow details rather than mixing it silently with human art.

## B — mixed-license and open-source game repositories

| Source | Animation value | Rights and provenance facts |
|---|---|---|
| [Battle for Wesnoth](https://github.com/wesnoth/wesnoth) | Large polished directional/action frame corpus, generally separate PNG frames | [Copyright policy](https://wiki.wesnoth.org/Wesnoth:Copyrights) describes GPLv2+ and newer CC-BY-SA 4.0 visual assets. Resolve from per-file copyright data/history. |
| [Flare: Empyrean Campaign](https://github.com/flareteam/flare-game) | Isometric character/creature animations; [engine format](https://github.com/flareteam/flare-engine/wiki/Attribute-Reference) exposes frame rectangles, durations, loops, and eight directions | Art/data CC-BY-SA 3.0+. Preserve credits, repository commit, and raw animation definitions. |
| [Shattered Pixel Dungeon](https://github.com/00-Evan/shattered-pixel-dungeon) | Pixel sheets for hero, mobs, items, and effects with compact action states | GPL3 repository. Keep the complete license/credit chain and a distinct GPL experiment pool. |
| [Dungeon Crawl Stone Soup tiles](https://github.com/crawl/crawl/tree/master/crawl-ref/source/rltiles) | Large mostly-static tile vocabulary with some state variants | Mixed project history. [Art requests policy](https://github.com/crawl/crawl/wiki/Art-Requests) defaults submissions to CC0 unless specified, while existing tiles also include public-domain and GPL material. Map per file. |
| [Open Surge](https://github.com/alemart/opensurge) | High-relevance 16-bit character and effect animation | Repository contains many third-party license families. Root GPL status alone does not establish the license of every asset; preserve `/licenses` and resolve each asset group. |

### Next structured-corpus queue (audited 2026-08-12)

This queue prioritizes native action/timing metadata and entity breadth over raw
image count. Counts describe repository-head metadata audits and must be repinned to
an immutable commit during acquisition.

| Rank | Corpus | Audited target and steering value | Main ingestion caveat |
|---:|---|---|---|
| 1 | [Space Station 14](https://github.com/space-wizards/space-station-14) | Mob slice: 184 RSI packs, 1,980 named state sheets, 209 animated states, 1,563 directional states; animals, aliens, robots, pets, species, humanoid layers, demons, and elementals. RSI JSON supplies cell size, directions, delays, SPDX license, copyright, and source. | Quarantine five noncommercial packs; distinguish complete mobs from component layers; 69 packs cite tgstation and require provenance-aware deduplication. |
| 2 | [Widelands](https://github.com/widelands/widelands) | 161 worker manifests, 10,613 worker PNGs and 128 critter PNGs; roughly 2,031 deduplicated action/direction carriers for idle, walk, loaded walk, work, dig, hack, plant, harvest, fish, attack, evade, and die. | Sparse acquisition is preferable to the roughly 3.1-GiB repository; collapse resolution and player-color variants while retaining local credit evidence. |
| 3 | [The Mana World client data](https://github.com/themanaworld/tmwa-client-data) | Monster slice: 176 XML, 169 sheets, 538 explicit action declarations across 143 files; broader tree includes modular humanoids and NPCs. XML records direction, ranges, timing, one-shot termination, includes, aliases, and palettes. | File-level GPL/CC mix and incomplete early artist records require include-aware attribution and quarantine of unresolved contributors. |
| 4 | [SuperTux](https://github.com/SuperTux/supertux) | 135 creature manifests, 1,940 frame PNGs, 1,036 declared actions and about 141 normalized action labels, including locomotion, combat, traversal, sleep, reaction, and death states. | Most data art is described as CC-BY-SA, but exceptions and credit scope must be resolved; aliases and effect layers are not independent identities. |
| 5 | [/tg/station](https://github.com/tgstation/tgstation) | 1,464 DMI carriers repository-wide; the mob slice includes simple animals, humans, nonhuman players, silicon, actions, clothing, and held overlays. DMI embeds state, direction, delay, movement, loop, and rewind data. | Requires a DMI decoder and compositing model; deduplicate upstream-linked SS14 assets without collapsing either provenance chain. |
| 6 | [Valyria Tear](https://github.com/ValyriaTear/ValyriaTear) | Roughly 120-150 entity sheets and well over 100 Lua animation descriptors for humanoids, animals, undead, monsters, bosses, battle actions, movement, magic, reactions, and emotes. | Detailed mixed-license file index and extensive TMW/LPC/OGA/Allacrost derivation require component-level attribution and duplicate links. |
| 7 | [Unknown Horizons](https://github.com/unknown-horizons/unknown-horizons) | 4,682 individual frames, 585 direction sequences, 74 identity/action pairs and 27 identities, usually in eight directions. | Regular but rendered/isometric rather than strict pixel art; preserve audiovisual exceptions and group color overlays. |
| 8 | [Stendhal](https://github.com/arianne/stendhal) | 1,023 compact monster/NPC/outfit sheets spanning broad humanoid, animal, undead, demon, golem, insect, reptile, and hybrid vocabulary. | Mostly stand/walk; reconcile historical attribution paths and expect overlap with LPC, OGA, TMW, and Wesnoth. |
| 9 | [Ninja Adventure](https://pixel-boy.itch.io/ninja-adventure-asset-pack) | Current pack advertises 50+ animated characters, 30+ monsters, nine bosses, and 30+ effects under CC0. | Current archive is manually gated; the public GitHub repository is stale and must not be treated as the advertised pack. Inspect Godot `SpriteFrames` metadata. |
| 10 | [Hedgewars](https://github.com/hedgewars/hw) | 71 character action sheets plus 304 hats, with unusually rich emote and weapon-action timing tables. | Low identity diversity and layered weapons/effects; art uses GNU FDL while code metadata is GPL. |

Recommended acquisition order is SS14, Widelands, TMW, SuperTux, then tgstation,
Valyria Tear, and Unknown Horizons. Stendhal, Ninja Adventure, and Hedgewars are
diversity supplements. No corpus in this queue depends on extracting imagery from
YouTube.

For any Git repository, a public clone is not equivalent to an asset license. GitHub's
[licensing guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
notes that an unlicensed public repository remains under default copyright. A root
license detector also cannot resolve embedded third-party packs or exceptions. Pin
the commit and snapshot `LICENSE`, `COPYING`, `CREDITS`, and asset-local notices.

## C — noncommercial sprite-rip/reference sources

These sources are unusually rich in coherent characters, actions, viewing angles,
effects, and historical console palettes. They also predominantly contain publisher-
owned game graphics. Index them as `C_noncommercial_reference`, retain franchise and
rightsholder observations, and keep the source pool mechanically separate from
creator-licensed and redistributable material.

| Source | Scope and animation | Publisher/site statements and access |
|---|---|---|
| [The Spriters Resource](https://www.spriters-resource.com/) | Large sheet-oriented library of official game rips and custom/fan sprites; action coverage is excellent | [Terms](https://www.spriters-resource.com/page/tou/) and [guidelines](https://www.spriters-resource.com/page/guidelines/) say official rips remain owned by game rightsholders and the site cannot license them. Custom work may require artist credit/permission. No public corpus API; use low-rate/manual retrieval and retain game/system/submitter/sheet-page provenance. |
| [Sprite Database](https://spritedatabase.net/) | Nearly 30,000 files was officially stated in 2022; mostly commercial-game sheets with substantial animation coverage | [Contact/copyright page](https://spritedatabase.net/contact) says official graphics remain publisher/developer property and asks users not to rehost the collection. No documented API. Preserve game, platform, submitter, and page URL. |
| [PokeAPI sprite repository](https://github.com/PokeAPI/sprites) | Versioned Pokémon sprites, including GIF animation from Crystal, Gen 5, and Showdown | Its [license file](https://github.com/PokeAPI/sprites/blob/master/LICENCE.txt) says image contents are copyright The Pokémon Company even though the repository is marked CC0; [PokeAPI About](https://pokeapi.co/about) says sprites were scraped. Treat the CC0 repository declaration and underlying character/image rights as separate recorded facts. |
| [Pokémon Showdown client/assets](https://github.com/smogon/pokemon-showdown-client) | Broad battle-sprite poses and animations | The client repository explicitly excludes `/sprites/` and `/audio/` from its AGPL coverage. Record artist credits where supplied and keep media rights distinct from the open client/server code. |

## D — UGC, mirrors, and discovery systems

| Source | Useful interface / content | Terms, provenance, and recommended acquisition mode |
|---|---|---|
| [itch.io](https://itch.io/game-assets) | Very broad asset-pack marketplace; many creator-authored pixel sheets and animated previews | Publishers retain rights under the [Terms](https://itch.io/docs/legal/terms); licenses vary by pack. The [API](https://itch.io/docs/api/overview) is mainly own-account/OAuth, while browse pages expose RSS. Use RSS/page metadata to discover packs, then archive the page and included license with any official pack download. |
| [Lospec](https://lospec.com/) | Pixel-art community, palettes, tutorials, gallery | [Official API](https://api.lospec.com/docs) covers palettes/prompts rather than gallery media. [Terms](https://lospec.com/terms-and-conditions) prohibit automated gallery crawling/scraping. Use API metadata and manual creator/source discovery. |
| [PixelJoint](https://pixeljoint.com/) | High-signal pixel-art gallery and animation examples | [Terms](https://pixeljoint.com/pixels/terms.asp) retain artist copyright and restrict copying/reproduction. No official gallery API was found; index manually and obtain creator-controlled originals where possible. |
| [Hugging Face Datasets](https://huggingface.co/datasets) | Mirrors, metadata tables, synthetic sets, and frame-extracted datasets | [Dataset cards](https://huggingface.co/docs/hub/datasets-cards) are publisher-authored, and the Hub's [license guidance](https://huggingface.co/docs/hub/repositories-licenses) requires users to verify rights. Treat the license tag as a claim, not per-sample proof. |
| [OpenGameArt-CC0 mirror](https://huggingface.co/datasets/nyuuzyou/OpenGameArt-CC0) | About 7.3k OGA metadata rows | Good discovery accelerator; revalidate every record at the canonical OGA item and fetch the original file there. |
| [spraix_1024](https://huggingface.co/datasets/pawkanarek/spraix_1024) / [sprite-animation](https://huggingface.co/datasets/Loacky/sprite-animation) | 560 composites and 1,287 extracted frames respectively | Cards use GPL3 tags but aggregate multiple upstream itch/GitHub sources without a complete per-sample rights manifest. Preserve upstream mappings and treat the mirror license separately from source assets. |
| [free-to-use-pixelart](https://huggingface.co/datasets/bghira/free-to-use-pixelart) | 7,273 Pixilart metadata/API rows | Visible records include fan art and repost-credit language, so use primarily as a provenance/risk-discovery table unless an item's original creator/license can be resolved. |
| [Internet Archive](https://archive.org/) | Searchable collections of images, software, game media, videos, and historical asset archives | Use [Advanced Search](https://archive.org/advancedsearch.php) and the [Metadata API](https://archive.org/developers/metadata.html). License fields are uploader-entered; retain uploader, identifier, files, hashes, metadata, and upstream evidence. Follow the [bot/LLM guide](https://archive.org/developers/bots.html). |
| [GIPHY](https://developers.giphy.com/docs/api/endpoint) | Millions of animated GIFs/stickers with creator and source metadata | [API terms](https://support.giphy.com/hc/en-us/articles/360028134111-GIPHY-API-Terms-of-Service) restrict building a separate content database and require branding/source attribution; [user terms](https://support.giphy.com/hc/en-us/articles/360020027752-GIPHY-User-Terms-of-Service) prohibit scraping. Use API discovery and creator/source resolution, not an automated media mirror. |
| [Tenor](https://tenor.com/) | GIF/sticker search; some pixel-art animation | Google states the [Tenor API was decommissioned on 2026-06-30](https://support.google.com/tenor/answer/10455265?hl=en). Historical [API terms](https://developers.google.com/tenor/guides/api-terms) treat content as third-party and constrain permanent copies. Keep as manual discovery only. |
| [Reddit](https://www.reddit.com/) | Artist posts, pack announcements, critique/process, and animation showcases | [Data API Terms](https://redditinc.com/policies/data-api-terms) preserve UGC ownership and expressly require rightsholder permission for ML/AI training; the [User Agreement](https://redditinc.com/policies/user-agreement) restricts unauthorized scraping. Use approved API/search discovery, then preserve post, author, outbound origin, and creator permission/license. |
| [Bluesky / AT Protocol](https://bsky.app/) | Public artist feeds with stable DID, AT URI, CID, embeds, and source links | Users retain ownership under [Terms](https://bsky.social/about/support/tos). The [author-feed API](https://docs.bsky.app/docs/api/app-bsky-feed-get-author-feed) and [repository/firehose protocol](https://atproto.com/specs/sync) provide strong metadata provenance but no blanket media license; mirror deletion/account status. |
| [Tumblr](https://www.tumblr.com/) | GIF-rich art blogs, reblogs, process posts, and asset showcases | Use the [official API](https://www.tumblr.com/docs/en/api/v2) for discovery. [Terms](https://www.tumblr.com/policy/terms-of-service) retain creator rights and restrict unauthorized scraping. Preserve original-blog/reblog/source chain rather than treating the latest reblog as author. |
| [GitHub repository discovery](https://github.com/topics/pixel-art) | Open-game repositories, pack archives, tools, metadata, and explicit license files | Use the [REST API](https://docs.github.com/en/rest) and license search for discovery, obey [rate-limit guidance](https://docs.github.com/en/rest/rate-limit/rate-limit), then inspect asset-local notices and exact commits. A root license is discovery evidence, not automatic coverage of every file. |

## E — YouTube video discovery

- Use the official [Data API `search.list`](https://developers.google.com/youtube/v3/docs/search/list)
  with `type=video` and optionally `videoLicense=creativeCommon`; resolve exact fields
  with [`videos.list`](https://developers.google.com/youtube/v3/docs/videos/list) using
  `snippet,status,contentDetails`.
- Capture video ID/watch URL, channel ID/name/URL, title, description, tags,
  publication time, duration, `status.license`, API ETag/retrieval time, relevant
  timestamp ranges, and creator-linked source pack/license.
- [YouTube license help](https://support.google.com/youtube/answer/2797468?hl=en)
  identifies `creativeCommon` as CC-BY and `youtube` as the Standard license. A label
  does not establish that an uploader owns every sprite, game capture, character,
  or soundtrack visible in the video.
- [Developer Policies](https://developers.google.com/youtube/terms/developer-policies)
  prohibit downloading, caching, separating, or modifying audiovisual content
  through an API client without the required approval and impose metadata refresh
  and deletion rules. Therefore the adapter is metadata/timestamp discovery only;
  reusable source media should come from the creator's linked pack/repository or a
  separately supplied file.
- High-yield searches include `pixel art sprite animation showcase`, `sprite sheet
  animation preview`, `pixel art walk cycle idle run attack death`, `animated game
  asset pack CC0`, `pixel art sprites CC BY`, `OpenGameArt animated sprite`,
  `Liberated Pixel Cup animation`, and `Aseprite walk cycle tutorial`.
- Prioritize asset authors/studios, open-source game maintainers, pack demos, and
  process/tutorial artists. Gameplay, trailers, fan-animation, ROM/rip, and
  copyrighted-character showcase channels are still useful discovery/reference
  signals but usually have weaker creator-to-source provenance.

## Required source-index evidence

Every retrieved carrier or discovery record should be able to represent:

- canonical item URL, platform source ID, creator/publisher and stable account ID;
- upstream origin URL and complete reblog/mirror/derivative chain when known;
- repository commit, page revision, archive release, or video ID and timestamp range;
- retrieval time, access adapter, response metadata, robots/terms/API evidence date;
- original filename/archive member, exact SHA-256, MIME type, dimensions, palette,
  frame count, frame durations, loop behavior, action, direction, and sequence group;
- raw title, description, tags, attribution notice, license text/URL, and normalized
  SPDX value only when justified;
- independent observations for copyright, trademark/franchise, fan-art/rip status,
  AI/synthetic origin, NSFW content, acquisition conditions, and provenance quality;
- exact/perceptual duplicate links without discarding any source or attribution edge;
- pool assignment, reviewer/rationale, experiment membership, and deletion/takedown
  status as changeable facts rather than destructive rewrites of history.

The highest-value operational rule is simple: keep the original bytes and the raw
evidence that explained where they came from. Normalized frames, generated captions,
and model-ready shards should always point back to that immutable carrier record.

## Verifiable report bundles

`spritelab reports export --output <new-directory>` renders the source registry,
complete inventory/rights JSONL, attribution Markdown, and corpus summary from one
SQLite query-only read transaction. `bundle-manifest.json` records the byte size,
SHA-256, and media type of every report file plus the exact database schema versions.
The manifest uses relative paths and no runtime timestamp, so an unchanged logical
index produces a byte-stable, portable citation bundle. It hashes the five report
payloads but does not claim a self-hash; hash the manifest file itself when pinning a
particular exported bundle.

The first checksum-bound live export after Flare, Widelands, and SS14 integration
is `data/index/reports/provenance-v3-checksummed-post-ss14`. Its
`bundle-manifest.json` SHA-256 is
`1e664fc065b47bea2296511d41269c9a22820cae8bba575478da1111342b070d`.
It binds five payloads totaling 402,480,780 bytes, including 401,855,384 bytes of
complete inventory JSONL, and records database schema migrations 1 through 5.

The post-TMWA schema-v2 export is
`data/index/reports/provenance-v5-checksummed-post-tmwa`. Its
`bundle-manifest.json` SHA-256 is
`1f0d678822f3ed36e0bbc374dc24b947b9c4fd36da958b806bd7325f8171aeff`.
All five manifest-bound payloads rehash exactly and total 402,480,925 bytes. Report
schema v2 adds the authoritative sequence-graph counts that schema v1 omitted:
25,186 sequences, 158,300 `sequence_frames`, 306,274 `sequence_occurrences`,
128,128 `sequence_subjects`, and 25,186 `sequence_source_keys`. The legacy `frames`
table remains separately reported as zero rather than being conflated with the
ordered `sequence_frames` table. The underlying checkpointed live SQLite file was
2,349,240,320 bytes with SHA-256
`840530fefbe869495127683e1b36208a1fdafc78edf331d8b1dbbc45c4ed7145`, and
`PRAGMA quick_check` returned `ok` immediately before this export.

`data/index/reports/provenance-v6-source-sequence-counts-post-tmwa` extends that
summary with source-resolved entity, sequence, ordered-frame, occurrence, and subject
counts. Its schema-v3 `bundle-manifest.json` SHA-256 is
`9662aa048e813a8b81bdafc265758fdb742b9fe56309bab8ceebd62c54c911f0`;
the five verified payloads total 402,495,612 bytes. For example, the report now
attributes 853 sequences / 4,153 frames to TMWA, 193 / 3,272 to Widelands, 246 / 297
to SS14, and 15,840 / 71,064 to Flare while preserving occurrence and subject edges
separately. These counts use `sequence_source_keys.source_id`, not an inferred item
or license inheritance rule.

The first complete post-SuperTux export is
`data/index/reports/provenance-v8-source-sequence-counts-post-supertux`. Its
schema-v3 `bundle-manifest.json` SHA-256 is
`64b8f0210583cbd5d994114a0490afa55d7821e115e12df19fdeceaee6e2a8e6`.
All five manifest-bound payloads rehash exactly and total 411,066,062 bytes. The
single query-only snapshot contains 26,196 sequences, 165,403 ordered frames,
318,009 archive/resource occurrences, 129,138 subject edges, and 5,111 entities.
The SuperTux source contributes 96 entities, 1,010 sequences, 7,103 ordered frames,
11,735 occurrences, and 1,010 subject edges. Its two rights observations remain
separately scoped; these counts do not imply that either rights claim applies to
every PNG. The checkpointed live database was 2,441,465,856 bytes with SHA-256
`6f3802aaed83cdc1ee4e3a8a3588328433f11b8c5b708c2aa1bb7c6a6a1dca87`, and
`PRAGMA quick_check` returned `ok` immediately before the export.
