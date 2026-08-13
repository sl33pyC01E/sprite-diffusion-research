export type SourceStage =
  | "model-ready"
  | "materialized"
  | "indexed"
  | "audited"
  | "blocked";

export type SourceRecord = {
  id: string;
  name: string;
  kind: string;
  entities: number;
  sequences: number;
  frames: number;
  occurrences: number;
  rights: number;
  stage: SourceStage;
  note: string;
};

export type GalleryRecord = {
  id: string;
  identity: string;
  entity: string;
  action: string;
  direction: string;
  run: string;
  target?: string;
  endpoint: string;
  euler?: string;
  label: string;
  caveat: string;
};

export const indexSnapshot = {
  exportedAt: "2026-08-12",
  bundleSha256:
    "64b8f0210583cbd5d994114a0490afa55d7821e115e12df19fdeceaee6e2a8e6",
  databaseSha256:
    "6f3802aaed83cdc1ee4e3a8a3588328433f11b8c5b708c2aa1bb7c6a6a1dca87",
  counts: {
    sources: 48,
    indexedSources: 12,
    entities: 5111,
    sequences: 26196,
    frames: 165403,
    occurrences: 318009,
    archiveMembers: 237749,
    blobs: 126712,
    mediaObservations: 123702,
    rightsObservations: 18,
  },
};

export const sources: SourceRecord[] = [
  {
    id: "flare_empyrean",
    name: "Flare: Empyrean Campaign",
    kind: "game animation definitions",
    entities: 1650,
    sequences: 15840,
    frames: 71064,
    occurrences: 202240,
    rights: 2,
    stage: "indexed",
    note: "Eight-direction fantasy entities; attachments remain separately modeled.",
  },
  {
    id: "openduelyst",
    name: "OpenDuelyst",
    kind: "TexturePacker atlases",
    entities: 2352,
    sequences: 5302,
    frames: 69020,
    occurrences: 77363,
    rights: 2,
    stage: "indexed",
    note: "Large atlas corpus with exact trim, rotation, alias, and role evidence.",
  },
  {
    id: "supertux",
    name: "SuperTux",
    kind: "creature manifests",
    entities: 96,
    sequences: 1010,
    frames: 7103,
    occurrences: 11735,
    rights: 2,
    stage: "blocked",
    note: "Indexed; model admission waits on exact flip and caller-controlled loop support.",
  },
  {
    id: "open_surge",
    name: "Open Surge",
    kind: "sprite declarations",
    entities: 353,
    sequences: 878,
    frames: 3484,
    occurrences: 4390,
    rights: 2,
    stage: "materialized",
    note: "Exact magenta color-key transform is now engine-evidenced and hash-bound.",
  },
  {
    id: "tmwa_client_data",
    name: "The Mana World",
    kind: "XML sprite definitions",
    entities: 107,
    sequences: 853,
    frames: 4153,
    occurrences: 5234,
    rights: 1,
    stage: "model-ready",
    note: "540 conservative model-ready multi-frame clips; current causal16 experiments use 16.",
  },
  {
    id: "wesnoth",
    name: "Battle for Wesnoth",
    kind: "WML frame sequences",
    entities: 248,
    sequences: 604,
    frames: 2526,
    occurrences: 6066,
    rights: 3,
    stage: "indexed",
    note: "Strict literal-animation subset; macros and conditional transforms stay quarantined.",
  },
  {
    id: "shattered_pixel_dungeon",
    name: "Shattered Pixel Dungeon",
    kind: "Java-defined sheets",
    entities: 103,
    sequences: 631,
    frames: 2439,
    occurrences: 4447,
    rights: 1,
    stage: "indexed",
    note: "Runtime inheritance and action timelines resolved from pinned source.",
  },
  {
    id: "freedoom",
    name: "Freedoom",
    kind: "Doom sprite families",
    entities: 19,
    sequences: 590,
    frames: 1633,
    occurrences: 1615,
    rights: 1,
    stage: "indexed",
    note: "Pose-heavy directional sprite families; timing remains deliberately unknown.",
  },
  {
    id: "space_station_14",
    name: "Space Station 14",
    kind: "RSI mob states",
    entities: 139,
    sequences: 246,
    frames: 297,
    occurrences: 2706,
    rights: 1,
    stage: "indexed",
    note: "Per-pack SPDX evidence; modular layers and noncanonical actions excluded.",
  },
  {
    id: "widelands",
    name: "Widelands",
    kind: "Lua workers & critters",
    entities: 22,
    sequences: 193,
    frames: 3272,
    occurrences: 2123,
    rights: 1,
    stage: "materialized",
    note: "Exact numbered-file timelines; action aliases are excluded from causal supervision.",
  },
  {
    id: "spritecook_free",
    name: "SpriteCook Free Assets",
    kind: "curated pixel sheets",
    entities: 22,
    sequences: 49,
    frames: 412,
    occurrences: 90,
    rights: 1,
    stage: "model-ready",
    note: "Small clean temporal corpus with 49 distinct sequences and seven action labels.",
  },
  {
    id: "universal_lpc",
    name: "Universal LPC",
    kind: "modular layer sheets",
    entities: 0,
    sequences: 0,
    frames: 0,
    occurrences: 0,
    rights: 1,
    stage: "audited",
    note: "87,962 layer sheets stream through a separate compositing manifest—not complete entities.",
  },
  {
    id: "tgstation",
    name: "/tg/station",
    kind: "DMI mob states",
    entities: 1150,
    sequences: 1680,
    frames: 60051,
    occurrences: 60051,
    rights: 0,
    stage: "audited",
    note: "Pinned pure audit: 401 DMIs and 1,680 whole-entity sequences; projection is not yet live.",
  },
];

export const actionCoverage = [
  ["idle", 3956],
  ["death", 3275],
  ["attack", 3144],
  ["hurt", 2820],
  ["run", 2212],
  ["cast", 1874],
  ["defend", 1874],
  ["shoot", 1801],
  ["walk", 1141],
  ["emote", 65],
] as const;

export const entityCoverage = [
  ["unknown", 3118],
  ["object", 1193],
  ["humanoid", 236],
  ["animal", 208],
  ["monster", 162],
  ["creature", 133],
  ["effect", 39],
  ["robot", 14],
] as const;

export const gallery: GalleryRecord[] = [
  {
    id: "sasquatch-walk",
    identity: "Sasquatch",
    entity: "monster",
    action: "walk",
    direction: "down",
    run: "TMWA causal16 · 2,000 steps",
    target: "/gallery/sasquatch-walk-target.png",
    endpoint: "/gallery/sasquatch-walk-endpoint.png",
    euler: "/gallery/sasquatch-walk-euler.png",
    label: "Matched target / generated walk",
    caveat: "Exact training identity and action; in-sample reconstruction, not novel generation.",
  },
  {
    id: "sasquatch-idle",
    identity: "Sasquatch",
    entity: "monster",
    action: "idle",
    direction: "down",
    run: "TMWA causal16 · 2,000 steps",
    target: "/gallery/sasquatch-idle-target.png",
    endpoint: "/gallery/sasquatch-idle-endpoint.png",
    label: "Matched target / generated idle",
    caveat: "Same identity, noise contract, canvas, phases, and decoder as the walk comparison.",
  },
  {
    id: "penguin-walk",
    identity: "Penguin",
    entity: "animal",
    action: "walk",
    direction: "down",
    run: "TMWA causal16 · 2,000 steps",
    target: "/gallery/penguin-walk-target.png",
    endpoint: "/gallery/penguin-walk-endpoint.png",
    euler: "/gallery/penguin-walk-euler.png",
    label: "Matched target / generated walk",
    caveat: "Endpoint sampling is numerically stronger, but visible output remains soft and speckled.",
  },
  {
    id: "logmonster-walk",
    identity: "Logmonster",
    entity: "monster",
    action: "walk",
    direction: "down",
    run: "TMWA causal16 · 2,000 steps",
    target: "/gallery/logmonster-walk-target.png",
    endpoint: "/gallery/logmonster-walk-endpoint.png",
    label: "Strongest causal pair",
    caveat: "This pair has the strongest target-distinct action response in the bounded batch.",
  },
  {
    id: "skull-ice-walk",
    identity: "Ice skull",
    entity: "monster",
    action: "walk",
    direction: "down",
    run: "TMWA causal16 · 2,000 steps",
    target: "/gallery/skull-ice-walk-target.png",
    endpoint: "/gallery/skull-ice-walk-endpoint.png",
    label: "High separation retention",
    caveat: "Generated idle/walk separation retains 88.5% of this target pair's distance.",
  },
  {
    id: "fetid-rat-idle",
    identity: "Fetid Rat",
    entity: "animal",
    action: "idle",
    direction: "unknown",
    run: "Fetid Rat · 1,500 steps · palette32",
    endpoint: "/gallery/fetid-rat-idle.png",
    label: "Decoded in-sample action",
    caveat: "Hard-alpha and 32-color display derivative; raw continuous RGBA remains canonical.",
  },
  {
    id: "fetid-rat-run",
    identity: "Fetid Rat",
    entity: "animal",
    action: "run",
    direction: "unknown",
    run: "Fetid Rat · 1,500 steps · palette32",
    endpoint: "/gallery/fetid-rat-run.png",
    label: "Decoded in-sample action",
    caveat: "One of four memorized actions sharing identity and matched-noise diagnostics.",
  },
  {
    id: "fetid-rat-attack",
    identity: "Fetid Rat",
    entity: "animal",
    action: "attack",
    direction: "unknown",
    run: "Fetid Rat · 1,500 steps · palette32",
    endpoint: "/gallery/fetid-rat-attack.png",
    label: "Decoded in-sample action",
    caveat: "Action-sensitive proof, not open-vocabulary semantics.",
  },
  {
    id: "fetid-rat-death",
    identity: "Fetid Rat",
    entity: "animal",
    action: "death",
    direction: "unknown",
    run: "Fetid Rat · 1,500 steps · palette32",
    endpoint: "/gallery/fetid-rat-death.png",
    label: "Decoded one-shot action",
    caveat: "Display derivative preserves eight authored output slots.",
  },
];

export const experimentSteps = [
  {
    step: 1000,
    pmMae: 0.051985,
    alphaIou: 0.795,
    separation: 37.52,
    idleToWalk: 0,
    walkCorrect: 1,
    endpointLoss: 0.115317,
    report: "147acc50a41b9bbffd905d0636520aeaebfc73a45755982a64d235dc5ac9fdbf",
  },
  {
    step: 2000,
    pmMae: 0.041747,
    alphaIou: 0.863808,
    separation: 61.79,
    idleToWalk: 6,
    walkCorrect: 4,
    endpointLoss: 0.082645,
    report: "80dab913e390e8bed42241f383b8b327d8053575a76f746a07b34dc2ebca4fa0",
  },
] as const;

export const causalGates = [
  { label: "PM-RGBA error", observed: "0.04175", threshold: "< 0.05199", pass: true },
  { label: "Action separation", observed: "61.79%", threshold: "> 37.52%", pass: true },
  { label: "Idle → walk movement", observed: "6 / 8", threshold: "> 0 / 8", pass: true },
  { label: "Walk target preference", observed: "4 / 8", threshold: "> 1 / 8", pass: true },
] as const;

export const pipelineStages = [
  {
    id: "discover",
    index: "01",
    title: "Discover",
    status: "operational",
    summary: "Registered adapters and metadata-only discovery collect stable source identities.",
    details: "48 registered source systems; YouTube is discovery metadata only, never a media downloader.",
  },
  {
    id: "acquire",
    index: "02",
    title: "Acquire + CAS",
    status: "operational",
    summary: "Guarded downloads become immutable SHA-256 objects before parsing.",
    details: "126,712 blobs; every write checks the 100 GiB free-space floor.",
  },
  {
    id: "inspect",
    index: "03",
    title: "Inspect",
    status: "operational",
    summary: "Archives, PNG/APNG/GIF media, sheets, palettes, and unsafe members are audited.",
    details: "237,749 archive members and 123,702 media observations in the latest bundle.",
  },
  {
    id: "project",
    index: "04",
    title: "Project",
    status: "operational",
    summary: "Source-specific runtime semantics become entities, actions, timelines, and evidence edges.",
    details: "26,196 indexed sequences; uncertain transforms and loop semantics fail closed.",
  },
  {
    id: "snapshot",
    index: "05",
    title: "Snapshot",
    status: "operational",
    summary: "Leakage-aware grouped splits preserve identity, pack, blob, and duplicate components.",
    details: "Temporal, pose-only, source-filtered, and model-ready policies are separately explicit.",
  },
  {
    id: "materialize",
    index: "06",
    title: "Materialize",
    status: "operational",
    summary: "Exact source rectangles become hash-verified RGBA tensors without interpolation.",
    details: "Native buckets, pixel transforms, phases, timing, and every pre/post hash remain linked.",
  },
  {
    id: "train",
    index: "07",
    title: "PixelDiT",
    status: "research",
    summary: "Native RGBA rectified flow with spatial/temporal attention and explicit control tokens.",
    details: "Current evidence is bounded memorization; semantic text and held-out generalization remain open.",
  },
  {
    id: "evaluate",
    index: "08",
    title: "Evaluate",
    status: "operational",
    summary: "Matched noise, target pairing, alpha, temporal, and causal action-swap metrics.",
    details: "Every preview points to canonical arrays; display decodes never replace raw evidence.",
  },
] as const;

export const evidenceHashes = [
  ["Provenance bundle", "64b8f0210583cbd5d994114a0490afa55d7821e115e12df19fdeceaee6e2a8e6"],
  ["Live index checkpoint", "6f3802aaed83cdc1ee4e3a8a3588328433f11b8c5b708c2aa1bb7c6a6a1dca87"],
  ["TMWA 2k checkpoint", "5b944cb24ded9a046a2e3d7af7a8d2264eb673added068f5f6218039d19cf8c8"],
  ["TMWA 2k matched eval", "2491272f1abf91784c4f2d4b3e066bf381f7bfe2ad9b4099f364e10600c4c15c"],
  ["TMWA 2k causal audit", "36679a485fd68673852f3866445fed7e3c1333c2e84d17b3ae1fe94e011dde66"],
  ["Preview index", "675660ca488f45d81f843efe2b490c8a10e577be39cb4244370ce531bc22b176"],
] as const;
