"use client";

/* Animated PNGs must bypass framework image optimization to preserve playback. */
/* eslint-disable @next/next/no-img-element */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  actionCoverage,
  causalGates,
  entityCoverage,
  evidenceHashes,
  experimentSteps,
  gallery,
  indexSnapshot,
  pipelineStages,
  sources,
  type GalleryRecord,
  type SourceStage,
} from "./data";

type View = "studio" | "gallery" | "experiments" | "corpus" | "pipeline" | "evidence";
type Sampler = "endpoint" | "euler";

const views: { id: View; key: string; label: string; short: string }[] = [
  { id: "studio", key: "1", label: "Studio", short: "ST" },
  { id: "gallery", key: "2", label: "Gallery", short: "GL" },
  { id: "experiments", key: "3", label: "Experiments", short: "EX" },
  { id: "corpus", key: "4", label: "Corpus", short: "CO" },
  { id: "pipeline", key: "5", label: "Pipeline", short: "PL" },
  { id: "evidence", key: "6", label: "Evidence", short: "EV" },
];

const stageLabels: Record<SourceStage, string> = {
  "model-ready": "Model-ready",
  materialized: "Materialized",
  indexed: "Indexed",
  audited: "Audited",
  blocked: "Held back",
};

const formatter = new Intl.NumberFormat("en-US");
const formatNumber = (value: number) => formatter.format(value);
const shortNumber = (value: number) =>
  value >= 1_000_000
    ? `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 1 : 2)}m`
    : value >= 1_000
      ? `${(value / 1_000).toFixed(value >= 10_000 ? 1 : 2)}k`
      : String(value);
const compactHash = (value: string) => `${value.slice(0, 12)}…${value.slice(-8)}`;

function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: string }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function Metric({ label, value, detail, accent = false }: { label: string; value: string; detail: string; accent?: boolean }) {
  return (
    <div className={`metric ${accent ? "metric-accent" : ""}`}>
      <span>{label}</span><strong>{value}</strong><small>{detail}</small>
    </div>
  );
}

function SpriteFrame({ src, alt, compact = false }: { src: string; alt: string; compact?: boolean }) {
  return (
    <div className={`sprite-frame ${compact ? "sprite-frame-compact" : ""}`}>
      <div className="checkerboard" />
      <img src={src} alt={alt} draggable={false} />
      <span className="frame-corner frame-corner-a" /><span className="frame-corner frame-corner-b" />
    </div>
  );
}

function CompareSlider({ record, sampler }: { record: GalleryRecord; sampler: Sampler }) {
  const [reveal, setReveal] = useState(50);
  const generated = sampler === "euler" && record.euler ? record.euler : record.endpoint;
  if (!record.target) return <SpriteFrame src={generated} alt={`${record.identity} ${record.action} generated animation`} />;
  return (
    <div className="compare-shell">
      <div className="compare-canvas">
        <div className="checkerboard" />
        <img className="compare-image" src={record.target} alt={`${record.identity} ${record.action} target`} draggable={false} />
        <div className="compare-generated" style={{ clipPath: `inset(0 ${100 - reveal}% 0 0)` }}>
          <img className="compare-image" src={generated} alt={`${record.identity} ${record.action} generated`} draggable={false} />
        </div>
        <div className="compare-line" style={{ left: `${reveal}%` }} aria-hidden="true"><span>↔</span></div>
        <span className="compare-label compare-label-left">generated</span>
        <span className="compare-label compare-label-right">target</span>
      </div>
      <label className="range-label">
        <span>Reveal comparison</span>
        <input type="range" min="0" max="100" value={reveal} onChange={(event) => setReveal(Number(event.target.value))} aria-label="Reveal generated versus target animation" />
        <output>{reveal}%</output>
      </label>
    </div>
  );
}

function StudioView() {
  const [identity, setIdentity] = useState("Sasquatch");
  const [action, setAction] = useState("walk");
  const [direction, setDirection] = useState("down");
  const [sampler, setSampler] = useState<Sampler>("endpoint");
  const [prompt, setPrompt] = useState("a shaggy forest sasquatch, chunky pixel art");
  const [active, setActive] = useState(gallery[0]);
  const [status, setStatus] = useState("Verified artifact loaded");
  const identities = useMemo(() => [...new Set(gallery.map((item) => item.identity))], []);
  const actions = useMemo(
    () => [...new Set(gallery.filter((item) => item.identity === identity).map((item) => item.action))],
    [identity],
  );

  function replay() {
    const candidate = gallery.find((item) => item.identity === identity && item.action === action) ?? gallery.find((item) => item.identity === identity) ?? gallery[0];
    setActive(candidate);
    if (sampler === "euler" && !candidate.euler) setSampler("endpoint");
    const directionNote = direction === candidate.direction
      ? ""
      : ` · requested ${direction}; showing verified ${candidate.direction}`;
    setStatus(`Bound to ${candidate.run}${directionNote}`);
  }

  return (
    <section className="view studio-view" aria-labelledby="studio-title">
      <div className="studio-copy">
        <div className="eyebrow"><span>01</span> CONDITIONED REPLAY LAB</div>
        <h1 id="studio-title">Describe the sprite.<br /><em>Steer the motion.</em></h1>
        <p className="lede">Explore identity text, entity class, action, direction, loop, phase, and sampler against hash-verified research outputs.</p>
        <div className="boundary-note"><span className="pulse-dot" /><div><strong>Hosted mode: verified replay</strong><small>Controls select completed artifacts. Live checkpoint inference remains local-GPU research.</small></div></div>
        <div className="studio-form">
          <label className="field field-wide"><span>Identity description</span><textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={3} maxLength={160} /><small>{prompt.length}/160 · text is not yet semantically generalizing</small></label>
          <label className="field"><span>Verified identity</span><select value={identity} onChange={(event) => { const nextIdentity = event.target.value; setIdentity(nextIdentity); setAction(gallery.find((item) => item.identity === nextIdentity)?.action ?? "idle"); }}>{identities.map((value) => <option key={value}>{value}</option>)}</select></label>
          <label className="field"><span>Direction</span><select value={direction} onChange={(event) => setDirection(event.target.value)}><option value="down">down</option><option value="left">left</option><option value="right">right</option><option value="up">up</option><option value="unknown">unknown</option></select></label>
          <fieldset className="field field-wide chip-field"><legend>Action</legend><div className="chip-row">{actions.map((value) => <button type="button" key={value} className={action === value ? "chip active" : "chip"} onClick={() => setAction(value)}>{value}</button>)}</div></fieldset>
          <fieldset className="field field-wide chip-field"><legend>Sampler</legend><div className="segmented"><button type="button" className={sampler === "endpoint" ? "active" : ""} onClick={() => setSampler("endpoint")}>Endpoint · 1 step</button><button type="button" className={sampler === "euler" ? "active" : ""} onClick={() => setSampler("euler")}>Euler · 32 steps</button></div></fieldset>
          <button type="button" className="primary-button" onClick={replay}><span>Replay verified artifact</span><b>↗</b></button>
          <div className="form-status" aria-live="polite"><span>●</span>{status}</div>
        </div>
      </div>
      <div className="studio-output">
        <div className="output-topline"><span>OUTPUT / 8F · 64² · RGBA</span><Badge tone="amber">IN-SAMPLE</Badge></div>
        <CompareSlider record={active} sampler={sampler} />
        <div className="output-title"><div><span>{active.entity} / {active.direction}</span><h2>{active.identity} · {active.action}</h2></div><span className="loop-mark">∞</span></div>
        <div className="output-facts"><div><span>Run</span><strong>{active.run}</strong></div><div><span>Decoder</span><strong>{sampler === "euler" && active.euler ? "Euler 32" : "Endpoint 1"}</strong></div><div><span>Claim</span><strong>Memorization diagnostic</strong></div></div>
        <p className="caveat">{active.caveat}</p>
      </div>
    </section>
  );
}

function GalleryView() {
  const [query, setQuery] = useState("");
  const [action, setAction] = useState("all");
  const [sampler, setSampler] = useState<Sampler>("endpoint");
  const actions = ["all", ...new Set(gallery.map((item) => item.action))];
  const filtered = gallery.filter((item) => `${item.identity} ${item.entity} ${item.action} ${item.run}`.toLowerCase().includes(query.toLowerCase()) && (action === "all" || item.action === action));
  return (
    <section className="view" aria-labelledby="gallery-title">
      <header className="section-header"><div><div className="eyebrow"><span>02</span> ANIMATION GALLERY</div><h1 id="gallery-title">Inspect the pixels,<br /><em>not just the score.</em></h1></div><p>Every animation is a hash-verified derivative of a canonical local array. Target comparisons share identity, action, phases, and canvas.</p></header>
      <div className="toolbar"><label className="search-box"><span>⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search identity, class, or run" /></label><div className="chip-row toolbar-chips">{actions.map((value) => <button key={value} type="button" className={action === value ? "chip active" : "chip"} onClick={() => setAction(value)}>{value}</button>)}</div><div className="segmented compact"><button type="button" className={sampler === "endpoint" ? "active" : ""} onClick={() => setSampler("endpoint")}>Endpoint</button><button type="button" className={sampler === "euler" ? "active" : ""} onClick={() => setSampler("euler")}>Euler</button></div><a className="text-button" href="/gallery/manifest.json" download>Manifest ↓</a></div>
      <div className="gallery-grid">{filtered.map((item) => { const src = sampler === "euler" && item.euler ? item.euler : item.endpoint; return <article className="gallery-card" key={item.id}><SpriteFrame src={src} alt={`${item.identity} ${item.action} animation`} compact /><div className="gallery-card-body"><div className="gallery-meta"><Badge tone={item.target ? "green" : "purple"}>{item.target ? "MATCHED" : "DECODED"}</Badge><span>{item.entity} · {item.direction}</span></div><h2>{item.identity}</h2><h3>{item.action}</h3><p>{item.label}</p><small>{item.caveat}</small></div></article>; })}</div>
      {!filtered.length && <div className="empty-state">No verified artifacts match this filter.</div>}
    </section>
  );
}

function ExperimentsView() {
  const [step, setStep] = useState<(typeof experimentSteps)[number]["step"]>(6000);
  const current = experimentSteps.find((item) => item.step === step)!;
  const before = experimentSteps[0];
  return (
    <section className="view" aria-labelledby="experiments-title">
      <header className="section-header"><div><div className="eyebrow"><span>03</span> MODEL EVIDENCE</div><h1 id="experiments-title">Causal controls,<br /><em>not victory laps.</em></h1></div><p>The current checkpoint learned a stronger idle/walk distinction on eight training identities. Held-out identity and open-vocabulary text remain unproven.</p></header>
      <div className="experiment-layout">
        <div className="experiment-main panel">
          <div className="panel-top"><div><span>TMWA CAUSAL16</span><h2>Continuation curve</h2></div><div className="step-switch">{experimentSteps.map((item) => <button type="button" key={item.step} className={step === item.step ? "active" : ""} onClick={() => setStep(item.step)}>{item.step.toLocaleString()} steps</button>)}</div></div>
          <div className="big-score"><span>PM-RGBA error</span><strong>{current.pmMae.toFixed(5)}</strong><small>{step >= 3000 ? "alpha-channel weight 4 · endpoint sampler" : "alpha-weight-one baseline"}</small></div>
          <div className="bar-chart" aria-label="Premultiplied RGBA error by training step">{experimentSteps.map((item) => <div className="bar-column" key={item.step}><div className="bar-track"><div className={item.step === step ? "bar-fill active" : "bar-fill"} style={{ height: `${(item.pmMae / 0.06) * 100}%` }}><span>{item.pmMae.toFixed(4)}</span></div></div><b>{item.step / 1000}k</b></div>)}</div>
          <div className="experiment-metrics"><Metric label="Alpha IoU @127" value={(current.alphaIou * 100).toFixed(1) + "%"} detail="matched target silhouette" accent /><Metric label="Action separation" value={current.separation.toFixed(1) + "%"} detail="of target pair distance" /><Metric label="Walk preference" value={`${current.walkCorrect} / 8`} detail="generated walk nearer walk" /><Metric label="Endpoint loss" value={current.endpointLoss.toFixed(4)} detail="fixed diagnostic" /></div>
        </div>
        <aside className="experiment-side">
          <div className="panel gate-panel"><div className="panel-top"><div><span>QUALITY PEAK</span><h2>5k → 6k decision</h2></div><Badge tone="amber">MIXED</Badge></div><div className="gate-list">{causalGates.map((gate) => <div className="gate-row" key={gate.label}><span className="gate-check">{gate.pass ? "✓" : "~"}</span><div><strong>{gate.label}</strong><small>{gate.observed} <i>{gate.threshold}</i></small></div></div>)}</div></div>
          <div className="panel claim-panel"><span className="claim-icon">!</span><div><h3>Evidence boundary</h3><p>All 16 requests, identities, actions, and targets occurred in training. Better causal response is real; generalization is untested.</p></div></div>
          <div className="panel mini-compare"><span>2k baseline → selected</span><div><b>{before.idleToWalk}/8</b><i>idle → walk</i><em>→</em><b>{current.idleToWalk}/8</b></div><small>movement toward replacement target</small></div>
        </aside>
      </div>
    </section>
  );
}

function CorpusView() {
  const [query, setQuery] = useState("");
  const [stage, setStage] = useState<"all" | SourceStage>("all");
  const [sort, setSort] = useState<"sequences" | "name">("sequences");
  const filtered = sources.filter((source) => `${source.name} ${source.kind} ${source.note}`.toLowerCase().includes(query.toLowerCase())).filter((source) => stage === "all" || source.stage === stage).sort((left, right) => sort === "sequences" ? right.sequences - left.sequences : left.name.localeCompare(right.name));
  return (
    <section className="view" aria-labelledby="corpus-title">
      <header className="section-header"><div><div className="eyebrow"><span>04</span> CORPUS EXPLORER</div><h1 id="corpus-title">Breadth with an<br /><em>evidence trail.</em></h1></div><div className="header-metrics"><strong>{formatNumber(indexSnapshot.counts.sequences)}</strong><span>indexed sequences</span><strong>{formatNumber(indexSnapshot.counts.frames)}</strong><span>ordered frames</span></div></header>
      <div className="toolbar corpus-toolbar"><label className="search-box"><span>⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search 13 active and staged corpora" /></label><select value={stage} onChange={(event) => setStage(event.target.value as "all" | SourceStage)} aria-label="Filter by corpus stage"><option value="all">All stages</option>{Object.entries(stageLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select><select value={sort} onChange={(event) => setSort(event.target.value as "sequences" | "name")} aria-label="Sort corpora"><option value="sequences">Most sequences</option><option value="name">Alphabetical</option></select></div>
      <div className="corpus-layout">
        <div className="source-table panel"><div className="source-row source-head"><span>Source / carrier</span><span>Entities</span><span>Sequences</span><span>Frames</span><span>State</span></div>{filtered.map((source) => <details className="source-details" key={source.id}><summary className="source-row"><span><b>{source.name}</b><small>{source.kind}</small></span><span>{formatNumber(source.entities)}</span><span>{formatNumber(source.sequences)}</span><span>{formatNumber(source.frames)}</span><span><Badge tone={source.stage === "model-ready" ? "green" : source.stage === "blocked" ? "red" : source.stage === "materialized" ? "purple" : "neutral"}>{stageLabels[source.stage]}</Badge></span></summary><div className="source-note"><p>{source.note}</p><div><span>{formatNumber(source.occurrences)} evidence occurrences</span><span>{source.rights || "—"} rights observations</span><code>{source.id}</code></div></div></details>)}</div>
        <aside className="coverage-stack"><CoveragePanel title="Action coverage" eyebrow="CONTROL LABELS" rows={actionCoverage.slice(0, 8)} /><CoveragePanel title="Subject classes" eyebrow="ENTITY TAXONOMY" rows={entityCoverage.slice(0, 7)} /></aside>
      </div>
    </section>
  );
}

function CoveragePanel({ title, eyebrow, rows }: { title: string; eyebrow: string; rows: readonly (readonly [string, number])[] }) {
  const max = rows[0]?.[1] ?? 1;
  return <div className="panel coverage-panel"><div className="panel-top"><div><span>{eyebrow}</span><h2>{title}</h2></div></div>{rows.map(([label, count]) => <div className="coverage-row" key={label}><span>{label}</span><div><i style={{ width: `${Math.max(3, (count / max) * 100)}%` }} /></div><b>{shortNumber(count)}</b></div>)}</div>;
}

function PipelineView() {
  const [selected, setSelected] = useState<(typeof pipelineStages)[number]>(pipelineStages[3]);
  return (
    <section className="view" aria-labelledby="pipeline-title">
      <header className="section-header"><div><div className="eyebrow"><span>05</span> END-TO-END SYSTEM</div><h1 id="pipeline-title">Every pixel knows<br /><em>where it came from.</em></h1></div><p>The pipeline is fail-closed. Unknown timing, unsafe paths, unsupported transforms, and ambiguous rights remain visible evidence—not silent training inputs.</p></header>
      <div className="pipeline-map">{pipelineStages.map((stage, index) => <button key={stage.id} type="button" className={selected.id === stage.id ? "pipeline-node active" : "pipeline-node"} onClick={() => setSelected(stage)}><span>{stage.index}</span><b>{stage.title}</b><i className={`status-${stage.status}`} />{index < pipelineStages.length - 1 && <em>→</em>}</button>)}</div>
      <div className="pipeline-detail panel"><div className="detail-index">{selected.index}</div><div><div className="eyebrow"><span>{selected.status === "operational" ? "●" : "◐"}</span> {selected.status.toUpperCase()}</div><h2>{selected.title}</h2><p>{selected.summary}</p><strong>{selected.details}</strong></div><div className="invariant-list"><span><b>SHA-256</b> immutable identity</span><span><b>NO-CLOBBER</b> derived artifacts</span><span><b>100 GiB</b> hard disk floor</span><span><b>QUERY-ONLY</b> readiness checks</span></div></div>
      <div className="feature-grid"><article><span>↳</span><h3>Source adapters</h3><p>Engine-aware parsers preserve native timing, direction, aliases, palettes, and runtime caveats.</p></article><article><span>⌗</span><h3>Exact media</h3><p>GIF/APNG compositing, PNG chunks, palettes, integer-nearest scaling, and source rectangles.</p></article><article><span>◇</span><h3>Grouped splits</h3><p>Identity, pack, exact blob, and duplicate components stay together across partitions.</p></article><article><span>∿</span><h3>Temporal control</h3><p>Authored durations, explicit phases, loop modes, repeat tails, and one-shot endpoints.</p></article><article><span>◎</span><h3>Native RGBA DiT</h3><p>Spatial/temporal attention with action, class, view, direction, loop, and phase tokens.</p></article><article><span>±</span><h3>Matched evaluation</h3><p>Same noise and non-action fields isolate causal effects instead of flattering aggregate loss.</p></article></div>
    </section>
  );
}

function EvidenceView() {
  const [copied, setCopied] = useState<string | null>(null);
  async function copyHash(label: string, value: string) { await navigator.clipboard.writeText(value); setCopied(label); window.setTimeout(() => setCopied(null), 1500); }
  return (
    <section className="view" aria-labelledby="evidence-title">
      <header className="section-header"><div><div className="eyebrow"><span>06</span> PROVENANCE LEDGER</div><h1 id="evidence-title">Citable by design.<br /><em>Honest by default.</em></h1></div><p>This console is a small static view of much larger canonical reports. The download preserves the summary facts and report identities used here.</p></header>
      <div className="metrics-strip"><Metric label="Sources" value={formatNumber(indexSnapshot.counts.sources)} detail="registered systems" /><Metric label="Archive members" value={formatNumber(indexSnapshot.counts.archiveMembers)} detail="path-safe inventory" /><Metric label="Media observations" value={formatNumber(indexSnapshot.counts.mediaObservations)} detail="format + geometry" /><Metric label="Evidence edges" value={formatNumber(indexSnapshot.counts.occurrences)} detail="sequence occurrences" accent /></div>
      <div className="evidence-layout">
        <div className="panel hash-ledger"><div className="panel-top"><div><span>IMMUTABLE REFERENCES</span><h2>Artifact hashes</h2></div><a className="text-button" href="/data/ui-snapshot.json" download>Download summary ↓</a></div>{evidenceHashes.map(([label, value]) => <button key={label} type="button" onClick={() => copyHash(label, value)}><span><b>{label}</b><small>SHA-256</small></span><code title={value}>{compactHash(value)}</code><em>{copied === label ? "copied" : "copy"}</em></button>)}</div>
        <aside className="evidence-notes"><PolicyCard code="RAW" title="Original bytes stay immutable">Normalized frames and tensors always point back to carrier URLs, archive members, and exact hashes.</PolicyCard><PolicyCard code="RGT" title="Rights are observations">License, attribution, scope, and uncertainty remain separate; collection claims never become per-asset claims.</PolicyCard><PolicyCard code="Q/A" title="Quarantine remains indexed">Malformed media, unsupported transforms, aliases, and uncertain loops are documented rather than erased.</PolicyCard></aside>
      </div>
      <div className="citation-card"><div><span>REPORT BUNDLE / SCHEMA 3</span><h2>provenance-v8 · post-SuperTux</h2><p>Five hash-bound payloads · 411,066,062 bytes · single query-only SQLite transaction</p></div><code>64b8f0210583cbd5d994114a0490afa55<br />d7821e115e12df19fdeceaee6e2a8e6</code></div>
    </section>
  );
}

function PolicyCard({ code, title, children }: { code: string; title: string; children: React.ReactNode }) {
  return <div className="panel policy-card"><span>{code}</span><div><h3>{title}</h3><p>{children}</p></div></div>;
}

export function ResearchConsole() {
  const [view, setView] = useState<View>("studio");
  const [navOpen, setNavOpen] = useState(false);
  const mainRef = useRef<HTMLElement>(null);
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const tag = (event.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      const target = views.find((item) => item.key === event.key);
      if (target) { setView(target.id); mainRef.current?.focus(); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const content = { studio: <StudioView />, gallery: <GalleryView />, experiments: <ExperimentsView />, corpus: <CorpusView />, pipeline: <PipelineView />, evidence: <EvidenceView /> }[view];
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">Skip to content</a>
      <aside className={navOpen ? "side-nav open" : "side-nav"}>
        <button className="brand" type="button" onClick={() => setView("studio")} aria-label="Open studio"><span className="brand-mark"><i /><i /><i /><i /></span><span><b>SPRITE</b><em>LAB</em></span></button>
        <nav aria-label="Research console sections">{views.map((item) => <button type="button" key={item.id} className={view === item.id ? "active" : ""} onClick={() => { setView(item.id); setNavOpen(false); }}><span>{item.short}</span><b>{item.label}</b><kbd>{item.key}</kbd></button>)}</nav>
        <div className="nav-status"><span className="pulse-dot" /><div><b>Research index</b><small>{formatNumber(indexSnapshot.counts.sequences)} sequences</small></div></div>
      </aside>
      <div className="app-main">
        <header className="topbar"><button type="button" className="menu-button" onClick={() => setNavOpen(!navOpen)} aria-label="Toggle navigation">☰</button><div className="crumb"><span>SPRITELAB</span><i>/</i><b>{views.find((item) => item.id === view)?.label}</b></div><div className="system-strip"><span><i className="online-dot" /> INDEX SNAPSHOT</span><span>64 × 64</span><span>8 FRAMES</span><Badge tone="amber">RESEARCH</Badge></div></header>
        <main id="main" ref={mainRef} tabIndex={-1}>{content}</main>
        <footer><span>Sprite Diffusion Research · proof of concept</span><span>Evidence snapshot · 2026-08-12</span><span>Not a production model</span></footer>
      </div>
    </div>
  );
}
