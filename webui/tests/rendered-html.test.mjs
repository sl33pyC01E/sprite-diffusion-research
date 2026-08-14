import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the SpriteLab research console", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>SpriteLab — Animated Sprite Research Console<\/title>/i);
  assert.match(html, /Describe the sprite/i);
  assert.match(html, /Hosted mode: verified replay/i);
  assert.match(html, /Effect-only and subject-absent targets are quarantined/i);
  assert.match(html, /Memorization diagnostic/i);
  assert.match(html, /26,196(?:<!-- -->)? sequences/i);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site|react-loading-skeleton/i);
});

test("publishes the curated evidence payload and representative animations", async () => {
  const [snapshot, galleryManifest] = await Promise.all([
    readFile(new URL("../public/data/ui-snapshot.json", import.meta.url), "utf8").then(JSON.parse),
    readFile(new URL("../public/gallery/manifest.json", import.meta.url), "utf8").then(JSON.parse),
  ]);

  assert.match(snapshot.claim_scope, /canonical provenance and model reports remain authoritative/i);
  assert.equal(snapshot.provenance.sequences, 26196);
  assert.equal(
    snapshot.provenance.bundle_sha256,
    "64b8f0210583cbd5d994114a0490afa55d7821e115e12df19fdeceaee6e2a8e6",
  );
  assert.match(snapshot.tmwa_causal16_step_2000.claim, /in-sample memorization/i);
  assert.equal(snapshot.tmwa_causal16_step_2000.idle_to_walk_moves_toward_replacement, 6);
  assert.equal(snapshot.tmwa_causal16_alpha4_step_6000.walk_correct_target_preference, 8);
  assert.equal(snapshot.tmwa_causal16_alpha4_step_6000.alpha_iou_at_127, 0.979635);
  assert.equal(snapshot.staged_mugen_latent_motion.canonical_sequences, 1443);
  assert.match(snapshot.staged_mugen_latent_motion.model, /latent space/i);
  assert.equal(galleryManifest.files.length, 21);
  assert.equal(
    galleryManifest.runs.tmwa_causal16_alpha4_step_6000.decode_bundle_sha256,
    "25c78b85b6dc9f6c463e081c5504203881d134842c3bc7427e2d0c2612a22aa0",
  );
  assert.match(galleryManifest.claim_scope, /not held-out/i);

  for (const file of galleryManifest.files) {
    const payload = await readFile(new URL(`../public/gallery/${file.path}`, import.meta.url));
    assert.equal(payload.byteLength, file.bytes);
    assert.equal(createHash("sha256").update(payload).digest("hex"), file.sha256);
  }

  await Promise.all([
    access(new URL("../public/gallery/sasquatch-walk-target.png", import.meta.url)),
    access(new URL("../public/gallery/sasquatch-walk-endpoint.png", import.meta.url)),
    access(new URL("../public/gallery/fetid-rat-idle.png", import.meta.url)),
    access(new URL("../public/gallery/skull-ice-walk-alpha4-6000.png", import.meta.url)),
  ]);
});
