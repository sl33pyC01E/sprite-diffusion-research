# Study gallery

This page gives a compact, honest visual index of the major model studies whose
weights are preserved in the `research-weights-2026-08-15` release. Files ending
in `.png` may be animated PNGs; open them directly if a Markdown viewer shows only
their first frame.

Labels matter:

- **target** is source supervision, not generated output;
- **in-sample** means the character/action was present during training;
- **held-out** means the character identity was not used for training;
- **display decode** is a visualization derivative, not the raw tensor;
- **rejected control** is retained to document a failed approach.

## 1. Fetid Rat PixelDiT — 1,500 steps

Best clean small-scale replay. Exact in-sample attack target versus endpoint output
with hard-alpha/palette display decoding. This proves memorization, not general generation.

| Target | Generated |
|---|---|
| ![Fetid Rat attack target](media/studies/01-fetid-rat-target-attack.png) | ![Fetid Rat generated attack](media/studies/01-fetid-rat-generated-attack.png) |

Weight: `fetid-rat-pixeldit-alpha4-1500.pt`

## 2. Corrected multi-identity PixelDiT — 3,000 steps

Neon walk target and same-request endpoint output. The generated form remains noisy
and does not reliably preserve identity.

| Target | Generated |
|---|---|
| ![Neon walk target](media/studies/02-multi-identity-target-neon-walk.png) | ![Neon generated walk](media/studies/02-multi-identity-generated-neon-walk.png) |

Weight: `multi-identity-pixeldit-3000.pt`

## 3. TMWA Causal16 PixelDiT

Matched Skull Ice walk replay. The 1,000-to-2,000 comparison uses the same target,
request order, phases, and noise. The released 6,000-step checkpoint continues this
same focused causal line; the images below are the most thoroughly audited 1k/2k pair.

| Target | 1,000 steps | 2,000 steps |
|---|---|---|
| ![Skull Ice walk target](media/studies/03-tmwa-causal-target-skull-ice-walk.png) | ![Skull Ice walk at 1000](media/studies/03-tmwa-causal-generated-1000.png) | ![Skull Ice walk at 2000](media/studies/03-tmwa-causal-generated-2000.png) |

Weight: `tmwa-causal16-pixeldit-alpha4-6000.pt`

## 4. Broad TMWA semantic endpoint model — 4,000 steps

Held-out character/action evaluation. The model learned broad shape and action cues,
but the result is not clean enough for asset production.

| Held-out target | Generated |
|---|---|
| ![Broad TMWA held-out attack target](media/studies/04-tmwa-semantic-target-attack.png) | ![Broad TMWA generated attack](media/studies/04-tmwa-semantic-generated-attack.png) |

Weight: `tmwa-broad-semantic-pixeldit-4000.pt`

## 5. Broad TMWA rectified-flow control — 10,000 steps

Held-out walk output from the rejected flow control. It is preserved because the
failure helped motivate the later staged MUGEN pipeline.

![TMWA flow generated walk](media/studies/05-tmwa-flow-generated-walk.png)

Weight: `tmwa-broad-flow-pixeldit-10000.pt`

## 6. SD 1.4 sprite-prior LoRA — 2,500 steps

Historical held-out text-to-image adapter. It produced a stronger generic image prior
than the early scratch models but depends on a separately obtained SD 1.4 base and is
not part of the intended efficient final architecture.

![SD 1.4 sprite LoRA held-out preview](media/studies/06-mugen-sd14-lora-heldout.png)

Weight: `mugen-sd14-sprite-lora-2500.pt`

## 7. Reference-conditioned latent motion with pixel refinement — 9,000 steps

The strongest in-sample MUGEN motion replay: exact normal-attack target and generated
trajectory for one reference character. Its quality is high because it is memorizing
one character, so it is a pipeline proof rather than a generalization result.

| Target | Generated |
|---|---|
| ![Reference motion attack target](media/studies/07-mugen-reference-target-attack.png) | ![Reference motion generated attack](media/studies/07-mugen-reference-generated-attack.png) |

Weight: `mugen-reference-latent-motion-pixel-refine-9000.pt`

## 8. Broad reference-conditioned latent motion — 15,000 steps

Broad 225-identity endpoint baseline. It responds to the reference/action inputs but
averages fine character detail.

| Target | Generated |
|---|---|
| ![Broad MUGEN motion target](media/studies/08-mugen-broad-motion-target-attack.png) | ![Broad MUGEN generated motion](media/studies/08-mugen-broad-motion-generated-attack.png) |

Weight: `mugen-broad-latent-motion-ema-15000.pt`

## 9. AnimateDiff temporal LoRA — 1,000 steps

Historical large-model motion adapter. It is included for comparison only; the base
model is not redistributed and this is not the chosen scratch-training direction.

| Target | Generated |
|---|---|
| ![AnimateDiff target](media/studies/09-mugen-animatediff-target-attack.png) | ![AnimateDiff output](media/studies/09-mugen-animatediff-generated-attack.png) |

Weight: `mugen-animatediff-temporal-lora-1000.pt`

## 10. Dense six-action latent motion from scratch — 10,000 steps

In-sample attack-B example from the dense six-action corpus. The generated clip shows
the central failure mode: recognizable subject mass but over-averaged action detail.

| Target | Generated |
|---|---|
| ![Dense motion target](media/studies/10-mugen-dense-latent-target-attack-b.png) | ![Dense motion generated](media/studies/10-mugen-dense-latent-generated-attack-b.png) |

Weight: `mugen-dense-latent-motion-scratch-10000.pt`

## 11. Fixed-middle keypose DiT — 30,000 steps

Single-image middle-pose prediction across the six canonical actions. This isolates
action pose from temporal trajectory generation.

![Fixed-middle keypose grid](media/studies/11-mugen-fixed-middle-keypose.png)

Weight: `mugen-fixed-middle-keypose-dit-30000.pt`

## 12. RGBA autoencoder — 20,000 steps

Held-out source/reconstruction gallery for the compact 2x spatial codec used by the
latent models. This measures the best detail the downstream latent stages can retain.

![RGBA autoencoder reconstructions](media/studies/12-mugen-rgba-autoencoder.png)

Weight: `mugen-rgba-autoencoder-2x-20000.pt`

## 13. Broad still-image DiT and crash-resume state

Representative target/output from the same broad still-image training line. The
released EMA is step 45,000 and the resume state is step 47,500; this visual was
exported at step 20,000 and is labeled as such rather than implied to be a later render.

| Target | Generated at step 20,000 |
|---|---|
| ![Broad still target](media/studies/13-mugen-broad-still-target.png) | ![Broad still generated](media/studies/13-mugen-broad-still-generated.png) |

Weights: `mugen-broad-still-dit-ema-45000.pt` and
`mugen-broad-still-dit-resume-47500.pt`

## 14. Identity-conditioned keypose U-Net — 30,000 steps

Alternative Stage-2 network tested against the keypose DiT on the same six-action task.

![Identity U-Net keypose grid](media/studies/14-mugen-keypose-unet.png)

Weight: `mugen-keypose-identity-unet-30000.pt`

## 15. Anchored trajectory model — 10,000 steps

Stage-3 experiment conditioned on start/peak/end anchors. The endpoints are explicit,
but the model can still average the trajectory between them.

| Target | Generated |
|---|---|
| ![Anchored motion target](media/studies/15-mugen-anchored-target-attack-b.png) | ![Anchored motion generated](media/studies/15-mugen-anchored-generated-attack-b.png) |

Weight: `mugen-anchored-motion-10000.pt`

## 16. Six-action latent classifier

This support model verifies that action information is present and measurable in the
latent residuals. Test characters are identity-disjoint from training characters.

![Six-action classifier test accuracy](media/studies/16-mugen-action-classifier.svg)

Weight: `mugen-six-action-classifier.pt`

## Integrity

The authoritative weight names, sizes, SHA-256 values, tiers, and local source paths
are in [`../releases/best-weights-v1.json`](../releases/best-weights-v1.json). The
gallery assets are visual summaries only and never replace the raw arrays, reports,
or checkpoint-linked evaluations.
