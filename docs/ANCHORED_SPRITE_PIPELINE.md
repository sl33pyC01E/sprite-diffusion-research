# Anchored sprite-generation pipeline

This is the current three-stage research contract. It replaces the earlier attempt to
make one motion model discover appearance, action pose, and trajectory simultaneously.

```text
character description
  -> text-to-image sprite still
  -> still + verb -> canonical middle action pose
  -> start/middle/start anchors + verb -> eight-frame loop
```

## Stage 1: description to canonical sprite

The still-image model owns character appearance: silhouette, anatomy, clothing,
palette, accessories, and pixel-art treatment. Dense MUGEN visual captions are being
produced once per identity reference rather than once per action. Motion and pose words
are excluded from the identity-appearance prompt where the caption contract can do so
reliably.

The active broad caption input contains 4,062 identity variants. Caption records are
append-only and hash-bound to the exact rendered reference image, prompt, caption model
identity, and source variant. A complete caption manifest must exist before a broad
still-image training plan can be published.

## Stage 2: canonical sprite and verb to one action pose

The key-pose model consumes the encoded reference sprite and one of the six canonical
MUGEN verbs:

- `idle`
- `walk`
- `jump`
- `block`
- `attack_a`
- `attack_b`

Its target is always source frame index 4, phase 0.5, from the identity's canonical
eight-frame action sequence. This is called the **canonical middle action pose**, not a
detected peak.

There is deliberately no per-sample peak-frame selector and no predicted frame index.
Training and inference therefore have the same temporal-slot contract. The model uses
all six actions for one identity in the same batch and the same noise realization for
all six, making the action condition the only requested change.

Implementation:

- `src/spritelab/latent_keypose_train.py`
- `scripts/run_mugen_latent_keypose_train_v1.py`

The current dense corpus contains 3,759 identity-disjoint characters and exactly
22,554 sequences. The training split contains 3,442 complete six-action identity
bundles; validation contains 162 and test contains 155.

## Stage 3: hard-anchored latent interpolation

The motion model receives an eight-frame latent canvas, the verb, the reference latent,
an anchor tensor, and an explicit boolean anchor mask.

| Frame | Contract |
| --- | --- |
| 0 | exact reference/start latent |
| 1–3 | predicted onset trajectory |
| 4 | exact Stage 2 middle-pose latent |
| 5–6 | predicted recovery trajectory |
| 7 | exact reference/start latent |

Frames 0, 4, and 7 are clamped before and after every sampling update. Velocity and
pixel reconstruction losses supervise only frames 1–3 and 5–6. Action-contrast metrics
also exclude anchors, so they cannot claim success merely because Stage 2 supplied a
different middle image.

The training target preserves authored source frames 1–6, replaces frame 0 with the
identity reference, and replaces frame 7 with that same reference. This is an output
normalization contract: one-shot source actions become production-friendly repeating
loops with an explicit recovery to the starting pose.

Implementation:

- `src/spritelab/models/anchored_latent_motion_dit.py`
- `src/spritelab/anchored_motion_train.py`
- `scripts/run_mugen_anchored_motion_train_v1.py`

Initial Stage 3 training uses the true frame-4 anchor. A later robustness phase must mix
in hash-bound Stage 2 predictions before end-to-end inference quality can be claimed.
Until that phase is run, evaluation must say **teacher-forced middle anchor**.

## Required evaluation order

1. Evaluate Stage 2 on training identities first. It must beat six-way chance target
   preference and produce visible action-specific poses before held-out results are
   interpreted.
2. Evaluate Stage 2 on identity-disjoint validation and test bundles.
3. Train Stage 3 with true frame-4 anchors and verify exact anchor preservation plus
   missing-frame reconstruction.
4. Replace true anchors with Stage 2 predictions and measure degradation.
5. Evaluate the complete description-to-loop path only after a Stage 1 checkpoint and
   all downstream checkpoints are fixed by hash.

No stage may use source action frames at inference other than through its trained
weights. Display previews must remain labeled as training, validation, or test and as
raw model output or postprocessed derivative.
