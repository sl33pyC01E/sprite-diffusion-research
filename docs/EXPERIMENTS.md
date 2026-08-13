# Model experiments

This log records saved model experiments and their evidence boundaries. The source
artifacts under `data/experiments/` are intentionally ignored by Git, so every run
listed here includes the SHA-256 of its canonical report. The reports, checkpoints,
samples, and preview sidecars retain the exact materialization, source-array, target,
condition, RNG, and runtime facts needed for audit.

Exact-target evaluation schema version 2 distinguishes the hash of the source
materialized `.npy` from the hash of the actual temporally selected training tensor.
It also records the target shape, dtype, intro/loop projection, temporal selection,
and duration method. Version-1 evaluation sidecars incorrectly labeled the source
array digest as `target_array_sha256` when temporal selection changed frame count;
their pixel metrics remain valid, but that field does not identify the tensor used
by the metric calculation. Version-1 files are retained rather than overwritten.

## Current claim

The executable path has passed a small, in-sample proof of concept:

- a factorized pixel-space diffusion transformer can memorize four 32x32x8 RGBA
  animations of the same Fetid Rat identity;
- the actions `idle`, `run`, `attack`, and `death` produce distinct outputs when the
  initial noise and every non-action condition are held fixed; and
- a causal endpoint-supervision ablation makes the model substantially more
  sensitive to the action token.

This is **not** evidence of open-vocabulary text-to-sprite generalization, novel
identity generation, held-out action generalization, or production-quality art. The
generated clips remain noisy and have soft alpha boundaries. All quality metrics
below are training-set diagnostics.

## Fetid Rat endpoint experiment v1

The first saved run used four actions from one identity, 300 CPU steps, and endpoint
weight 1.0. Its report SHA-256 is
`23cfcb60af3c8ceda0b7abe00490209832c97acfb0ef496b831e2e802155cce2`.

The fixed ordinary-flow loss fell from 3.14651 to 0.32563, and the fixed matched
endpoint loss fell from 3.14565 to 0.25344. Cyclic action substitution increased the
endpoint loss by only 0.03354. The run proved that the end-to-end tensor, optimizer,
checkpoint, sampler, and RGBA export path worked, but it did not isolate action
control rigorously enough. It is retained as an immutable baseline.

## Matched 1,000-step causal ablation

Two runs use the same four source clips, seed (`20260813`), model configuration,
diagnostic tensors, and 1,000 CPU steps. They differ only in whether the matched
endpoint loss contributes to optimization. Both use a 2.4-million-parameter-class
PixelDiT with width 128, four blocks, four heads, 4-pixel patches, float32, and a
32-step backward-Euler sampler.

| Diagnostic | Endpoint weight 0 | Endpoint weight 1 |
| --- | ---: | ---: |
| Report SHA-256 | `c6633d014d4659b7d9f8612da3e0a527290afd54e69f3f7424751ec67ec6b8d6` | `d0a17412c653a3e125a2ed1f7ad4887ab1636d8227fb1030692f1c23032b12b7` |
| Fixed ordinary loss, initial -> final | 3.12660 -> 0.37163 | 3.12660 -> 0.39933 |
| Fixed endpoint loss, initial -> final | 3.14602 -> 0.26721 | 3.14602 -> 0.10815 |
| Endpoint action-permutation delta | +0.05050 | +0.28524 |
| Mean premultiplied-RGBA MAE | 0.10392 | 0.06882 |
| Mean alpha MAE | 0.23197 | 0.12930 |
| Mean temporal-delta MAE | 0.07215 | 0.06028 |
| Idle/run output separation relative to target | 41.83% | 70.28% |
| Attack/death output separation relative to target | 41.54% | 84.13% |

All four directed action swaps increased endpoint loss in both runs. With endpoint
weight 1, the deltas were +0.16669 for idle->run, +0.22630 for run->idle, +0.36469
for attack->death, and +0.38327 for death->attack. The aggregate action-permutation
delta is 5.65 times the weight-zero control. Because the paired runs hold stochastic
inputs and non-action conditions fixed, this is evidence that the model uses the
categorical action token on this memorized identity.

The result is still narrowly scoped. Four clips from one identity are insufficient
for a generalization claim, and the alpha-IoU comparison is not monotonic with the
other reconstruction metrics (0.52137 for weight zero versus 0.47946 for weight
one). Raw contact sheets show identifiable poses but visible speckle, translucent
edges, and occasional detached pixels.

## Raw and derived outputs

Canonical outputs are the uint8 straight-RGBA `.npy` files under each experiment's
`samples/` directory. Animated PNGs and contact sheets under `previews/` are
display-only nearest-neighbor derivatives with hash sidecars. They are not used for
training or evaluation.

For inspection, the endpoint-weight-one run also has a derived hard-alpha decode at
threshold 192. It maps alpha to 0 or 255, zeros newly invisible RGB, and performs no
spatial cleanup, palette reduction, resizing, or frame interpolation. A sweep over
thresholds 32, 64, 96, 128, 160, 192, and 224 on these same four training targets
selected 192 for the best measured premultiplied-RGBA MAE (0.04689) and alpha IoU
(0.83883). The canonical calibration JSON SHA-256 is
`69b6481a8d042de868d9366269225b666132537bd0103ad061368ed6aab2c31a`.
Its four target-array hashes exactly match the version-2 evaluation sidecar, whose
SHA-256 is `4848b7e5ffe5733c72ddbba94578a4d447052de58ff048d8f02a42f4ba3767e4`.
This selection is optimistic because it was calibrated on the memorized training
clips; the raw samples remain authoritative.

## Next experimental gates

The next model result should use identity-disjoint train/validation partitions and
report both raw continuous-alpha and explicitly decoded outputs. It should require:

1. a positive held-out action-swap loss delta with matched noise and non-action
   conditions;
2. held-out action/identity separation above a declared baseline;
3. alpha, temporal, loop-seam, and nearest-neighbor memorization metrics; and
4. exact prompt/condition replay from the checkpoint without relying on an external
   mutable manifest.

Until those gates pass, these runs remain an action-conditioned memorization proof,
not a text-to-animated-sprite model result.

## Checkpoint-only replay and text probe

The endpoint-weight-one checkpoint can now be loaded without its training manifest:
the inference path verifies the checkpoint byte hash before deserialization, uses
restricted weight-only loading, reconstructs the saved model and condition encoder,
and records the exact request, phases, initial noise, runtime, arrays, and hashes in
a no-clobber report. The checkpoint-only replay report SHA-256 is
`f24d8b3c198d9973c72449ba475737f769282045bfe8f86cf81fa62ec0666d17`.
Using the original sample seed and shared-noise contract reproduced `attack`, `run`,
and `death` byte-for-byte. The `idle` array differed in one of 131,072 uint8 channel
values by one least-significant bit. The saved training runtime had deterministic
algorithms disabled, so this is a numerically near-exact replay, not a bit-exact
cross-run guarantee.

A matched-noise text probe kept action (`idle`), entity class, loop mode, view,
direction, phases, and latent noise fixed while changing only the description among
`fetid rat sprite`, `blue armored wolf sprite`, and `tiny wizard sprite`. Its report
SHA-256 is `8b384fea7a8fa715da6d274c8aa54598c60628deedecd45474fdbd6ab302a2a2`.
The pairwise premultiplied-RGBA distances were 0.01312 (rat/wolf), 0.00858
(rat/wizard), and 0.00807 (wolf/wizard). Thus the byte-level text channel affects
the output, but all three results remain visibly rat-like. This is evidence of text
sensitivity only, not semantic text generalization.

## Multi-identity 64-pixel diagnostic (invalid transparency targets)

The first broad run is retained at `multi-identity-b64-v1-1000` as a data-pipeline
diagnostic. It trained for 1,000 GPU steps on 48 clips spanning 13 identities and 15
actions. The report SHA-256 is
`132dcd6bfc4baec31682a12be49554f93128ed4e1e03b519058c5846143a13c8`,
the checkpoint SHA-256 is
`f0bd0392fb87dfe3eb27b5e5744de2831aa465cdf373717d07ea50517819a8c4`,
and the exact-target evaluation-v2 SHA-256 is
`920ac84cb3dead89ee8e80cb4fe6f5959ca8233228ccfb017b79ebb0a5f43463`.

The optimization path worked: fixed ordinary loss fell from 3.47408 to 0.55773,
fixed endpoint loss fell from 3.61996 to 0.28817, and matched action substitution
raised endpoint loss by 0.03659. Of 120 directed action swaps, 108 had positive and
12 had negative loss deltas (mean +0.03744; range -0.00961 to +0.20035). Mean
premultiplied-RGBA MAE was 0.10771, alpha IoU 0.51514, alpha MAE 0.09854, and
temporal-delta MAE 0.07189. Generated action-pair separation averaged only 29.92%
of target separation (median 26.66%).

Those numbers must not be interpreted as transparent-sprite quality. Forty-six of
the 48 selected clips came from Open Surge. The engine treats exact RGB magenta
`[255, 0, 255]` as transparent, but the pre-fix materializer retained it as opaque.
In the exact eight-frame training tensors, 31 of those 46 Open Surge clips and 248
frames were affected: 399,403 opaque-magenta pixels, or 26.50% of all selected Open
Surge pixels and 25.39% of the full 48-clip batch. The two SpriteCook clips had zero
such pixels. The model consequently learned source-key rectangles visible in its
targets and samples. The checkpoint is preserved for audit and regression testing,
but it is invalid evidence for sprite appearance, alpha quality, or generalization.
The colocated machine-readable validity annotation has SHA-256
`4736c7b4f3c293b5ded377a640b8e35c81feb4d110f8261924029a81887b8584`.

Open Surge projection version 2 now carries the exact engine color-key transform
and materialization evidence. Corrected snapshots, materializations, and model runs
must use new versioned paths; the pre-fix artifacts above will not be overwritten.

## Sample-quality follow-up

The valid four-action Fetid Rat checkpoint was subsequently evaluated with exact
matched noise and targets to distinguish sampler error from model error. The
canonical diagnostic is
`data/experiments/fetid-rat-sample-quality-diagnostic-v2/diagnostic.json`, SHA-256
`df52fd8c30f95978f2f50fcebf73ced8378c3209b84c3ae478922101ccc1b928`.
Direct reconstruction from the explicitly supervised `t=1` endpoint improved raw
premultiplied-RGBA MAE from 0.068823 for Euler-32 to 0.063713. After the existing
hard-alpha-192 derivative, the corresponding values were 0.046890 and 0.037271.
Heun integration was worse, so this checkpoint's learned velocity field should not
be described as a well-solved continuous ODE.

A safe-load, matched 500-step continuation then tested the original objective
against an alpha-channel residual multiplier of four. The complete A/B report at
`data/experiments/fetid-rat-continuation-ablation-v4/report.json` has SHA-256
`a22c7ad9a55ebc732e7efeabfbb8d4973e54dfa0e9c06273e3892ec69486fbb0`.
The alpha-weighted continuation reached hard-alpha MAE 0.024542 and alpha IoU
0.950537, versus 0.026353 and 0.931376 for the matched weight-one continuation.
This supports an opt-in alpha-loss ablation, not a universal hyperparameter claim.

For display only, a generated-only, clip-global 32-color derivative further reduced
MAE to 0.024062 and temporal-delta MAE to 0.023463. Its evaluation SHA-256 is
`245c86a25a28edd94f4514f98d177cd482736940681042ac3081066356d55ef1`.
The output remains visibly noisy and dark, but idle, run, attack, and death are
recognizable and distinct. Raw continuous-RGBA arrays remain authoritative, and all
of these measurements are in-sample memorization diagnostics rather than semantic
text-to-sprite generalization.

## Sparse transparent-canvas metric contract

A global canvas-mean error can look small when a sprite occupies only a small part
of a transparent canvas: errors on the subject and colored alpha noise in the
background are diluted by many correct transparent pixels. Future exact-target
evaluations therefore retain the existing global metrics and add target-aware
matched metrics. Existing sidecars remain immutable and are not retroactively
rewritten.

Visibility is always explicit: a pixel is visible exactly when
`alpha > alpha_visibility_threshold`, where the threshold is an integer from 0
through 254 recorded in both the report and each matched sample row. The additional
fields are:

- `alpha_precision` and `alpha_recall` over the predicted and target visibility
  masks;
- `target_visible_premultiplied_rgba_mae`, evaluated only where the target is
  visible;
- `target_background_premultiplied_rgba_mae`, evaluated only where the target is
  not visible;
- `predicted_visible_canvas_fraction` and `target_visible_canvas_fraction`; and
- `predicted_to_target_visible_canvas_ratio`, the predicted visible-pixel count
  divided by the target visible-pixel count.

All premultiplied errors remain normalized to `[0, 1]`. A masked error is `0` when
its target region has no pixels. If both visibility masks are empty, precision,
recall, IoU, and the visible-canvas ratio are `1`. If the target is visible but the
prediction is empty, precision, recall, and the ratio are `0`. If the prediction is
visible but the target is empty, precision and the ratio are `0`, while recall is
`1` because there are no target-positive pixels to miss; the background error and
predicted visible fraction quantify the false-positive content. The two one-sided
zero-denominator cases can therefore be distinguished without non-finite JSON
values.

## Corrected 48-clip alpha-weight ablation

The corrected broad training batch was held fixed at 48 clips, including exact
ordered sequence IDs, input arrays, target geometry, architecture, seed, optimizer,
and 1,000 steps. The only configuration difference was the alpha-channel residual
multiplier: 1 for the control and 4 for the ablation. The control report SHA-256 is
`aa04d212be069771875d6716a2bcf7b23456325e688656adb528cae6c2865bb6` and
its checkpoint SHA-256 is
`8707d60974765a4323c28ae30430cbf2a3564993f23df2c887d077ad3cdb2b82`.
The alpha-4 report SHA-256 is
`509dedc7e91965559ec5a0731f19a9f8fe53f50387dacfb047067ea26adf82a0` and
its checkpoint SHA-256 is
`185c296c066b3c17d0c7c1863dfe47d06cf597a52f9ff896cae9690c4aa770e4`.

On the saved training-run Euler samples, alpha 4 reduced global premultiplied-RGBA
MAE from 0.113222 to 0.099203 and alpha MAE from 0.152486 to 0.119989. Alpha IoU
rose from 0.269330 to 0.304373. The sparse-target metrics show where that gain came
from: target-background premultiplied MAE fell from 0.091047 to 0.071192 and the
predicted-to-target visible-canvas ratio fell from 4.16543 to 3.48007, while
target-visible foreground MAE worsened from 0.241393 to 0.249985 and alpha recall
fell from 0.974618 to 0.966303. Thus this ablation primarily suppressed false
positive background occupancy; it did not improve reconstruction on the target
subject. The alpha-4 sparse exact-target evaluation SHA-256 is
`b572d78675ffdc6c742fd5c2ee3e7067f1cdb9bda2bd0a936e3dda49cd3da012`;
the recomputed control sidecar SHA-256 is
`79ae24398f0c3aeb3e40e809e599c655e9ec2dc573477c91a8e34242b3d2390e`.

A second comparison used all 48 exact requests and phases with shared CUDA noise,
seed `20260816`, and noise SHA-256
`7ef7e72e54533b35a199f31d9df89093994b4c9f5d1ab32dc313896e7a2feaf0`.
For Euler-32, alpha 4 changed global premultiplied MAE from 0.112567 to 0.101308,
background MAE from 0.089966 to 0.074067, and foreground MAE from 0.242626 to
0.246660. For the direct endpoint sampler, the corresponding values changed from
0.099013 to 0.089487, 0.075818 to 0.061361, and 0.232251 to 0.237629. The complete
matched report SHA-256 is
`49ca0d1c768c25a28df139bd8bba4e1c5f00dd41f7f6af1681f9b5d0820ab977`.
Its alpha-4 Euler and endpoint inference report SHA-256 values are respectively
`7b4b588a4f5613f34323744f2bcbe843cc2ff293d6f0ec53de5934509df8df97`
and `d0fea6d5338bc75a647ffd61a00ba5898e941d569f4f9756b883bee8e0e07907`.

Visual inspection across animal/walk, humanoid/attack, creature/shoot, and
object/spawn examples agreed with the target-aware metrics: alpha-4 Euler removed
many detached colored background pixels, but neither sampler reliably reconstructed
the target identity or action. Endpoint outputs remained softer, more translucent,
and more static. These are exact training-target diagnostics on one CUDA stack, not
held-out or semantic-generalization evidence.

## Immutable overfit continuation contract

`continue_tiny_overfit` extends a saved `run_tiny_overfit` experiment only into a
new output directory. It requires the exact parent checkpoint and report SHA-256
values, uses `torch.load(weights_only=True)`, rejects zero additional steps, and
verifies the manifest, source snapshot, ordered sequence IDs, input file/array
hashes, target geometry, model architecture, full optimization configuration,
endpoint contrast plan, runtime, optimizer, diagnostic tensors, and RNG state
contracts before training. Model, optimizer, global Python/NumPy/Torch RNG, and the
dedicated ordinary and endpoint training generators are restored. Diagnostic and
matched sample generators are recreated from their fixed seeds and checked against
the parent's tensor/state hashes.

The child bundle is built in a private sibling staging directory and promoted only
after every sample, checkpoint, and report is complete. Existing parent or child
artifacts are never overwritten. Each child report and checkpoint records the
parent paths, hashes, parent step, added steps, cumulative step, and replayed parent
diagnostic losses. A deterministic CPU split-run test requires resumed model,
encoder, optimizer, RNG, history, samples, and final losses to equal an uninterrupted
run. CUDA equality remains limited to a compatible runtime stack and the parent's
recorded deterministic-algorithm setting.

## Corrected 48-clip weight-one continuation to 3,000 steps

The immutable continuation path was exercised on the corrected 48-clip weight-one
control. The original 1,000-step report and checkpoint retained SHA-256 values
`aa04d212be069771875d6716a2bcf7b23456325e688656adb528cae6c2865bb6` and
`8707d60974765a4323c28ae30430cbf2a3564993f23df2c887d077ad3cdb2b82`
after the run. The new child adds 2,000 steps for 3,000 cumulative steps. Its report
SHA-256 is
`dd469d924e4aa3f104d5eb00ec0961d6ee1d0a1d04091b53d1ff1c3ed4bfa83c`,
and its checkpoint SHA-256 is
`2585665eceb5f0d8f844a53b7071ba3b57812620e86a32b37ef439acb2cce124`.
The resumed fixed ordinary diagnostic exactly reproduced the parent's final value
of 0.466799 before falling to 0.318813. The resumed matched-endpoint diagnostic was
0.248565 and fell to 0.163150. These are optimization results on the same 48
training clips, not evidence about unseen inputs.

On the saved training-run Euler samples, continuation reduced global
premultiplied-RGBA MAE from 0.113222 to 0.080157, target-visible foreground MAE
from 0.241393 to 0.191515, and target-background MAE from 0.091047 to 0.059806.
Alpha IoU rose from 0.269330 to 0.346648, alpha precision from 0.270772 to
0.347963, and alpha recall from 0.974618 to 0.984419. The predicted-to-target
visible-canvas ratio fell from 4.16543 to 3.07128, although it remained far above
the ideal value of one. The child's sparse exact-target evaluation SHA-256 is
`2eaea6e9f160a12dbcea3992d88ae78e2a0f63a90a5130b3b32b49e0102f69eb`.

A matched comparison then replayed all 48 exact requests and phases with CUDA seed
`20260816`, deterministic algorithms, and the same shared noise SHA-256
`7ef7e72e54533b35a199f31d9df89093994b4c9f5d1ab32dc313896e7a2feaf0`.
For Euler-32, continuation reduced global, foreground, and background
premultiplied MAE from 0.112567, 0.242626, and 0.089966 to 0.080271, 0.190816,
and 0.059936. For the direct endpoint sampler, the same values fell from 0.099013,
0.232251, and 0.075818 to 0.065057, 0.170303, and 0.044822. At 3,000 steps the
endpoint sampler therefore had lower global and region-specific errors and lower
temporal-delta error (0.038164 versus 0.051903), while Euler-32 retained slightly
higher alpha IoU (0.355589 versus 0.348764), higher alpha crispness (0.607406
versus 0.520116), and less translucent visible alpha (0.783677 versus 0.914064).
The complete matched sidecar SHA-256 is
`05379f957600ed409d894d27b42ba15115fae6ce5798fba48f3fcc8af1792d86`.
The child Euler-32 and endpoint inference report SHA-256 values are respectively
`1c85a4096f01a3246069d5f252b135bc29942e549f209755051f0435c3e86c48`
and `911108dc045de59343f28024b3da45c6ba639a893171537ec69030aa309d67f7`.

Raw-alpha nearest-neighbor previews were inspected for the same fixed animal/walk,
humanoid/attack, creature/shoot, and object/spawn representatives. Continued
Euler-32 outputs had fewer detached colored pixels and more recognizable central
forms than their 1,000-step counterparts, but remained substantially noisy and did
not reliably reproduce the target identity or action, especially for Lady Bugsy
and the spring booster. Endpoint outputs had lower numeric errors but remained
softer, more translucent, and visually less crisp. The preview index SHA-256 is
`159509b96e279e4e0baf196f72ab8729568c22488dd6a24022f23f134b8d4c0c`;
the linked raw NPY arrays remain authoritative. All results are in-sample
memorization diagnostics on one recorded CUDA stack and make no held-out,
open-vocabulary, or semantic-generalization claim.

## TMWA causal16 baseline at 1,000 steps

The first TMWA causal run uses sixteen exact fixed-eight, 64 x 64 training clips:
down-facing idle and walk for eight target-distinct identities (five animal, two
monster, and one humanoid identity). The complete immutable training report has
SHA-256 `147acc50a41b9bbffd905d0636520aeaebfc73a45755982a64d235dc5ac9fdbf`;
its safely loadable checkpoint has SHA-256
`7b844bb14276c64c16bc4bd723e8ac55d0d9cdd6b01079e957df0fc5aac01f7c`.
`torch.load(weights_only=True, map_location="cpu")` reconstructed the exact stored
architecture and all 88 denoiser plus 16 condition-encoder state tensors.

An independent replay on the recorded Torch 2.6.0/CUDA 12.4 stack reproduced all
three fixed diagnostic tensor hashes, both initial and final ordinary and endpoint
losses, both final action-permutation losses, and all sixteen directed swap rows
exactly. Fixed ordinary loss fell from 2.595508 to 0.247897 (90.45%), while the
matched pure-noise endpoint loss fell from 2.597627 to 0.115317 (95.56%). The final
endpoint action permutation raised aggregate loss by only 0.004227. Twelve of
sixteen directed swaps raised loss, but four lowered it; all four negative rows are
idle-to-walk substitutions. This is weak and asymmetric action-token sensitivity,
not a complete steering result.

Four new no-clobber inference runs compare Euler-32 and the direct endpoint sampler
under exact and action-swapped requests. Every run uses seed `20260812`, shared noise
SHA-256 `d1851b428b3f39de2665f5a9900dc387f0419a33e350ed4d6ea2b7d0f8c30fa7`,
identical request order and phase rows, and identical per-row noise hashes. The
inference report SHA-256 values are:

- Euler-32 exact:
  `b6635a0615d7c8bac74ef11b14a1144ff8ddae61f6e59275ead8e8ecc6e352a7`;
- endpoint exact:
  `bbab0c0b025586b6ccd660def730ba3d698c25e97528c3171a98bfba4a801602`;
- Euler-32 action swap:
  `da1c6b77771ec09ae2e8606746e5aa77d10281ff143f6bc1235b744865ce80cd`;
  and
- endpoint action swap:
  `c369fdd47b1b92ed67917a59d76beb1486038146d3b572720692c8f4e6d4b65b`.

At alpha threshold zero, the endpoint sampler improves exact-request global,
target-visible, and target-background premultiplied-RGBA MAE from Euler's 0.067756,
0.134501, and 0.051493 to 0.051985, 0.117301, and 0.035189. At threshold 127,
endpoint alpha IoU is 0.795010 versus Euler's 0.745535; precision is 0.854059
versus 0.795113 and recall is 0.920073 versus 0.921897. Endpoint temporal-delta MAE
is also lower, 0.027063 versus 0.034863. It is therefore the numerically stronger
sampler for this endpoint-supervised checkpoint. Raw nearest-neighbor previews are
still visibly blurry and speckled, with detached colored alpha noise. Identities are
recognizable, but the output is not clean pixel art.

The action-separation audit is more important than the aggregate reconstruction
gain. The eight target idle/walk pairs have mean premultiplied-RGBA separation
0.032581. Euler retains mean generated separation 0.006593, or 28.38% of target;
endpoint retains 0.009063, or 37.52%. Endpoint pair ratios for serqet, crystal
spider, ice skull, tortuga, logmonster, penguin, golden skull, and sasquatch are
respectively 0.2174, 0.5146, 0.5291, 0.4152, 0.2176, 0.4798, 0.5301, and 0.0977.
For both samplers every idle generation is closer to its idle target, while only
one of eight walk generations (logmonster) is closer to its walk target. Swapping
walk to idle moves all eight generations toward the idle replacement target;
swapping idle to walk moves zero of eight toward the walk replacement target. The
learned control is strongly idle-biased rather than symmetric idle/walk steering.

The complete matched evaluation has SHA-256
`569ebfacb35fd753cbfb78ec3d6e6fe3e1cdaa7d20150a1b6a79cbe9b97ec2e1`.
The dedicated causal-separation sidecar has SHA-256
`dfd81cfc7b2077ca8aa0ea7782405f41be7a08c11700ce7b9aa89578c0f365ba`.
The display-only target/Euler/endpoint preview index has SHA-256
`889553ac095e60e6beef6c279edb0d118d67b08148fe5831c651be958befc1c3`.
All requests, identities, action labels, and targets occur in training, so this is
memorization and token-sensitivity evidence only, not held-out generalization.

The pinned next experiment is an immutable 1,000-step continuation to 2,000
cumulative steps with the exact parent configuration unchanged: 64 x 64, eight
frames, patch size four, model dimension 128, depth four, four heads, condition
dimension 128, learning rate `3e-4`, foreground weight two, alpha weight one, and
matched-endpoint weight one. A second 1,000-step continuation to 3,000 is justified
only if matched endpoint evaluation beats all of these stage gates: global
premultiplied-RGBA MAE below 0.051985, action-separation ratio above 0.375176, at
least one idle-to-walk substitution moving toward the walk target, and more than
one walk generation preferring its walk target. Loss reduction alone is not a
continuation gate. If steering remains idle-biased, keep the continuation as
evidence and test a separate from-scratch endpoint-weight ablation rather than
mutating the continuation configuration.

## TMWA alpha-weight quality sweep and selected checkpoint

The exact sixteen-clip TMWA causal subset was used for a controlled quality
sweep. All compared runs retain the same ordered inputs, eight 64 x 64 RGBA
frames, architecture, seed, optimizer, learning rate, foreground weight two,
matched-endpoint weight one, and shared inference noise SHA-256
`d1851b428b3f39de2665f5a9900dc387f0419a33e350ed4d6ea2b7d0f8c30fa7`.
The successful ablation changes only the alpha-channel residual multiplier from
one to four. All results are exact in-sample reconstruction and action-token
sensitivity measurements, not held-out or open-vocabulary generation.

At 3,000 steps, alpha weight four decisively beat the alpha-weight-one control.
Endpoint premultiplied-RGBA MAE fell from `0.042123` to `0.030303`, target
background MAE from `0.033598` to `0.014410`, temporal-delta MAE from `0.026199`
to `0.020968`, and alpha IoU at 127 rose from `0.882087` to `0.929190`. Mean
generated-to-target idle/walk separation rose from `0.764567` to `0.825175`, and
both actions preferred their correct target in seven of eight identity pairs.
The alpha-four checkpoint/report SHA-256 values are respectively
`804e4077e0e8d6a20237cd078b97b8069058fd7635390c8d0764ec9eea847826` and
`1c471220c2b47cdbd821d8921520db97722abe808f07f02a50f07a499fe035c5`;
the matched evaluation is
`7e69c119370a9e32ad82c75fec72ec88a730a24d0051ec86d322e4d1348e7ccd`.

Immutable 1,000-step continuations were evaluated at 4,000, 5,000, and 6,000
cumulative steps. Endpoint results were:

| Step | PM-RGBA MAE | Foreground MAE | Background MAE | Alpha IoU @127 | Temporal MAE | Action separation | Both correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3,000 | 0.030303 | 0.088261 | 0.014410 | 0.929190 | 0.020968 | 0.825175 | 7/8 |
| 4,000 | 0.025557 | 0.075670 | 0.011045 | 0.949561 | 0.017328 | 0.934535 | 7/8 |
| 5,000 | 0.022242 | 0.068638 | 0.009056 | 0.970442 | 0.017293 | 1.0134 | 7/8 |
| 6,000 | 0.024113 | 0.062863 | 0.012851 | 0.979635 | 0.016642 | 1.1021 | 8/8 |

Step 6,000 is selected for strongest subject reconstruction, animation, silhouette,
and symmetric action response: all eight idle-to-walk swaps move toward the walk
target and all eight walk outputs prefer the walk target. Step 5,000 remains the
cleaner global/background-error alternative and is not superseded or deleted.
The selected checkpoint SHA-256 is
`394880c5e067059f01b1f9c2462e75bae66705944e11f04c0a5b058e9689b761`,
report SHA-256 is
`7fef539de909b6612de95c40168c98de02f25f93078433adcd6ec529f26e01de`,
and matched evaluation SHA-256 is
`a1e151788c00f977d5b167e1e39a71bb02682ac8234cbd235c6a496afb5504cd`.

Display calibration is explicitly an optimistic training-target estimate.
Step 6,000 selects hard-alpha threshold 144 and a generated-only, clip-global
64-color palette. The calibrated decode has PM-RGBA MAE `0.020253`, alpha IoU
`0.911034`, and temporal-delta MAE `0.011305`. Its no-clobber bundle index SHA-256
is `25c78b85b6dc9f6c463e081c5504203881d134842c3bc7427e2d0c2612a22aa0`.
Visual inspection still finds occasional isolated high-confidence background
pixels; they are retained as model errors rather than removed by an uncalibrated
connected-component heuristic.

Two additional 3,000-step ablations were rejected. Foreground weight four improved
foreground MAE only slightly (`0.083632` versus `0.088261`) while worsening global
MAE, background MAE, IoU, temporal error, and action separation; its matched report
SHA-256 is
`afe7b5f353184126660245bcbfd13a6fd7e526fff0c8f06c72012028f9a15e63`.
Alpha weight eight also worsened global (`0.033606`), foreground (`0.088758`),
background (`0.017874`), and temporal (`0.024233`) errors without improving action
separation; its matched report SHA-256 is
`3703cd17e82b9e13c9595a7bc04ec207ed6c2b4494e5bbe520f347e80f4b8964`.
Alpha weight four and foreground weight two therefore remain the evidence-backed
objective for this bounded experiment.
