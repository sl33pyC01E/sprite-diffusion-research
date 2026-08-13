# Widelands target-distinct carry/walk baseline

This is a one-identity, in-sample memorization and conditioning diagnostic. It is
not evidence of cross-identity, open-vocabulary, or semantic generalization.

The source materialization has SHA-256
`8503d46d9305890393df7e31421e2f4261d9e8ae620b688a4112fa24d1973616`.
Its corrected schema-v2 fixed-eight-frame audit has SHA-256
`4bf003409b5d514c8dec7a2a94f38f3be44a7abb5098503b03f057451bcd3af0`.
The training split has 166 clips, but only 6 of its 42 nominal action-only groups
have different exact target tensors between actions. The other 36 are authored
aliases, not learnable steering evidence:

- Twelve critter `eat`/`idle` groups resolve to the same `idle_1.png` carrier.
  For example, the pinned Widelands `data/world/critters/wisent/init.lua`
  declaration sets `eating.basename = "idle"` and says the animation remains a
  TODO.
- Twenty-four animal `carry`/`walk` groups resolve to the same directional
  `walk_*.png` carrier. For example, the pinned
  `data/tribes/workers/empire/donkey/init.lua` declaration sets
  `walkload.basename = "walk"` and says the animation remains a TODO.
- The Atlantean spiderbreeder is the sole target-distinct identity. Its six
  directional `walkload_*.png` carriers differ from its six `walk_*.png`
  carriers.

The endpoint planner now keeps at most one action per exact target digest and
requires at least two distinct digests. Alias rows remain in ordinary stochastic
training; they are excluded only from the causal endpoint term and are reported
with explicit reasons.

## Immutable subset and audits

The no-clobber subset manifest is
`data/processed/widelands-ss14-model-ready-v1/widelands-causal-carry-walk-12-materialization-v1.json`,
SHA-256 `25b4fc9dd3e58ab2336ca0201419bb8c01c6bfdbc94e89f02ba5a6eb818fb505`.
It contains one humanoid identity, six `carry` clips, six `walk` clips, and two
clips for each of six directions. Every clip is a 64x64 loop projected to eight
frames. All 12 rows form six target-distinct endpoint groups with zero exclusions.

The subset schema-v2 training-readiness audit has SHA-256
`ed7d9985e3923d70f9cf0c2c5f98f525e97d2279cb3f6c64143c825b4ee8b460`.
The subset pixel-quality audit has SHA-256
`96a6fa87628e7b7b2bec6772e1b0e4724a6ef87f0bd1e6270700ca918cf2926c`.
It finds no fully transparent clips, no fully opaque clips, no opaque magenta
sentinel pixels, and no occupied corners. All 120 native frames touch a canvas
border because bottom-center normalization grounds the sprite; the occupied
border fraction is 4.81%, so this fact must remain visible in interpretation.
The earlier schema-v1 audit files remain preserved; they are superseded for the
target-distinct endpoint contract and are not rewritten.

The canonical design audit at
`data/index/reports/widelands-model-ready-training-design-v1.json`, SHA-256
`fd6d9bff4ad6f481d8b0c869ea28a606db3869bc3dbe405df2b9647d5e3499cc`,
records every exact target-duplicate component, source-blob component, nominal
condition group, endpoint exclusion, carrier member, and selected row. Across all
183 splits it finds 40 fixed-target duplicate components with 49 excess rows and
40 shared source-blob components; the train-only condition inventory is exactly 36
cross-action aliases plus 6 target-distinct groups.

## Bounded baseline

The pinned launcher is `scripts/run_widelands_causal12_baseline_v1.py`. Its
configuration is 64x64x8, patch size 4, model width 128, depth 4, four heads,
condition width 128, float32, learning rate `3e-4`, foreground multiplier 2,
alpha multiplier 1, endpoint multiplier 1, seed 0, 1,000 steps, and 32 Euler
sample steps. The output is the new path
`data/experiments/widelands-spiderbreeder-carry-walk-b64-f8-baseline-v1-1000`;
the launcher never overwrites it and reserves two GiB above the 100-GiB disk
floor.

First verify the complete hash-pinned plan without touching the GPU:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
python scripts\run_widelands_causal12_baseline_v1.py --preflight-only
```

The exact production architecture and all 12 rows also passed a one-step CPU-only
wiring smoke with one Euler sample step. Its immutable report is
`data/experiments/widelands-causal12-b64-f8-cpu-smoke-v1/overfit-report.json`,
SHA-256 `14619790eae06ffcd0a00442b8f25872188eb39a09f33329769d4c1e8ba0d087`;
the checkpoint SHA-256 is
`059f051b85930fc46bcb5f4411bc82b33d863bc7ff7a96d31432653ac3da92ed`.
The exact-target evaluation has SHA-256
`0b99bc427979c2d17f3954b2dd5302c1d2fdd6978b02e26b5b49a31ebb18c910`.
All six generated carry/walk pair distances are zero after one step, as expected
for an untrained smoke; this proves only end-to-end execution and is not a visual
or steering result. The raw target and CPU-smoke APNG/contact-sheet preview index
has SHA-256
`ae6e2c40fc7c665e5f5d7ef88e1cf6a47e65447ad5a0d8ff70567ba39dcefa5e`.

Before launch, inspect `nvidia-smi` and every live Python command line. Do not
launch while the user's `neural-game` training or any other GPU job is active.
Only after that external audit is clean, use a fresh no-clobber log pair and the
required attestation flag:

```powershell
$stdout = 'data\index\reports\widelands-causal12-baseline-v1-1000.out'
$stderr = 'data\index\reports\widelands-causal12-baseline-v1-1000.err'
if ((Test-Path $stdout) -or (Test-Path $stderr)) { throw 'Refusing to replace logs' }
python scripts\run_widelands_causal12_baseline_v1.py --confirm-gpu-idle 1> $stdout 2> $stderr
```

Evaluation must use the subset manifest and exact generated sample hashes. Compare
carry and walk only within the same direction and matched stochastic input. Report
per-swap deltas, exact-target metrics, temporal metrics, alpha precision/recall,
and raw nearest-neighbor APNG/contact-sheet previews. Do not interpret this one
spiderbreeder as evidence that the model understands `carry` across entities.
