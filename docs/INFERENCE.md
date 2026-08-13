# Verified PixelDiT checkpoint inference

`spritelab.inference` is the standalone replay path for the current tiny-corpus
PixelDiT checkpoints. It reconstructs the exact stored `PixelDiTConfig`, nested
`ConditioningSchema`, and `SpriteConditionEncoder`, then runs the existing backward
Euler rectified-flow sampler.

This is an overfit-replay and conditioning-diagnostic API. The description field may
contain arbitrary text, but accepting arbitrary bytes is not evidence that a tiny
checkpoint understands unseen concepts or generalizes beyond its training clips.

## Example

```python
from pathlib import Path

from spritelab.captions import SpriteGenerationRequest
from spritelab.inference import (
    CheckpointInferenceConfig,
    run_checkpoint_inference,
)
from spritelab.storage import DiskGuard

checkpoint = Path("data/experiments/example/checkpoint.pt")
training_report = Path("data/experiments/example/overfit-report.json")

result = run_checkpoint_inference(
    checkpoint,
    Path("data/inference/example-run"),
    requests=(
        SpriteGenerationRequest(
            description="fetid rat",
            entity_class="animal",
            action="idle",
            view="side",
            direction="right",
            loop_mode="loop",
        ),
        SpriteGenerationRequest(
            description="fetid rat",
            entity_class="animal",
            action="run",
            view="side",
            direction="right",
            loop_mode="loop",
        ),
    ),
    frame_phases=(
        (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875),
        (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875),
    ),
    expected_checkpoint_sha256="<64 lowercase hex characters from the trusted report>",
    source_report_path=training_report,
    expected_source_report_sha256="<64 lowercase hex characters from the source index>",
    config=CheckpointInferenceConfig(
        seed=20260812,
        sample_steps=32,
        noise_strategy="shared",
        device="cpu",
    ),
    disk_guard=DiskGuard(Path("C:/"), min_free_bytes=100 * 1024**3),
)
```

Supply one explicit phase row per request and exactly the checkpoint's stored number
of frames. Loop and ping-pong phases must lie in `[0, 1)`; one-shot phases may include
`1.0` and must be nondecreasing. Structured labels must belong to the conditioning
vocabulary stored in the checkpoint.

`noise_strategy="independent"` draws one generator stream across the batch.
`noise_strategy="shared"` repeats one initial-noise clip for every request. Shared
noise is the appropriate controlled comparison when only an action or another
condition should change; it removes stochastic input as a confound.

## Verification and safety contract

The caller must provide the expected checkpoint SHA-256. The file is hashed before
deserialization. Loading is explicitly limited to
`torch.load(weights_only=True, map_location="cpu")`; there is no unsafe pickle
fallback. Model and encoder state dictionaries must contain finite tensors and must
load strictly against the reconstructed configuration. The stored training geometry,
encoder width, step, precision label, and runtime facts are type-checked and checked
for internal agreement.

A source overfit report is optional but recommended. If supplied, its observed hash
is recorded, an expected hash can be required, and its declared checkpoint hash must
match the checkpoint being loaded.

Existing sample or report paths are never overwritten. NumPy samples and the report
are written through same-directory temporary files and atomically published. Passing
a `DiskGuard` performs a capacity preflight and guards each write. The project-wide
100 GiB floor is represented by `100 * 1024**3`, as shown above.

## Output contract

Each sample is a canonical uint8 straight-RGBA NumPy array with shape `[T, H, W, 4]`.
Names use the request index and a hash of the complete request plus explicit phases.
`inference-report.json` is canonical, sorted UTF-8 JSON and records:

- checkpoint and optional source-report paths and SHA-256 hashes;
- the exact stored model, conditioning, training, and training-runtime facts;
- original request fields, the rendered documentation prompt, the exact stripped
  description seen by the encoder, every structured label, token IDs, attention mask,
  and both the requested and model-facing float32 frame phases;
- seed, noise strategy, generator-state hashes, whole-batch and per-row noise hashes;
- sampler algorithm and step count plus the inference Torch/device/runtime facts; and
- each output's relative path, dtype, shape, byte count, file SHA-256, and typed-array
  content SHA-256.

No timestamp is inserted, so compatible same-seed deterministic replays can produce
the same report hash in separate destination directories. Byte-identical replay is
only expected on a compatible deterministic Torch/device stack; the recorded hashes
make any divergence explicit.

## Claim boundary

The API proves that a verified checkpoint can be reconstructed, conditioned, sampled,
and attributed without depending on an external materialization manifest. For the
current tiny overfit experiments, outputs remain in-sample diagnostics. Novel prompts,
entities, poses, actions, or compositions require held-out evaluation before any
generalization claim is appropriate.
