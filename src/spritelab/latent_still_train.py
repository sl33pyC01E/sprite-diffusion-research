"""Resumable identity-balanced rectified-flow training for MUGEN still latents."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from spritelab.models.latent_still_dit import LatentStillDiT, LatentStillDiTConfig
from spritelab.storage import DiskGuard

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency boundary
    torch = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None

Precision = Literal["float32", "bfloat16"]


class LatentStillTrainingError(ValueError):
    """Raised when an input, resume, or training artifact violates its contract."""


@dataclass(frozen=True, slots=True)
class LatentStillTrainingConfig:
    """Quality-first scratch-training contract for one 4090-class GPU."""

    batch_size: int = 4
    gradient_accumulation: int = 8
    learning_rate: float = 1e-4
    minimum_learning_rate: float = 1e-5
    warmup_steps: int = 1_000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.9999
    conditioning_dropout_probability: float = 0.1
    endpoint_probability: float = 0.25
    steps: int = 30_000
    log_every: int = 25
    validate_every: int = 500
    checkpoint_every: int = 5_000
    recovery_checkpoint_every: int | None = None
    recovery_checkpoint_slots: int = 2
    validation_rows: int = 32
    seed: int = 20260818
    device: str = "cuda"
    precision: Precision = "bfloat16"
    model: LatentStillDiTConfig = LatentStillDiTConfig()

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "gradient_accumulation",
            "steps",
            "log_every",
            "validate_every",
            "checkpoint_every",
            "recovery_checkpoint_slots",
            "validation_rows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.recovery_checkpoint_every is not None and (
            isinstance(self.recovery_checkpoint_every, bool)
            or not isinstance(self.recovery_checkpoint_every, int)
            or self.recovery_checkpoint_every <= 0
        ):
            raise ValueError("recovery_checkpoint_every must be null or a positive integer")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int):
            raise ValueError("warmup_steps must be an integer")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("minimum_learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1)")
        for name in ("conditioning_dropout_probability", "endpoint_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be non-empty text")


@dataclass(frozen=True, slots=True)
class LatentStillRow:
    sequence_id: str
    identity_id: str
    verb: str
    split: str
    prompt: str
    prompt_row: int
    latent_path: Path
    latent_file_sha256: str
    latent_array_sha256: str
    eligible_frame_indices: tuple[int, ...] = tuple(range(8))


@dataclass(frozen=True, slots=True)
class LatentStillCorpus:
    rows: tuple[LatentStillRow, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    context_embeddings: np.ndarray
    context_mask: np.ndarray
    channel_mean: tuple[float, ...]
    channel_standard_deviation: tuple[float, ...]
    contract: dict[str, Any]
    resident_latents: tuple[np.ndarray, ...] | None = None


@dataclass(frozen=True, slots=True)
class LatentStillTrainingResult:
    output_directory: Path
    report_path: Path
    training_checkpoint_path: Path
    inference_checkpoint_path: Path
    report_sha256: str


def build_hierarchical_sampler_index(
    rows: tuple[LatentStillRow, ...], indices: tuple[int, ...]
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Build identity -> verb -> sequence row indices with stable ordering."""

    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in indices:
        row = rows[index]
        grouped[row.identity_id][row.verb].append(index)
    return {
        identity: {
            verb: tuple(sorted(values))
            for verb, values in sorted(verbs.items(), key=lambda item: item[0].encode())
        }
        for identity, verbs in sorted(grouped.items(), key=lambda item: item[0].encode())
    }


def sample_hierarchical_batch(
    index: dict[str, dict[str, tuple[int, ...]]],
    *,
    batch_size: int,
    generator: Any,
    frame_indices_by_row: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Sample identities and their verbs uniformly, then sequence and frame."""

    runtime = _require_torch()
    if not index:
        raise ValueError("sampler index cannot be empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    identities = tuple(index)
    output = []
    for _ in range(batch_size):
        identity = identities[int(runtime.randint(len(identities), (1,), generator=generator))]
        verbs = tuple(index[identity])
        verb = verbs[int(runtime.randint(len(verbs), (1,), generator=generator))]
        candidates = index[identity][verb]
        row = candidates[int(runtime.randint(len(candidates), (1,), generator=generator))]
        eligible = tuple(range(8)) if frame_indices_by_row is None else frame_indices_by_row[row]
        if not eligible:
            raise ValueError(f"sampler row {row} has no eligible frames")
        frame = eligible[int(runtime.randint(len(eligible), (1,), generator=generator))]
        output.append((row, frame))
    return tuple(output)


def normalize_latents(
    value: np.ndarray,
    mean: tuple[float, ...],
    standard_deviation: tuple[float, ...],
) -> np.ndarray:
    """Apply the immutable train-split channel normalization contract."""

    if value.ndim != 4 or value.shape[1:] != (8, 64, 64):
        raise ValueError("latent batch must have shape [B,8,64,64]")
    if len(mean) != 8 or len(standard_deviation) != 8:
        raise ValueError("latent normalization must contain eight channels")
    if any(not math.isfinite(item) for item in (*mean, *standard_deviation)) or any(
        item <= 0 for item in standard_deviation
    ):
        raise ValueError("latent normalization values must be finite with positive std")
    means = np.asarray(mean, dtype=np.float32)[None, :, None, None]
    stds = np.asarray(standard_deviation, dtype=np.float32)[None, :, None, None]
    return np.ascontiguousarray((value.astype(np.float32) - means) / stds)


def load_latent_still_corpus(
    plan_path: Path | str,
    latent_manifest_path: Path | str,
    text_manifest_path: Path | str,
    *,
    verify_latent_files: bool = True,
) -> LatentStillCorpus:
    """Load and hash-verify the complete plan/latent/text closure."""

    plan_file = Path(plan_path).resolve()
    latent_file = Path(latent_manifest_path).resolve()
    text_file = Path(text_manifest_path).resolve()
    plan_bytes = plan_file.read_bytes()
    latent_bytes = latent_file.read_bytes()
    text_bytes = text_file.read_bytes()
    plan = _json_object(plan_bytes, "training plan")
    latent = _json_object(latent_bytes, "latent manifest")
    text = _json_object(text_bytes, "text manifest")
    plan_records = _records(plan, "records", "training plan")
    latent_records = _records(latent, "records", "latent manifest")
    text_rows = _records(text, "rows", "text manifest")
    if plan.get("artifact_kind") != "mugen_latent_still_sequence_training_plan":
        raise LatentStillTrainingError("training plan has the wrong artifact kind")
    if latent.get("artifact_kind") != "mugen_frozen_rgba_autoencoder_latent_cache":
        raise LatentStillTrainingError("latent cache has the wrong artifact kind")
    if text.get("artifact_kind") != "frozen_clip_token_hidden_state_cache":
        raise LatentStillTrainingError("text cache has the wrong artifact kind")
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    text_source = text.get("source")
    if (
        not isinstance(text_source, dict)
        or text_source.get("training_plan_file_sha256") != plan_sha256
    ):
        raise LatentStillTrainingError("text cache was not encoded from this exact plan")
    plan_source = plan.get("source")
    latent_source = latent.get("source")
    if not isinstance(plan_source, dict) or not isinstance(latent_source, dict):
        raise LatentStillTrainingError("plan/latent source contracts are missing")
    materialization_lineage = _materialization_sources_compatible(plan_source, latent_source)
    if materialization_lineage is None:
        raise LatentStillTrainingError("plan and latent cache materializations differ")
    latent_by_id = _unique(latent_records, "sequence_id", "latent manifest")
    prompt_by_text = _unique(text_rows, "prompt", "text manifest")
    plan_sequence_ids = {row.get("sequence_id") for row in plan_records}
    if not plan_sequence_ids.issubset(latent_by_id):
        raise LatentStillTrainingError("latent cache lacks plan sequences")

    text_root = text_file.parent
    arrays = text.get("arrays")
    if not isinstance(arrays, dict):
        raise LatentStillTrainingError("text cache array records are missing")
    embeddings = _load_text_array(text_root, arrays, "embeddings", np.float16, (77, 768))
    masks = _load_text_array(text_root, arrays, "attention_mask", np.bool_, (77,))
    if embeddings.shape[0] != len(text_rows) or masks.shape[0] != len(text_rows):
        raise LatentStillTrainingError("text cache row count differs from arrays")

    latent_root = latent_file.parent
    rows = []
    resident_latents: list[np.ndarray] | None = [] if verify_latent_files else None
    for plan_record in sorted(plan_records, key=lambda item: str(item.get("sequence_id")).encode()):
        sequence_id = _required_text(plan_record, "sequence_id")
        identity_id = _required_text(plan_record, "identity_id")
        split = _required_text(plan_record, "split")
        prompt = _required_text(plan_record, "prompt")
        conditioning = plan_record.get("conditioning")
        if not isinstance(conditioning, dict):
            raise LatentStillTrainingError(f"conditioning is missing for {sequence_id}")
        verb = _required_text(conditioning, "verb")
        target = plan_record.get("target")
        if target is None:
            raw_eligible = list(range(8))
        elif not isinstance(target, dict):
            raise LatentStillTrainingError(f"target is missing for {sequence_id}")
        else:
            raw_eligible = target.get("eligible_frame_indices", list(range(8)))
        if (
            not isinstance(raw_eligible, list)
            or not raw_eligible
            or any(
                isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < 8
                for frame in raw_eligible
            )
            or raw_eligible != sorted(set(raw_eligible))
        ):
            raise LatentStillTrainingError(f"eligible frames are invalid for {sequence_id}")
        latent_record = latent_by_id[sequence_id]
        if latent_record.get("identity_id") != identity_id or latent_record.get("split") != split:
            raise LatentStillTrainingError(f"latent identity/split differs for {sequence_id}")
        target = plan_record.get("target")
        latent_source_record = latent_record.get("source")
        if isinstance(target, dict):
            if not isinstance(latent_source_record, dict):
                raise LatentStillTrainingError(f"latent source is absent for {sequence_id}")
            for key in ("array_content_sha256", "file_sha256", "relative_path"):
                if latent_source_record.get(key) != target.get(key):
                    raise LatentStillTrainingError(
                        f"latent source target differs for {sequence_id} at {key}"
                    )
        text_row = prompt_by_text.get(prompt)
        if text_row is None:
            raise LatentStillTrainingError(f"text cache lacks prompt for {sequence_id}")
        prompt_row = text_row.get("row_index")
        if isinstance(prompt_row, bool) or not isinstance(prompt_row, int):
            raise LatentStillTrainingError("text row index is invalid")
        relative = _required_text(latent_record, "relative_path")
        path = (latent_root / relative).resolve()
        if latent_root not in path.parents:
            raise LatentStillTrainingError(f"latent path escapes root for {sequence_id}")
        row = LatentStillRow(
            sequence_id=sequence_id,
            identity_id=identity_id,
            verb=verb,
            split=split,
            prompt=prompt,
            prompt_row=prompt_row,
            latent_path=path,
            latent_file_sha256=_required_text(latent_record, "file_sha256"),
            latent_array_sha256=_required_text(latent_record, "array_content_sha256"),
            eligible_frame_indices=tuple(raw_eligible),
        )
        if resident_latents is not None:
            latent_value = _load_latent(row, verify_hashes=True)
            latent_value.setflags(write=False)
            resident_latents.append(latent_value)
        rows.append(row)
    train_indices = tuple(index for index, row in enumerate(rows) if row.split == "train")
    validation_indices = tuple(index for index, row in enumerate(rows) if row.split == "validation")
    if not train_indices or not validation_indices:
        raise LatentStillTrainingError("training and validation splits must both be non-empty")
    if {rows[index].identity_id for index in train_indices}.intersection(
        rows[index].identity_id for index in validation_indices
    ):
        raise LatentStillTrainingError("training and validation identities overlap")
    normalization = latent.get("normalization")
    if not isinstance(normalization, dict):
        raise LatentStillTrainingError("latent normalization is missing")
    mean = _float_tuple(normalization.get("channel_mean"), "channel mean")
    std = _float_tuple(
        normalization.get("channel_standard_deviation"), "channel standard deviation"
    )
    contract = {
        "latent_manifest_file_sha256": hashlib.sha256(latent_bytes).hexdigest(),
        "materialization_lineage": materialization_lineage,
        "plan_file_sha256": plan_sha256,
        "record_count": len(rows),
        "text_manifest_file_sha256": hashlib.sha256(text_bytes).hexdigest(),
        "train_identities": len({rows[index].identity_id for index in train_indices}),
        "train_rows": len(train_indices),
        "validation_identities": len({rows[index].identity_id for index in validation_indices}),
        "validation_rows": len(validation_indices),
    }
    contract["canonical_sha256"] = hashlib.sha256(_canonical_json(contract)).hexdigest()
    return LatentStillCorpus(
        rows=tuple(rows),
        train_indices=train_indices,
        validation_indices=validation_indices,
        context_embeddings=embeddings,
        context_mask=masks,
        channel_mean=mean,
        channel_standard_deviation=std,
        contract=contract,
        resident_latents=(tuple(resident_latents) if resident_latents is not None else None),
    )


def _materialization_sources_compatible(
    plan_source: dict[str, Any], latent_source: dict[str, Any]
) -> str | None:
    """Prove direct identity or a common immutable dense-manifest ancestor."""

    plan_sha256 = plan_source.get("materialization_file_sha256")
    latent_sha256 = latent_source.get("materialization_file_sha256")
    if isinstance(plan_sha256, str) and plan_sha256 == latent_sha256:
        return "exact_materialization_file_sha256"
    plan_dense = _dense_manifest_ancestor(plan_source, "plan")
    latent_dense = _dense_manifest_ancestor(latent_source, "latent cache")
    if plan_dense is not None and plan_dense == latent_dense:
        return f"common_dense_manifest_sha256:{plan_dense}"
    return None


def _dense_manifest_ancestor(source: dict[str, Any], label: str) -> str | None:
    path_value = source.get("materialization_path")
    expected_sha256 = source.get("materialization_file_sha256")
    if not isinstance(path_value, str) or not isinstance(expected_sha256, str):
        return None
    path = Path(path_value).resolve()
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise LatentStillTrainingError(f"{label} source materialization is unreadable") from error
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise LatentStillTrainingError(f"{label} source materialization hash differs")
    materialization = _json_object(payload, f"{label} source materialization")
    ancestor_source = materialization.get("source")
    if not isinstance(ancestor_source, dict):
        return None
    dense_sha256 = ancestor_source.get("dense_manifest_file_sha256")
    if not isinstance(dense_sha256, str) or len(dense_sha256) != 64:
        return None
    if any(character not in "0123456789abcdef" for character in dense_sha256):
        return None
    return dense_sha256


def run_latent_still_training(
    plan_path: Path | str,
    latent_manifest_path: Path | str,
    text_manifest_path: Path | str,
    output_directory: Path | str,
    *,
    config: LatentStillTrainingConfig | None = None,
    resume_checkpoint_path: Path | str | None = None,
    expected_resume_sha256: str | None = None,
    disk_guard: DiskGuard | None = None,
) -> LatentStillTrainingResult:
    """Train the scratch latent DiT into an immutable, resumable bundle."""

    runtime = _require_torch()
    experiment = config or LatentStillTrainingConfig()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace latent-still output: {output}")
    if (resume_checkpoint_path is None) != (expected_resume_sha256 is None):
        raise ValueError("resume checkpoint and expected SHA-256 must be supplied together")
    device = runtime.device(experiment.device)
    if device.type == "cuda" and not runtime.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if experiment.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("bfloat16 latent-still training requires CUDA")
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(3 * 1024**3, label="latent-still training checkpoint")
    corpus = load_latent_still_corpus(
        plan_path, latent_manifest_path, text_manifest_path, verify_latent_files=True
    )
    output.mkdir(parents=True, exist_ok=False)
    history_path = output / "training-history.jsonl"
    with history_path.open("x", encoding="utf-8", newline="\n") as history:
        return _train(
            runtime,
            corpus=corpus,
            output=output,
            history=history,
            config=experiment,
            device=device,
            resume_checkpoint_path=(
                Path(resume_checkpoint_path).resolve()
                if resume_checkpoint_path is not None
                else None
            ),
            expected_resume_sha256=expected_resume_sha256,
            disk_guard=guard,
        )


def _train(
    runtime: Any,
    *,
    corpus: LatentStillCorpus,
    output: Path,
    history: Any,
    config: LatentStillTrainingConfig,
    device: Any,
    resume_checkpoint_path: Path | None,
    expected_resume_sha256: str | None,
    disk_guard: DiskGuard,
) -> LatentStillTrainingResult:
    runtime.manual_seed(config.seed)
    if device.type == "cuda":
        runtime.cuda.manual_seed_all(config.seed)
    model = LatentStillDiT(config.model).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = runtime.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    sampler_generator = runtime.Generator(device="cpu").manual_seed(config.seed + 1)
    flow_generator = runtime.Generator(device=device).manual_seed(config.seed + 2)
    dropout_generator = runtime.Generator(device=device).manual_seed(config.seed + 3)
    start_step = 0
    lineage = None
    if resume_checkpoint_path is not None:
        assert expected_resume_sha256 is not None
        parent = _load_resume(
            runtime,
            resume_checkpoint_path,
            expected_sha256=expected_resume_sha256,
            corpus=corpus,
            config=config,
        )
        start_step = int(parent["step"])
        if start_step >= config.steps:
            raise LatentStillTrainingError("resume step must be below cumulative steps")
        model.load_state_dict(parent["raw_model"], strict=True)
        ema.load_state_dict(parent["ema_model"], strict=True)
        optimizer.load_state_dict(parent["optimizer"])
        sampler_generator.set_state(parent["rng_state"]["sampler"])
        flow_generator.set_state(parent["rng_state"]["flow"])
        dropout_generator.set_state(parent["rng_state"]["dropout"])
        runtime.set_rng_state(parent["rng_state"]["torch_cpu"])
        if device.type == "cuda":
            runtime.cuda.set_rng_state(parent["rng_state"]["cuda"], device=device)
        lineage = {
            "parent_checkpoint_path": str(resume_checkpoint_path),
            "parent_checkpoint_sha256": expected_resume_sha256,
            "parent_step": start_step,
        }
    sampler_index = build_hierarchical_sampler_index(corpus.rows, corpus.train_indices)
    eligible_frames = tuple(row.eligible_frame_indices for row in corpus.rows)
    training_evaluation_selection = _split_selection(
        corpus, corpus.train_indices, config.validation_rows
    )
    validation_selection = _validation_selection(corpus, config.validation_rows)
    dtype = runtime.bfloat16 if config.precision == "bfloat16" else runtime.float32
    autocast = config.precision == "bfloat16"
    model.train()
    latest_validation = None
    for step_index in range(start_step, config.steps):
        step = step_index + 1
        learning_rate = _learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        endpoint_fraction = 0.0
        dropout_fraction = 0.0
        for _ in range(config.gradient_accumulation):
            selection = sample_hierarchical_batch(
                sampler_index,
                batch_size=config.batch_size,
                generator=sampler_generator,
                frame_indices_by_row=eligible_frames,
            )
            clean, context, mask = _training_batch(runtime, corpus, selection, device=device)
            noise = runtime.randn(
                clean.shape, device=device, dtype=clean.dtype, generator=flow_generator
            )
            timesteps = runtime.rand(
                (clean.shape[0],), device=device, dtype=clean.dtype, generator=flow_generator
            )
            endpoint_rows = (
                runtime.rand((clean.shape[0],), device=device, generator=flow_generator)
                < config.endpoint_probability
            )
            timesteps = runtime.where(endpoint_rows, runtime.ones_like(timesteps), timesteps)
            expanded = timesteps[:, None, None, None]
            noisy = (1 - expanded) * clean + expanded * noise
            target = noise - clean
            dropout_rows = (
                runtime.rand((clean.shape[0],), device=device, generator=dropout_generator)
                < config.conditioning_dropout_probability
            )
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                predicted = model(
                    noisy,
                    timesteps,
                    context,
                    context_mask=mask,
                    context_dropout_mask=dropout_rows,
                )
                loss = runtime.nn.functional.mse_loss(predicted.float(), target.float())
                scaled_loss = loss / config.gradient_accumulation
            if not bool(runtime.isfinite(scaled_loss)):
                raise RuntimeError(f"non-finite latent-still loss at step {step}")
            scaled_loss.backward()
            accumulated += float(loss.detach().cpu())
            endpoint_fraction += float(endpoint_rows.float().mean().cpu())
            dropout_fraction += float(dropout_rows.float().mean().cpu())
        gradient_norm = float(
            runtime.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            .detach()
            .cpu()
        )
        optimizer.step()
        _ema_update(
            runtime,
            ema,
            model,
            0.0 if step <= config.warmup_steps else config.ema_decay,
        )
        validation = None
        if step == 1 or step % config.validate_every == 0 or step == config.steps:
            latest_validation = _validate(
                runtime,
                corpus,
                validation_selection,
                ema,
                device=device,
                dtype=dtype,
                autocast=autocast,
                seed=config.seed + 20_000,
            )
            validation = latest_validation
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            record = {
                "conditioning_dropout_fraction": dropout_fraction / config.gradient_accumulation,
                "endpoint_fraction": endpoint_fraction / config.gradient_accumulation,
                "gradient_norm_before_clip": gradient_norm,
                "learning_rate": learning_rate,
                "loss": accumulated / config.gradient_accumulation,
                "step": step,
                "validation": validation,
            }
            history.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
            history.flush()
            os.fsync(history.fileno())
        permanent_checkpoint = step % config.checkpoint_every == 0 or step == config.steps
        if permanent_checkpoint:
            _write_training_checkpoint(
                runtime,
                output / f"training-step-{step:07d}.pt",
                corpus=corpus,
                config=config,
                step=step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                sampler_generator=sampler_generator,
                flow_generator=flow_generator,
                dropout_generator=dropout_generator,
                device=device,
                disk_guard=disk_guard,
            )
        elif (
            config.recovery_checkpoint_every is not None
            and step % config.recovery_checkpoint_every == 0
        ):
            recovery_index = (
                (step // config.recovery_checkpoint_every - 1) % config.recovery_checkpoint_slots
            ) + 1
            _write_training_checkpoint(
                runtime,
                output / f"recovery-slot-{recovery_index}.pt",
                corpus=corpus,
                config=config,
                step=step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                sampler_generator=sampler_generator,
                flow_generator=flow_generator,
                dropout_generator=dropout_generator,
                device=device,
                disk_guard=disk_guard,
                replace_existing=True,
            )
    final_checkpoint = output / f"training-step-{config.steps:07d}.pt"
    inference_checkpoint = output / "checkpoint-ema.pt"
    _atomic_torch_save(
        runtime,
        inference_checkpoint,
        {
            "artifact_kind": "mugen_latent_still_dit_ema_inference_checkpoint",
            "config": asdict(config),
            "corpus": corpus.contract,
            "ema_policy": _ema_policy(config),
            "ema_model": ema.state_dict(),
            "normalization": {
                "channel_mean": list(corpus.channel_mean),
                "channel_standard_deviation": list(corpus.channel_standard_deviation),
            },
            "step": config.steps,
        },
        disk_guard=disk_guard,
    )
    final_training_evaluation = _validate(
        runtime,
        corpus,
        training_evaluation_selection,
        ema,
        device=device,
        dtype=dtype,
        autocast=autocast,
        seed=config.seed + 30_000,
    )
    report = {
        "artifact_kind": "mugen_latent_still_dit_training",
        "claim": (
            "identity-disjoint validation of scratch latent text-to-sprite generation; "
            "no held-out prompt generalization claim without decoded fixed-noise evaluation"
        ),
        "config": asdict(config),
        "corpus": corpus.contract,
        "ema_policy": _ema_policy(config),
        "final_training_evaluation": final_training_evaluation,
        "final_validation": latest_validation,
        "history": {
            "file_sha256": _file_sha256(output / "training-history.jsonl"),
            "path": "training-history.jsonl",
        },
        "inference_checkpoint": {
            "file_sha256": _file_sha256(inference_checkpoint),
            "path": inference_checkpoint.name,
        },
        "lineage": lineage,
        "runtime": _runtime_facts(runtime, device),
        "step": config.steps,
        "training_checkpoint": {
            "file_sha256": _file_sha256(final_checkpoint),
            "path": final_checkpoint.name,
        },
    }
    report_path = output / "training-report.json"
    report_payload = _canonical_json(report)
    _atomic_bytes(report_path, report_payload, disk_guard=disk_guard)
    return LatentStillTrainingResult(
        output_directory=output,
        report_path=report_path,
        training_checkpoint_path=final_checkpoint,
        inference_checkpoint_path=inference_checkpoint,
        report_sha256=hashlib.sha256(report_payload).hexdigest(),
    )


def _training_batch(
    runtime: Any,
    corpus: LatentStillCorpus,
    selection: tuple[tuple[int, int], ...],
    *,
    device: Any,
) -> tuple[Any, Any, Any]:
    latent_rows = []
    prompt_rows = []
    for row_index, frame_index in selection:
        row = corpus.rows[row_index]
        if corpus.resident_latents is None:
            latent = _load_latent(row, verify_hashes=False)
        else:
            latent = corpus.resident_latents[row_index]
        latent_rows.append(latent[frame_index])
        prompt_rows.append(row.prompt_row)
    normalized = normalize_latents(
        np.stack(latent_rows), corpus.channel_mean, corpus.channel_standard_deviation
    )
    clean = runtime.from_numpy(normalized).to(device)
    context = runtime.from_numpy(
        np.asarray(corpus.context_embeddings[prompt_rows], dtype=np.float32)
    ).to(device)
    mask = runtime.from_numpy(np.asarray(corpus.context_mask[prompt_rows], dtype=np.bool_)).to(
        device
    )
    return clean, context, mask


def _validation_selection(
    corpus: LatentStillCorpus, maximum_rows: int
) -> tuple[tuple[int, int], ...]:
    return _split_selection(corpus, corpus.validation_indices, maximum_rows)


def _split_selection(
    corpus: LatentStillCorpus,
    indices: tuple[int, ...],
    maximum_rows: int,
) -> tuple[tuple[int, int], ...]:
    by_identity = {}
    for index in indices:
        by_identity.setdefault(corpus.rows[index].identity_id, index)
    selected = [by_identity[key] for key in sorted(by_identity, key=str.encode)[:maximum_rows]]
    return tuple((index, corpus.rows[index].eligible_frame_indices[0]) for index in selected)


def _validate(
    runtime: Any,
    corpus: LatentStillCorpus,
    selection: tuple[tuple[int, int], ...],
    model: Any,
    *,
    device: Any,
    dtype: Any,
    autocast: bool,
    seed: int,
) -> dict[str, float]:
    clean, context, mask = _training_batch(runtime, corpus, selection, device=device)
    generator = runtime.Generator(device=device).manual_seed(seed)
    noise = runtime.randn(clean.shape, device=device, dtype=clean.dtype, generator=generator)
    model.eval()
    output = {}
    with runtime.no_grad():
        for label, timestep in (("midpoint", 0.5), ("endpoint", 1.0)):
            times = runtime.full((clean.shape[0],), timestep, device=device, dtype=clean.dtype)
            noisy = (1 - timestep) * clean + timestep * noise
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                predicted = model(noisy, times, context, context_mask=mask)
            output[f"{label}_velocity_mse"] = float(
                runtime.nn.functional.mse_loss(predicted.float(), (noise - clean).float()).cpu()
            )
    model.train()
    return output


def _write_training_checkpoint(
    runtime: Any,
    path: Path,
    *,
    corpus: LatentStillCorpus,
    config: LatentStillTrainingConfig,
    step: int,
    model: Any,
    ema: Any,
    optimizer: Any,
    sampler_generator: Any,
    flow_generator: Any,
    dropout_generator: Any,
    device: Any,
    disk_guard: DiskGuard,
    replace_existing: bool = False,
) -> None:
    writer = _atomic_torch_replace if replace_existing else _atomic_torch_save
    writer(
        runtime,
        path,
        {
            "artifact_kind": "mugen_latent_still_dit_resume_checkpoint",
            "config": asdict(config),
            "corpus": corpus.contract,
            "ema_policy": _ema_policy(config),
            "ema_model": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "raw_model": model.state_dict(),
            "rng_state": {
                "cuda": runtime.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "dropout": dropout_generator.get_state(),
                "flow": flow_generator.get_state(),
                "sampler": sampler_generator.get_state(),
                "torch_cpu": runtime.get_rng_state(),
            },
            "runtime": _runtime_facts(runtime, device),
            "step": step,
        },
        disk_guard=disk_guard,
    )


def _load_resume(
    runtime: Any,
    path: Path,
    *,
    expected_sha256: str,
    corpus: LatentStillCorpus,
    config: LatentStillTrainingConfig,
) -> dict[str, Any]:
    if _file_sha256(path) != expected_sha256:
        raise LatentStillTrainingError("resume checkpoint SHA-256 mismatch")
    try:
        value = runtime.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise LatentStillTrainingError("resume checkpoint failed safe load") from error
    if not isinstance(value, dict) or value.get("artifact_kind") != (
        "mugen_latent_still_dit_resume_checkpoint"
    ):
        raise LatentStillTrainingError("resume checkpoint has the wrong artifact kind")
    if value.get("corpus") != corpus.contract:
        raise LatentStillTrainingError("resume corpus contract differs")
    parent_config = value.get("config")
    if not isinstance(parent_config, dict):
        raise LatentStillTrainingError("resume config is missing")
    current = asdict(config)
    for key, parent_value in parent_config.items():
        if key != "steps" and current.get(key) != parent_value:
            raise LatentStillTrainingError(f"resume config differs at {key!r}")
    return value


def _load_latent(row: LatentStillRow, *, verify_hashes: bool) -> np.ndarray:
    payload = row.latent_path.read_bytes()
    if verify_hashes and hashlib.sha256(payload).hexdigest() != row.latent_file_sha256:
        raise LatentStillTrainingError(f"latent file hash mismatch: {row.sequence_id}")
    try:
        value = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise LatentStillTrainingError(f"latent file is unreadable: {row.sequence_id}") from error
    if value.dtype != np.float16 or value.shape != (8, 8, 64, 64):
        raise LatentStillTrainingError(f"latent geometry is invalid: {row.sequence_id}")
    if verify_hashes and _array_sha256(value) != row.latent_array_sha256:
        raise LatentStillTrainingError(f"latent array hash mismatch: {row.sequence_id}")
    if not bool(np.isfinite(value).all()):
        raise LatentStillTrainingError(f"latent contains non-finite values: {row.sequence_id}")
    return value


def _load_text_array(
    root: Path,
    arrays: dict[str, Any],
    name: str,
    dtype: Any,
    trailing_shape: tuple[int, ...],
) -> np.ndarray:
    record = arrays.get(name)
    if not isinstance(record, dict):
        raise LatentStillTrainingError(f"text array record is missing: {name}")
    relative = _required_text(record, "path")
    path = (root / relative).resolve()
    if root not in path.parents or _file_sha256(path) != record.get("file_sha256"):
        raise LatentStillTrainingError(f"text array file mismatch: {name}")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if value.dtype != dtype or value.shape[1:] != trailing_shape:
        raise LatentStillTrainingError(f"text array geometry differs: {name}")
    if _array_sha256(value) != record.get("array_content_sha256"):
        raise LatentStillTrainingError(f"text array content differs: {name}")
    return value


def _learning_rate(step: int, config: LatentStillTrainingConfig) -> float:
    if step <= config.warmup_steps and config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))
    return (
        config.minimum_learning_rate
        + (config.learning_rate - config.minimum_learning_rate) * cosine
    )


def _ema_update(runtime: Any, ema: Any, model: Any, decay: float) -> None:
    with runtime.no_grad():
        for target, source in zip(ema.parameters(), model.parameters(), strict=True):
            target.lerp_(source.detach(), 1 - decay)
        for target, source in zip(ema.buffers(), model.buffers(), strict=True):
            target.copy_(source)


def _ema_policy(config: LatentStillTrainingConfig) -> dict[str, Any]:
    return {
        "decay_after_warmup": config.ema_decay,
        "policy": "copy_raw_through_learning_rate_warmup_then_fixed_decay",
        "warmup_steps": config.warmup_steps,
    }


def _records(value: dict[str, Any], key: str, label: str) -> list[dict[str, Any]]:
    records = value.get(key)
    count = value.get("record_count")
    if key == "records" and count is None:
        count = (
            value.get("counts", {}).get("sequences")
            if isinstance(value.get("counts"), dict)
            else None
        )
    if key == "rows":
        count = value.get("prompt_count")
    if not isinstance(records, list) or count != len(records):
        raise LatentStillTrainingError(f"{label} record count differs")
    if not all(isinstance(record, dict) for record in records):
        raise LatentStillTrainingError(f"{label} records must be objects")
    return records


def _unique(records: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output = {}
    for record in records:
        value = _required_text(record, key)
        if value in output:
            raise LatentStillTrainingError(f"{label} has duplicate {key}: {value}")
        output[value] = record
    return output


def _required_text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise LatentStillTrainingError(f"field {key} must be non-empty text")
    return result


def _float_tuple(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 8:
        raise LatentStillTrainingError(f"{label} must contain eight values")
    output = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in output):
        raise LatentStillTrainingError(f"{label} values must be finite")
    return output


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LatentStillTrainingError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise LatentStillTrainingError(f"{label} must contain an object")
    return value


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_facts(runtime: Any, device: Any) -> dict[str, Any]:
    return {
        "cuda_version": runtime.version.cuda,
        "device": str(device),
        "device_name": runtime.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(runtime.__version__),
    }


def _atomic_bytes(path: Path, payload: bytes, *, disk_guard: DiskGuard) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace artifact: {path}")
    disk_guard.require_capacity(len(payload), label=path.name)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.rename(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_torch_save(
    runtime: Any,
    path: Path,
    payload: dict[str, Any],
    *,
    disk_guard: DiskGuard,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace checkpoint: {path}")
    disk_guard.require_capacity(3 * 1024**3, label=path.name)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        runtime.save(payload, temporary)
        os.rename(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_torch_replace(
    runtime: Any,
    path: Path,
    payload: dict[str, Any],
    *,
    disk_guard: DiskGuard,
) -> None:
    """Atomically rotate a bounded recovery slot without risking the old slot."""

    disk_guard.require_capacity(3 * 1024**3, label=path.name)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        runtime.save(payload, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("latent still training requires PyTorch") from _TORCH_IMPORT_ERROR
    return torch
