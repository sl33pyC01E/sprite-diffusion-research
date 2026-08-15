"""Identity-disjoint action recognizer for dense canonical MUGEN motion."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.latent_motion_train import (
    LatentMotionTrainingCorpus,
    build_matched_action_index,
    load_latent_motion_training_corpus,
)
from spritelab.spark_caption import canonical_json_bytes
from spritelab.storage import DiskGuard

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency boundary
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class MugenActionClassifierError(ValueError):
    """Raised when a classifier artifact or dense corpus contract is invalid."""


@dataclass(frozen=True, slots=True)
class MugenActionClassifierConfig:
    input_channels: int = 8
    base_channels: int = 32
    feature_dim: int = 192
    action_count: int = 6

    def __post_init__(self) -> None:
        for name in ("input_channels", "base_channels", "feature_dim", "action_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.base_channels % 8 or self.feature_dim % 8:
            raise ValueError("base_channels and feature_dim must be divisible by 8")


@dataclass(frozen=True, slots=True)
class MugenActionClassifierTrainingConfig:
    epochs: int = 6
    identities_per_batch: int = 8
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    seed: int = 20260829
    device: str = "cuda"
    precision: str = "bfloat16"
    model: MugenActionClassifierConfig = MugenActionClassifierConfig()

    def __post_init__(self) -> None:
        for name in ("epochs", "identities_per_batch", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "minimum_learning_rate"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")


@dataclass(frozen=True, slots=True)
class MugenActionClassifierResult:
    output_directory: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    report_path: Path
    report_sha256: str


if torch is not None and nn is not None:

    class MugenLatentActionClassifier(nn.Module):
        """Classify canonical action from reference-relative latent motion."""

        def __init__(self, config: MugenActionClassifierConfig | None = None) -> None:
            super().__init__()
            self.config = config or MugenActionClassifierConfig()
            width = self.config.base_channels
            self.encoder = nn.Sequential(
                nn.Conv3d(
                    self.config.input_channels,
                    width,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=(1, 2, 2),
                ),
                nn.GroupNorm(8, width),
                nn.SiLU(),
                nn.Conv3d(width, width * 2, 3, stride=(1, 2, 2), padding=1),
                nn.GroupNorm(8, width * 2),
                nn.SiLU(),
                nn.Conv3d(width * 2, width * 4, 3, stride=2, padding=1),
                nn.GroupNorm(8, width * 4),
                nn.SiLU(),
                nn.Conv3d(
                    width * 4,
                    self.config.feature_dim,
                    3,
                    stride=2,
                    padding=1,
                ),
                nn.GroupNorm(8, self.config.feature_dim),
                nn.SiLU(),
            )
            self.head = nn.Linear(self.config.feature_dim, self.config.action_count)

        def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if residual.ndim != 5 or residual.shape[1] != self.config.input_channels:
                raise ValueError("residual must have shape [B,C,T,H,W]")
            features = self.encoder(residual).mean(dim=(2, 3, 4))
            return self.head(features), features

else:

    class MugenLatentActionClassifier:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("MugenLatentActionClassifier requires PyTorch") from (
                _TORCH_IMPORT_ERROR
            )


def dense_action_bundles(
    corpus: LatentMotionTrainingCorpus, indices: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """Return one exact vocabulary-ordered six-action bundle per identity."""

    index = build_matched_action_index(corpus.rows, indices)
    expected = set(corpus.action_vocabulary)
    bundles = []
    for identity, actions in index.items():
        if set(actions) != expected:
            raise MugenActionClassifierError(f"identity lacks dense action bundle: {identity}")
        bundles.append(tuple(actions[action] for action in corpus.action_vocabulary))
    return tuple(bundles)


def latent_action_batch(
    runtime: Any,
    corpus: LatentMotionTrainingCorpus,
    bundles: tuple[tuple[int, ...], ...],
    *,
    device: Any,
) -> tuple[Any, Any]:
    """Load exact normalized reference-relative latent motion and labels."""

    indices = tuple(index for bundle in bundles for index in bundle)
    array_indices = list(indices)
    target = runtime.from_numpy(corpus.target_latents[array_indices].astype(np.float32)).to(device)
    reference = runtime.from_numpy(corpus.reference_latents[array_indices].astype(np.float32)).to(
        device
    )
    standard_deviation = runtime.tensor(
        corpus.channel_standard_deviation, device=device, dtype=runtime.float32
    ).view(1, 1, 8, 1, 1)
    residual = (target - reference.unsqueeze(1)) / standard_deviation
    residual = residual.permute(0, 2, 1, 3, 4).contiguous()
    labels = runtime.tensor(
        [corpus.rows[index].action_index for index in indices],
        device=device,
        dtype=runtime.long,
    )
    return residual, labels


def train_mugen_action_classifier(
    manifest_path: Path | str,
    output_directory: Path | str,
    *,
    config: MugenActionClassifierTrainingConfig | None = None,
    disk_guard: DiskGuard | None = None,
) -> MugenActionClassifierResult:
    """Train and publish an exact identity-disjoint six-action recognizer."""

    runtime = _require_torch()
    config = config or MugenActionClassifierTrainingConfig()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace action classifier output: {output}")
    root = Path.cwd().resolve()
    guard = disk_guard or DiskGuard(root, min_free_bytes=100 * 1024**3)
    guard.require_capacity(512 * 1024**2, label="MUGEN action classifier")
    corpus = load_latent_motion_training_corpus(
        manifest_path, verify_hashes=True, array_loading="lazy"
    )
    if tuple(corpus.action_vocabulary) != (
        "attack_a",
        "attack_b",
        "block",
        "idle",
        "jump",
        "walk",
    ):
        raise MugenActionClassifierError("dense six-action vocabulary differs")
    if config.model.action_count != len(corpus.action_vocabulary):
        raise MugenActionClassifierError("classifier action count differs")
    bundles = {
        "train": dense_action_bundles(corpus, corpus.train_indices),
        "validation": dense_action_bundles(corpus, corpus.validation_indices),
        "test": dense_action_bundles(corpus, corpus.test_indices),
    }
    device = runtime.device(config.device)
    if device.type == "cuda" and not runtime.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    runtime.manual_seed(config.seed)
    if device.type == "cuda":
        runtime.cuda.manual_seed_all(config.seed)
    model = MugenLatentActionClassifier(config.model).to(device)
    optimizer = runtime.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    generator = runtime.Generator(device="cpu").manual_seed(config.seed + 1)
    dtype = runtime.bfloat16 if config.precision == "bfloat16" else runtime.float32
    autocast = config.precision == "bfloat16"
    total_steps = config.epochs * math.ceil(len(bundles["train"]) / config.identities_per_batch)
    step = 0
    history = []
    best_state = None
    best_validation = -1.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = runtime.randperm(len(bundles["train"]), generator=generator).tolist()
        losses = []
        correct = 0
        seen = 0
        for start in range(0, len(order), config.identities_per_batch):
            selected = tuple(
                bundles["train"][index]
                for index in order[start : start + config.identities_per_batch]
            )
            residual, labels = latent_action_batch(runtime, corpus, selected, device=device)
            step += 1
            progress = step / max(total_steps, 1)
            learning_rate = config.minimum_learning_rate + 0.5 * (
                config.learning_rate - config.minimum_learning_rate
            ) * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                logits, _features = model(residual)
                loss = runtime.nn.functional.cross_entropy(logits.float(), labels)
            loss.backward()
            runtime.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(1) == labels).sum().detach().cpu())
            seen += labels.numel()
        validation = evaluate_mugen_action_classifier(
            runtime,
            model,
            corpus,
            bundles["validation"],
            device=device,
            dtype=dtype,
            autocast=autocast,
            identities_per_batch=config.identities_per_batch,
        )
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_accuracy": correct / seen,
            "train_loss": sum(losses) / len(losses),
            "validation": validation,
        }
        history.append(row)
        if validation["accuracy"] > best_validation:
            best_validation = validation["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    test_metrics = evaluate_mugen_action_classifier(
        runtime,
        model,
        corpus,
        bundles["test"],
        device=device,
        dtype=dtype,
        autocast=autocast,
        identities_per_batch=config.identities_per_batch,
    )
    report = {
        "artifact_kind": "mugen_dense_six_action_classifier_training",
        "claim": (
            "action recognition on exact dense MUGEN latent residuals; test identities "
            "are disjoint from training identities"
        ),
        "config": asdict(config),
        "corpus": corpus.contract,
        "counts": {
            "actions": len(corpus.action_vocabulary),
            "test_identities": len(bundles["test"]),
            "train_identities": len(bundles["train"]),
            "validation_identities": len(bundles["validation"]),
        },
        "history": history,
        "test": test_metrics,
        "vocabulary": list(corpus.action_vocabulary),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        checkpoint = {
            "artifact_kind": "mugen_dense_six_action_classifier",
            "config": asdict(config.model),
            "corpus": corpus.contract,
            "model": best_state,
            "test": test_metrics,
            "vocabulary": list(corpus.action_vocabulary),
        }
        buffer = io.BytesIO()
        runtime.save(checkpoint, buffer)
        checkpoint_payload = buffer.getvalue()
        guard.require_capacity(len(checkpoint_payload), label="action classifier checkpoint")
        (stage / "checkpoint.pt").write_bytes(checkpoint_payload)
        report["checkpoint_sha256"] = hashlib.sha256(checkpoint_payload).hexdigest()
        report_payload = canonical_json_bytes(report)
        (stage / "training-report.json").write_bytes(report_payload)
        (stage / "training-history.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in history
            ),
            encoding="utf-8",
            newline="\n",
        )
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return MugenActionClassifierResult(
        output_directory=output,
        checkpoint_path=output / "checkpoint.pt",
        checkpoint_sha256=report["checkpoint_sha256"],
        report_path=output / "training-report.json",
        report_sha256=hashlib.sha256(report_payload).hexdigest(),
    )


def evaluate_mugen_action_classifier(
    runtime: Any,
    model: Any,
    corpus: LatentMotionTrainingCorpus,
    bundles: tuple[tuple[int, ...], ...],
    *,
    device: Any,
    dtype: Any,
    autocast: bool,
    identities_per_batch: int,
) -> dict[str, Any]:
    """Evaluate exact accuracy and confusion counts without identity leakage."""

    model.eval()
    action_count = len(corpus.action_vocabulary)
    confusion = np.zeros((action_count, action_count), dtype=np.int64)
    losses = []
    with runtime.no_grad():
        for start in range(0, len(bundles), identities_per_batch):
            residual, labels = latent_action_batch(
                runtime,
                corpus,
                bundles[start : start + identities_per_batch],
                device=device,
            )
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                logits, _features = model(residual)
                loss = runtime.nn.functional.cross_entropy(logits.float(), labels)
            losses.append(float(loss.detach().cpu()))
            predictions = logits.argmax(1).detach().cpu().numpy()
            actual = labels.detach().cpu().numpy()
            for expected, predicted in zip(actual, predictions, strict=True):
                confusion[int(expected), int(predicted)] += 1
    per_action = {}
    for index, action in enumerate(corpus.action_vocabulary):
        total = int(confusion[index].sum())
        per_action[action] = {
            "accuracy": int(confusion[index, index]) / total,
            "correct": int(confusion[index, index]),
            "total": total,
        }
    return {
        "accuracy": int(np.trace(confusion)) / int(confusion.sum()),
        "confusion_matrix": confusion.tolist(),
        "cross_entropy": sum(losses) / len(losses),
        "per_action": per_action,
    }


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("MUGEN action classification requires PyTorch") from (
            _TORCH_IMPORT_ERROR
        )
    return torch
