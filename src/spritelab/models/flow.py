"""Optional-PyTorch rectified-flow objective and deterministic Euler sampler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised in the torch-free venv
    torch = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class MissingFlowTorchError(RuntimeError):
    """Raised when a flow tensor operation is requested without PyTorch."""


@dataclass(frozen=True, slots=True)
class RectifiedFlowBatch:
    """One linear probability-path training batch.

    ``clean`` is a native RGBA clip in ``[-1, 1]`` and ``noise`` is the
    Gaussian endpoint. The path convention is ``x_t=(1-t)*clean+t*noise``;
    the model target is therefore the constant velocity ``noise-clean``.
    """

    clean: Any
    noise: Any
    noisy: Any
    timesteps: Any
    target_velocity: Any


def sample_rectified_flow_batch(
    clean: Any,
    *,
    timesteps: Any | None = None,
    noise: Any | None = None,
    generator: Any | None = None,
) -> RectifiedFlowBatch:
    """Sample points on the clean-to-noise line without mutating the input."""

    runtime = _require_torch()
    _validate_clip_tensor(clean, name="clean", require_unit_range=True)
    batch_size = clean.shape[0]
    if noise is None:
        noise = runtime.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
    else:
        _validate_like(noise, clean, name="noise")
    if timesteps is None:
        timesteps = runtime.rand(
            (batch_size,),
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
    else:
        _validate_timesteps(timesteps, clean)
    expanded = timesteps.reshape(batch_size, *([1] * (clean.ndim - 1)))
    target_velocity = noise - clean
    noisy = (1.0 - expanded) * clean + expanded * noise
    return RectifiedFlowBatch(
        clean=clean,
        noise=noise,
        noisy=noisy,
        timesteps=timesteps,
        target_velocity=target_velocity,
    )


def predict_clean_from_velocity(noisy: Any, predicted_velocity: Any, timesteps: Any) -> Any:
    """Recover the clean endpoint implied by a velocity prediction at ``t``."""

    _require_torch()
    _validate_clip_tensor(noisy, name="noisy", require_unit_range=False)
    _validate_like(predicted_velocity, noisy, name="predicted_velocity")
    _validate_timesteps(timesteps, noisy)
    expanded = timesteps.reshape(noisy.shape[0], *([1] * (noisy.ndim - 1)))
    return noisy - expanded * predicted_velocity


def rectified_flow_mse(
    predicted_velocity: Any,
    batch: RectifiedFlowBatch,
    *,
    foreground_weight: float = 0.0,
    alpha_channel_weight: float = 1.0,
) -> Any:
    """Return velocity MSE with independent foreground and alpha emphasis.

    ``foreground_weight`` emphasizes every RGBA residual at visible target
    pixels. ``alpha_channel_weight`` independently scales the alpha-channel
    residual everywhere; keeping it at one preserves the original objective.
    """

    runtime = _require_torch()
    _validate_like(predicted_velocity, batch.target_velocity, name="predicted_velocity")
    if not math.isfinite(foreground_weight) or foreground_weight < 0:
        raise ValueError("foreground_weight must be finite and non-negative")
    if not math.isfinite(alpha_channel_weight) or alpha_channel_weight < 0:
        raise ValueError("alpha_channel_weight must be finite and non-negative")
    squared = (predicted_velocity - batch.target_velocity).square()
    if alpha_channel_weight != 1.0:
        channel_weights = runtime.ones(
            (1, 1, squared.shape[2], 1, 1),
            device=squared.device,
            dtype=squared.dtype,
        )
        channel_weights[:, :, 3] = alpha_channel_weight
        squared = squared * channel_weights
    if foreground_weight:
        alpha = ((batch.clean[:, :, 3:4] + 1.0) * 0.5).clamp(0.0, 1.0)
        squared = squared * (1.0 + foreground_weight * alpha)
    return runtime.mean(squared)


def endpoint_sample_velocity_model(
    model: Any,
    noise: Any,
    *,
    conditioning: Any | None = None,
    conditioning_mask: Any | None = None,
    frame_phase: Any | None = None,
) -> Any:
    """Predict the clean endpoint directly from pure noise at ``t=1``.

    This is algebraically identical to one backward-Euler step. It is named
    explicitly because checkpoints trained with matched pure-noise endpoint
    supervision can be materially better under this one-call reconstruction
    than under a multi-step flow solve. It is not a general replacement for ODE
    integration; sampler choice must be validated per training objective.
    """

    runtime = _require_torch()
    _validate_clip_tensor(noise, name="noise", require_unit_range=False)
    timesteps = runtime.ones(
        (noise.shape[0],),
        device=noise.device,
        dtype=noise.dtype,
    )
    with runtime.no_grad():
        velocity = model(
            noise,
            timesteps,
            conditioning,
            conditioning_mask=conditioning_mask,
            frame_phase=frame_phase,
        )
        _validate_like(velocity, noise, name="model velocity")
        return noise - velocity


def euler_sample_velocity_model(
    model: Any,
    noise: Any,
    *,
    steps: int,
    conditioning: Any | None = None,
    conditioning_mask: Any | None = None,
    frame_phase: Any | None = None,
) -> Any:
    """Integrate the learned velocity field from noise ``t=1`` to data ``t=0``."""

    runtime = _require_torch()
    _validate_clip_tensor(noise, name="noise", require_unit_range=False)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    state = noise
    times = runtime.linspace(1.0, 0.0, steps + 1, device=noise.device, dtype=noise.dtype)
    with runtime.no_grad():
        for index in range(steps):
            current = times[index]
            following = times[index + 1]
            timestep_batch = current.expand(noise.shape[0])
            velocity = model(
                state,
                timestep_batch,
                conditioning,
                conditioning_mask=conditioning_mask,
                frame_phase=frame_phase,
            )
            _validate_like(velocity, state, name="model velocity")
            state = state + (following - current) * velocity
    return state


def _require_torch() -> Any:
    if torch is None:
        raise MissingFlowTorchError(
            "rectified-flow tensor operations require a platform-appropriate PyTorch install"
        ) from _TORCH_IMPORT_ERROR
    return torch


def _validate_clip_tensor(value: Any, *, name: str, require_unit_range: bool) -> None:
    runtime = _require_torch()
    if not isinstance(value, runtime.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 5 or value.shape[2] != 4:
        raise ValueError(f"{name} must have shape [B, T, 4, H, W]; got {tuple(value.shape)!r}")
    if value.shape[0] < 1 or value.shape[1] < 1 or value.shape[3] < 1 or value.shape[4] < 1:
        raise ValueError(f"{name} dimensions must be non-zero")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(runtime.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    if require_unit_range and (bool((value < -1).any()) or bool((value > 1).any())):
        raise ValueError(f"{name} must be normalized to [-1, 1]")


def _validate_like(value: Any, reference: Any, *, name: str) -> None:
    _validate_clip_tensor(value, name=name, require_unit_range=False)
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} must match reference shape {tuple(reference.shape)!r}; "
            f"got {tuple(value.shape)!r}"
        )
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f"{name} must match reference device and dtype")


def _validate_timesteps(timesteps: Any, reference: Any) -> None:
    runtime = _require_torch()
    if not isinstance(timesteps, runtime.Tensor):
        raise TypeError("timesteps must be a torch.Tensor")
    if tuple(timesteps.shape) != (reference.shape[0],):
        raise ValueError(
            f"timesteps must have shape {(reference.shape[0],)!r}; got {tuple(timesteps.shape)!r}"
        )
    if timesteps.device != reference.device or timesteps.dtype != reference.dtype:
        raise ValueError("timesteps must match the reference device and dtype")
    if not bool(runtime.isfinite(timesteps).all()):
        raise ValueError("timesteps must contain only finite values")
    if bool((timesteps < 0).any()) or bool((timesteps > 1).any()):
        raise ValueError("timesteps must be in [0, 1]")
