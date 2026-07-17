from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gui_live_tracking.frame_source import TrackingSample

PROTOCOL_VERSION = 1
IMAGE_ENCODING = "bgr8"
SUPPORTED_IMAGE_DTYPES = {"uint8"}
SUPPORTED_JOINT_DTYPES = {"float32", "float64"}


@dataclass(frozen=True)
class TrackingResult:
    timestamp_ns: int
    source_index: int
    frame_id: str
    child_frame_id: str
    translation: Sequence[float]
    quaternion_xyzw: Sequence[float]
    optimized_joint_angles: Sequence[float]
    loss: float
    fps: float
    overlay_bgr: np.ndarray


def _metadata(part: bytes) -> dict:
    try:
        data = json.loads(part.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON metadata: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Transport metadata must be a JSON object.")
    if data.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported transport version: {data.get('version')!r}")
    return data


def _dtype(name: str, supported: set[str]) -> np.dtype:
    if name not in supported:
        raise ValueError(f"Unsupported dtype: {name!r}")
    return np.dtype(name)


def _shape(value, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(v, int) and v > 0 for v in value):
        raise ValueError(f"{label} must be a nonempty list of positive integers.")
    return tuple(value)


def _array_from_bytes(raw: bytes, shape: tuple[int, ...], dtype: np.dtype, label: str) -> np.ndarray:
    expected = int(np.prod(shape)) * dtype.itemsize
    if len(raw) != expected:
        raise ValueError(f"{label} byte length mismatch: expected {expected}, got {len(raw)}.")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def serialize_tracking_sample(sample: TrackingSample) -> list[bytes]:
    frame = np.ascontiguousarray(sample.frame_bgr)
    joints = np.ascontiguousarray(sample.raw_joint_angles)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Tracking sample image must have shape HxWx3.")
    if str(frame.dtype) not in SUPPORTED_IMAGE_DTYPES:
        raise ValueError("Tracking sample image must use uint8 dtype.")
    if joints.ndim != 1:
        raise ValueError("Tracking sample joints must be a 1D array.")
    if str(joints.dtype) not in SUPPORTED_JOINT_DTYPES:
        joints = joints.astype(np.float64)

    meta = {
        "version": PROTOCOL_VERSION,
        "kind": "tracking_sample",
        "timestamp_ns": int(sample.timestamp_ns),
        "source_index": int(sample.source_index),
        "image_shape": list(frame.shape),
        "image_dtype": str(frame.dtype),
        "image_encoding": IMAGE_ENCODING,
        "joint_shape": list(joints.shape),
        "joint_dtype": str(joints.dtype),
    }
    return [json.dumps(meta, separators=(",", ":")).encode("utf-8"), frame.tobytes(), joints.tobytes()]


def deserialize_tracking_sample(parts: Sequence[bytes]) -> TrackingSample:
    if len(parts) != 3:
        raise ValueError(f"Tracking sample must have 3 parts, got {len(parts)}.")
    meta = _metadata(parts[0])
    if meta.get("kind") != "tracking_sample":
        raise ValueError(f"Expected tracking_sample, got {meta.get('kind')!r}.")
    if meta.get("image_encoding") != IMAGE_ENCODING:
        raise ValueError(f"Unsupported image encoding: {meta.get('image_encoding')!r}")

    image_shape = _shape(meta.get("image_shape"), "image_shape")
    if len(image_shape) != 3 or image_shape[2] != 3:
        raise ValueError("image_shape must be HxWx3.")
    joint_shape = _shape(meta.get("joint_shape"), "joint_shape")
    if len(joint_shape) != 1:
        raise ValueError("joint_shape must be 1D.")

    frame = _array_from_bytes(parts[1], image_shape, _dtype(meta.get("image_dtype"), SUPPORTED_IMAGE_DTYPES), "Image")
    joints = _array_from_bytes(parts[2], joint_shape, _dtype(meta.get("joint_dtype"), SUPPORTED_JOINT_DTYPES), "Joint")
    return TrackingSample(
        frame_bgr=frame,
        raw_joint_angles=joints,
        timestamp_ns=int(meta["timestamp_ns"]),
        source_index=int(meta["source_index"]),
    )


def serialize_tracking_result(result: TrackingResult) -> list[bytes]:
    overlay = np.ascontiguousarray(result.overlay_bgr)
    if overlay.ndim != 3 or overlay.shape[2] != 3:
        raise ValueError("Tracking result overlay must have shape HxWx3.")
    if str(overlay.dtype) not in SUPPORTED_IMAGE_DTYPES:
        raise ValueError("Tracking result overlay must use uint8 dtype.")

    meta = {
        "version": PROTOCOL_VERSION,
        "kind": "tracking_result",
        "timestamp_ns": int(result.timestamp_ns),
        "source_index": int(result.source_index),
        "frame_id": str(result.frame_id),
        "child_frame_id": str(result.child_frame_id),
        "translation": [float(v) for v in result.translation],
        "quaternion_xyzw": [float(v) for v in result.quaternion_xyzw],
        "optimized_joint_angles": [float(v) for v in result.optimized_joint_angles],
        "loss": float(result.loss),
        "fps": float(result.fps),
        "overlay_shape": list(overlay.shape),
        "overlay_dtype": str(overlay.dtype),
        "overlay_encoding": IMAGE_ENCODING,
    }
    return [json.dumps(meta, separators=(",", ":")).encode("utf-8"), overlay.tobytes()]


def deserialize_tracking_result(parts: Sequence[bytes]) -> TrackingResult:
    if len(parts) != 2:
        raise ValueError(f"Tracking result must have 2 parts, got {len(parts)}.")
    meta = _metadata(parts[0])
    if meta.get("kind") != "tracking_result":
        raise ValueError(f"Expected tracking_result, got {meta.get('kind')!r}.")
    if meta.get("overlay_encoding") != IMAGE_ENCODING:
        raise ValueError(f"Unsupported overlay encoding: {meta.get('overlay_encoding')!r}")
    overlay_shape = _shape(meta.get("overlay_shape"), "overlay_shape")
    if len(overlay_shape) != 3 or overlay_shape[2] != 3:
        raise ValueError("overlay_shape must be HxWx3.")
    overlay = _array_from_bytes(parts[1], overlay_shape, _dtype(meta.get("overlay_dtype"), SUPPORTED_IMAGE_DTYPES), "Overlay")
    return TrackingResult(
        timestamp_ns=int(meta["timestamp_ns"]),
        source_index=int(meta["source_index"]),
        frame_id=str(meta["frame_id"]),
        child_frame_id=str(meta["child_frame_id"]),
        translation=[float(v) for v in meta["translation"]],
        quaternion_xyzw=[float(v) for v in meta["quaternion_xyzw"]],
        optimized_joint_angles=[float(v) for v in meta["optimized_joint_angles"]],
        loss=float(meta["loss"]),
        fps=float(meta["fps"]),
        overlay_bgr=overlay,
    )
