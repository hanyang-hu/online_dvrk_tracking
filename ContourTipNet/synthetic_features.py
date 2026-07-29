import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch


EDGE_KEYS = ("edge_features", "edges", "edge_lines", "lines", "cylinders")
KEYPOINT_KEYS = ("keypoints", "grippers", "points")
MASK_KEYS = ("mask_path", "image_path", "file_name", "filename", "mask", "image")


def load_feature_records(features_json):
    with open(features_json, "r") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported feature JSON format in {features_json}")

    for key in ("samples", "records", "frames", "images", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    records = []
    for key, value in payload.items():
        if isinstance(value, dict):
            item = dict(value)
            item.setdefault("id", key)
            records.append(item)

    if records:
        return records

    raise ValueError(f"No records found in {features_json}")


def record_has_edges(record):
    return any(key in record and record[key] is not None for key in EDGE_KEYS)


def choose_feature_file(data_root=".", features_json=None, require_edges=None):
    if features_json and features_json != "auto":
        return features_json

    roots = [Path.cwd(), Path(data_root)]
    candidates = []
    for root in roots:
        for name in ("features_v4.json", "features_v3.json"):
            path = root / name
            if path.exists() and path not in candidates:
                candidates.append(path)

    if not candidates:
        return None

    inspected = []
    for path in candidates:
        records = load_feature_records(path)
        has_edges = any(record_has_edges(record) for record in records)
        inspected.append((path, has_edges))

    if require_edges is True:
        for path, has_edges in inspected:
            if has_edges:
                return str(path)
    elif require_edges is False:
        for path, has_edges in inspected:
            if not has_edges:
                return str(path)
    else:
        for path, has_edges in inspected:
            if path.name == "features_v4.json" and has_edges:
                return str(path)
        return str(inspected[0][0])

    return None


def resolve_record_path(record, features_json, data_root="."):
    for key in MASK_KEYS:
        value = record.get(key)
        if not value:
            continue
        path = Path(value)
        if path.is_absolute() and path.exists():
            return str(path)
        for base in (Path(features_json).parent, Path(data_root), Path.cwd()):
            candidate = base / path
            if candidate.exists():
                return str(candidate)

    record_id = str(record.get("id", record.get("frame", "")))
    if record_id:
        for base in (Path(data_root), Path(features_json).parent, Path.cwd()):
            matches = sorted(base.rglob(f"*{record_id}*.png"))
            if matches:
                return str(matches[0])

    raise FileNotFoundError(f"Could not resolve mask path for record {record}")


def first_present(record, keys):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def normalize_points(value):
    if value is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    arr = arr.reshape(-1, arr.shape[-1])
    return arr[:, :2]


def normalize_lines(value):
    if value is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.ndim == 3 and arr.shape[-2:] == (2, 2):
        arr = arr.reshape(-1, 2, 2)
    return arr


def gaussian_heatmap(points_xy, out_size, orig_hw, sigma):
    heatmap = np.zeros((out_size, out_size), dtype=np.float32)
    if len(points_xy) == 0:
        return heatmap

    yy, xx = np.mgrid[:out_size, :out_size].astype(np.float32)
    orig_h, orig_w = orig_hw
    for x, y in points_xy:
        xr = x * out_size / max(orig_w, 1)
        yr = y * out_size / max(orig_h, 1)
        if 0 <= xr < out_size and 0 <= yr < out_size:
            g = np.exp(-((xx - xr) ** 2 + (yy - yr) ** 2) / (2.0 * sigma ** 2))
            heatmap = np.maximum(heatmap, g.astype(np.float32))
    return heatmap


def _line_heatmap_from_endpoints(line, out_size, orig_hw, sigma):
    orig_h, orig_w = orig_hw
    x1, y1, x2, y2 = line.astype(np.float32)
    x1 *= out_size / max(orig_w, 1)
    x2 *= out_size / max(orig_w, 1)
    y1 *= out_size / max(orig_h, 1)
    y2 *= out_size / max(orig_h, 1)

    yy, xx = np.mgrid[:out_size, :out_size].astype(np.float32)
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom < 1e-8:
        return gaussian_heatmap([[x1, y1]], out_size, (out_size, out_size), sigma)

    t = ((xx - x1) * dx + (yy - y1) * dy) / denom
    t = np.clip(t, 0.0, 1.0)
    px = x1 + t * dx
    py = y1 + t * dy
    dist2 = (xx - px) ** 2 + (yy - py) ** 2
    return np.exp(-dist2 / (2.0 * sigma ** 2)).astype(np.float32)


def _line_heatmap_from_ab(line, out_size, orig_hw, sigma):
    orig_h, orig_w = orig_hw
    a, b = line.astype(np.float32)
    ar = a * orig_w / max(out_size, 1)
    br = b * orig_h / max(out_size, 1)
    yy, xx = np.mgrid[:out_size, :out_size].astype(np.float32)
    dist = np.abs(ar * xx + br * yy - 1.0) / (np.sqrt(ar * ar + br * br) + 1e-8)
    return np.exp(-(dist ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


def line_heatmaps(lines, out_size, orig_hw, sigma, channels=2):
    heatmaps = np.zeros((channels, out_size, out_size), dtype=np.float32)
    lines = normalize_lines(lines)
    if lines.size == 0:
        return heatmaps

    if lines.ndim == 3 and lines.shape[-2:] == (2, 2):
        flat = lines
    else:
        flat = lines.reshape(lines.shape[0], -1)

    for idx, line in enumerate(flat[:channels]):
        if np.asarray(line).shape == (2, 2):
            coords = np.asarray(line, dtype=np.float32).reshape(-1)
            heatmaps[idx] = _line_heatmap_from_endpoints(coords, out_size, orig_hw, sigma)
        else:
            line = np.asarray(line, dtype=np.float32).reshape(-1)
            if line.size >= 4:
                heatmaps[idx] = _line_heatmap_from_endpoints(line[:4], out_size, orig_hw, sigma)
            elif line.size >= 2:
                heatmaps[idx] = _line_heatmap_from_ab(line[:2], out_size, orig_hw, sigma)

    return heatmaps


def read_mask(path, out_size):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    orig_hw = mask.shape[:2]
    mask = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
    return mask, orig_hw


def write_feature_manifest(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"samples": records}, f, indent=2)
