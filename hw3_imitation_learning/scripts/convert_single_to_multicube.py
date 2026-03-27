"""Convert raw single-cube zarr datasets to multi-cube compatible format.

Adds the missing arrays so single-cube data can be merged with multi-cube data:
  - pos_cube_red:   copy of state_cube (treat the single cube as "red")
  - pos_cube_green: copy of state_cube (same position — won't be selected)
  - pos_cube_blue:  copy of state_cube (same position — won't be selected)
  - state_goal:     randomly assigned one-hot per episode (balanced across red/green/blue)
  - goal_pos:       constant bin center from single-cube scene [-0.2, 0.7, 0.021]

Removes state_cube (not present in multi-cube raw data).

Usage:
    python scripts/convert_single_to_multicube.py
    python scripts/convert_single_to_multicube.py --input-dir datasets/raw/single_cube --output-dir datasets/raw/multi_cube/from_single_cube
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numcodecs
import numpy as np
import zstandard

# Bin center in single-cube scene (world frame):
#   body pos = (-0.2, 0.7, 0.001), site local offset = (0, 0, 0.02)
SINGLE_CUBE_GOAL_POS = np.array([-0.2, 0.7, 0.021], dtype=np.float32)

ONEHOT = np.eye(3, dtype=np.float32)  # [red, green, blue]


def _read_zarr_v3_array(array_dir: Path) -> np.ndarray:
    """Read a zarr v3 array by decoding all chunks with numcodecs."""
    meta = json.loads((array_dir / "zarr.json").read_text())
    shape = tuple(meta["shape"])
    chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    dtype = np.dtype(meta["data_type"])

    # Determine codec from metadata
    codec_names = [c["name"] for c in meta.get("codecs", [])]
    if "zstd" in codec_names:
        codec = None  # handled inline below
    else:
        codec = numcodecs.Blosc()

    # Calculate number of chunks per dimension
    n_chunks = [
        (s + cs - 1) // cs for s, cs in zip(shape, chunk_shape)
    ]

    result = np.zeros(shape, dtype=dtype)

    # Read each chunk
    for idx in np.ndindex(*n_chunks):
        chunk_path = array_dir / "c" / "/".join(str(i) for i in idx)
        if not chunk_path.exists():
            continue
        raw = chunk_path.read_bytes()
        if codec is not None:
            decompressed = codec.decode(raw)
        else:
            expected_size = int(np.prod(chunk_shape)) * dtype.itemsize
            decompressed = zstandard.ZstdDecompressor().decompress(
                raw, max_output_size=expected_size
            )
        chunk_arr = np.frombuffer(decompressed, dtype=dtype).copy()

        # Calculate slice for this chunk
        slices = []
        for ci, cs, s in zip(idx, chunk_shape, shape):
            start = ci * cs
            end = min(start + cs, s)
            slices.append(slice(start, end))

        actual_shape = tuple(sl.stop - sl.start for sl in slices)
        chunk_arr = chunk_arr[: int(np.prod(actual_shape))].reshape(actual_shape)
        result[tuple(slices)] = chunk_arr

    return result


def _write_zarr_v3_array(array_dir: Path, data: np.ndarray) -> None:
    """Write a zarr v3 array as a single zstd-compressed chunk."""
    if array_dir.exists():
        shutil.rmtree(array_dir)
    array_dir.mkdir(parents=True)

    meta = {
        "shape": list(data.shape),
        "data_type": data.dtype.name,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(data.shape)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0.0,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
        ],
        "attributes": {},
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }
    (array_dir / "zarr.json").write_text(json.dumps(meta, indent=2))

    compressed = zstandard.ZstdCompressor(level=0).compress(
        np.ascontiguousarray(data).tobytes()
    )

    # Single chunk at c/0/0 (2D) or c/0 (1D)
    chunk_path = array_dir / "c" / "/".join(["0"] * len(data.shape))
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_bytes(compressed)


def convert_zarr(src: Path, dst: Path, rng: np.random.Generator) -> None:
    """Read a raw single-cube zarr and write a multi-cube compatible copy."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    data_dir = dst / "data"
    meta_dir = dst / "meta"

    state_cube = _read_zarr_v3_array(data_dir / "state_cube")
    episode_ends = _read_zarr_v3_array(meta_dir / "episode_ends")
    n_steps = state_cube.shape[0]

    # Build per-step state_goal with random color per episode
    state_goal = np.zeros((n_steps, 3), dtype=np.float32)
    starts = np.concatenate([[0], episode_ends[:-1]])
    color_counts = [0, 0, 0]
    for start, end in zip(starts, episode_ends):
        color_idx = rng.integers(0, 3)
        state_goal[int(start):int(end)] = ONEHOT[color_idx]
        color_counts[color_idx] += 1

    # Write multi-cube arrays
    _write_zarr_v3_array(data_dir / "pos_cube_red", state_cube)
    _write_zarr_v3_array(data_dir / "pos_cube_green", state_cube)
    _write_zarr_v3_array(data_dir / "pos_cube_blue", state_cube)
    _write_zarr_v3_array(data_dir / "state_goal", state_goal)
    _write_zarr_v3_array(
        data_dir / "goal_pos",
        np.tile(SINGLE_CUBE_GOAL_POS, (n_steps, 1)),
    )

    # Remove state_cube (not present in multi-cube format)
    shutil.rmtree(data_dir / "state_cube")

    # Update data group zarr.json to remove state_cube reference if present
    data_zarr = data_dir / "zarr.json"
    if data_zarr.exists():
        dmeta = json.loads(data_zarr.read_text())
        data_zarr.write_text(json.dumps(dmeta, indent=2))

    n_ep = len(episode_ends)
    print(
        f"  Converted: {src.name} -> {dst.name}  "
        f"({n_steps} steps, {n_ep} episodes, "
        f"goals: R={color_counts[0]} G={color_counts[1]} B={color_counts[2]})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw single-cube zarrs to multi-cube format."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./datasets/raw/single_cube"),
        help="Root dir with single-cube .zarr stores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./datasets/raw/multi_cube/from_single_cube"),
        help="Output dir for converted zarrs.",
    )
    args = parser.parse_args()

    zarr_paths = sorted(args.input_dir.rglob("*.zarr"))
    if not zarr_paths:
        print(f"No .zarr stores found under {args.input_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(zarr_paths)} zarr(s) from {args.input_dir} -> {args.output_dir}")

    rng = np.random.default_rng(42)
    for src in zarr_paths:
        rel = src.relative_to(args.input_dir)
        dst = args.output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        convert_zarr(src, dst, rng)

    print(f"\nDone. Now run:\n  python scripts/compute_actions.py --action-space ee --datasets-dir datasets/raw/multi_cube")


if __name__ == "__main__":
    main()
