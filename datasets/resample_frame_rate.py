import argparse
import os
from typing import List, Tuple

import numpy as np


def parse_frame_line(line: str, expected_dim: int, file_path: str) -> Tuple[int, np.ndarray, bool]:
    if ":" not in line:
        raise ValueError(f"{file_path}: invalid frame line without ':' -> {line}")
    frame_str, coords_str = line.split(":", 1)
    frame_id = int(frame_str.strip())
    coords = np.array([float(x) for x in coords_str.split(",")], dtype=np.float32)
    if len(coords) < expected_dim:
        raise ValueError(
            f"{file_path} frame {frame_id}: expected {expected_dim} coordinates, got {len(coords)}"
        )
    has_extra_coords = len(coords) > expected_dim
    if has_extra_coords:
        coords = coords[:expected_dim]
    return frame_id, coords, has_extra_coords


def parse_label_line(line: str, file_path: str) -> Tuple[int, np.ndarray]:
    if ":" not in line:
        raise ValueError(f"{file_path}: invalid label line without ':' -> {line}")
    frame_str, coords_str = line.split(":", 1)
    frame_id = int(frame_str.strip())
    coords = np.array([float(x) for x in coords_str.split(",")], dtype=np.float32)
    if len(coords) != 3:
        raise ValueError(f"{file_path} label frame {frame_id}: expected 3 coordinates, got {len(coords)}")
    return frame_id, coords


def format_number(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def format_coords(coords: np.ndarray) -> str:
    return ",".join(format_number(v) for v in coords)


def nearest_resample(frame_ids: np.ndarray, coords: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
    nearest_indices = np.abs(frame_ids[:, None] - target_positions[None, :]).argmin(axis=0)
    return coords[nearest_indices]


def linear_resample(frame_ids: np.ndarray, coords: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
    output = np.empty((len(target_positions), coords.shape[1]), dtype=np.float32)
    for dim in range(coords.shape[1]):
        output[:, dim] = np.interp(target_positions, frame_ids, coords[:, dim])
    return output


def load_sequence(file_path: str, output_len: int, expected_dim: int):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < output_len + 1:
        raise ValueError(f"{file_path}: expected at least {output_len + 1} non-empty lines, got {len(lines)}")

    sequence_lines = lines[:output_len]
    label_line = lines[-1]

    frame_ids: List[int] = []
    coords: List[np.ndarray] = []
    warned_extra_coords = False
    for line in sequence_lines:
        frame_id, frame_coords, has_extra_coords = parse_frame_line(line, expected_dim, file_path)
        if has_extra_coords and not warned_extra_coords:
            print(
                f"Warning: {file_path} has extra coordinates; first seen at frame {frame_id}. "
                f"Expected {expected_dim} coordinates. Extra coordinates will be truncated."
            )
            warned_extra_coords = True
        frame_ids.append(frame_id)
        coords.append(frame_coords)

    frame_ids_array = np.array(frame_ids, dtype=np.float32)
    if np.any(np.diff(frame_ids_array) <= 0):
        raise ValueError(f"{file_path}: sequence frame ids must be strictly increasing")

    drop_frame, drop_coords = parse_label_line(label_line, file_path)
    return frame_ids_array, np.stack(coords, axis=0), drop_frame, drop_coords


def resample_file(input_path: str,
                  output_path: str,
                  source_fps: float,
                  target_fps: float,
                  method: str,
                  point_num: int,
                  hit_index: int,
                  output_len: int):
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if hit_index < 1 or hit_index > output_len:
        raise ValueError(f"hit_index must be in [1, {output_len}], got {hit_index}")

    expected_dim = point_num * 3
    frame_ids, coords, drop_frame, drop_coords = load_sequence(input_path, output_len, expected_dim)

    hit_pos = hit_index - 1
    orig_hit_frame = frame_ids[hit_pos]
    target_indices = np.arange(1, output_len + 1, dtype=np.float32)
    target_positions = orig_hit_frame + (target_indices - hit_index) * source_fps / target_fps

    min_target = target_positions.min()
    max_target = target_positions.max()
    if min_target < frame_ids[0] or max_target > frame_ids[-1]:
        raise ValueError(
            f"{input_path}: target positions [{format_number(min_target)}, {format_number(max_target)}] "
            f"exceed source range [{format_number(frame_ids[0])}, {format_number(frame_ids[-1])}]"
        )

    if method == "linear":
        resampled = linear_resample(frame_ids, coords, target_positions)
    elif method == "nearest":
        resampled = nearest_resample(frame_ids, coords, target_positions)
    else:
        raise ValueError(f"unsupported method: {method}")

    new_drop_frame = int(hit_index + round((drop_frame - orig_hit_frame) * target_fps / source_fps))

    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for out_frame_id, frame_coords in enumerate(resampled, start=1):
            f.write(f"{out_frame_id}:{format_coords(frame_coords)}\n")
        f.write(f"{new_drop_frame}:{format_coords(drop_coords)}\n")


def resample_directory(input_dir: str,
                       output_dir: str,
                       source_fps: float,
                       target_fps: float,
                       method: str,
                       point_num: int,
                       hit_index: int,
                       output_len: int):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    txt_files = [fn for fn in sorted(os.listdir(input_dir)) if fn.endswith(".txt")]
    if not txt_files:
        raise ValueError(f"no .txt files found in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    for fn in txt_files:
        input_path = os.path.join(input_dir, fn)
        output_path = os.path.join(output_dir, fn)
        resample_file(
            input_path=input_path,
            output_path=output_path,
            source_fps=source_fps,
            target_fps=target_fps,
            method=method,
            point_num=point_num,
            hit_index=hit_index,
            output_len=output_len,
        )

    print(f"Processed {len(txt_files)} files: {input_dir} -> {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Resample badminton sequence files to a target frame rate.")
    parser.add_argument("--input_dir", required=True, help="Input dataset directory containing .txt samples")
    parser.add_argument("--output_dir", required=True, help="Output dataset directory")
    parser.add_argument("--source_fps", type=float, required=True, help="Source frame rate")
    parser.add_argument("--target_fps", type=float, required=True, help="Target frame rate")
    parser.add_argument("--method", choices=["linear", "nearest"], default="linear", help="Resampling method")
    parser.add_argument("--points_num", type=int, default=22, help="Number of 3D points per frame")
    parser.add_argument("--hit_index", type=int, default=100, help="1-based hit frame index in the output sequence")
    parser.add_argument("--output_len", type=int, default=105, help="Number of sequence frames to write")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resample_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        method=args.method,
        point_num=args.points_num,
        hit_index=args.hit_index,
        output_len=args.output_len,
    )
