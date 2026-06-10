import argparse
import importlib.util
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGINAL_DATASETS = [
    ROOT / "datasets" / "20251217_scene1",
    ROOT / "datasets" / "20260202_scene2",
    ROOT / "datasets" / "20260418_scene3",
]

SOURCE_FPS = {
    "20251217_scene1": 245,
    "20260202_scene2": 160,
    "20260418_scene3": 300,
}

FPS300_ROOT = ROOT / "datasets" / "fps300"
FPS300_DATASETS = [FPS300_ROOT / path.name for path in ORIGINAL_DATASETS]

ORIGINAL_BEFORE_MODEL = ROOT / "models" / "ImprovedTransformerModel_20260521_201718.pt"
ORIGINAL_AFTER_MODEL = ROOT / "models" / "ImprovedTransformerModel_20260521_203745.pt"


class SweepLogger:
    def __init__(self, time_name):
        self.time = time_name

    def info(self, message):
        encoding = sys.stdout.encoding or "utf-8"
        safe_message = str(message).encode(encoding, errors="replace").decode(encoding)
        print(safe_message)


def load_resample_module():
    module_path = ROOT / "datasets" / "resample_frame_rate.py"
    spec = importlib.util.spec_from_file_location("resample_frame_rate", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_paths(values):
    return [Path(value).resolve() for value in values]


def validate_resampled_dataset(folder, point_num=22, output_len=105):
    expected_dim = point_num * 3
    txt_files = sorted(Path(folder).glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {folder}")

    for path in txt_files:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != output_len + 1:
            raise ValueError(f"{path}: expected {output_len + 1} non-empty lines, got {len(lines)}")

        for index, line in enumerate(lines[:output_len], start=1):
            frame_str, coords_str = line.split(":", 1)
            if int(frame_str) != index:
                raise ValueError(f"{path}: expected output frame {index}, got {frame_str}")
            coord_count = len(coords_str.split(","))
            if coord_count != expected_dim:
                raise ValueError(f"{path}: expected {expected_dim} coordinates, got {coord_count}")

        label_coord_count = len(lines[-1].split(":", 1)[1].split(","))
        if label_coord_count != 3:
            raise ValueError(f"{path}: expected 3 label coordinates, got {label_coord_count}")


def resample_to_fps300(force=False):
    resample_module = load_resample_module()
    for input_dir, output_dir in zip(ORIGINAL_DATASETS, FPS300_DATASETS):
        if output_dir.exists() and any(output_dir.glob("*.txt")) and not force:
            print(f"Skip existing fps300 dataset: {output_dir}")
        else:
            print(f"Resampling {input_dir} -> {output_dir}")
            resample_module.resample_directory(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                source_fps=SOURCE_FPS[input_dir.name],
                target_fps=300,
                method="linear",
                point_num=22,
                hit_index=100,
                output_len=105,
            )
        validate_resampled_dataset(output_dir, point_num=22, output_len=105)


def model_norm_stats_path(model_path):
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_norm_stats.npz")


def save_norm_stats_for_model(model_path, data_folders, point_num, force=False,
                              use_time_pos_encoding=False, time_label_unit="frames",
                              reference_fps=300.0, hit_index=100, dataset_fps=None):
    stats_path = model_norm_stats_path(model_path)
    if stats_path.exists() and not force:
        return stats_path

    from util.dataset import BadmintonDataset, parse_dataset_fps_values, split_samples_by_dataset

    dataset_fps = parse_dataset_fps_values(dataset_fps)

    random.seed(42)
    np.random.seed(42)
    train_samples, _, dataset_stats = split_samples_by_dataset(
        [str(path) for path in data_folders],
        point_num=point_num,
        train_ratio=0.8,
        dataset_fps=dataset_fps,
    )
    train_dataset = BadmintonDataset(
        train_samples,
        mode="train",
        min_len=10,
        max_len=50,
        use_time_pos_encoding=use_time_pos_encoding,
        time_label_unit=time_label_unit,
        reference_fps=reference_fps,
        hit_index=hit_index,
    )
    feature_mean, feature_std, label_mean, label_std = train_dataset.get_norm_stats()

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        stats_path,
        feature_mean=feature_mean,
        feature_std=feature_std,
        label_mean=label_mean,
        label_std=label_std,
        time_label_unit=np.array(time_label_unit),
        use_time_pos_encoding=np.array(use_time_pos_encoding),
        reference_fps=np.array(reference_fps),
    )
    print(f"Saved norm stats for {model_path} -> {stats_path}")
    for stat in dataset_stats:
        print(
            f"  {stat['dataset']}: total={stat['total']}, train={stat['train']}, test={stat['test']}"
        )
    return stats_path


def run_command(command):
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    print("Running:", " ".join(str(part) for part in command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def newest_new_model(before_paths, model_dir):
    current = set(Path(model_dir).glob("*.pt"))
    new_paths = sorted(current - before_paths, key=lambda path: path.stat().st_mtime, reverse=True)
    if not new_paths:
        raise RuntimeError(f"Training finished but no new .pt model was found in {model_dir}")
    return new_paths[0]


def train_fps300_model(scenario, args):
    provided = args.before_fps300_model if scenario == "before" else args.after_fps300_model
    if provided:
        return Path(provided).resolve()

    model_dir = ROOT / "models"
    before_paths = set(model_dir.glob("*.pt"))
    if scenario == "before":
        points_num = 21
        offset_args = ["--min_offset_len", "5", "--max_offset_len", "25", "--temp_test_offset", "5"]
    else:
        points_num = 22
        offset_args = ["--min_offset_len", "0", "--max_offset_len", "4", "--temp_test_offset", "0"]

    command = [
        sys.executable,
        "train/train_landpoint_pred.py",
        "--data_folders",
        *[str(path) for path in FPS300_DATASETS],
        "--points_num",
        str(points_num),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--batch_size",
        str(args.batch_size),
        *offset_args,
    ]
    run_command(command)
    model_path = newest_new_model(before_paths, model_dir)
    print(f"New fps300 {scenario} model: {model_path}")
    return model_path


def train_timepe_model(scenario, args):
    provided = args.before_timepe_model if scenario == "before" else args.after_timepe_model
    if provided:
        return Path(provided).resolve()

    model_dir = ROOT / "models"
    before_paths = set(model_dir.glob("*.pt"))
    if scenario == "before":
        points_num = 21
        offset_args = ["--min_offset_len", "5", "--max_offset_len", "25", "--temp_test_offset", "5"]
    else:
        points_num = 22
        offset_args = ["--min_offset_len", "0", "--max_offset_len", "4", "--temp_test_offset", "0"]

    command = [
        sys.executable,
        "train/train_landpoint_pred.py",
        "--data_folders",
        *[str(path) for path in ORIGINAL_DATASETS],
        "--points_num",
        str(points_num),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--batch_size",
        str(args.batch_size),
        "--use_time_pos_encoding",
        "--time_label_unit",
        "seconds",
        "--reference_fps",
        str(args.reference_fps),
        "--hit_index",
        str(args.hit_index),
        *offset_args,
    ]
    run_command(command)
    model_path = newest_new_model(before_paths, model_dir)
    print(f"New time PE {scenario} model: {model_path}")
    return model_path


def load_norm_stats(path):
    data = np.load(path)
    return {
        "feature_mean": data["feature_mean"],
        "feature_std": data["feature_std"],
        "label_mean": data["label_mean"],
        "label_std": data["label_std"],
    }


def evaluate_model(model_path, stats_path, data_folders, point_num, offset, scenario, model_variant, args,
                   samples=None, use_time_pos_encoding=False, time_label_unit="frames"):
    import torch  # noqa: F401
    from util.dataset import BadmintonDataset, load_samples_from_folders
    from util.model import ImprovedTransformerModel, ImprovedTransformerTimePEModel
    from util.trainer import Trainer

    stats = load_norm_stats(stats_path)
    if samples is None:
        samples = load_samples_from_folders([str(path) for path in data_folders], point_num=point_num)
    eval_args = SimpleNamespace(
        max_len=50,
        min_len=10,
        min_offset_len=0,
        max_offset_len=0,
        temp_test_offset=offset,
        num_subsamples=1,
        aug_method="None",
        delta=1.0,
        lambda_time=0.1,
        lambda_direction=0.1,
        use_time_pos_encoding=use_time_pos_encoding,
        time_label_unit=time_label_unit,
        reference_fps=args.reference_fps,
        hit_index=args.hit_index,
    )
    dataset = BadmintonDataset(
        samples,
        mode="test",
        max_len=eval_args.max_len,
        min_offset_len=eval_args.min_offset_len,
        max_offset_len=eval_args.max_offset_len,
        temp_test_offset=eval_args.temp_test_offset,
        feature_mean=stats["feature_mean"],
        feature_std=stats["feature_std"],
        label_mean=stats["label_mean"],
        label_std=stats["label_std"],
        use_time_pos_encoding=use_time_pos_encoding,
        time_label_unit=time_label_unit,
        reference_fps=args.reference_fps,
        hit_index=args.hit_index,
    )
    if use_time_pos_encoding:
        model = ImprovedTransformerTimePEModel(
            seq_len=eval_args.max_len,
            num_points=point_num,
            reference_fps=args.reference_fps,
        )
    else:
        model = ImprovedTransformerModel(seq_len=eval_args.max_len, num_points=point_num)
    logger_name = f"offset_sweep_{scenario}_{model_variant}_offset{offset}"
    trainer = Trainer(
        args=eval_args,
        logger=SweepLogger(logger_name),
        model=model,
        train_dataset=dataset,
        test_dataset=dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=str(ROOT / "models"),
    )
    trainer.best_model_path = str(model_path)
    detail_dir = ROOT / "results" / "offset_sweep_details"
    df = trainer.test_and_save(save_dir=str(detail_dir))

    err_xy = np.sqrt((df["pred_x"] - df["label_x"]) ** 2 + (df["pred_y"] - df["label_y"]) ** 2)
    result_csv = detail_dir / f"{logger_name}.csv"
    return {
        "scenario": scenario,
        "model_variant": model_variant,
        "model_path": str(model_path),
        "temp_test_offset": offset,
        "mean_xy_error": float(err_xy.mean()),
        "result_csv": str(result_csv),
    }


def plot_summary(records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    results_dir = ROOT / "results"
    vis_dir = ROOT / "visualization"
    results_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    labels = {
        "original_mixed": "Original mixed",
        "fps300": "FPS 300",
        "time_pe": "Time PE",
    }
    axis_titles = {
        "before": ("frames in advance", "Mean XY Error vs Frames in Advance"),
        "after": ("frames extend", "Mean XY Error vs Frames Extended"),
    }
    for scenario in ["before", "after"]:
        sub = df[df["scenario"] == scenario].copy()
        sub = sub.sort_values(["model_variant", "temp_test_offset"])
        sub.to_csv(results_dir / f"offset_sweep_{scenario}.csv", index=False)
        if scenario == "after":
            max_offset = int(sub["temp_test_offset"].max())
            sub["plot_x"] = max_offset - sub["temp_test_offset"] + 1
            x_ticks = [1, 2, 3, 4, 5]
        else:
            sub["plot_x"] = sub["temp_test_offset"] - 5
            x_ticks = [20, 15, 10, 5, 0]
        x_label, title = axis_titles[scenario]

        fig, ax = plt.subplots(figsize=(8, 5))
        for variant in ["original_mixed", "fps300", "time_pe"]:
            line = sub[sub["model_variant"] == variant].sort_values("plot_x")
            if line.empty:
                continue
            ax.plot(
                line["plot_x"],
                line["mean_xy_error"],
                marker="o",
                label=labels.get(variant, variant),
            )
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(value) for value in x_ticks])
        if scenario == "before":
            ax.set_xlim(21, -1)
        else:
            ax.set_xlim(0.5, 5.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean XY error")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(vis_dir / f"offset_sweep_{scenario}.png", dpi=200)
        plt.close(fig)


def run_evaluations(before_fps300_model, after_fps300_model, before_timepe_model, after_timepe_model, args):
    from util.dataset import load_samples_from_folders

    configs = [
        {
            "scenario": "before",
            "offsets": [5, 10, 15, 20, 25],
            "points_num": 21,
            "models": [
                ("original_mixed", Path(args.before_original_model).resolve(), ORIGINAL_DATASETS, False, "frames"),
                ("fps300", Path(before_fps300_model).resolve(), FPS300_DATASETS, False, "frames"),
                ("time_pe", Path(before_timepe_model).resolve(), ORIGINAL_DATASETS, True, "seconds"),
            ],
        },
        {
            "scenario": "after",
            "offsets": [0, 1, 2, 3, 4],
            "points_num": 22,
            "models": [
                ("original_mixed", Path(args.after_original_model).resolve(), ORIGINAL_DATASETS, False, "frames"),
                ("fps300", Path(after_fps300_model).resolve(), FPS300_DATASETS, False, "frames"),
                ("time_pe", Path(after_timepe_model).resolve(), ORIGINAL_DATASETS, True, "seconds"),
            ],
        },
    ]

    records = []
    sample_cache = {}
    for config in configs:
        for variant, model_path, data_folders, use_time_pos_encoding, time_label_unit in config["models"]:
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found for {config['scenario']} {variant}: {model_path}")
            stats_path = save_norm_stats_for_model(
                model_path=model_path,
                data_folders=data_folders,
                point_num=config["points_num"],
                force=args.force_norm_stats,
                use_time_pos_encoding=use_time_pos_encoding,
                time_label_unit=time_label_unit,
                reference_fps=args.reference_fps,
                hit_index=args.hit_index,
            )
            cache_key = (
                tuple(str(Path(path).resolve()) for path in data_folders),
                config["points_num"],
            )
            if cache_key not in sample_cache:
                sample_cache[cache_key] = load_samples_from_folders(
                    list(cache_key[0]),
                    point_num=config["points_num"],
                )
            for offset in config["offsets"]:
                records.append(
                    evaluate_model(
                        model_path=model_path,
                        stats_path=stats_path,
                        data_folders=data_folders,
                        point_num=config["points_num"],
                        offset=offset,
                        scenario=config["scenario"],
                        model_variant=variant,
                        args=args,
                        samples=sample_cache[cache_key],
                        use_time_pos_encoding=use_time_pos_encoding,
                        time_label_unit=time_label_unit,
                    )
                )
    plot_summary(records)


def parse_args():
    parser = argparse.ArgumentParser(description="Run FPS300 training and offset sweep evaluation.")
    parser.add_argument("--skip_resample", action="store_true", help="Skip generating fps300 datasets")
    parser.add_argument("--force_resample", action="store_true", help="Regenerate fps300 datasets if they already exist")
    parser.add_argument("--skip_train", action="store_true", help="Skip all training and require model paths")
    parser.add_argument("--skip_fps300_train", action="store_true", help="Skip fps300 training")
    parser.add_argument("--skip_timepe_train", action="store_true", help="Skip time PE training")
    parser.add_argument("--skip_eval", action="store_true", help="Skip offset sweep evaluation")
    parser.add_argument("--force_norm_stats", action="store_true", help="Regenerate norm stats even if present")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--reference_fps", type=float, default=300.0)
    parser.add_argument("--hit_index", type=int, default=100)
    parser.add_argument("--before_original_model", default=str(ORIGINAL_BEFORE_MODEL))
    parser.add_argument("--after_original_model", default=str(ORIGINAL_AFTER_MODEL))
    parser.add_argument("--before_fps300_model", default=None)
    parser.add_argument("--after_fps300_model", default=None)
    parser.add_argument("--before_timepe_model", default=None)
    parser.add_argument("--after_timepe_model", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.skip_train:
        args.skip_fps300_train = True
        args.skip_timepe_train = True

    if not args.skip_resample:
        resample_to_fps300(force=args.force_resample)

    before_fps300_model = Path(args.before_fps300_model).resolve() if args.before_fps300_model else None
    after_fps300_model = Path(args.after_fps300_model).resolve() if args.after_fps300_model else None
    before_timepe_model = Path(args.before_timepe_model).resolve() if args.before_timepe_model else None
    after_timepe_model = Path(args.after_timepe_model).resolve() if args.after_timepe_model else None

    if not args.skip_fps300_train:
        before_fps300_model = train_fps300_model("before", args)
        after_fps300_model = train_fps300_model("after", args)
    if not args.skip_timepe_train:
        before_timepe_model = train_timepe_model("before", args)
        after_timepe_model = train_timepe_model("after", args)

    if not args.skip_eval:
        missing = []
        if before_fps300_model is None:
            missing.append("--before_fps300_model")
        if after_fps300_model is None:
            missing.append("--after_fps300_model")
        if before_timepe_model is None:
            missing.append("--before_timepe_model")
        if after_timepe_model is None:
            missing.append("--after_timepe_model")
        if missing:
            raise ValueError("Evaluation requires model paths or training enabled for: " + ", ".join(missing))

    if not args.skip_eval:
        run_evaluations(before_fps300_model, after_fps300_model, before_timepe_model, after_timepe_model, args)


if __name__ == "__main__":
    main()
