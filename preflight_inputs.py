import argparse
import json
import os
from pathlib import Path

import opensmile
import pandas as pd

from dataset import load_eval_frame
from factor_data import SOURCES
from integrity import file_sha256
from prepare_crop_features import CROP_SECONDS


CAMEO_TARGETS = (
    "cafe",
    "emns",
    "emozionalmente",
    "enterface",
    "jl_corpus",
    "mesd",
    "nemo",
    "oreau",
    "pavoque",
    "ravdess",
    "resd",
    "subesco",
)


def parse_args():
    parser = argparse.ArgumentParser("Validate every input before formal training")
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--source-feature-root", required=True)
    parser.add_argument("--prepared-feature-root", required=True)
    parser.add_argument("--cameo-csv-root", required=True)
    parser.add_argument("--cameo-audio-root", required=True)
    return parser.parse_args()


def validate_paths(csv_path, audio_root):
    frame = load_eval_frame(csv_path, audio_root)
    missing = [path for path in frame["audio_path"] if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"{csv_path}: {len(missing)} audio files missing; first: {missing[0]}"
        )
    print(f"Validated {len(frame)} audio paths: {csv_path}")


def validate_prepared_features(
    source,
    split_root,
    source_feature_root,
    prepared_feature_root,
):
    train = pd.read_csv(
        split_root / source / "train.csv",
        usecols=["sample_id"],
    )
    source_feature_path = (
        source_feature_root / f"{source}_train_eGeMAPSv02.csv"
    )
    feature_path = (
        prepared_feature_root / f"{source}_train_eGeMAPSv02.csv"
    )
    manifest_path = feature_path.with_suffix(".json")
    features = pd.read_csv(feature_path, usecols=["sample_id"])
    manifest = json.loads(manifest_path.read_text())
    expected_manifest = {
        "crop_seconds": CROP_SECONDS,
        "selection": "center",
        "opensmile_version": opensmile.__version__,
        "train_sha256": file_sha256(split_root / source / "train.csv"),
        "feature_sha256": file_sha256(source_feature_path),
        "output_sha256": file_sha256(feature_path),
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise ValueError(f"{source}: manifest mismatch for {name}")
    if manifest.get("crop_seconds") != CROP_SECONDS:
        raise ValueError(f"{source}: prepared feature crop is not five seconds")
    if manifest.get("selection") != "center":
        raise ValueError(f"{source}: prepared features are not center crops")
    if manifest.get("rows") != len(train):
        raise ValueError(f"{source}: manifest row count mismatch")
    if features["sample_id"].duplicated().any():
        raise ValueError(f"{source}: duplicate prepared feature sample_id")
    if set(features["sample_id"]) != set(train["sample_id"]):
        raise ValueError(f"{source}: prepared features do not cover Train")
    print(
        f"Validated prepared {source} features: {len(features)} rows, "
        f"{manifest.get('recomputed_rows')} center crops"
    )


def main():
    args = parse_args()
    split_root = Path(args.split_root)
    source_feature_root = Path(args.source_feature_root)
    feature_root = Path(args.prepared_feature_root)
    cameo_csv_root = Path(args.cameo_csv_root)
    cameo_audio_root = Path(args.cameo_audio_root)

    for source in SOURCES:
        validate_prepared_features(
            source,
            split_root,
            source_feature_root,
            feature_root,
        )
        for split in ("train", "development", "test"):
            validate_paths(split_root / source / f"{split}.csv", "/")
    validate_paths(split_root / "ravdess" / "test.csv", "/")
    validate_paths(split_root / "tess" / "full.csv", "/")
    for target in CAMEO_TARGETS:
        validate_paths(
            cameo_csv_root / f"{target}.csv",
            cameo_audio_root / target / "audio",
        )

    ready = feature_root / "READY"
    ready.write_text("crop=center\ncrop_seconds=5.0\n")
    print(f"All inputs validated; wrote {ready}")


if __name__ == "__main__":
    main()
