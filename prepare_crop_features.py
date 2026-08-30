import argparse
import json
import os
from pathlib import Path

import opensmile
import pandas as pd

from integrity import file_sha256


SOURCES = ("msp", "iemocap", "crema_d")
CROP_SECONDS = 5.0


def parse_args():
    parser = argparse.ArgumentParser(
        "Create eGeMAPS features matched to deterministic center crops"
    )
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def center_intervals(durations, crop_seconds=CROP_SECONDS):
    starts = ((durations - crop_seconds) / 2.0).clip(lower=0.0)
    ends = starts + durations.clip(upper=crop_seconds)
    return starts, ends


def output_complete(
    output,
    manifest,
    train,
    feature_columns,
    train_sha256,
    feature_sha256,
):
    if not output.exists() or not manifest.exists():
        return False
    metadata = json.loads(manifest.read_text())
    expected = {
        "crop_seconds": CROP_SECONDS,
        "selection": "center",
        "train_sha256": train_sha256,
        "feature_sha256": feature_sha256,
        "opensmile_version": opensmile.__version__,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return False
    if metadata.get("output_sha256") != file_sha256(output):
        return False
    frame = pd.read_csv(output, low_memory=False)
    return (
        list(frame.columns) == feature_columns
        and len(frame) == len(train)
        and not frame["sample_id"].duplicated().any()
        and set(frame["sample_id"]) == set(train["sample_id"])
    )


def prepare_source(source, split_root, feature_root, output_root, num_workers):
    train_path = split_root / source / "train.csv"
    feature_path = feature_root / f"{source}_train_eGeMAPSv02.csv"
    output = output_root / f"{source}_train_eGeMAPSv02.csv"
    manifest = output.with_suffix(".json")

    train = pd.read_csv(
        train_path,
        usecols=["sample_id", "audio_path"],
    )
    features = pd.read_csv(feature_path, low_memory=False)
    if train["sample_id"].duplicated().any():
        raise ValueError(f"{source} Train has duplicate sample_id values")
    if features["sample_id"].duplicated().any():
        raise ValueError(f"{source} features have duplicate sample_id values")
    if set(train["sample_id"]) != set(features["sample_id"]):
        raise ValueError(f"{source} Train/features sample_id mismatch")
    train_sha256 = file_sha256(train_path)
    feature_sha256 = file_sha256(feature_path)
    if output_complete(
        output,
        manifest,
        train,
        list(features.columns),
        train_sha256,
        feature_sha256,
    ):
        print(f"Reusing complete crop-matched features: {output}")
        return

    metadata = train.merge(
        features[["sample_id", "Duration"]],
        on="sample_id",
        validate="one_to_one",
    )
    long = metadata[metadata["Duration"] > CROP_SECONDS].copy()
    missing = [path for path in long["audio_path"] if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"{source}: {len(missing)} audio files are missing; first: {missing[0]}"
        )

    matched = features.set_index("sample_id")
    if not long.empty:
        starts, ends = center_intervals(long["Duration"])
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
            num_workers=num_workers,
            verbose=True,
        )
        extracted = smile.process_files(
            long["audio_path"].tolist(),
            starts=starts.tolist(),
            ends=ends.tolist(),
        ).reset_index()
        if len(extracted) != len(long):
            raise ValueError(
                f"{source}: extracted {len(extracted)} of {len(long)} long crops"
            )
        expected_paths = [os.path.realpath(path) for path in long["audio_path"]]
        observed_paths = [os.path.realpath(str(path)) for path in extracted["file"]]
        if observed_paths != expected_paths:
            raise ValueError(f"{source}: openSMILE output order/path mismatch")
        extracted["sample_id"] = long["sample_id"].to_numpy()
        extracted["Duration"] = (
            extracted["end"] - extracted["start"]
        ).dt.total_seconds()
        replacement_columns = [
            column
            for column in extracted
            if column not in {"file", "start", "end", "sample_id"}
        ]
        missing_columns = sorted(set(replacement_columns) - set(matched.columns))
        if missing_columns:
            raise ValueError(
                f"{source}: unexpected extracted columns: {missing_columns}"
            )
        replacement = extracted.set_index("sample_id")[replacement_columns]
        matched.loc[replacement.index, replacement_columns] = replacement

    matched = matched.reset_index()[features.columns]
    if matched.isna().any().any():
        bad = matched.columns[matched.isna().any()].tolist()
        raise ValueError(f"{source}: crop-matched features contain NaN: {bad}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    matched.to_csv(temporary, index=False)
    os.replace(temporary, output)
    output_sha256 = file_sha256(output)
    manifest.write_text(
        json.dumps(
            {
                "source": source,
                "crop_seconds": CROP_SECONDS,
                "selection": "center",
                "opensmile_version": opensmile.__version__,
                "train_sha256": train_sha256,
                "feature_sha256": feature_sha256,
                "output_sha256": output_sha256,
                "rows": len(matched),
                "recomputed_rows": len(long),
                "source_train_csv": str(train_path),
                "source_feature_csv": str(feature_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Saved {len(matched)} rows to {output}; "
        f"recomputed center crops for {len(long)} utterances"
    )


def main():
    args = parse_args()
    split_root = Path(args.split_root)
    feature_root = Path(args.feature_root)
    output_root = Path(args.output_root)
    for source in SOURCES:
        prepare_source(
            source,
            split_root,
            feature_root,
            output_root,
            args.num_workers,
        )


if __name__ == "__main__":
    main()
