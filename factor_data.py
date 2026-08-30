import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import WeightedRandomSampler

from dataset import TemplateDataset, load_training_frame


SOURCES = ("msp", "iemocap", "crema_d")
SOURCE_EMOTIONS = {
    "iemocap": ("angry", "happy", "neutral", "sad", "excited", "frustrated"),
}
FACTOR_GROUPS = {
    "pitch": (
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
        "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
        "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
        "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
    ),
    "energy": (
        "loudness_sma3_stddevNorm",
        "loudness_sma3_pctlrange0-2",
        "loudness_sma3_meanRisingSlope",
        "loudness_sma3_meanFallingSlope",
    ),
    "rhythm": (
        "VoicedSegmentsPerSec",
        "MeanVoicedSegmentLengthSec",
        "MeanUnvoicedSegmentLength",
    ),
}
FACTOR_COLUMNS = tuple(
    column for columns in FACTOR_GROUPS.values() for column in columns
)
TARGET_COLUMNS = tuple(f"factor_target.{column}" for column in FACTOR_COLUMNS)


class PooledTrainingDataset(TemplateDataset):
    def __init__(self, frame, include_factor_targets):
        super().__init__(
            frame,
            condition="only_emo",
            caption_pairing="utterance",
            acoustic_tag_schema="crosscorpus",
            emotion_wording="original",
            canonical_tag_content="full",
        )
        self.include_factor_targets = include_factor_targets

    def __getitem__(self, item):
        sample = super().__getitem__(item)
        if self.include_factor_targets:
            target = torch.tensor(
                self.df.loc[item, list(TARGET_COLUMNS)].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
        else:
            target = torch.empty(0, dtype=torch.float32)
        return (*sample, target)


def add_source_quantile_targets(frame):
    missing = sorted(set(FACTOR_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing factor columns: {missing}")
    values = frame.loc[:, FACTOR_COLUMNS].replace([np.inf, -np.inf], np.nan)
    if values.isna().any().any():
        bad = values.columns[values.isna().any()].tolist()
        raise ValueError(f"Non-finite factor values: {bad}")
    for column in FACTOR_COLUMNS:
        transformer = QuantileTransformer(
            n_quantiles=min(1000, len(frame)),
            output_distribution="uniform",
            subsample=None,
        )
        frame[f"factor_target.{column}"] = transformer.fit_transform(
            frame[[column]]
        ).ravel()
    return frame


def shuffle_targets_within_source_emotion(frame, seed):
    shuffled = frame.copy()
    rng = np.random.default_rng(seed)
    for _, positions in shuffled.groupby(
        ["_source_corpus", "emotion"], sort=False
    ).indices.items():
        positions = np.asarray(positions)
        if len(positions) > 1:
            permutation = rng.permutation(positions)
            while np.any(permutation == positions):
                permutation = rng.permutation(positions)
            shuffled.loc[positions, list(TARGET_COLUMNS)] = frame.loc[
                permutation, list(TARGET_COLUMNS)
            ].to_numpy()
    return shuffled


def corpus_emotion_sampler(frame):
    counts = frame.groupby(["_source_corpus", "emotion"]).size()
    classes_per_source = frame.groupby("_source_corpus")["emotion"].nunique()
    weights = [
        1.0 / (classes_per_source[source] * counts[source, emotion])
        for source, emotion in frame[["_source_corpus", "emotion"]].itertuples(
            index=False, name=None
        )
    ]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(frame),
        replacement=True,
    )


def load_pooled_training_data(
    split_root,
    feature_root,
    shuffle_factor_targets=False,
    seed=3407,
):
    frames = []
    emotions_by_source = {}
    train_csvs = {}
    dev_csvs = {}
    for source in SOURCES:
        train_csv = os.path.join(split_root, source, "train.csv")
        dev_csv = os.path.join(split_root, source, "development.csv")
        feature_csv = os.path.join(feature_root, f"{source}_train_eGeMAPSv02.csv")
        frame = load_training_frame(
            train_csv,
            feature_csv,
            "/",
            emotions=SOURCE_EMOTIONS.get(source),
            include_dimensions=False,
        )
        frame["_source_corpus"] = source
        frame = add_source_quantile_targets(frame)
        emotions = sorted(frame["emotion"].unique())
        dev_emotions = set(
            pd.read_csv(dev_csv, usecols=["gt_emo"])["gt_emo"]
            .astype(str)
            .str.lower()
        )
        selected_emotions = SOURCE_EMOTIONS.get(source)
        if selected_emotions:
            missing_train = sorted(set(selected_emotions) - set(emotions))
            missing_dev = sorted(set(selected_emotions) - dev_emotions)
            if missing_train or missing_dev:
                raise ValueError(
                    f"{source} selected labels missing from "
                    f"Train={missing_train}, Development={missing_dev}"
                )
        else:
            invalid = sorted(dev_emotions - set(emotions))
            if invalid:
                raise ValueError(
                    f"{source} Development labels absent from Train: {invalid}"
                )
        frames.append(frame)
        emotions_by_source[source] = emotions
        train_csvs[source] = train_csv
        dev_csvs[source] = dev_csv

    pooled = pd.concat(frames, ignore_index=True)
    if shuffle_factor_targets:
        pooled = shuffle_targets_within_source_emotion(pooled, seed)
    return (
        pooled,
        corpus_emotion_sampler(pooled),
        emotions_by_source,
        train_csvs,
        dev_csvs,
    )
