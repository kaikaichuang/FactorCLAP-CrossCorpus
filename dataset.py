import math
import os
import random

import audiofile
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from sklearn.preprocessing import QuantileTransformer


SAMPLE_RATE = 16000
AUDIO_SAMPLES = 5 * SAMPLE_RATE
TAG_FEATURE_COLUMNS = (
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "equivalentSoundLevel_dBp",
    "loudness_sma3_amean",
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
    "Duration",
)


def quantile_level(value, middle="normal"):
    if value >= 0.7:
        return "high"
    if value <= 0.3:
        return "low"
    return middle


def intensity_level(row):
    if (
        row["equivalentSoundLevel_dBp.quantile"] >= 0.7
        or row["loudness_sma3_amean.quantile"] >= 0.7
    ):
        return "high"
    if (
        row["equivalentSoundLevel_dBp.quantile"] <= 0.3
        or row["loudness_sma3_amean.quantile"] <= 0.3
    ):
        return "low"
    return "normal"


def build_canonical_tags(row):
    tags = [f'emotion {row["emotion"]}']
    if row["gender"] != "unknown":
        tags.append(f'gender {row["gender"]}')
    for column, name in (
        ("F0semitoneFrom27.5Hz_sma3nz_amean.quantile", "pitch"),
        ("F0semitoneFrom27.5Hz_sma3nz_stddevNorm.quantile", "pitch variation"),
        ("jitterLocal_sma3nz_amean.quantile", "jitter"),
        ("shimmerLocaldB_sma3nz_amean.quantile", "shimmer"),
    ):
        tags.append(f"{quantile_level(row[column])} {name}")
    tags.append(f"{intensity_level(row)} intensity")
    duration = quantile_level(row["Duration.quantile"])
    duration_word = {"high": "long", "low": "short", "normal": "normal"}[duration]
    tags.append(f"{duration_word} duration")
    return ", ".join(tags)


def resolve_audio_path(wav_name, audio_root):
    if os.path.isabs(wav_name):
        if os.path.exists(wav_name) or not audio_root:
            return wav_name
        return os.path.join(audio_root, os.path.basename(wav_name))
    return os.path.join(audio_root, wav_name)


def load_waveform(path):
    audio, sample_rate = audiofile.read(path, always_2d=True)
    audio = audio.mean(axis=0)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sample_rate // divisor)
    return np.asarray(audio, dtype=np.float32)


def waveform_num_samples(path):
    samples = audiofile.samples(path)
    sample_rate = audiofile.sampling_rate(path)
    return math.ceil(samples * SAMPLE_RATE / sample_rate)


def center_crop_or_pad(audio, num_samples=AUDIO_SAMPLES):
    attention_mask = np.ones(num_samples, dtype=np.int64)
    if len(audio) < num_samples:
        attention_mask[len(audio) :] = 0
        audio = np.pad(audio, (0, num_samples - len(audio)))
    elif len(audio) > num_samples:
        start = (len(audio) - num_samples) // 2
        audio = audio[start : start + num_samples]
    return audio, attention_mask


def load_training_frame(
    csv_path,
    feature_csv,
    audio_root,
    emotions=None,
    known_speakers_only=False,
    include_dimensions=False,
):
    if include_dimensions:
        raise ValueError("This experiment intentionally excludes VAD dimensions")
    labels = pd.read_csv(csv_path)
    features = pd.read_csv(feature_csv)
    merge_key = (
        "sample_id"
        if "sample_id" in labels and "sample_id" in features
        else "wav_name"
    )
    frame = labels.merge(features, on=merge_key, how="inner", validate="one_to_one")
    if len(frame) != len(labels):
        raise ValueError(
            f"Feature CSV covers {len(frame)} of {len(labels)} training rows"
        )
    frame["emotion"] = frame["gt_emo"].astype(str).str.lower()
    if emotions:
        frame = frame[frame["emotion"].isin(emotions)].reset_index(drop=True)
    if known_speakers_only:
        frame = frame[
            frame["speaker_id"].astype(str) != "Unknown"
        ].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No training rows remain")
    frame["gender"] = (
        frame["gender"].fillna("unknown").astype(str).str.lower()
    )
    frame["audio_path"] = frame["wav_name"].map(
        lambda name: resolve_audio_path(name, audio_root)
    )
    for column in TAG_FEATURE_COLUMNS:
        if column not in frame:
            raise ValueError(f"Missing tag feature column: {column}")
        transformer = QuantileTransformer(
            n_quantiles=min(1000, len(frame)),
            output_distribution="uniform",
            subsample=None,
        )
        frame[f"{column}.quantile"] = transformer.fit_transform(
            frame[[column]]
        ).ravel()
    return frame


def load_eval_frame(csv_path, audio_root=None):
    frame = pd.read_csv(csv_path)
    frame["emotion"] = frame["gt_emo"].astype(str).str.lower()
    frame["audio_path"] = frame["wav_name"].map(
        lambda name: resolve_audio_path(name, audio_root) if audio_root else name
    )
    return frame


def emotion_queries(emotion):
    return (
        f"sentence is {emotion}",
        f"this is a {emotion} instance",
        f"emotion is {emotion}",
        f"speaker is {emotion}",
    )


class TemplateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame,
        condition,
        caption_pairing,
        acoustic_tag_schema,
        emotion_wording,
        canonical_tag_content,
    ):
        expected = (
            condition,
            caption_pairing,
            acoustic_tag_schema,
            emotion_wording,
            canonical_tag_content,
        )
        if expected != (
            "only_emo",
            "utterance",
            "crosscorpus",
            "original",
            "full",
        ):
            raise ValueError(f"Unsupported experiment dataset settings: {expected}")
        self.df = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        audio, attention_mask = center_crop_or_pad(load_waveform(row["audio_path"]))
        caption = random.choice(emotion_queries(row["emotion"]))
        return (
            torch.from_numpy(audio),
            torch.from_numpy(attention_mask),
            caption,
            build_canonical_tags(row),
            row["emotion"],
            row["wav_name"],
        )


class TargetDataset(torch.utils.data.Dataset):
    def __init__(self, frame):
        self.df = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        return (
            torch.from_numpy(load_waveform(row["audio_path"])),
            row["emotion"],
            row["wav_name"],
        )


def collate_eval(batch, fixed_length=None):
    waveforms, labels, wav_names = zip(*batch)
    max_length = fixed_length or max(len(waveform) for waveform in waveforms)
    audio = torch.zeros(len(waveforms), max_length, dtype=torch.float32)
    attention_mask = torch.zeros(len(waveforms), max_length, dtype=torch.long)
    for index, waveform in enumerate(waveforms):
        length = len(waveform)
        audio[index, :length] = waveform
        attention_mask[index, :length] = 1
    return audio, attention_mask, list(labels), list(wav_names)
