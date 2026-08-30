import math
import os
import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

from dataset import (
    SAMPLE_RATE,
    TargetDataset,
    collate_eval,
    load_eval_frame,
    waveform_num_samples,
)
from utils import compute_similarity


def collate_length_bucket(batch):
    max_length = max(len(item[0]) for item in batch)
    padded_length = math.ceil(max_length / SAMPLE_RATE) * SAMPLE_RATE
    return collate_eval(batch, fixed_length=padded_length)


def evaluate_csv(
    model,
    tokenizer,
    csv_path,
    device,
    batch_size,
    audio_root=None,
    emotions=None,
    candidate_emotions=None,
    label_mapping=None,
    known_speakers_only=False,
    query_style="label",
    query_aliases=None,
    output_csv=None,
    tqdm_disable=False,
):
    df = load_eval_frame(csv_path, audio_root)
    if label_mapping:
        df["emotion"] = df["emotion"].replace(label_mapping)
    if known_speakers_only:
        df = df[df["speaker_id"].astype(str) != "Unknown"].reset_index(drop=True)
    if emotions:
        missing = sorted(set(emotions) - set(df["emotion"].unique()))
        if missing:
            raise ValueError(f"Evaluation emotions not found in CSV: {missing}")
        df = df[df["emotion"].isin(emotions)].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No evaluation rows match emotions: {emotions}")
    candidates = sorted(candidate_emotions or df["emotion"].unique())
    queries = candidates
    if query_style == "sentence":
        aliases = query_aliases or {}
        queries = [f"this person is feeling {aliases.get(emotion, emotion)}." for emotion in candidates]

    df["_eval_order"] = range(len(df))
    df["_num_samples"] = [
        waveform_num_samples(path) for path in df["audio_path"]
    ]
    df = df.sort_values("_num_samples", kind="stable").reset_index(drop=True)
    candidate_tokens = tokenizer.batch_encode_plus(
        queries,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    loader = DataLoader(
        TargetDataset(df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_length_bucket,
    )

    targets = []
    predictions = []
    score_rows = []
    wav_names = []
    model.eval()
    with torch.no_grad():
        text_features = model.text_projection(model.text_branch(candidate_tokens))
        logit_scale = model.logit_scale.exp()
        for audio, attention_mask, labels, names in tqdm.tqdm(
            loader,
            desc="Evaluate",
            disable=tqdm_disable,
        ):
            audio_features = model.audio_projection(
                model.audio_branch(
                    audio.to(device),
                    attention_mask=attention_mask.to(device),
                )
            )
            similarity = compute_similarity(
                logit_scale, audio_features, text_features
            )
            predictions.extend(candidates[index] for index in similarity.argmax(1).tolist())
            score_rows.extend(similarity.float().cpu().tolist())
            targets.extend(labels)
            wav_names.extend(names)

    results = {
        "WA": float(accuracy_score(targets, predictions)),
        "UAR": float(
            recall_score(
                targets,
                predictions,
                labels=candidates,
                average="macro",
                zero_division=0,
            )
        ),
        "Macro_F1": float(
            f1_score(
                targets,
                predictions,
                labels=candidates,
                average="macro",
                zero_division=0,
            )
        ),
        "WF1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
    }
    chance = 1.0 / len(candidates)
    results["Chance_Corrected_UAR"] = float(
        (results["UAR"] - chance) / (1.0 - chance)
    )
    prediction_counts = pd.Series(predictions).value_counts().reindex(candidates, fill_value=0)
    prediction_probabilities = prediction_counts.to_numpy(dtype=float) / len(predictions)
    nonzero = prediction_probabilities[prediction_probabilities > 0]
    results["Max_Predicted_Share"] = float(prediction_probabilities.max())
    results["Missing_Predicted_Classes"] = int((prediction_counts == 0).sum())
    results["Effective_Predicted_Classes"] = float(
        np.exp(-(nonzero * np.log(nonzero)).sum())
    )
    per_class_recall = recall_score(
        targets,
        predictions,
        labels=candidates,
        average=None,
        zero_division=0,
    )
    results["Worst_Class_Recall"] = float(per_class_recall.min())
    print(
        f"Evaluation rows: {len(df)}, classes: {candidates}, "
        f"length-bucket batch size: {batch_size}"
    )
    print(yaml.safe_dump(results, sort_keys=False), end="")

    if output_csv:
        output = pd.DataFrame(
            {
                "wav_name": wav_names,
                "prediction": predictions,
                "emotion": targets,
                "_eval_order": df["_eval_order"],
            }
        )
        for index, candidate in enumerate(candidates):
            column = f"score_{candidate.replace(' ', '_')}"
            output[column] = [scores[index] for scores in score_rows]
        output.sort_values("_eval_order").drop(columns="_eval_order").to_csv(
            output_csv,
            index=False,
        )
    return results
