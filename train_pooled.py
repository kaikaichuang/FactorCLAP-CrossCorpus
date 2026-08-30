import argparse
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from transformers import logging as transformers_logging

from dataset import AUDIO_SAMPLES, SAMPLE_RATE
from evaluate import evaluate_csv
from factor_data import FACTOR_GROUPS, PooledTrainingDataset, SOURCES
from factor_data import load_pooled_training_data
from factor_model import FactorCLAP
from integrity import file_sha256
from loss_smoothclap import SmoothCLAPLoss
from losses import class_aware_clap_loss, grouped_factor_loss
from models_xin import ParaCLAP, SmoothCLAP


REPO_ROOT = Path(__file__).resolve().parent
CONDITIONS = ("e0_emotion", "e1_smooth", "e2_factor", "e3_shuffled_factor")
SEED = 3407
BATCH_SIZE = 32
EPOCHS = 30


def parse_args():
    parser = argparse.ArgumentParser("Train one pooled first-principles condition")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--initial-state", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--factor-weight", type=float, default=64.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tqdm-disable", action="store_true")
    return parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_torch_save(value, path):
    temporary = f"{path}.tmp"
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_metrics(path, row):
    frame = pd.DataFrame([row])
    if os.path.exists(path):
        history = pd.read_csv(path)
        history = history[history["epoch"] != row["epoch"]]
        frame = pd.concat([history, frame], ignore_index=True)
    frame.sort_values("epoch").to_csv(path, index=False)


def build_model(condition, config):
    common = {
        "speech_name": config["models"]["speech"],
        "text_name": config["models"]["text"],
        "embedding_dim": 768,
        "train_audio_encoder": True,
    }
    if condition == "e1_smooth":
        return SmoothCLAP(
            **common,
            local_speech_name=config["models"]["local_speech"],
        )
    if condition in {"e2_factor", "e3_shuffled_factor"}:
        return FactorCLAP(**common)
    return ParaCLAP(**common)


def experiment_contract(
    args,
    config,
    train_csvs,
    dev_csvs,
):
    input_paths = {
        "initial_state": args.initial_state,
        "config": str(REPO_ROOT / "configs/config.yaml"),
    }
    for source in SOURCES:
        input_paths[f"{source}_train"] = train_csvs[source]
        input_paths[f"{source}_development"] = dev_csvs[source]
        input_paths[f"{source}_features"] = os.path.join(
            args.feature_root,
            f"{source}_train_eGeMAPSv02.csv",
        )
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    factor_condition = args.condition in {
        "e2_factor",
        "e3_shuffled_factor",
    }
    return {
        "condition": args.condition,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "factor_weight": args.factor_weight if factor_condition else 0.0,
        "factor_groups": FACTOR_GROUPS,
        "factor_shuffle": (
            "within_source_emotion_derangement"
            if args.condition == "e3_shuffled_factor"
            else "none"
        ),
        "sampling": "corpus_then_emotion_uniform",
        "audio_view": {
            "selection": "center",
            "samples": AUDIO_SAMPLES,
            "sample_rate": SAMPLE_RATE,
        },
        "main_caption": "emotion_only",
        "main_loss": (
            "smoothclap"
            if args.condition == "e1_smooth"
            else "class_aware_multi_positive"
        ),
        "train_audio_encoder": True,
        "train_text_encoder": True,
        "config": config,
        "code_commit": code_commit,
        "input_paths": input_paths,
        "input_sha256": {
            name: file_sha256(path)
            for name, path in input_paths.items()
        },
    }


def load_shared_initial_state(
    model,
    condition,
    initial_state_path,
    device,
    config,
):
    initial = torch.load(initial_state_path, map_location=device, weights_only=False)
    if initial["seed"] != SEED:
        raise ValueError("Initial-state seed is not 3407")
    if initial.get("models") != config["models"]:
        raise ValueError("Initial-state pretrained model metadata mismatch")
    if initial.get("embedding_dim") != 768:
        raise ValueError("Initial-state embedding dimension is not 768")
    state = initial["model"]
    if condition == "e1_smooth":
        model.load_state_dict(state)
        return
    model_keys = set(model.state_dict())
    shared = {name: value for name, value in state.items() if name in model_keys}
    missing = model_keys - set(shared)
    expected_missing = (
        {name for name in model_keys if name.startswith("factor_heads.")}
        if condition in {"e2_factor", "e3_shuffled_factor"}
        else set()
    )
    if missing != expected_missing:
        raise ValueError(f"Initial state missing unexpected model keys: {sorted(missing)}")
    model.load_state_dict(shared, strict=False)


def build_optimizer(model, config):
    encoder_lr = config["hparams"]["encoder_learning_rate"]
    other_lr = config["hparams"]["other_learning_rate"]
    encoder_parameters = [
        parameter
        for module in (model.audio_branch, model.text_branch)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_ids
    ]
    return torch.optim.Adam(
        [
            {"params": encoder_parameters, "lr": encoder_lr},
            {"params": other_parameters, "lr": other_lr},
        ]
    )


def train_epoch(
    model,
    loader,
    tokenizer,
    optimizer,
    condition,
    smooth_criterion,
    factor_weight,
    device,
    writer,
    epoch,
    tqdm_disable,
):
    model.train()
    totals = {"loss": 0.0, "emotion": 0.0, "factor": 0.0}
    progress = tqdm.tqdm(loader, desc=f"Train epoch {epoch}", disable=tqdm_disable)
    for batch_index, batch in enumerate(progress):
        audio, audio_mask, captions, tags, labels, _, factor_targets = batch
        caption_tokens = tokenizer.batch_encode_plus(
            list(captions), padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        tag_tokens = None
        if condition == "e1_smooth":
            tag_tokens = tokenizer.batch_encode_plus(
                list(tags), padding=True, truncation=True, return_tensors="pt"
            ).to(device)
        audio = audio.to(device)
        audio_mask = audio_mask.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(
                audio,
                caption_tokens,
                tag_tokens,
                audio_attention_mask=audio_mask,
            )
            factor_loss = torch.zeros((), device=device)
            if condition == "e1_smooth":
                emotion_loss = smooth_criterion(*outputs)
            elif condition in {"e2_factor", "e3_shuffled_factor"}:
                text_features, audio_features, factor_predictions, logit_scale = outputs
                emotion_loss = class_aware_clap_loss(
                    text_features, audio_features, labels, logit_scale
                )
                factor_loss, _ = grouped_factor_loss(
                    factor_predictions, factor_targets.to(device)
                )
            else:
                emotion_loss = class_aware_clap_loss(
                    outputs[0], outputs[1], labels, outputs[4]
                )
            loss = emotion_loss + factor_weight * factor_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(0, math.log(100))
        totals["loss"] += loss.item()
        totals["emotion"] += emotion_loss.item()
        totals["factor"] += factor_loss.item()
        progress.set_postfix(loss=f"{loss.item():.5f}")
        if batch_index == 0 or (batch_index + 1) % 50 == 0:
            step = (epoch - 1) * len(loader) + batch_index
            writer.add_scalar("loss/batch_total", loss.item(), step)
            writer.add_scalar("loss/batch_emotion", emotion_loss.item(), step)
            writer.add_scalar("loss/batch_factor", factor_loss.item(), step)
    return {name: value / len(loader) for name, value in totals.items()}


def evaluate_development(
    model,
    tokenizer,
    dev_csvs,
    emotions_by_source,
    device,
    output_root,
    tqdm_disable,
):
    results = {}
    for source in SOURCES:
        results[source] = evaluate_csv(
            model=model,
            tokenizer=tokenizer,
            csv_path=dev_csvs[source],
            audio_root="/",
            candidate_emotions=emotions_by_source[source],
            device=device,
            batch_size=BATCH_SIZE,
            output_csv=os.path.join(output_root, f"dev_latest_{source}.csv"),
            tqdm_disable=tqdm_disable,
        )
    return results


def main():
    args = parse_args()
    if args.factor_weight < 0:
        raise ValueError("--factor-weight must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
    transformers_logging.set_verbosity_error()
    setup_seed(SEED)
    with open(REPO_ROOT / "configs/config.yaml") as file:
        config = yaml.safe_load(file)

    factor_condition = args.condition in {"e2_factor", "e3_shuffled_factor"}
    pooled, sampler, emotions_by_source, train_csvs, dev_csvs = (
        load_pooled_training_data(
            args.split_root,
            args.feature_root,
            shuffle_factor_targets=args.condition == "e3_shuffled_factor",
            seed=SEED,
        )
    )
    loader = DataLoader(
        PooledTrainingDataset(pooled, include_factor_targets=factor_condition),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    device = torch.device(args.device)
    model = build_model(args.condition, config).to(device)
    load_shared_initial_state(
        model,
        args.condition,
        args.initial_state,
        device,
        config,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["text"])
    optimizer = build_optimizer(model, config)
    smooth_settings = dict(config["smoothclap"])
    smooth_settings["detach_targets"] = False
    smooth_criterion = SmoothCLAPLoss(**smooth_settings)
    contract = experiment_contract(args, config, train_csvs, dev_csvs)
    # Model construction consumes a condition-dependent number of random draws.
    # Reset before the first loader iteration so E0/E2/E3 share sampler, crop,
    # caption-template, and dropout streams. Resume restores the saved stream.
    setup_seed(SEED)

    output_root = os.path.join(args.results, "out")
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(args.results, "log"), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.results, "log"))

    start_epoch = 1
    best_uar = -1.0
    best_epoch = 0
    resume_path = os.path.join(args.results, "resume_latest.pth.tar")
    if args.resume:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        saved_contract = checkpoint.get("contract")
        if saved_contract != contract:
            keys = sorted(
                key
                for key in set(saved_contract or {}) | set(contract)
                if (saved_contract or {}).get(key) != contract.get(key)
            )
            raise ValueError(f"Resume contract mismatch: {keys}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng_state(checkpoint["rng_state"])
        # Resume state takes precedence over the fresh-run reset above.
        start_epoch = checkpoint["epoch"] + 1
        best_uar = checkpoint["best_uar"]
        best_epoch = checkpoint["best_epoch"]

    print(f"Condition: {args.condition}")
    print("Sources: MSP, IEMOCAP, CREMA-D")
    print("Sampling: corpus-uniform, emotion-uniform within corpus")
    print("Checkpoint: equal mean of three source Development native UARs")
    print("Main captions: emotion-only; audio and text encoders: trainable")
    print(f"Factor groups: {FACTOR_GROUPS if factor_condition else 'none'}")
    print(f"Factor weight: {args.factor_weight if factor_condition else 0.0}")
    for source in SOURCES:
        rows = int((pooled["_source_corpus"] == source).sum())
        print(f"{source}: Train rows={rows}; emotions={emotions_by_source[source]}")
    print(f"Rows/draws per epoch: {len(pooled)}; epochs: {start_epoch}-{EPOCHS}")

    metrics_path = os.path.join(args.results, "metrics.csv")
    for epoch in range(start_epoch, EPOCHS + 1):
        started = time.time()
        train_losses = train_epoch(
            model,
            loader,
            tokenizer,
            optimizer,
            args.condition,
            smooth_criterion,
            args.factor_weight,
            device,
            writer,
            epoch,
            args.tqdm_disable,
        )
        dev = evaluate_development(
            model,
            tokenizer,
            dev_csvs,
            emotions_by_source,
            device,
            output_root,
            args.tqdm_disable,
        )
        mean_uar = sum(dev[source]["UAR"] for source in SOURCES) / len(SOURCES)
        row = {
            "epoch": epoch,
            "train_loss": train_losses["loss"],
            "train_emotion_loss": train_losses["emotion"],
            "train_factor_loss": train_losses["factor"],
            "mean_dev_uar": mean_uar,
        }
        for source in SOURCES:
            row[f"{source}_dev_uar"] = dev[source]["UAR"]
            writer.add_scalar(f"eval/{source}_UAR", dev[source]["UAR"], epoch)
        save_metrics(metrics_path, row)
        for name, value in train_losses.items():
            writer.add_scalar(f"loss/epoch_{name}", value, epoch)
        writer.add_scalar("eval/mean_UAR", mean_uar, epoch)

        if mean_uar > best_uar:
            best_uar = mean_uar
            best_epoch = epoch
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "condition": args.condition,
                    "seed": SEED,
                    "factor_groups": FACTOR_GROUPS if factor_condition else None,
                    "contract": contract,
                },
                os.path.join(args.results, "best.pth.tar"),
            )
            for source in SOURCES:
                pd.read_csv(
                    os.path.join(output_root, f"dev_latest_{source}.csv")
                ).to_csv(
                    os.path.join(output_root, f"dev_best_{source}.csv"),
                    index=False,
                )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_uar": best_uar,
            "best_epoch": best_epoch,
            "contract": contract,
            "rng_state": rng_state(),
        }
        atomic_torch_save(checkpoint, resume_path)
        elapsed = (time.time() - started) / 3600
        print(
            f"Epoch {epoch}: total={train_losses['loss']:.6f}, "
            f"emotion={train_losses['emotion']:.6f}, "
            f"factor={train_losses['factor']:.6f}, "
            f"mean Dev UAR={mean_uar:.6f}, best={best_uar:.6f} "
            f"at epoch {best_epoch}, time={elapsed:.2f}h"
        )
    writer.close()


if __name__ == "__main__":
    main()
