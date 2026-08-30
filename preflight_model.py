import argparse
import gc

import torch
import yaml
from transformers import AutoTokenizer

from train_pooled import (
    REPO_ROOT,
    build_model,
    load_shared_initial_state,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Offline pretrained-cache and shared-initial-state preflight"
    )
    parser.add_argument("--initial-state", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    with open(REPO_ROOT / "configs/config.yaml") as file:
        config = yaml.safe_load(file)
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["text"])
    tokens = tokenizer(
        ["angry", "happy", "neutral", "sad"],
        padding=True,
        return_tensors="pt",
    )
    if tokens["input_ids"].shape[0] != 4:
        raise ValueError("Tokenizer preflight failed")

    device = torch.device(args.device)
    for condition in ("e0_emotion", "e1_smooth", "e2_factor"):
        model = build_model(condition, config).to(device)
        load_shared_initial_state(
            model,
            condition,
            args.initial_state,
            device,
            config,
        )
        print(f"Validated offline model and initial state: {condition}")
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
