import argparse
import os

import torch
import yaml
from transformers import AutoTokenizer

from evaluate import evaluate_csv
from factor_model import FactorCLAP
from models_xin import ParaCLAP, SmoothCLAP


def parse_args():
    parser = argparse.ArgumentParser("Evaluate a CLAP checkpoint on an emotion CSV")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--audio-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--emotions", nargs="+")
    parser.add_argument("--candidate-emotions", nargs="+")
    parser.add_argument("--known-speakers-only", action="store_true")
    parser.add_argument("--merge-excited-to-happy", action="store_true")
    parser.add_argument("--map-cameo-labels", action="store_true")
    parser.add_argument(
        "--query-style",
        choices=("label", "sentence", "iemocap"),
        default="label",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.emotions:
        args.emotions = [emotion.lower() for emotion in args.emotions]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested, but torch.cuda.is_available() is false")

    with open("configs/config.yaml") as file:
        config = yaml.safe_load(file)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    if any(name.startswith("factor_heads.") for name in state_dict):
        model = FactorCLAP(
            speech_name=config["models"]["speech"],
            text_name=config["models"]["text"],
            embedding_dim=768,
            train_audio_encoder=True,
        )
    elif any(name.startswith("local_audio_branch.") for name in state_dict):
        model = SmoothCLAP(
            speech_name=config["models"]["speech"],
            text_name=config["models"]["text"],
            local_speech_name=config["models"]["local_speech"],
            embedding_dim=768,
        )
    else:
        model = ParaCLAP(
            speech_name=config["models"]["speech"],
            text_name=config["models"]["text"],
            embedding_dim=768,
        )
    model.load_state_dict(state_dict)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["text"])

    os.makedirs(args.results, exist_ok=True)
    query_style = "sentence" if args.query_style == "iemocap" else args.query_style
    query_aliases = {"angry": "anger", "sad": "sadness"} if args.query_style == "iemocap" else None
    label_mapping = {}
    if args.merge_excited_to_happy:
        label_mapping["excited"] = "happy"
    if args.map_cameo_labels:
        label_mapping.update(
            {"anger": "angry", "happiness": "happy", "sadness": "sad"}
        )

    metrics = evaluate_csv(
        model=model,
        tokenizer=tokenizer,
        csv_path=args.csv,
        audio_root=args.audio_root,
        emotions=args.emotions,
        candidate_emotions=args.candidate_emotions,
        label_mapping=label_mapping or None,
        known_speakers_only=args.known_speakers_only,
        device=args.device,
        batch_size=args.batch_size,
        query_style=query_style,
        query_aliases=query_aliases,
        output_csv=os.path.join(args.results, "predictions.csv"),
    )
    with open(os.path.join(args.results, "metrics.yaml"), "w") as file:
        yaml.safe_dump(metrics, file, sort_keys=False)


if __name__ == "__main__":
    main()
