import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from models_xin import SmoothCLAP


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser("Create one complete shared SmoothCLAP initial state")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open("configs/config.yaml") as file:
        config = yaml.safe_load(file)
    setup_seed(args.seed)
    model = SmoothCLAP(
        speech_name=config["models"]["speech"],
        text_name=config["models"]["text"],
        local_speech_name=config["models"]["local_speech"],
        embedding_dim=768,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{output}.tmp")
    torch.save(
        {
            "seed": args.seed,
            "model": model.state_dict(),
            "models": config["models"],
            "embedding_dim": 768,
        },
        temporary,
    )
    os.replace(temporary, output)
    print(f"Saved complete seed-{args.seed} initial state to {output}")


if __name__ == "__main__":
    main()
