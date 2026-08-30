#!/usr/bin/env bash
set -euo pipefail

ml purge
ml load miniconda3/24.11.1
conda activate clap

repo_root=/work/u1667110/clap_series/FactorCLAP-CrossCorpus
old_initial=/work/u1667110/clap_series/SmoothCLAP-CrossCorpus/runs/_initial_states/smoothclapbase_seed3407.pth.tar
initial="$repo_root/runs/_initial_states/smoothclapbase_seed3407.pth.tar"
split_root=/work/u1667110/clap_series/dataset/CrossCorpus/splits
feature_root=/work/u1667110/clap_series/dataset/CrossCorpus/features

cd "$repo_root"
if [[ -n $(git status --porcelain) ]]; then
    echo "Repository has uncommitted files; refuse formal experiment submission." >&2
    git status --short >&2
    exit 1
fi
mkdir -p "$(dirname "$initial")"
if [[ ! -f "$initial" ]]; then
    if [[ -f "$old_initial" ]]; then
        cp "$old_initial" "$initial"
    else
        env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            python create_initial_state.py --seed 3407 --output "$initial"
    fi
fi

for source in msp iemocap crema_d; do
    test -f "$split_root/$source/train.csv"
    test -f "$split_root/$source/development.csv"
    test -f "$split_root/$source/test.csv"
    test -f "$feature_root/${source}_train_eGeMAPSv02.csv"
done
test -f "$split_root/ravdess/test.csv"
test -f "$split_root/tess/full.csv"
test -f "$initial"
python -m unittest discover -s tests -v
echo "Preflight passed. Run scripts/nchc/submit_all.sh to prepare matched features and train."
