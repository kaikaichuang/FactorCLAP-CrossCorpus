#!/bin/bash
#SBATCH --job-name=fclap-feat
#SBATCH --partition=normal
#SBATCH --account=MST115095
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/u1667110/clap_series/FactorCLAP-CrossCorpus
#SBATCH --output=/work/u1667110/clap_series/FactorCLAP-CrossCorpus/slurm-%x-%j.out
#SBATCH --error=/work/u1667110/clap_series/FactorCLAP-CrossCorpus/slurm-%x-%j.err

ml purge
ml load miniconda3/24.11.1
conda activate clap
set -euo pipefail

repo_root=/work/u1667110/clap_series/FactorCLAP-CrossCorpus
cd "$repo_root"
bash scripts/nchc/prepare_once.sh
python -u prepare_crop_features.py \
    --split-root /work/u1667110/clap_series/dataset/CrossCorpus/splits \
    --feature-root /work/u1667110/clap_series/dataset/CrossCorpus/features \
    --output-root "$repo_root/runs/prepared_features/center5" \
    --num-workers 32
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python -u preflight_model.py \
    --initial-state "$repo_root/runs/_initial_states/smoothclapbase_seed3407.pth.tar" \
    --device cuda:0

python -u preflight_inputs.py \
    --split-root /work/u1667110/clap_series/dataset/CrossCorpus/splits \
    --prepared-feature-root "$repo_root/runs/prepared_features/center5" \
    --source-feature-root /work/u1667110/clap_series/dataset/CrossCorpus/features \
    --cameo-csv-root /work/u1667110/clap_series/dataset/CAMEO \
    --cameo-audio-root /work/u1667110/EMOTION_DATASETS/CAMEO
