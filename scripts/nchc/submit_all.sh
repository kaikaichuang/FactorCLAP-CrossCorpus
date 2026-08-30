#!/usr/bin/env bash
set -euo pipefail

repo_root=/work/u1667110/clap_series/FactorCLAP-CrossCorpus
cd "$repo_root"
bash scripts/nchc/prepare_once.sh

feature_job=$(sbatch --parsable scripts/nchc/prepare_features.sbatch)
echo "Feature preparation and preflight job: $feature_job"
conditions=(
    e0_emotion
    e1_smooth
    e2_factor
    e3_shuffled_factor
)
for condition in "${conditions[@]}"; do
    job=$(sbatch \
        --parsable \
        --dependency="afterok:$feature_job" \
        scripts/nchc/train_case.sbatch \
        "$condition")
    echo "$condition job (after feature job): $job"
done
