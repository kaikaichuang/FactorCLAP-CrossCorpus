#!/usr/bin/env bash
set -euo pipefail

repo_root=/work/u1667110/clap_series/FactorCLAP-CrossCorpus
cd "$repo_root"

feature_job=$(sbatch --parsable scripts/nchc/prepare_features.sbatch)
echo "Feature preparation and preflight job: $feature_job"
sources=(msp iemocap crema_d)
conditions=(e0_emotion e2_factor e3_shuffled_factor)
for source in "${sources[@]}"; do
    for condition in "${conditions[@]}"; do
        job=$(sbatch \
            --parsable \
            --dependency="afterok:$feature_job" \
            scripts/nchc/train_case.sbatch \
            "$source" "$condition")
        echo "$source/$condition job (after feature job): $job"
    done
done
