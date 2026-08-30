#!/bin/bash
#SBATCH --job-name=fclap-train
#SBATCH --partition=normal
#SBATCH --account=MST115095
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --chdir=/work/u1667110/clap_series/FactorCLAP-CrossCorpus
#SBATCH --output=/work/u1667110/clap_series/FactorCLAP-CrossCorpus/slurm-%x-%j.out
#SBATCH --error=/work/u1667110/clap_series/FactorCLAP-CrossCorpus/slurm-%x-%j.err

ml purge
ml load miniconda3/24.11.1
conda activate clap
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 <condition>" >&2
    exit 2
fi
cd /work/u1667110/clap_series/FactorCLAP-CrossCorpus
bash scripts/run_case.sh "$1"
