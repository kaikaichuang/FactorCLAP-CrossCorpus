#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <e0_emotion|e1_smooth|e2_factor|e3_shuffled_factor>" >&2
    exit 2
fi
condition=$1
case "$condition" in
    e0_emotion|e1_smooth|e2_factor|e3_shuffled_factor) ;;
    *) echo "Invalid condition: $condition" >&2; exit 2 ;;
esac

repo_root=${REPO_ROOT:-/work/u1667110/clap_series/FactorCLAP-CrossCorpus}
split_root=${SPLIT_ROOT:-/work/u1667110/clap_series/dataset/CrossCorpus/splits}
feature_root=${FEATURE_ROOT:-$repo_root/runs/prepared_features/center5}
cameo_csv_root=${CAMEO_CSV_ROOT:-/work/u1667110/clap_series/dataset/CAMEO}
cameo_audio_root=${CAMEO_AUDIO_ROOT:-/work/u1667110/EMOTION_DATASETS/CAMEO}
initial_state=${INITIAL_STATE:-$repo_root/runs/_initial_states/smoothclapbase_seed3407.pth.tar}
python_bin=${PYTHON_BIN:-$(command -v python)}
run="$repo_root/runs/first_principles_pooled_seed3407/$condition"

required_files=("$initial_state" "$feature_root/READY")
for source in msp iemocap crema_d; do
    required_files+=(
        "$split_root/$source/train.csv"
        "$split_root/$source/development.csv"
        "$split_root/$source/test.csv"
        "$feature_root/${source}_train_eGeMAPSv02.csv"
    )
done
for required in "${required_files[@]}"; do
    [[ -f "$required" ]] || { echo "Missing: $required" >&2; exit 1; }
done

cameo_targets=(
    cafe emns emozionalmente enterface jl_corpus mesd
    nemo oreau pavoque ravdess resd subesco
)
for target in "${cameo_targets[@]}"; do
    [[ -f "$cameo_csv_root/$target.csv" ]] || {
        echo "Missing CAMEO CSV: $cameo_csv_root/$target.csv" >&2
        exit 1
    }
    [[ -d "$cameo_audio_root/$target/audio" ]] || {
        echo "Missing CAMEO audio: $cameo_audio_root/$target/audio" >&2
        exit 1
    }
done

mkdir -p "$run"
cd "$repo_root"
resume=()
if [[ -f "$run/resume_latest.pth.tar" ]]; then
    resume=(--resume)
else
    {
        echo "condition: $condition"
        echo "sources: [msp, iemocap, crema_d]"
        echo "seed: 3407"
        echo "epochs: 30"
        echo "selection: equal_source_dev_native_uar"
        echo "sampling: corpus_then_emotion_uniform"
        echo "audio_view: deterministic_center_5_seconds"
        echo "main_caption: emotion_only"
        echo "audio_encoder: trainable"
        echo "text_encoder: trainable"
        echo "git_commit: $(git rev-parse HEAD)"
        echo "factor_weight: 64.0"
    } > "$run/run_config.yaml"
    : > "$run/train.log"
fi
train_command=(
    "$python_bin" -u train_pooled.py
    --condition "$condition"
    --split-root "$split_root"
    --feature-root "$feature_root"
    --initial-state "$initial_state"
    --results "$run"
    --device cuda:0
    --tqdm-disable
    "${resume[@]}"
)
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${train_command[@]}" >> "$run/train.log" 2>&1

checkpoint="$run/best.pth.tar"
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }

evaluate_one() {
    local csv=$1
    local output=$2
    local audio_root=$3
    shift 3
    mkdir -p "$output"
    command=(
        "$python_bin" -u eval_csv.py
        --csv "$csv"
        --checkpoint "$checkpoint"
        --audio-root "$audio_root"
        --device cuda:0
        --batch-size 32
        --results "$output"
        "$@"
    )
    env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${command[@]}" > "$output/eval.log" 2>&1
}

for target in msp iemocap crema_d ravdess tess; do
    split=test
    [[ "$target" == tess ]] && split=full
    csv="$split_root/$target/$split.csv"
    evaluate_one "$csv" "$run/test/$target/native" /
    shared_args=(--emotions angry happy neutral sad)
    [[ "$target" == iemocap ]] && shared_args+=(--merge-excited-to-happy)
    evaluate_one "$csv" "$run/test/$target/shared4" / "${shared_args[@]}"
done

for target in "${cameo_targets[@]}"; do
    csv="$cameo_csv_root/$target.csv"
    audio_root="$cameo_audio_root/$target/audio"
    evaluate_one "$csv" "$run/test/cameo/$target/native" "$audio_root"
    shared_args=(--map-cameo-labels --emotions angry happy neutral sad)
    if [[ "$target" == enterface ]]; then
        shared_args=(--map-cameo-labels --emotions angry happy sad)
        shared_name=shared3
    else
        shared_name=shared4
    fi
    evaluate_one "$csv" "$run/test/cameo/$target/$shared_name" "$audio_root" "${shared_args[@]}"
done

touch "$run/COMPLETED"
echo "Completed training and inference: $run"
