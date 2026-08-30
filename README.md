# FactorCLAP-CrossCorpus

First-principles experiments for cross-corpus zero-shot speech emotion
recognition (SER). The repository tests whether correctly paired,
utterance-level paralinguistic dynamics improve an emotion-aligned CLAP model,
rather than treating corpus identity or discrete caption tags as the mechanism.

The first experiment is intentionally small: four pooled models, one seed, and
one controlled causal null. It does not include routing.

## Experiment in plain language

All models train on MSP-Podcast, IEMOCAP, and CREMA-D with balanced sampling.
At inference, every model still compares an audio embedding with text emotion
labels such as `angry` or `sad`.

IEMOCAP is defined as six classes: angry, happy, neutral, sad, excited, and
frustrated. Its disgust, fear, other, and surprise rows are excluded from
Train, Development checkpoint selection, and Native Test evaluation.

- **E0 emotion:** emotion-only, class-aware contrastive baseline.
- **E1 smooth:** original SmoothCLAP soft alignment with full canonical
  paralinguistic tags, but emotion-only main captions.
- **E2 factor:** E0 plus separate linear heads that predict continuous,
  source-normalized pitch, energy, and rhythm dynamics during training.
- **E3 deranged factor:** exactly E2, except the complete factor vector is
  reassigned within each source and true-emotion stratum with zero fixed points.

The factor heads are training-only. E2 versus E3 asks whether the correct
utterance-to-prosody correspondence matters; E2 versus E0 asks whether the
auxiliary task helps at all.

## Files

- [RESEARCH.md](RESEARCH.md): hypotheses, design rationale, literature, and
  interpretation boundaries.
- [EXPERIMENT.md](EXPERIMENT.md): immutable contract, run registry, commands,
  status, and eventually results.
- `train_pooled.py`: pooled E0-E3 trainer.
- `scripts/run_case.sh`: one condition, followed by Native and Shared-4
  inference on all configured targets.
- `prepare_crop_features.py`: exact center-five-second eGeMAPS preparation.
- `scripts/nchc/`: dependency-safe NCHC preparation and training jobs.

These three Markdown files are deliberate: only `EXPERIMENT.md` is a live
experiment log, so status is not duplicated across documents.

## NCHC execution

Expected checkout:

```text
/work/u1667110/clap_series/FactorCLAP-CrossCorpus
```

Expected datasets are the existing CrossCorpus and CAMEO folders under
`/work/u1667110/clap_series/dataset`. The preparation script reuses the
seed-3407 initial state from SmoothCLAP-CrossCorpus when present; otherwise it
creates the same architecture from the cached pretrained models.

All four conditions use the same deterministic center five seconds. For Train
utterances longer than five seconds, eGeMAPS is recomputed on precisely that
interval before either SmoothCLAP tags or factor targets are built.

Submit preparation first. All initialization, tests, crop-feature extraction,
and input preflight execute inside this Slurm job, not in the login terminal:

```bash
cd /work/u1667110/clap_series/FactorCLAP-CrossCorpus
sbatch scripts/nchc/prepare_features.sbatch
```

After that job exits successfully and
`runs/prepared_features/center5/READY` exists, submit the four models:

```bash
sbatch scripts/nchc/train_case.sbatch e0_emotion
sbatch scripts/nchc/train_case.sbatch e1_smooth
sbatch scripts/nchc/train_case.sbatch e2_factor
sbatch scripts/nchc/train_case.sbatch e3_shuffled_factor
```

Nano5 permits two concurrent jobs per user, so these run in two waves. The
preprocessing job requests one H100 because Nano5 partitions are GPU partitions,
although extraction itself is CPU-bound. Each model resumes independently from
`resume_latest.pth.tar` and performs all inference after training.

Alternatively, `bash scripts/nchc/submit_all.sh` only issues `sbatch` calls and
returns immediately; it no longer runs preparation in the login terminal.

Results are written to:

```text
runs/first_principles_pooled_seed3407/<condition>/
```

Copy that complete directory back to the same repository path on AMD for
analysis. Training and checkpoint selection never read Test labels.

## Local checks

Use the existing `clap` environment:

```bash
cd /homes/kevin/clap_series/FactorCLAP-CrossCorpus
/homes/kevin/.conda/envs/clap/bin/python -m unittest discover -s tests -v
bash -n scripts/run_case.sh scripts/nchc/prepare_once.sh \
    scripts/nchc/prepare_features.sbatch scripts/nchc/train_case.sbatch \
    scripts/nchc/submit_all.sh
```
