# FactorCLAP-CrossCorpus

First-principles experiments for cross-corpus zero-shot speech emotion
recognition. The first wave tests whether correctly paired continuous
paralinguistic supervision improves source-specific emotion experts.

## First wave

Train nine models: three source corpora × three conditions. Every model starts
from the same seed-3407 initialization, trains its audio and text encoders for
30 epochs, and selects its checkpoint using only its own source Development
Native UAR.

- **E0 emotion:** class-aware emotion-only CLAP.
- **E2 factor:** E0 plus training-only heads for correctly paired continuous
  pitch, energy, and rhythm dynamics.
- **E3 deranged factor:** the same model, loss, and factor distributions as E2,
  but complete factor vectors are reassigned within source × true emotion with
  zero fixed points.

E2 versus E3 is the causal comparison. E2 versus E0 only tests whether adding
the auxiliary task is useful. IEMOCAP is fixed to angry, happy, neutral, sad,
excited, and frustrated; every selected Train class has at least 300 examples.

Each expert samples emotions uniformly, with draws per epoch equal to that
source's Train size. This removes label-frequency imbalance and the old pooled
corpus oversampling, but does not remove corpus-size, class-inventory, speaker,
or recording differences.

## Repository files

- [RESEARCH.md](RESEARCH.md): hypotheses, rationale, decision rules, and
  interpretation boundaries.
- [EXPERIMENT.md](EXPERIMENT.md): immutable contract, run registry, commands,
  status, and results.
- `train_expert.py`: one source/condition training job.
- `prepare_crop_features.py`: center-five-second eGeMAPS preparation.
- `scripts/run_case.sh`: training followed by all Native and Shared-4/Shared-3
  evaluations, including CAMEO.
- `scripts/nchc/`: NCHC Slurm preparation and training scripts.

The three Markdown files have separate roles; status is maintained only in
`EXPERIMENT.md`.

## NCHC execution

Expected checkout:

```text
/work/u1667110/clap_series/FactorCLAP-CrossCorpus
```

Preparation, tests, initial-state validation, and center-crop feature creation
must run inside Slurm:

```bash
cd /work/u1667110/clap_series/FactorCLAP-CrossCorpus
sbatch scripts/nchc/prepare_features.sbatch
```

After that job succeeds and `runs/prepared_features/center5/READY` exists,
submit the nine independent jobs. Submitting them is quick; Slurm queues them
when only two H100 jobs may run concurrently.

```bash
sbatch scripts/nchc/train_case.sbatch msp e0_emotion
sbatch scripts/nchc/train_case.sbatch msp e2_factor
sbatch scripts/nchc/train_case.sbatch msp e3_shuffled_factor
sbatch scripts/nchc/train_case.sbatch iemocap e0_emotion
sbatch scripts/nchc/train_case.sbatch iemocap e2_factor
sbatch scripts/nchc/train_case.sbatch iemocap e3_shuffled_factor
sbatch scripts/nchc/train_case.sbatch crema_d e0_emotion
sbatch scripts/nchc/train_case.sbatch crema_d e2_factor
sbatch scripts/nchc/train_case.sbatch crema_d e3_shuffled_factor
```

`bash scripts/nchc/submit_all.sh` is an optional equivalent that only issues
`sbatch` commands and returns; it does not run training in the login terminal.

Each job resumes from `resume_latest.pth.tar`, then runs MSP, IEMOCAP, CREMA-D,
RAVDESS, TESS, and all configured CAMEO inference. Results are written to:

```text
runs/first_principles_experts_seed3407/<source>/<condition>/
```

Copy that complete directory back to the same path on AMD for analysis. Test
data never participates in training or checkpoint selection.

## Local checks

```bash
cd /homes/kevin/clap_series/FactorCLAP-CrossCorpus
/homes/kevin/.conda/envs/clap/bin/python -m unittest discover -s tests -v
bash -n scripts/run_case.sh scripts/nchc/prepare_once.sh \
    scripts/nchc/prepare_features.sbatch scripts/nchc/train_case.sbatch \
    scripts/nchc/submit_all.sh
```
