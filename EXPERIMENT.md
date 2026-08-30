# Experiment ledger

Last updated: 2026-08-30

## Current status

- [x] Research question and leakage boundary frozen.
- [x] E0-E3 architecture and causal null defined.
- [x] Dynamic eGeMAPS columns checked on all three source Train sets: no
  missing or infinite values.
- [x] Training, evaluation, resume, and NCHC scripts implemented.
- [x] Full-utterance/random-crop mismatch identified before training.
- [x] Deterministic center-crop and crop-matched eGeMAPS pipeline implemented.
- [x] Local unit/static tests and real-data loading dry run passed.
- [x] E1 and E2 real-batch AMD forward/backward smoke tests passed.
- [x] Factor-loss scale measured on five source-Train batches; weight frozen.
- [x] Final code review passed with no blocking logic findings.
- [x] Initial Git commit created and GitHub remote configured.
- [ ] NCHC initial-state/data preflight passed.
- [ ] E0-E3 training completed.
- [ ] Native and Shared-4/Shared-3 inference completed.
- [ ] Per-target analysis completed.
- [ ] Additional seeds authorized. Do not start before interpreting seed 3407.

Previous SmoothCLAP expert, routing, fusion, and factorial-tag experiments remain
background evidence in the old repository. They are not silently counted as
runs of this new design.

## Immutable first-wave contract

- Sources: MSP-Podcast, IEMOCAP, CREMA-D.
- IEMOCAP classes: angry, happy, neutral, sad, excited, frustrated. The same
  subset is used for Train, Development selection, and Native Test.
- Seed: 3407.
- Epochs: 30.
- Audio encoder: trainable.
- Text encoder: trainable.
- Main captions: emotion-only.
- Sampling: equal corpus probability, then equal emotion probability within
  each corpus.
- Audio view: deterministic center five seconds; shorter clips are used in full
  and padded with an attention mask. Identical for E0-E3.
- Checkpoint selection: equal mean of the three source Development Native UARs.
- eGeMAPS for clips over five seconds: recomputed on the exact center crop.
- Factor normalization: crop-matched source Train quantiles only.
- Factor loss weight: 64, fixed before formal training from source-Train-only
  initialization gradients (five batches; median equal-gradient ratio 248,
  range 222–371). No Development/Test tuning.
- Test: prohibited from training, normalization, tuning, and checkpoint
  selection.
- BIIC: excluded.
- Primary analysis: each unseen target's Shared-4 UAR separately; eNTERFACE is
  Shared-3 and stays separate.
- Secondary analysis: Native UAR, class recall, and collapse diagnostics.

## Run registry

| ID | Condition | NCHC batch | Output path | Status |
|---|---|---|---|---|
| E0 | class-aware emotion-only | independent job | `runs/first_principles_pooled_seed3407/e0_emotion` | pending |
| E1 | original SmoothCLAP, emotion-only main caption/full tags | independent job | `runs/first_principles_pooled_seed3407/e1_smooth` | pending |
| E2 | E0 + paired continuous factor heads | independent job | `runs/first_principles_pooled_seed3407/e2_factor` | pending |
| E3 | E0 + within-source/emotion zero-fixed-point deranged factor heads | independent job | `runs/first_principles_pooled_seed3407/e3_shuffled_factor` | pending |

All four models must load the same file:

```text
runs/_initial_states/smoothclapbase_seed3407.pth.tar
```

## Execution record

NCHC checkout:

```text
/work/u1667110/clap_series/FactorCLAP-CrossCorpus
```

Commands, to be run after the Git remote is created and the repo is cloned or
updated on NCHC:

```bash
cd /work/u1667110/clap_series/FactorCLAP-CrossCorpus
bash scripts/nchc/submit_all.sh
```

All four one-model jobs have an `afterok` dependency on crop-feature preparation
and full input preflight. Nano5 runs at most two concurrently, so the four jobs
naturally execute in two waves without risking two models inside one 48-hour
allocation.

Each case trains first, then evaluates:

- MSP and CREMA-D source Test: Native and Shared-4.
- IEMOCAP source Test: six-class Native and Shared-4 (excited merged into
  happy; frustrated excluded from Shared-4).
- RAVDESS official split Test: Native and Shared-4.
- TESS full: Native and Shared-4.
- CAMEO cafe, emns, emozionalmente, eNTERFACE, jl_corpus, mesd, nemo, oreau,
  pavoque, RAVDESS-full, resd, and subesco: Native plus Shared-4, except
  eNTERFACE Shared-3.

## Result tables

Fill only after all four cases pass completeness checks. Never average away the
target rows.

### Shared-4 UAR (eNTERFACE Shared-3)

| Target | E0 | E1 | E2 | E3 | E2-E0 | E2-E3 |
|---|---:|---:|---:|---:|---:|---:|
| RAVDESS Test | | | | | | |
| TESS | | | | | | |
| CAMEO cafe | | | | | | |
| CAMEO emns | | | | | | |
| CAMEO emozionalmente | | | | | | |
| CAMEO eNTERFACE (Shared-3) | | | | | | |
| CAMEO jl_corpus | | | | | | |
| CAMEO mesd | | | | | | |
| CAMEO nemo | | | | | | |
| CAMEO oreau | | | | | | |
| CAMEO pavoque | | | | | | |
| CAMEO RAVDESS-full | | | | | | |
| CAMEO resd | | | | | | |
| CAMEO subesco | | | | | | |

### Interpretation checklist

- Number of external targets with E2 > E0; equal; lower.
- Number with E2 > E3; equal; lower.
- Median deltas, shown only beside the complete per-target table.
- Whether any positive count disappears when TESS is removed.
- Whether RAVDESS Test and RAVDESS-full tell the same story.
- Per-class source of each material delta.
- Prediction collapse or missing-class changes.
- Only if E2 consistently beats both controls: add seeds and then reconsider
  factor-aware routing or stronger factor subspaces.
