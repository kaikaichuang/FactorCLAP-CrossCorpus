# Experiment ledger

Last updated: 2026-08-30

## Current status

- [x] Research question and leakage boundary frozen.
- [x] Pooled design rejected because source-relative factor coordinates conflict
  under one shared head and corpus-balanced sampling duplicates small corpora.
- [x] IEMOCAP fixed to six classes; every selected source Train class has at
  least 300 examples.
- [x] E0/E2/E3 source-expert architecture and matched causal null defined.
- [x] Deterministic center crop and crop-matched eGeMAPS pipeline implemented.
- [x] Per-source sampling, own-Development checkpoint selection, evaluation,
  resume, and nine NCHC jobs implemented.
- [x] Local static/unit tests and real-data loader/sampler checks passed.
- [x] E0, E2, and E3 real-IEMOCAP one-batch AMD forward/backward checks passed.
- [x] Seed-3407 decision rule, size-matched contingency, and E1 contingency
  frozen before training.
- [ ] NCHC feature/initial-state/data preflight passed for this commit.
- [ ] Nine seed-3407 experts trained.
- [ ] Native and Shared-4/Shared-3 inference completed.
- [ ] E2/E3 mechanism gate evaluated per target.
- [ ] Complementarity analysis authorized. Only proceed if the mechanism gate
  passes.
- [ ] Additional seeds authorized. Only proceed if the mechanism gate passes.

Previous SmoothCLAP expert, routing, fusion, and factorial-tag experiments are
background evidence only. They are not comparable runs of this pipeline.

## Immutable first-wave contract

- Sources/experts: MSP-Podcast, IEMOCAP, CREMA-D trained separately.
- Conditions per source: E0 emotion, E2 paired factor, E3 deranged factor.
- IEMOCAP classes: angry, happy, neutral, sad, excited, frustrated for Train,
  Development Native selection, and Native Test.
- Minimum selected source Train class size: 300.
- Seed: 3407.
- Epochs: 30.
- Audio and text encoders: trainable.
- Main captions: emotion-only.
- Main loss: class-aware multi-positive contrastive loss.
- Sampling: emotion-uniform within one source; draws per epoch equal that
  source's selected Train rows.
- Checkpoint: own source Development Native UAR only.
- Audio view: deterministic center five seconds; shorter clips use their full
  audio plus attention-mask padding.
- eGeMAPS for clips over five seconds: recomputed on the exact center crop.
- Factor normalization: expert source Train quantiles only.
- Factor loss weight: 64, fixed before formal training from five pooled
  source-Train batches at initialization and not retuned per expert; no
  Development/Test tuning.
- E3: complete factor vectors deranged within the expert's true-emotion strata,
  with zero fixed points.
- Test: prohibited from training, normalization, tuning, and checkpoint
  selection.
- BIIC: excluded.
- Primary analysis: each external target's Shared-4 UAR separately;
  eNTERFACE is Shared-3.
- Secondary analysis: Native UAR, class recall, maximum predicted-class share,
  missing predicted classes, and worst-class recall.

Emotion-uniform sampling removes within-expert label-frequency imbalance. It
does not equalize the three experts' total updates, class sets, speakers,
recordings, or acting styles. No result may be described as purely
paralinguistic without the matched E2/E3 evidence.

## Predeclared seed-3407 mechanism gate

The 13 independent external target families are TESS, RAVDESS, and the 11
non-RAVDESS CAMEO corpora. RAVDESS official Test supplies the counted value;
RAVDESS-full is corroboration and is not double counted.

For each source expert:

1. Compute E2−E3 Shared-4 UAR in percentage points for every target family
   (Shared-3 for eNTERFACE).
2. Count a target only if the delta is at least +2.0 pp, E2 does not newly lose
   a predicted class relative to E3, and E2 maximum predicted-class share is
   below 0.80.
3. The source expert passes at 7 of 13 counted target families.

The mechanism advances only if at least two of three source experts pass. For
each passing expert, at least five counted families must be neither TESS nor
RAVDESS. Seed 3407 is exploratory; a passing result requires additional seeds
before a paper claim. If it fails, stop this factor/complementarity route
rather than adding routing experiments.

E2 versus E0 remains a secondary auxiliary-task comparison. It cannot replace
E2 versus E3 because E0 differs in architecture and gradient budget.

## Run registry

All models load:

```text
runs/_initial_states/smoothclapbase_seed3407.pth.tar
```

| Source | Condition | Output path | Status |
|---|---|---|---|
| MSP | E0 | `runs/first_principles_experts_seed3407/msp/e0_emotion` | pending |
| MSP | E2 | `runs/first_principles_experts_seed3407/msp/e2_factor` | pending |
| MSP | E3 | `runs/first_principles_experts_seed3407/msp/e3_shuffled_factor` | pending |
| IEMOCAP | E0 | `runs/first_principles_experts_seed3407/iemocap/e0_emotion` | pending |
| IEMOCAP | E2 | `runs/first_principles_experts_seed3407/iemocap/e2_factor` | pending |
| IEMOCAP | E3 | `runs/first_principles_experts_seed3407/iemocap/e3_shuffled_factor` | pending |
| CREMA-D | E0 | `runs/first_principles_experts_seed3407/crema_d/e0_emotion` | pending |
| CREMA-D | E2 | `runs/first_principles_experts_seed3407/crema_d/e2_factor` | pending |
| CREMA-D | E3 | `runs/first_principles_experts_seed3407/crema_d/e3_shuffled_factor` | pending |

## NCHC execution record

Checkout:

```text
/work/u1667110/clap_series/FactorCLAP-CrossCorpus
```

First submit and wait for preparation:

```bash
cd /work/u1667110/clap_series/FactorCLAP-CrossCorpus
sbatch scripts/nchc/prepare_features.sbatch
```

After `runs/prepared_features/center5/READY` exists:

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

Nano5 queues jobs beyond the two concurrent-job limit. Each job trains one
expert and then evaluates MSP, IEMOCAP, CREMA-D, RAVDESS Test, TESS, and all
configured CAMEO targets in Native and Shared-4/Shared-3 form.

## Deferred work, not authorized now

- If the mechanism gate passes: additional E2/E3 seeds, then exploratory
  cross-expert wrong→correct/correct→wrong and oracle-headroom comparison
  against E3.
- If a paralinguistic result remains: size-matched E0/E2/E3 control, with MSP
  and CREMA-D fixed to the 4,246-example IEMOCAP scale and equal update counts.
- If comparison with discrete SmoothCLAP tags is needed: train E1 using this
  same crop, class-aware loss, source-specific sampling, and checkpoint
  pipeline. Old E1 results cannot substitute.

## Result checklist

Before analysis, require all nine runs to contain `best.pth.tar`, `metrics.csv`,
`train.log`, `COMPLETED`, and every configured Native/Shared prediction and
metrics file. Report each target separately. Also report how many targets
improve, remain within ±2.0 pp, or decline; do not replace the target table with
an average.
