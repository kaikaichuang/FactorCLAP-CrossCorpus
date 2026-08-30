# Research design

## Goal

The main task remains cross-corpus zero-shot SER. The narrower question is:

> Can correctly paired, continuous paralinguistic dynamics shape transferable
> source experts and later support useful expert complementarity?

The previous SmoothCLAP experiments showed expert complementarity and ensemble
headroom, but did not show that paralinguistic alignment caused either. This
repository therefore tests the representation mechanism before returning to
routing.

## Why use source-specific experts

The abandoned pooled design had two problems.

First, factor quantiles are necessarily fitted within each source Train split.
A pitch-variation rank of 0.8 in MSP and 0.8 in CREMA-D therefore has different
absolute acoustics. One pooled factor head would have to infer corpus identity
before reproducing the appropriate within-corpus rank. That can encourage the
exact corpus information the experiment is trying not to mistake for a
transferable paralinguistic mechanism.

Second, equal-corpus pooled sampling repeated IEMOCAP and CREMA-D roughly in
proportion to MSP's size. Source-specific epochs instead draw exactly the size
of that source Train split:

| Source | Classes | Train rows/draws per epoch | Smallest class |
|---|---:|---:|---:|
| MSP-Podcast | 9 | 68,328 | 1,120 |
| IEMOCAP | 6 | 4,246 | 387 |
| CREMA-D | 6 | 5,316 | 776 |

Within each expert, emotion-uniform sampling removes the training-frequency
prior that dominated the old analysis. It does **not** equalize total updates,
class inventories, speakers, acting style, or recording conditions. Those
remaining differences are limitations and possible sources of expert
specialization, not evidence of a paralinguistic cause.

## Common formulation

All models use a class-aware multi-positive contrastive target: every item with
the same emotion is a positive. This avoids treating duplicate emotion-only
captions as negatives.

All models use the same deterministic center-five-second audio view. Shorter
utterances are used in full and padded with an attention mask. For longer
utterances, eGeMAPS is recomputed on exactly that interval. This removes the
old mismatch between random audio crops and full-utterance acoustic targets.

Only continuous within-utterance dynamics are included initially:

- Pitch: normalized F0 variation, percentile range, rising slope, falling
  slope.
- Energy: normalized loudness variation, percentile range, rising slope,
  falling slope.
- Rhythm: voiced segments per second, mean voiced length, mean unvoiced
  length.

Gender, VAD labels, absolute mean F0, absolute sound level, and raw duration
are excluded because they are especially likely to encode speaker, recording,
or corpus identity. Every factor is converted to a quantile fitted only on the
expert's own source Train split.

## Three conditions

All conditions start from the same seed-3407 initialization and train both
main encoders for 30 epochs.

| ID | Main alignment | Paralinguistic supervision |
|---|---|---|
| E0 | Class-aware emotion-only CLAP | None |
| E2 | Same as E0 | Three linear heads regress correctly paired continuous factor ranks |
| E3 | Same as E0 | Same heads and targets, but complete target vectors are deranged within true emotion with zero fixed points |

The factor heads are training-only. Their three Smooth-L1 losses are averaged
and added with fixed weight 64. This common weight was selected before formal
training from five pooled source-Train batches at initialization and is not
retuned per expert. E2 and E3 have identical architecture, supervision
marginals, and gradient budget; only correct utterance-to-factor pairing
differs.

Therefore:

- E2 > E3 supports a contribution from correct utterance-level factor pairing.
- E2 > E0 but E2 ≈ E3 supports only generic auxiliary-task regularization or
  emotion-conditional factor distributions.
- E2 ≤ E3 is a stop signal for this factor mechanism.

E2 versus E3 is the primary causal comparison. E0 is secondary and must not
replace the matched null.

## Predeclared first-seed decision

Use 13 external target families: TESS, RAVDESS, and the 11 non-RAVDESS CAMEO
corpora. RAVDESS official Test is primary; RAVDESS-full is corroboration and is
not counted as a second independent family. Shared-4 is primary, except
eNTERFACE Shared-3.

A source expert passes only if E2−E3 is at least +2.0 UAR percentage points on
at least 7 of 13 target families. A target does not count if E2 newly loses a
predicted class relative to E3 or its maximum predicted-class share is at least
0.80. The mechanism advances only if at least two of the three source experts
pass. For each passing expert, at least five counted families must be neither
TESS nor RAVDESS.

This is a direction screen, not publication evidence. Any passing result must
be repeated with additional seeds before a causal claim. If E2 ≈ E3, stop the
factor/complementarity route instead of adding routers or small ablations.

Expert complementarity is exploratory at seed 3407. Only after the E2/E3 gate
passes should we compare E2 experts' per-target wrong→correct and correct→wrong
transitions or oracle headroom against E3. A few points of extra oracle
headroom at one seed are not evidence of mechanism.

## Deferred controls

- **Size matched:** if the mechanism passes, downsample MSP and CREMA-D to the
  fixed 4,246-row IEMOCAP scale and equalize update counts for E0/E2/E3. This
  tests whether the result survives source-data volume differences.
- **E1 SmoothCLAP:** if E2 wins and the paper needs a comparison with discrete
  tag supervision, train E1 in this exact new pipeline. Old SmoothCLAP results
  used different crops, loss, sampling, and tags, so they are not comparable.
- **Additional seeds:** authorize only after the mechanism gate, focusing on
  E2/E3 and any sources that pass.

## Evaluation boundaries

Each checkpoint uses only its own source Development Native UAR. Test data is
never used for training, factor normalization, tuning, or checkpoint
selection. Report every target separately; do not hide target dependence in a
single average.

Primary reporting is per-target Shared-4 UAR (Shared-3 for eNTERFACE).
Secondary reporting includes Native UAR, per-class recall, maximum
predicted-class share, missing classes, and worst-class recall. Source Test is
in-domain diagnosis, not cross-corpus evidence. BIIC remains excluded.

## Relation to prior work

- [SmoothCLAP](https://arxiv.org/abs/2601.12591) motivates relational soft
  targets but does not isolate utterance-level paralinguistic causality here.
- [ParaMETA](https://ojs.aaai.org/index.php/AAAI/article/view/40505) motivates
  task-specific paralinguistic subspaces; separate factor heads are the minimal
  version tested here.
- [EmotionRankCLAP](https://arxiv.org/abs/2505.23732) supports continuous or
  ordinal affect supervision, but not this cross-corpus causal question.
- [CLSP](https://arxiv.org/abs/2601.03065) motivates multi-granular supervision
  at larger scale; it does not establish bundled captions as the mechanism.
- [CLEP-DG](https://www.isca-archive.org/interspeech_2025/shi25_interspeech.html)
  is a future domain-generalization comparison only if this screen passes.
