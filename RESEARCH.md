# Research design

## Goal

The main problem is cross-corpus zero-shot SER: train only on source corpora and
recognize emotions in unseen target corpora through audio-text alignment. The
research question is narrower than “does adding more caption text help?”:

> Can explicit, correctly paired paralinguistic dynamics shape a more
> transferable emotion space, beyond corpus/emotion priors and generic
> regularization?

The previous expert-routing work established complementarity and useful
ensemble headroom, but did not show that paralinguistic alignment caused it.
This repository therefore starts with representation learning and withholds
routing until a paralinguistic mechanism survives a direct null control.

## Why reset the formulation

Three problems make the old positive cells insufficient as causal evidence:

1. Discrete captions mix emotion, gender, absolute pitch/intensity, duration,
   and wording augmentation in one sentence.
2. Source dataset properties can make those attributes proxies for corpus or
   emotion priors.
3. Standard diagonal CLIP loss treats another utterance with the same emotion
   caption as a negative. With emotion-only text, that contradicts the desired
   class geometry.

The new baseline uses a class-aware multi-positive contrastive target: every
same-emotion item in the batch is positive. This is shared by E0, E2, and E3.
It is a deliberate first-principles correction, not an extra ablation.

## Factors

Only continuous within-utterance dynamics are used in the first experiment.
Each value is converted to a quantile using that source corpus's **Train split
only**, making scales comparable without fitting on Development or Test.

- Pitch: normalized F0 variation, percentile range, rising slope, falling slope.
- Energy: normalized loudness variation, percentile range, rising slope,
  falling slope.
- Rhythm: voiced segments per second, mean voiced length, mean unvoiced length.

The first experiment excludes gender, VAD labels, absolute mean F0, absolute
sound level, and raw duration. These are either unavailable across corpora or
especially likely to encode speaker, recording, or corpus identity. Spectral
flux is deferred to avoid broadening the initial claim.


All conditions use a deterministic center five-second view; shorter utterances
are used in full and padded with an attention mask. This replaces inherited
random cropping because 51.6% of MSP and 32.9% of IEMOCAP Train utterances
exceed five seconds. For every longer utterance, eGeMAPS is recomputed on the
exact center interval. E0-E3 therefore receive identical audio views, E1 tags
describe its observed crop, and E2/E3 targets describe that same crop. The
first pilot deliberately gives up random-crop augmentation rather than accept
unmeasured supervision noise or introduce a multi-window pipeline.

## Model and four conditions

All conditions start from one seed-3407 SmoothCLAP initial state, use trainable
main audio and text encoders, train for 30 epochs, and sample source corpus
uniformly then emotion uniformly within source.

| ID | Main alignment | Paralinguistic supervision |
|---|---|---|
| E0 | Class-aware emotion-only CLAP | None |
| E1 | Original SmoothCLAP soft target; emotion-only main caption | Full canonical text tags plus frozen local audio similarity |
| E2 | Same as E0 | Three linear heads regress the correctly paired continuous factor ranks |
| E3 | Same as E0 | Same heads and targets as E2, but whole target vectors deranged within source × true emotion with zero fixed points |

The E2/E3 heads operate on the pooled audio-encoder representation. Their three
losses are averaged and added to the emotion loss with fixed weight 64. They
are discarded at inference, so zero-shot prediction remains audio-versus-label
cosine similarity.

The weight was frozen before formal training using five seed-3407 pooled Train
batches at initialization. The median emotion-to-factor gradient-norm ratio at
the shared audio representation was 248 (range 222–371); weight 64 gives the
auxiliary task roughly one quarter of the emotion gradient at initialization.
No Development or Test result was used to choose it.

Derangement preserves each source/emotion factor distribution and all
within-vector correlations while ensuring that no utterance keeps its own
factor vector. It destroys only the utterance-level pairing.
Therefore:

- E2 > E0 but E2 ≈ E3 indicates generic multitask regularization or
  source/emotion-distribution effects, not useful utterance-level alignment.
- E2 > E0 and E2 > E3 supports a contribution from correctly paired continuous
  factors.
- E1 > E0 but E2 does not win suggests the original relational/text mechanism,
  not the proposed continuous factor route.
- E2 ≤ E0 is a stop signal for factor-based routing in this form.

This is evidence about this factor set and architecture, not proof that every
paralinguistic feature causes generalization.

## Evaluation contract

Training sources are MSP-Podcast, IEMOCAP, and CREMA-D. Checkpoint selection is
the equal-weight mean of their three Development Native UARs. Test data never
participates in optimization, normalization, hyperparameter choice, or
checkpoint selection.

Report each target separately:

- Primary: Shared-4 UAR on unseen corpora, with eNTERFACE reported as Shared-3
  rather than silently mixed into Shared-4.
- Secondary: Native UAR, per-class recall, maximum predicted-class share,
  missing classes, and worst-class recall.
- In-domain source Test results are diagnostics, not cross-corpus evidence.
- RAVDESS split Test and CAMEO RAVDESS-full must be labeled separately because
  they overlap in corpus identity and are not independent targets.
- Do not replace the target table with one average. Summaries may state how
  many targets improve or decline and give the median delta, alongside all
  per-target values.

The first seed is a direction screen. A credible positive result must be
distributed across several genuinely different targets, not driven only by
TESS or one corpus family, and must not trade improvement for prediction
collapse. Only after that result should the winning comparison be repeated
with additional seeds.

## Relation to prior work

- [SmoothCLAP](https://arxiv.org/abs/2601.12591) motivates relational A2A/T2T
  soft targets, but does not isolate utterance-level paralinguistic causality
  for this cross-corpus setting.
- [ParaMETA](https://ojs.aaai.org/index.php/AAAI/article/view/40505) supports
  task-specific subspaces to reduce interference among paralinguistic tasks.
  Separate factor heads are the minimal version of that principle here.
- [EmotionRankCLAP](https://arxiv.org/abs/2505.23732) shows that continuous or
  ordinal affect supervision can shape CLAP representations, but its VA
  setting and task are not a direct cross-corpus SER answer.
- [CLSP](https://arxiv.org/abs/2601.03065) uses multi-granular supervision at a
  much larger data scale; its results do not establish that bundled captions
  are sufficient in this small supervised regime.
- [ParaSpeechCLAP](https://arxiv.org/abs/2603.28737) supports specialized
  paralinguistic modeling, while leaving the present zero-shot SER question
  open.
- [CLEP-DG](https://www.isca-archive.org/interspeech_2025/shi25_interspeech.html)
  is a relevant domain-generalization comparison if the first experiment
  passes, not a component added to this pilot.

The ICASSP 2027 paper deadline is listed as 16 September 2026 on the
[official call for papers](https://2027.ieeeicassp.org/call-for-papers/).
