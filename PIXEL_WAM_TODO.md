# Pixel WAM Research TODO

Last synchronized: 2026-08-26

This is the persistent implementation and experiment checklist for the
Pixel WAM project. It consolidates, in order of precedence:

1. the latest proposal in `../overleaf/main.tex`;
2. decisions confirmed in the subsequent discussion;
3. the earlier Pixel WAM revision plan; and
4. the existing memory TODO for Pixel Fast-WAM inference solvers.

When these sources differ, the latest proposal and the most recent explicit
decision take precedence. Proposal prose should remain concise; implementation
details and unresolved choices belong here.

## Research objective

The main research question is whether direct pixel-space video prediction can
be applied within a joint World--Action Model (WAM) and, relative to a matched
latent WAM, preserve more manipulation-relevant visual information, learn more
action-relevant latent visual representations, and improve action performance.

The research chain is:

> Pixel-space video prediction -> retained visual information -> latent visual
> representations used by the action component -> closed-loop action
> performance.

Pixel generation in a WAM is the central topic. Latent-to-pixel adaptation is
an enabling starting point, not the main contribution by itself. Fine-detail
modeling and visual-representation learning are complementary directions: the
project is not limited to making rollouts look sharper.

## Fixed project decisions

- [x] Use Fast-WAM-Joint as the initial matched WAM framework.
- [x] Use the local latent Fast-WAM-Joint checkpoint as the primary matched
  latent reference; retain the paper-reported checkpoint result as context.
- [x] Use the projection fitted on all valid clips (`A_all`) for the main pixel
  baseline. Treat `A_100K` as a sensitivity diagnostic rather than the default.
- [x] Name the current pixel target `scale-calibrated rank-asymmetric velocity`;
  do not use ambiguous labels such as `base` or `ordinary FM`.
- [x] Treat clean-pixel `x_0` prediction as one parameterization ablation, not
  as a separate research direction.
- [x] Measure action performance throughout the project with matched
  closed-loop success rate. Do not introduce a redundant inverse-dynamics
  probe as the primary representation metric.
- [x] Test representation utility with closed-loop interventions: remove the
  representation objective or conditioning path, perturb/remove predicted
  representation tokens, and compare with a matched no-rollout variant.
- [x] Keep the current Wan-compatible spatial and temporal video packing for
  the core study. Revisit tokenization only if a new backbone or boundary
  design makes the comparison technically meaningful.

## Completed preliminary baseline

- [x] Adapt latent Fast-WAM-Joint to pixel-space input and output boundaries
  while reusing the pretrained Transformer blocks.
- [x] Implement AsymFlow scale calibration, rank-asymmetric velocity, fixed
  projection bases, and recovery of the full pixel-space velocity for training
  and sampling.
- [x] Train and evaluate the rank-asymmetric pixel variant.
- [x] Train and evaluate the rank-asymmetric pixel variant with joint VR and
  time-gated LPIPS.
- [x] Complete the initial matched LIBERO closed-loop comparison:

  | Model | Overall success rate |
  | --- | ---: |
  | Latent Fast-WAM-Joint, paper reported | 98.50% |
  | Latent Fast-WAM-Joint, local reproduction | 98.80% |
  | Pixel Fast-WAM-Joint, asym. velocity | 95.95% |
  | Pixel Fast-WAM-Joint, asym. velocity + VR/LPIPS | 97.70% |

- [x] Produce a matched qualitative comparison showing that the current pixel
  variants retain coarse scene structure but generate substantially blurrier
  rollouts than the latent reference.
- [x] Compare `A_all` with `A_100K` (about 45% of valid clips) and record the
  isolated severe artifact observed with the `A_100K` asym-velocity model.
- [ ] Determine whether projection fitting is systematically sensitive to
  fitting-data coverage or subset randomness. Keep this as a short diagnostic
  unless repeated evidence makes it a central issue.

The exact checkpoint, evaluation-log, clip, frame, and seed provenance for the
completed comparisons is recorded in comments around the tables and figures in
`../overleaf/main.tex`.

## M1: Pixel-space adaptation, parameterization, and sampling

Goal: establish the strongest controlled Pixel Fast-WAM-Joint foundation before
adding detail or representation modules.

- [ ] Freeze a matched experiment specification for the local latent reference
  and all pixel variants: initialization, training data, action pathway,
  training budget, checkpoint type, and evaluation protocol.
- [ ] Reproduce the `A_all` rank-asymmetric and `A_all` VR/LPIPS baselines under
  that specification when a new code revision requires it.
- [ ] Compare the current scale-calibrated rank-asymmetric velocity with direct
  clean-pixel `x_0` prediction.
- [ ] Calibrate pixel noise scale / SNR (`gamma`) and training timestep sampling
  for the pixel prediction space.
- [ ] Audit training and inference time conventions together while keeping the
  following choices experimentally separate:
  - [ ] prediction parameterization;
  - [ ] training timestep sampling;
  - [ ] inference timestep schedule and shift; and
  - [ ] ODE solver.
- [ ] Add an inference-only solver ablation from the existing memory TODO:
  - [ ] retain the current explicit-Euler, 10-step, shift-5 path as the recorded
    baseline;
  - [ ] implement a mathematically joint video--action Heun update first; and
  - [ ] consider UniPC or another few-step solver after the Heun comparison.
- [ ] Compare inference solvers at a matched number of model evaluations.
- [ ] Implement an L2P-inspired two-stage adaptation:
  - [ ] adapt the pixel-specific input and output modules while reusing the
    pretrained Transformer blocks;
  - [ ] then jointly fine-tune video and action modeling in the full WAM.
- [ ] Select the Pixel WAM foundation used by M2 and M3 using video metrics,
  qualitative rollouts, stability, and closed-loop success rate.

Estimated duration from the current RP: 2 weeks.

## M2: Spatial detail refinement and frequency-aware video modeling

Goal: address the blurred pixel rollouts while checking action performance at
each step.

- [ ] Adapt DiP's convolutional Patch Detailer Head to robot video so that
  Transformer context and noised video patches refine local spatial structure
  consistently across frames.
- [ ] Adapt WaiT's lossless spatiotemporal wavelet decomposition and
  frequency-dependent flow schedules to robot video.
- [ ] Implement a distinct PixelGen-style loss variant on predicted clean-frame
  estimates:
  - [ ] noise-gated LPIPS for local perceptual structure; and
  - [ ] noise-gated P-DINO for semantic visual structure.
- [ ] For the rank-asymmetric branch, verify the clean-frame recovery boundary
  before applying PixelGen-style losses.
- [ ] Evaluate the detail head, wavelet/frequency method, and PixelGen-style
  supervision individually.
- [ ] Combine only the selected components needed to test complementary effects
  on video information and action performance.
- [ ] Select the detail-aware variant(s) passed to M3.

Estimated duration from the current RP: 2 weeks.

## M3: Action-relevant visual representation learning

Goal: determine whether pixel-space supervision can produce latent visual
representations that more effectively support action generation, including
cases where rendered rollouts remain imperfect.

- [ ] Implement Representation Forcing as the primary representation study:
  - [ ] choose and document the target visual feature extractor (DINOv3 is the
    current leading candidate);
  - [ ] define the online quantizer, codebook, number of representation tokens,
    and temporal alignment;
  - [ ] predict discrete representation tokens before pixel generation; and
  - [ ] condition both the video and action components on their embeddings.
- [ ] Compare with a continuous representation-prediction variant if it remains
  informative for temporally continuous robot video.
- [ ] Run matched closed-loop representation interventions:
  - [ ] remove the added representation objective;
  - [ ] remove the representation-conditioning pathway;
  - [ ] remove or perturb predicted representation tokens at inference; and
  - [ ] compare explicit video rollout with a matched Fast-WAM-style no-rollout
    variant.
- [ ] Decide which secondary representation method, if any, adds a genuinely
  distinct test:
  - [ ] RepWAM-style semantic visual--action targets;
  - [ ] DreamWAM-style motion, geometry, or semantic targets; or
  - [ ] PixelREPA/AGRA-style lightweight feature alignment.
- [ ] Select the representation-guided Pixel WAM using closed-loop action
  performance together with its matched intervention results.

Estimated duration from the current RP: 1--2 weeks.

## M4: Matched evaluation and benchmark extension

### Core matched evaluation

- [ ] Use the same pretrained initialization, training data, action pathway,
  training budget, and evaluation protocol for latent and pixel variants.
- [ ] Evaluate each M1--M3 component individually before selected combinations.
- [ ] Report held-out video quality with:
  - [ ] gFVD;
  - [ ] PSNR and SSIM;
  - [ ] LPIPS; and
  - [ ] wavelet-band reconstruction error.
- [ ] Report closed-loop success rate for every principal model and ablation.
- [ ] Analyze video fidelity, latent visual representation interventions, and
  action performance jointly, without using visual quality alone as a proxy for
  control.
- [ ] Preserve result provenance for every reported number and image: resolved
  config, checkpoint path, raw/EMA status, projection artifact, sampler,
  evaluation seed/protocol, and `summary.json` path.

### Benchmark priority

- [ ] LIBERO: primary matched benchmark.
- [ ] MimicGen Threading and Three-Piece Assembly: precision- and contact-rich
  evaluation.
- [ ] LIBERO-Plus: controlled visual and spatial perturbations.
- [ ] ManiSkill 3 PegInsertionSide: broader precision validation.
- [ ] RoboTwin 2.0 bimanual tasks: later broader validation.

The exact task subset and evaluation budget remain to be finalized before each
benchmark is promoted from extension to core commitment.

Estimated duration from the current RP: 2 weeks.

## M5: Scale and architecture generality

- [ ] Apply the selected Pixel WAM components to a larger video backbone, with
  a 14B-scale Wan model as one candidate beyond the current 5B backbone.
- [ ] Select one or more other latent WAM architectures where the same
  pixel-space ideas can be implemented without changing the research question.
- [ ] Re-run only the decisive matched ablations needed to test whether the
  findings generalize across backbone scale or WAM design.

Estimated duration from the current RP: 2 weeks.

## M6: Consolidation and writing

- [ ] Consolidate quantitative results, matched rollout figures, and failure
  cases.
- [ ] Separate feasibility findings, detail-modeling findings, representation
  findings, and action-performance findings in the analysis.
- [ ] Update the RP/paper only after implementation choices and evidence are
  fixed; keep implementation-only details in this TODO or experiment records.
- [ ] Keep `references.bib` and citations synchronized with the final methods.
- [ ] Before each Overleaf update, run `git diff --check` and a clean LaTeX
  build with `main.tex`, `references.bib`, and `figures/` present.

Estimated duration from the current RP: 2--3 weeks.

## Open decisions requiring review

- [ ] Final M1 parameterization retained for later stages.
- [ ] Pixel `gamma`, training timestep distribution, inference schedule/shift,
  ODE solver, and selection criteria.
- [ ] Freeze/unfreeze boundary and timing for the two-stage adaptation.
- [ ] Whether projection sensitivity merits a systematic study beyond the
  current one-anomaly diagnostic.
- [ ] Temporal design of the video Patch Detailer Head.
- [ ] Wavelet bands and band-specific schedules for robot video.
- [ ] Which M2 components should be combined after individual ablations.
- [ ] Representation teacher, quantizer, token interface, and insertion point;
  only Representation Forcing is currently a committed primary direction.
- [ ] Whether a continuous representation target and any secondary alignment
  target add enough information to retain.
- [ ] Final held-out video split and exact metric protocols.
- [ ] Which external benchmarks are core versus broader validation.
- [ ] Larger backbone and alternative WAM candidates.
- [ ] Exact experiment counts, compute budget, and completion thresholds for
  each milestone.

## Immediate next actions

- [ ] Review this consolidated TODO and mark any item that should be promoted,
  deferred, or removed.
- [ ] Lock the M1 matched baseline specification.
- [ ] Implement the joint video--action Heun solver ablation already recorded in
  memory.
- [ ] Choose the first M1 parameterization/SNR experiment after the baseline is
  locked.
