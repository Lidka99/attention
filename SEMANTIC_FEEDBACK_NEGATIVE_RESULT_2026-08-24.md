# Training-only semantic-feedback attention — negative feasibility result

**Date:** 2026-08-24  
**Decision:** close this implementation; do not spend further seeds on it.

## Matched seed-42 result

Both arms use Tiny ImageNet, ResNet-50 for small images, 16 stage-1 groups,
10 retained groups, AdamW, 20 epochs, 64 images/GPU and four-GPU DDP.  The
only change is the training-only deep semantic feedback correction.

| Arm | Last deployment Top-1 | Best deployment Top-1 | Delta to static, last |
|---|---:|---:|---:|
| Static channel selection | 55.19% | 56.26% | — |
| Semantic feedback | 49.83% | 52.88% | −5.36 pp |

All values use the deployment path.  In particular, feedback is disabled at
validation, so this is not a comparison of a stronger dynamic teacher against
a static baseline.

## Interpretation

The test falsifies the present mechanism as a viable way to improve a static
mask.  During training, image-specific feedback substantially changes the
hard group choice; at deployment only one fixed ranking remains.  This
train/deployment decision mismatch is the likely primary failure mode.  The
feedback-score magnitude remains nonzero through training, so the correction
did not simply collapse to the static arm.

The result is too negative for a seed replication.  Repeating it would not be
an efficient use of compute unless the mechanism changes in a way that removes
the mismatch and is preregistered as a new hypothesis.

## What survives

The static channel-selection control is useful: it reaches 55.19% at 62.5%
of stage-1 groups.  However, masking activations remains a functional proxy;
it neither proves structural feasibility nor measured speedup.

The recommended next direction is therefore **static, structurally compiled
pruning with representation-preservation targets**, not a feedback gate.  A
teacher can supervise a fixed student mask directly (for example, matching
post-stage representations after physical channel removal), but must not make
per-image decisions that disappear at deployment.  Before implementation this
needs a literature audit against structured-pruning and pruning-distillation
baselines, followed by an oracle test: whether any fixed 10/16 stage-1 subset
has substantially better held-out accuracy than the learned static selection.
