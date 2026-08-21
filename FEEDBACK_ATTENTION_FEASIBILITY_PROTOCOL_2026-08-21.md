# Feasibility protocol — training-only semantic-feedback attention

**Date:** 2026-08-21  
**Status:** implementation and smoke-test stage; no empirical claim yet.

## Question

Can deep semantic information improve *which fixed channel groups survive* in a
structuredly pruned CNN, without a feedback module, input-dependent gate, or
second pass at deployment?

This is deliberately narrower than the earlier UCLA hypothesis.  UCLA's
per-image channel allocation did not beat utility-only routing and did not
provide structural acceleration.  A positive result here must survive after
the feedback machinery is removed.

## Mechanism

The candidate model gates the 16 groups at ResNet-50 stage 1.  It keeps 10
groups (62.5%).  During training only, it first obtains a *detached* stage-4
global representation from an unmasked teacher pass.  A small feedback head
maps that representation to a group-score correction.  Classification is then
trained through a hard straight-through group gate.

At validation and deployment, the correction head is not invoked.  The gate is
the same fixed top-10 selection for every image, determined solely by learned
`static_scores`.  Therefore the deployed decision is structurally eligible for
channel removal; zeroing activations remains only a functional feasibility
proxy until the physical compaction experiment is implemented.

## Controls and falsification

| Arm | Training score | Deployment score | Required interpretation |
|---|---|---|---|
| Static | `static_scores` | `static_scores` | baseline structured selection |
| Feedback | `static_scores + deep feedback` | `static_scores` | tests training-only semantic feedback |
| Dense | no gate | no gate | upper reference |

All arms use identical data split, optimiser, epochs, seed list and target
keep ratio.  The primary endpoint is held-out accuracy using the deployment
path, not training-time dynamic accuracy.  A useful signal requires a
replicated gain over Static at the same 10/16 groups; otherwise this direction
is closed.  A later positive result must also demonstrate compacted-model
latency, because masking alone is not speedup.

## Immediate experiment

Run a 20-epoch Tiny ImageNet screen with seeds 42, 123 and 2026 after the
currently active static-depth control releases the GPUs.  The initial screen
is an elimination test only, not publication evidence.  Promote only a
consistent signal to 100/200-epoch training, physical channel compaction and
real latency measurement.

## Novelty boundary

The claim is **not** "attention improves pruning" nor "the first dynamic
pruner".  It is conditional: training-only semantic feedback can improve a
static, deployable structured mask.  This still requires a focused literature
audit against dynamic/static channel pruning before any novelty claim.
