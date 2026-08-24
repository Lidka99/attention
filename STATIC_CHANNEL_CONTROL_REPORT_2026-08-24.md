# Static channel-selection control — Tiny ImageNet, seed 42

**Date:** 2026-08-24  
**Protocol:** ResNet-50 for small images; 16 stage-1 groups, fixed retained
set of 10 groups (62.5%); 20 epochs; AdamW; 64 images/GPU; four-GPU DDP.

## Result

The deployment path, which uses only the learned sample-independent group
scores, reached **55.19% validation Top-1** at the last epoch.  The best
validation checkpoint across the frozen 20-epoch trajectory was **56.26%** at
epoch 17.

## Interpretation

This is the required no-feedback control for the semantic-feedback study.  It
does not yet establish a speedup because the current implementation masks
features rather than physically compacts convolutions.  It is also not directly
comparable to the earlier stage-3 static-depth baseline: the computation and
the retained structure differ.  The only immediate comparison is the upcoming
feedback arm with the same model, seed, group count, schedule and deployment
evaluation.
