# Static-depth control — seed 123 update

**Date:** 2026-08-21  
**Status:** completed; replicate only, not a final multi-seed conclusion.

## Result

The fixed stage-3 ResNet-50 control completed its frozen 20-epoch Tiny ImageNet
protocol at **53.67% validation Top-1** (last epoch).  Its cost proxy is
81.25% of the cumulative residual-block cost of the full network.

| Seed | Static stage 3 | Adaptive confidence policy | Adaptive full-depth |
|---|---:|---:|---:|
| 42 | 53.55% | 54.96% | 54.96% |
| 123 | 53.67% | 53.46% | 53.78% |

## Consequence

The seed-42 point Pareto improvement does **not** replicate directionally in
seed 123: the adaptive confidence policy is 0.21 pp below the static stage-3
control and 0.32 pp below full depth.  The observation does not falsify an
adaptive-depth benefit, but it does falsify any claim that one seed establishes
dominance over a trained fixed-depth model.

Seed 2026 and a mean ± SD comparison remain required.  The static channel
selection control for the separate training-only semantic-feedback study was
started after this run released the GPUs.
