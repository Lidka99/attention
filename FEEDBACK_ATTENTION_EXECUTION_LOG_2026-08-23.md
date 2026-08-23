# Execution log — semantic-feedback feasibility screen

**Date:** 2026-08-23

This log separates infrastructure failures from empirical results.  None of
the entries below is evidence for or against the feedback hypothesis.

| Attempt | Outcome | Corrective action |
|---|---|---|
| 1 | `ModuleNotFoundError: attentionv3` under `torchrun` | Run with `PYTHONPATH=src`, as required by the repository's source layout. |
| 2 | CUDA OOM at 128 images/GPU | Freeze the established feasible batch size of 64/GPU for both static and feedback arms. |
| 3 | DDP reduction error in the static arm because `feedback_head` is intentionally unused | Enable `find_unused_parameters=True` in this experiment's DDP wrapper. |

The next run is valid only if it starts from a clean GPU state, records all 20
epochs, and evaluates the static deployment path.  No partial run is retained
as a result.
