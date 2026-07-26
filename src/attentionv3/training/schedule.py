"""Curriculum for warm-up, sparsification and fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn

from attentionv3.budget_controller import BudgetController


@dataclass(frozen=True)
class CurriculumState:
    """Hyperparameters applied at the beginning of a training epoch."""

    phase: str
    budget: float
    max_budget: float
    brier_weight: float
    budget_weight: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


class BudgetCurriculum:
    """Three-phase schedule that never prunes abruptly at epoch zero."""

    def __init__(self, warmup_epochs: int, pilot_epochs: int, fine_tune_epochs: int,
                 target_budget: float, target_max_budget: float,
                 brier_weight: float, budget_weight: float) -> None:
        if min(warmup_epochs, pilot_epochs, fine_tune_epochs) < 0:
            raise ValueError("epoch counts must be non-negative")
        if pilot_epochs == 0 and target_budget < 1:
            raise ValueError("sparsification requires at least one pilot epoch")
        if not 0 < target_budget <= target_max_budget <= 1:
            raise ValueError("require 0 < target_budget <= target_max_budget <= 1")
        self.warmup_epochs = warmup_epochs
        self.pilot_epochs = pilot_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.target_budget = target_budget
        self.target_max_budget = target_max_budget
        self.brier_weight = brier_weight
        self.budget_weight = budget_weight

    @property
    def total_epochs(self) -> int:
        return self.warmup_epochs + self.pilot_epochs + self.fine_tune_epochs

    def state_for_epoch(self, epoch: int) -> CurriculumState:
        """Return a 0-indexed epoch state with linear sparsification ramp."""
        if not 0 <= epoch < self.total_epochs:
            raise IndexError("epoch is outside the curriculum")
        if epoch < self.warmup_epochs:
            return CurriculumState("warmup", 1.0, 1.0, 0.0, 0.0)
        pilot_index = epoch - self.warmup_epochs
        if pilot_index < self.pilot_epochs:
            progress = (pilot_index + 1) / self.pilot_epochs
            return CurriculumState(
                "sparsification",
                1.0 + progress * (self.target_budget - 1.0),
                1.0 + progress * (self.target_max_budget - 1.0),
                progress * self.brier_weight,
                progress * self.budget_weight,
            )
        return CurriculumState("fine_tune", self.target_budget, self.target_max_budget,
                               self.brier_weight, self.budget_weight)


def apply_curriculum(model: nn.Module, criterion: object, state: CurriculumState) -> None:
    """Apply a state to all controllers, including controllers inside DDP."""
    for module in model.modules():
        if isinstance(module, BudgetController):
            module.budget = state.budget
            module.max_budget = state.max_budget
    # Criterion is intentionally duck-typed here so the schedule remains free
    # of a circular import and can be used with future loss implementations.
    criterion.target_budget = state.budget
    criterion.brier_weight = state.brier_weight
    criterion.budget_weight = state.budget_weight
