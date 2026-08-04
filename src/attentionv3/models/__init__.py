"""Models used by attentionv3 experiments."""

from .resnet_ucla import GlobalUCLAResNet50, UCLAResNet50
from .resnet_value_budget import GlobalValueBudgetResNet50

__all__ = ["UCLAResNet50", "GlobalUCLAResNet50", "GlobalValueBudgetResNet50"]
