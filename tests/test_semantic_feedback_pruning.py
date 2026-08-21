import torch

from attentionv3.models.semantic_feedback_pruning import SemanticFeedbackPrunedResNet50


def test_feedback_model_deploys_static_group_gate_without_feedback():
    model = SemanticFeedbackPrunedResNet50(11, groups=16, keep_ratio=0.625, small_images=True)
    model.eval()
    logits, info = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 11)
    assert info["feedback_scores"] is None
    assert info["gate"].shape == (2, 256)
    assert torch.equal(info["gate"][0], info["gate"][1])
    assert info["gate"][0].sum().item() == 160


def test_feedback_path_is_training_only_and_receives_gradients():
    model = SemanticFeedbackPrunedResNet50(3, groups=16, keep_ratio=0.5, small_images=True)
    logits, info = model(torch.randn(1, 3, 32, 32))
    assert info["feedback_scores"] is not None
    logits.sum().backward()
    assert model.feedback_head[-1].weight.grad is not None
