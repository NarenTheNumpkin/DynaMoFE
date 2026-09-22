"""Unit tests for DynaMoFE components (v2 28-D and 16-D)."""

import numpy as np
import torch

from dynamofe.degradation import DegradationSignatureExtractor
from dynamofe.router import DynamicGatingRouter, DynaMoFEDetector


def test_degradation_extractor():
    # 1. Test 28-D Extractor (Default)
    extractor28 = DegradationSignatureExtractor(size=64, version="v2_28d")
    frames = np.random.randint(0, 256, (32, 112, 112, 3), dtype=np.uint8)
    features28 = extractor28.extract_from_numpy(frames)
    assert isinstance(features28, np.ndarray)
    assert features28.shape == (28,)
    assert not np.isnan(features28).any()
    assert not np.isinf(features28).any()

    # Test tensor extraction directly
    tensor = torch.rand(10, 3, 128, 128)
    feat_t28 = extractor28.extract_from_tensor(tensor)
    assert feat_t28.shape == (28,)
    assert not torch.isnan(feat_t28).any()

    # 2. Test 16-D Extractor (Legacy)
    extractor16 = DegradationSignatureExtractor(size=64, version="v1_16d")
    features16 = extractor16.extract_from_numpy(frames)
    assert features16.shape == (16,)
    assert not np.isnan(features16).any()


def test_router_forward_and_gradients():
    # Test 28-D Router
    router = DynamicGatingRouter(in_dim=28, num_experts=4, hidden_dim=32, routing_mode="residual")
    x = torch.randn(8, 28)
    weights = router(x)
    assert weights.shape == (8, 4)
    # Check that weights sum to 1 across experts
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # Check all weights are positive
    assert (weights >= 0).all()

    # Check predicted risk
    pred_risk = router.predict_risk(x)
    assert pred_risk.shape == (8, 4)

    # Check backprop
    loss = weights.sum() + pred_risk.sum()
    loss.backward()
    for param in router.parameters():
        assert param.grad is not None


def test_dynamofe_detector():
    router = DynamicGatingRouter(in_dim=28, num_experts=3, hidden_dim=32)
    means = [1.0, 2.0, 3.0]
    stds = [0.5, 1.0, 1.5]
    detector = DynaMoFEDetector(router, expert_names=["fcg", "tall", "f3net"], means=means, stds=stds)

    deg = torch.randn(4, 28)
    scores = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.5],
        [0.5, 1.0, 1.5],
        [1.5, 2.5, 3.5],
    ])
    fused, weights = detector(deg, scores)
    assert fused.shape == (4,)
    assert weights.shape == (4, 3)
    # In row 0: scores equal means, so standardized scores are [0, 0, 0] -> fused score must be 0
    assert torch.allclose(fused[0], torch.tensor(0.0), atol=1e-5)


def test_deterministic_ranking_pipeline():
    """Verify ranking loss sign, risk ordering, Top-1, Top-2, Kendall tau, and inverse-risk softmax."""
    from scipy.stats import kendalltau
    from dynamofe.train_router import pairwise_ranking_loss, confidence_weighted_ranking_loss

    # Synthetic expert losses: E1 is best (0.1), E4 is worst (1.2)
    true_losses = torch.tensor([[0.1, 0.4, 0.8, 1.2]])

    # 1. Correct risk ordering: R1 < R2 < R3 < R4
    pred_risk_correct = torch.tensor([[0.1, 0.4, 0.8, 1.2]])

    # 2. Inverted risk ordering: R1 > R2 > R3 > R4 (worst expert has lowest risk)
    pred_risk_inverted = torch.tensor([[1.2, 0.8, 0.4, 0.1]])

    # Verify ranking loss decreases for correct ordering
    loss_corr = pairwise_ranking_loss(pred_risk_correct, true_losses).item()
    loss_inv = pairwise_ranking_loss(pred_risk_inverted, true_losses).item()
    assert loss_corr < loss_inv, f"Correct loss ({loss_corr}) must be < inverted loss ({loss_inv})"

    loss_conf_corr = confidence_weighted_ranking_loss(pred_risk_correct, true_losses).item()
    loss_conf_inv = confidence_weighted_ranking_loss(pred_risk_inverted, true_losses).item()
    assert loss_conf_corr < loss_conf_inv, f"Conf-weighted loss ({loss_conf_corr}) must be < inverted ({loss_conf_inv})"

    # Verify Top-1 selects expert 0 (E1)
    risk_np = pred_risk_correct.numpy()[0]
    loss_np = true_losses.numpy()[0]
    best_true = np.argmin(loss_np)
    best_pred = np.argmin(risk_np)
    assert best_true == 0, "Expert 0 should be true best"
    assert best_pred == 0, "Top-1 should predict expert 0"

    # Verify Top-2 returns experts 0 and 1
    sorted_pred = np.argsort(risk_np)
    assert sorted_pred[0] == 0 and sorted_pred[1] == 1, "Top-2 must return experts 0 and 1"

    # Verify Kendall tau is strictly positive (+1.0)
    true_ranks = np.argsort(np.argsort(loss_np))
    pred_ranks = np.argsort(np.argsort(risk_np))
    tau, _ = kendalltau(true_ranks, pred_ranks)
    assert tau > 0.99, f"Kendall tau should be +1.0, got {tau}"

    # Verify inverse-risk softmax assigns largest weight to expert 0
    weights = torch.softmax(-pred_risk_correct / 1.0, dim=-1).numpy()[0]
    assert np.argmax(weights) == 0, "Expert 0 must receive largest weight"
    assert weights[0] > weights[1] > weights[2] > weights[3], "Weights must decrease with risk"


def test_trimoe_architecture_and_pipeline():
    from dynamofe.trimoe import TRIMOE, TRIMOELoss, extract_multidomain_inputs

    # 1. Verify parameter count matches paper (28.40M)
    model = TRIMOE(num_classes=2, top_k=2, pretrained=False)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    assert abs(total_params - 28.40) < 0.05, f"Expected ~28.40M params, got {total_params:.2f}M"

    # 2. Test multi-domain input extraction on batch [2, 16, 224, 224, 3]
    dummy_frames = torch.randint(0, 255, (2, 16, 224, 224, 3), dtype=torch.uint8)
    sp, spec, temp = extract_multidomain_inputs(dummy_frames)
    assert sp.shape == (2, 16, 3, 224, 224), f"Unexpected spatial shape: {sp.shape}"
    assert spec.shape == (2, 16, 1, 224, 224), f"Unexpected spectral shape: {spec.shape}"
    assert temp.shape == (2, 16, 4), f"Unexpected temporal shape: {temp.shape}"

    # 3. Test forward pass
    y_hat, details = model(sp, spec, temp)
    assert y_hat.shape == (2,), f"Unexpected prediction shape: {y_hat.shape}"
    assert details["prob"].shape == (2,), f"Unexpected probability shape: {details['prob'].shape}"
    assert (details["prob"] >= 0.0).all() and (details["prob"] <= 1.0).all(), "Probabilities must be in [0, 1]"

    # 4. Test loss with auxiliary load balancing
    loss_fn = TRIMOELoss(lambda_aux=0.01)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loss = loss_fn(details, labels)
    assert loss.item() > 0.0, "Loss must be positive"
    loss.backward()

    # 5. Test spatial to spectral parameter transfer
    model.transfer_spatial_to_spectral()


if __name__ == "__main__":
    test_degradation_extractor()
    test_router_forward_and_gradients()
    test_dynamofe_detector()
    test_deterministic_ranking_pipeline()
    test_trimoe_architecture_and_pipeline()
    print("All DynaMoFE & TriMoE unit tests passed successfully!")

