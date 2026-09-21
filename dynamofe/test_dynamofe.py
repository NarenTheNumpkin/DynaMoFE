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


if __name__ == "__main__":
    test_degradation_extractor()
    test_router_forward_and_gradients()
    test_dynamofe_detector()
    print("All DynaMoFE unit tests passed successfully!")
