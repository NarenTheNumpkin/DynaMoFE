"""Unit tests for DynaMoFE components."""

import numpy as np
import torch

from dynamofe.degradation import DegradationSignatureExtractor
from dynamofe.router import DynamicGatingRouter, DynaMoFEDetector


def test_degradation_extractor():
    extractor = DegradationSignatureExtractor(size=64)
    # Test with random frames (32 frames, 112x112, 3 channels)
    frames = np.random.randint(0, 256, (32, 112, 112, 3), dtype=np.uint8)
    features = extractor.extract_from_numpy(frames)
    assert isinstance(features, np.ndarray)
    assert features.shape == (16,)
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()

    # Test tensor extraction directly
    tensor = torch.rand(10, 3, 224, 224)
    feat_t = extractor.extract_from_tensor(tensor)
    assert feat_t.shape == (16,)
    assert not torch.isnan(feat_t).any()


def test_router_forward_and_gradients():
    router = DynamicGatingRouter(in_dim=16, num_experts=4, hidden_dim=32)
    x = torch.randn(8, 16)
    weights = router(x)
    assert weights.shape == (8, 4)
    # Check that weights sum to 1 across experts
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # Check all weights are positive
    assert (weights >= 0).all()

    # Check backprop
    loss = weights.sum()
    loss.backward()
    for param in router.parameters():
        assert param.grad is not None


def test_dynamofe_detector():
    router = DynamicGatingRouter(in_dim=16, num_experts=3, hidden_dim=32)
    means = [1.0, 2.0, 3.0]
    stds = [0.5, 1.0, 1.5]
    detector = DynaMoFEDetector(router, expert_names=["fcg", "tall", "f3net"], means=means, stds=stds)

    deg = torch.randn(4, 16)
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
