import torch

from src.model import build_model


def test_model_output_shape():
    model = build_model(pretrained=False).eval()
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2)