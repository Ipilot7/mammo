import numpy as np
import torch
from torch import nn

from train import train_screening


class StableModel(nn.Module):
    def forward(self, x):
        return torch.tensor([[1000.0, -1000.0, 0.0, 1.0, 2.0]]).repeat(len(x), 1)


def test_screening_prediction_avoids_fp16_training_autocast(monkeypatch):
    def forbidden_autocast(_device):
        raise AssertionError("validation must use full precision")

    monkeypatch.setattr(train_screening, "autocast", forbidden_autocast)
    loader = [(torch.zeros(2, 3, 8, 8), torch.tensor([1, 2]))]

    probabilities = train_screening.predict(StableModel(), loader, torch.device("cpu"))

    assert probabilities.shape == (2, 5)
    assert np.isfinite(probabilities).all()
