import torch
from torch import nn

from train.train_classifier import evaluate, scaled_optimizer_step


class FixedClassifier(nn.Module):
    def forward(self, images):
        batch = images.shape[0]
        assert batch == 2
        proxy_logits = torch.tensor([-2.0, 2.0], device=images.device)
        birads_logits = torch.tensor(
            [[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]], device=images.device
        )
        return proxy_logits, birads_logits


def test_classifier_metrics_name_birads_proxy_instead_of_malignancy():
    batch = (
        torch.zeros((2, 3, 8, 8)),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0, 2]),
    )

    metrics = evaluate(FixedClassifier(), [batch], torch.device("cpu"))

    assert metrics["birads_4_5_auc"] == 1.0
    assert metrics["birads_4_5_acc"] == 1.0
    assert "malignant_auc" not in metrics
    assert "malignant_acc" not in metrics


def test_scheduler_does_not_advance_when_grad_scaler_skips_optimizer_step():
    class Scaler:
        def __init__(self):
            self.scale = 1024.0

        def get_scale(self):
            return self.scale

        def step(self, optimizer):
            optimizer.steps += 1

        def update(self):
            self.scale = 512.0

    class Counter:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    scaler, optimizer, scheduler = Scaler(), Counter(), Counter()

    stepped = scaled_optimizer_step(scaler, optimizer, scheduler)

    assert stepped is False
    assert optimizer.steps == 1
    assert scheduler.steps == 0
