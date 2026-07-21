import torch
from torch.utils.data import DataLoader, TensorDataset

from model import ResNetCIFAR
from train import evaluate, train_one_epoch


def _make_loader(num_samples=16, num_classes=10, batch_size=4):
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_train_one_epoch_returns_expected_keys():
    net = ResNetCIFAR(blocks_per_stage=1, stochastic_depth=True)
    loader = _make_loader()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    result = train_one_epoch(net, loader, optimizer, device=torch.device("cpu"))
    assert set(result.keys()) == {"loss", "accuracy", "time_seconds"}
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["time_seconds"] > 0


def test_evaluate_returns_expected_keys():
    net = ResNetCIFAR(blocks_per_stage=1, stochastic_depth=True)
    loader = _make_loader()
    result = evaluate(net, loader, device=torch.device("cpu"))
    assert set(result.keys()) == {"loss", "accuracy"}
    assert 0.0 <= result["accuracy"] <= 1.0


def test_evaluate_does_not_change_model_parameters():
    net = ResNetCIFAR(blocks_per_stage=1, stochastic_depth=True)
    loader = _make_loader()
    before = [p.clone() for p in net.parameters()]
    evaluate(net, loader, device=torch.device("cpu"))
    after = list(net.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_train_one_epoch_reduces_loss_over_several_epochs():
    net = ResNetCIFAR(blocks_per_stage=1, stochastic_depth=False)
    loader = _make_loader(num_samples=16, batch_size=4)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    first = train_one_epoch(net, loader, optimizer, device=torch.device("cpu"))
    last = first
    for _ in range(5):
        last = train_one_epoch(net, loader, optimizer, device=torch.device("cpu"))
    assert last["loss"] < first["loss"]
