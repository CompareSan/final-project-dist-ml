import torch
from final_project_dist_ml.get_data import get_data_loaders


def test_data_loader():
    trainloader, testloader = get_data_loaders(4)

    assert isinstance(trainloader, torch.utils.data.DataLoader)
    assert isinstance(testloader, torch.utils.data.DataLoader)


def test_normalization():
    trainloader, _ = get_data_loaders(4)
    images, _ = next(iter(trainloader))
    assert images.min() >= -1.0
    assert images.max() <= 1.0
