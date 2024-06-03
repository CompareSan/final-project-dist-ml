import torch


def test_data_loader(get_data_loaders):
    trainloader, testloader = get_data_loaders
    assert isinstance(trainloader, torch.utils.data.DataLoader)
    assert isinstance(testloader, torch.utils.data.DataLoader)


def test_normalization(get_data_loaders):
    trainloader, _ = get_data_loaders
    images, _ = next(iter(trainloader))
    assert images.min() >= -1.0
    assert images.max() <= 1.0
