from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms


def load_and_transform_data() -> (
    Tuple[
        torchvision.datasets.mnist.FashionMNIST,
        torchvision.datasets.mnist.FashionMNIST,
    ]
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    trainset = torchvision.datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform,
    )

    testset = torchvision.datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=transform,
    )

    return trainset, testset


def get_data_loaders(
    batch_size: int,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    trainset, testset = load_and_transform_data()
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return trainloader, testloader


if __name__ == "__main__":
    batch_size = 64
    get_data_loaders(batch_size)
