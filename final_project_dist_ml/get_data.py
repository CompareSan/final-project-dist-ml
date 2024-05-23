import torch
import torchvision
import torchvision.transforms as transforms
from typing import Tuple


def load_and_process_data() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(
        root='../data',
        train=True,
        download=True,
        transform=transform
    )

    testset = torchvision.datasets.FashionMNIST(
        root='../data',
        train=False,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    return trainloader, testloader


if __name__ == "__main__":
    load_and_process_data()