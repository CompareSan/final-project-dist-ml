from typing import Tuple

import pytest
import torch

from final_project_dist_ml.get_data import load_and_transform_data


@pytest.fixture(params=[64, 32], scope="module")
def get_data_loaders(
    request: pytest.FixtureRequest,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    trainloader, testloader = load_and_transform_data(request.param)
    return trainloader, testloader
