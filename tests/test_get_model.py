import pytest
import torch

from final_project_dist_ml.get_model import CNNModel


@pytest.mark.parametrize(
    "input_shape, expected_shape",
    [
        ((1, 1, 28, 28), (1, 10)),
        ((10, 1, 28, 28), (10, 10)),
    ],
)
def test_cnn_model_dimensions(input_shape, expected_shape):
    model = CNNModel()
    input_tensor = torch.randn(input_shape)  # batch_size, channels, height, width
    output = model(input_tensor)

    assert output.size() == torch.Size(expected_shape)  # batch_size, num_classes
