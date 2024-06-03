import os
from io import BytesIO

import mlflow
import torch
import torchvision.transforms as transforms
import uvicorn
from fastapi import FastAPI, File
from PIL import Image

app = FastAPI()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
model_name = "pytorch-cnn-model"
model_version_alias = "champion"

model_uri = f"models:/{model_name}@{model_version_alias}"

model = mlflow.pytorch.load_model(model_uri)
model.eval()

label_classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def read_image(file: bytes) -> Image.Image:
    pil_image = Image.open(BytesIO(file))
    return pil_image


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return transform(image).unsqueeze(0)


@app.post("/predict")
def predict(file: bytes = File(...)) -> str:
    image = read_image(file)
    tensor_image = preprocess_image(image)
    with torch.no_grad():
        output = model(tensor_image)
        _, predicted = torch.max(output.data, 1)
    return f"prediction: {label_classes[int(predicted.item())]}"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
