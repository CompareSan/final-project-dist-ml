from get_data import load_and_transform_data
from get_model import CNNModel
from metaflow import FlowSpec, pypi_base, step
from trainer import Trainer


@pypi_base(
    python="3.10.9",
    packages={
        "torchvision": "0.18.0",
        "torch": "2.3.0",
        "torchinfo": "1.8.0",
        "mlflow": "2.13.0",
    },
)
class TrainFlow(FlowSpec):
    @step
    def start(self):
        print("Starting the training flow")
        self.next(self.data_processing)

    @step
    def data_processing(self):
        print("Loading and processing data")
        self.batch_size = 32
        self.trainloader, self.testloader = load_and_transform_data(self.batch_size)
        self.next(self.train)

    @step
    def train(self):
        import mlflow
        import torch

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("FashionMNIST-Classification")
        print("Training the model")
        model = CNNModel()
        with mlflow.start_run():
            self.learning_rate = 3e-4
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.save_every = 2
            self.max_epochs = 4
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("device", self.device)
            mlflow.log_param("optimizer", self.optimizer)
            mlflow.log_param("max_epochs", self.max_epochs)
            mlflow.log_param("batch_size", self.batch_size)
            self.trainer = Trainer(
                model,
                self.trainloader,
                self.testloader,
                self.optimizer,
                self.device,
                self.save_every,
            )
            self.trainer.fit(self.max_epochs)
        self.next(self.end)

    @step
    def end(self):
        print("Ending the training flow")


if __name__ == "__main__":
    TrainFlow()
