import os
from typing import Tuple

import mlflow
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        save_every: int,
    ) -> None:
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.device = device
        self.save_every = save_every

    def _run_batch(self, samples: torch.Tensor, labels: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        output = self.model(samples)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch: int) -> None:
        batch_size = len(next(iter(self.train_data))[0])
        total_loss = 0.0
        for samples, labels in self.train_data:
            samples = samples.to(self.device)
            labels = labels.to(self.device)
            loss = self._run_batch(samples, labels)
            total_loss += loss
        avg_loss = total_loss / len(self.train_data)
        print(
            f"Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_data)} | Training Loss: {avg_loss}"
        )

    def _save_checkpoint(self, epoch: int) -> None:
        ckp = self.model.state_dict()
        PATH = f"../checkpoints/checkpoint_{epoch}.pt"
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def _evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for samples, labels in data_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                output = self.model(samples)
                loss = F.cross_entropy(output, labels)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)
        return accuracy, avg_loss

    def fit(self, max_epochs: int) -> None:
        val_losses = []
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            _, val_loss = self._evaluate(self.val_data)
            val_losses.append(val_loss)
            print(f"Epoch {epoch} | Validation Loss: {val_loss}")
        test_accuracy, _ = self._evaluate(self.val_data)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.pytorch.log_model(
            self.model,
            "models",  # Artifact path after "artifacts" folder
            registered_model_name="pytorch-cnn-model",
        )
        print(f"Test Accuracy: {test_accuracy} | Final Validation Loss: {val_losses[-1]}")
