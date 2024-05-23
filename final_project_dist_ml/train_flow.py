from metaflow import FlowSpec, step, pypi_base
from get_data import load_and_process_data
from get_model import CNNModel
from trainer import Trainer

@pypi_base(
    python="3.10.9",
    packages={
        "torchvision": "0.18.0",
        "torch": "2.3.0",
    },
)
class TrainFlow(FlowSpec):

    @step
    def start(self):
        print('Starting the training flow')
        self.next(self.load_data)
    
    @step 
    def load_data(self):
        print('Loading and processing data')
        self.trainloader, self.testloader = load_and_process_data()
        self.next(self.train)
    
    @step 
    def train(self):
        import torch
        print('Training the model')
        model = CNNModel()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_every = 2
        self.max_epochs = 4
        self.trainer = Trainer(model, self.trainloader, self.testloader, self.optimizer, self.device, self.save_every)
        self.trainer.train(self.max_epochs)
        self.next(self.end)

    @step 
    def end(self):
        print('Ending the training flow')
        
if __name__ == "__main__":

    TrainFlow()