import torch
from torch import nn
from torch.utils.data import DataLoader

from torchmetrics.classification import BinaryAccuracy

from os import Path

class Hybrid(nn.Module):
	def __init__(self, name: str, device: torch.device):
		super().__init__()

		self.name = name
		self.epochs = 0
		self.train_loss = []
		self.val_loss = []
		self.val_accuracy = []

		self.criterion = nn.BCELoss()
		self.binary_accuracy = BinaryAccuracy()

		self.optimiser = torch.optim.Adam(self.parameters())
		self.device = device
  
    def save(self):
        Path("checkpoints").mkdir(parents=True, exist_ok=True)
        
        torch.save({
   			"train_loss": self.train_loss,
			"val_loss": self.val_loss,
			"val_accuracy": self.val_accuracy
   
			"model_state_dict": self.state_dict(),
			"optimizer_state_dict": self.optimiser.state_dict(),

		}, f"checkpoints/{self.name}_{self.epochs}.pt")
        
        
    def load(self, epochs: int = None):
        checkpoint = torch.load(f"checkpoints/{self.name}_{epochs}.pt", map_location=self.device)
        
        self.epochs = epochs
        self.train_loss = checkpoint["train_loss"]
        self.val_loss = checkpoint["val_loss"]
        self.val_accuracy = checkpoint["val_accuracy"]
        
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(x, y)
        return loss
        
        
    def train_model(self, epochs: int, train: DataLoader, val: DataLoader):
        self.train()
        
        time = datetime.now()
        
        for epoch in range(epochs):
            train_loss = 0
            
            for i, (x, concept, y) in enumerate(train):
                self.optimiser.zero_grad()
                
                y_pred, encoding = self(x, concept)
                
                loss = self.loss(y_pred, encoding, y)
                loss.backward()
                self.optimiser.step()
                
                train_loss += loss.item()
                self.train_loss.append(loss.item())

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Hybrid("Hybrid", device)
    print(model)