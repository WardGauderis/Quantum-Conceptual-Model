from ast import Tuple
import lightning as l
import torch as t
from jaxtyping import Float
from typing import Tuple
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy

from . import VQC, Decoder, Encoder


class Hybrid(l.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        
        embedding_dim = 12
        
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim)
        
        self.vqc = VQC()

        # self.binary_accuracy = BinaryAccuracy()
    
    def configure_optimizers(self) -> t.optim.Optimizer:
        return t.optim.Adam(self.parameters())
    
    def forward(self, x, y) -> Tuple[Float[Tensor, "batch color height width"], Float[Tensor, "batch label"]]:
        encoding = self.encoder(x)
        y_pred = self.vqc(encoding)
        x_pred = self.decoder(encoding)
        return x_pred, y_pred
    
    def loss(self, x, y, x_pred, y_pred) -> Float[Tensor, ""]:
        return nn.functional.mse_loss(x_pred, x)  + nn.functional.binary_cross_entropy(y_pred, y)
        
    def training_step(self, batch, batch_idx) -> Float[Tensor, ""]:
        x, y = batch
        x_pred, y_pred = self(x, y)
        loss = self.loss(x, y, x_pred, y_pred)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx) -> Float[Tensor, ""]:
        x, y = batch
        x_pred, y_pred = self(x, y)
        loss = self.loss(x, y, x_pred, y_pred)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx) -> Float[Tensor, ""]:
        x, y = batch
        x_pred, y_pred = self(x, y)
        loss = self.loss(x, y, x_pred, y_pred)
        self.log("test_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx) -> Tuple[Float[Tensor, "batch color height width"], Float[Tensor, "batch label"]]:
        x, y = batch
        x_pred, y_pred = self(x, y)
        return x_pred, y_pred
    
    


if __name__ == "__main__":
    model = Hybrid()
    model.cuda()
    print(model.device)