from . import Encoder, Decoder
from . import VQC
import lightning as l
import torch as t
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy


class Hybrid(l.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        
        embedding_dim = 12
        
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim)
        
        self.vqc = VQC()

        # self.binary_accuracy = BinaryAccuracy()
    
    def configure_optimizers(self):
        return t.optim.Adam(self.parameters())
    
    def forward(self, x, y):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return decoding, y
    
    def loss(self, x, y, x_pred, y_pred):
        
        return nn.functional.mse_loss(x_pred, x) # + nn.functional.binary_cross_entropy(y_pred.float(), y.float())
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_pred, y_pred = self(x, y)
        loss = self.loss(x, y, x_pred, y_pred)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_pred, y_pred = self(x, y)
        loss = self.loss(x, y, x_pred, y_pred)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x_pred, y_pred = self(x, y)
        loss = self.loss(x, y, x_pred, y_pred)
        self.log("test_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x_pred, y_pred = self(x, y)
        return x_pred, y_pred
    
    


if __name__ == "__main__":
    model = Hybrid()
    model.cuda()
    print(model.device)