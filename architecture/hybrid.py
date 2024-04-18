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
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        loss = nn.functional.mse_loss(decoding, x)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_pred = self(x)
    #     loss = nn.functional.binary_cross_entropy(y_pred, y)
    #     self.log("val_loss", loss)
    #     return loss
    
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_pred = self(x)
    #     loss = nn.functional.binary_cross_entropy(y_pred, y)
    #     self.log("test_loss", loss)
    #     return loss
    
    


if __name__ == "__main__":
    model = Hybrid()
    model.cuda()
    print(model.device)