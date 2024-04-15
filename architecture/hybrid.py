import classical
import lightning as l
import torch as t
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy


class Hybrid(l.LightningModule):
    def __init__(self):
        super().__init__()
        
        embedding = 12
        
        self.encoder = classical.Encoder(embedding)
        self.decoder = classical.Decoder(embedding)
        
        self.

        # self.binary_accuracy = BinaryAccuracy()
    
    def configure_optimizers(self):
        return t.optim.Adam(self.parameters())
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.binary_cross_entropy(y_pred, y)
        self.log("train_loss", loss)
        return loss


if __name__ == "__main__":
    model = Hybrid()
    model.cuda()
    print(model.device)