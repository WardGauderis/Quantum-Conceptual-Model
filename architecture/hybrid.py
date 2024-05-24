from ast import Tuple
from typing import Tuple

import lightning as l
import torch as t
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
from einops import reduce

from utils import Config

from architecture import VQC, Decoder, Encoder


class Hybrid(l.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.vqc = VQC(config)

        self.accuracy = BinaryAccuracy()

    def configure_optimizers(self) -> t.optim.Optimizer:
        return t.optim.Adam(self.parameters())

    def forward(
        self, x, index, product=True
    ) -> Tuple[
        Float[Tensor, "batch color height width"], Float[Tensor, "batch domain"]
    ]:
        encoding = self.encoder(x)
        y_pred = self.vqc(encoding, index)
        if product:
            y_pred = reduce(y_pred, "batch domain -> batch", "prod")
        x_pred = self.decoder(encoding)
        return x_pred, y_pred

    def loss(self, x, y, x_pred, y_pred) -> Float[Tensor, ""]:
        y_pred = y_pred.clamp(0, 1)
        return (
            nn.functional.binary_cross_entropy(y_pred, y)
            + nn.functional.mse_loss(x_pred, x) * self.config.decoder_multiplier
        )

    def training_step(self, batch, batch_idx) -> Float[Tensor, ""]:
        x, index, y = batch
        x_pred, y_pred = self(x, index)
        loss = self.loss(x, y, x_pred, y_pred)
        self.log("train_loss", loss)

        return loss
    
    def evaluate_indices(self, x, index):
        # take first correct half of the batch
        x = x[: x.shape[0] // 2]
        index = index[: index.shape[0] // 2]

        all_indices = t.empty((*index.shape, self.config.num_properties), device=self.device)
        for i in range(self.config.num_properties):
            fake_index = self.config.add_offset(t.full_like(index, i))
            _, all_indices[..., i] = self(x, fake_index, product=False)
        index_pred = t.argmax(all_indices, dim=-1)

        index_accuracy = (
            t.sum(index_pred == self.config.remove_offset(index), dim=0)
            / index.shape[0]
        )
        
        return index_accuracy

    def validation_step(self, batch, batch_idx) -> Float[Tensor, ""]:
        x, index, y = batch
        x_pred, y_pred = self(x, index)

        loss = self.loss(x, y, x_pred, y_pred)
        self.log("val_loss", loss, prog_bar=True)
        
        if self.config.num_properties == 1:
            accuracy = self.accuracy(y_pred, y)
            self.log("val_accuracy", accuracy, prog_bar=True)
        else:
            index_accuracy = self.evaluate_indices(x, index)
            for i in range(self.config.num_domains):
                self.log(f"val_index_accuracy_{i}", index_accuracy[i], prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx) -> Float[Tensor, ""]:
        x, index, y = batch
        x_pred, y_pred = self(x, index)

        loss = self.loss(x, y, x_pred, y_pred)
        self.log("test_loss", loss)

        if self.config.num_properties == 1:
            accuracy = self.accuracy(y_pred, y)
            self.log("val_accuracy", accuracy, prog_bar=True)
        else:
            index_accuracy = self.evaluate_indices(x, index)
            for i in range(self.config.num_domains):
                self.log(f"val_index_accuracy_{i}", index_accuracy[i], prog_bar=True)

        return loss

    def predict_step(
        self, batch, batch_idx
    ) -> Tuple[Float[Tensor, "batch color height width"], Float[Tensor, "batch label"]]:
        x, index, y = batch
        x_pred, y_pred = self(x, index)
        return x_pred, y_pred


if __name__ == "__main__":
    model = Hybrid()
    model.cuda()
    print(model.device)
