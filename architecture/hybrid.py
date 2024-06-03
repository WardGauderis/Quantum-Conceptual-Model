from ast import Tuple
from typing import Tuple

import lightning as l
import torch as t
from einops import rearrange, reduce
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy

from architecture import VQC, Decoder, Encoder
from utils import Config


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

    def forward(self, x, index, product=True) -> Tuple[
        Float[Tensor, "batch color height width"],
        Float[Tensor, "batch domain"],
        Float[Tensor, "batch domain weights"],
    ]:
        encoding = self.encoder(x, self.config.images_per_instance)
        y_pred = self.vqc(encoding, index)
        if product:
            y_pred = reduce(y_pred, "batch domain -> batch", "prod")
        # x_pred = self.decoder(encoding)
        return x, y_pred, encoding

    def loss(self, x, y, x_pred, y_pred) -> Float[Tensor, ""]:
        y_pred = y_pred.clamp(0, 1)
        return (
            nn.functional.binary_cross_entropy(y_pred, y)
            # + nn.functional.mse_loss(x_pred, x) * self.config.decoder_multiplier
        )

    def evaluate_indices(self, x, index):
        # take first correct half of the batch
        x = x[: x.shape[0] // 2]
        index = index[: index.shape[0] // 2]

        all_indices = t.empty(
            (*index.shape, self.config.num_properties), device=self.device
        )
        for i in range(self.config.num_properties):
            fake_index = self.config.add_offset(t.full_like(index, i))
            _, all_indices[..., i], _ = self(x, fake_index, product=False)
        index_pred = t.argmax(all_indices, dim=-1)

        index_accuracy = (
            t.sum(index_pred == self.config.remove_offset(index), dim=0)
            / index.shape[0]
        )

        return index_accuracy
    
    def step(self, batch, name: str) -> Float[Tensor, ""]:
        x, index, y = batch
        x_pred, y_pred, _ = self(x, index)
        loss = self.loss(x, y, x_pred, y_pred)
        self.log(f"{name}_loss", loss, prog_bar=True)
        
        
        # if name == "train":
        #     return loss
        
        if self.config.is_product_concept:
            index_accuracy = self.evaluate_indices(x, index)
            for i in range(self.config.num_domains):
                self.log(f"{name}_accuracy_{i}", index_accuracy[i])
            self.log(f"{name}_accuracy", t.mean(index_accuracy), prog_bar=True)
        else:
            accuracy = self.accuracy(y_pred, y)
            self.log(f"{name}_accuracy", accuracy, prog_bar=True)
        
        return loss

    def training_step(self, batch, batch_idx) -> Float[Tensor, ""]:
        return self.step(batch, "train")
        
    def validation_step(self, batch, batch_idx) -> Float[Tensor, ""]:
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx) -> Float[Tensor, ""]:
        return self.step(batch, "test")

    def predict_step(self, batch, batch_idx) -> Tuple[
        Float[Tensor, "batch color height width"],
        Float[Tensor, "batch label"],
        Float[Tensor, "batch domain weights"],
    ]:
        x, index, y = batch
        x_pred, y_pred, encoding = self(x, index)
        return x_pred, y_pred, encoding
