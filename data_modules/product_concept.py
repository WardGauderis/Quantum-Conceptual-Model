# %%

from os.path import join
from typing import Tuple

import lightning as l
import numpy as np
import torch as t
from jaxtyping import Float, Int
from pandas import read_csv
from skimage.io import imread_collection
from sympy import false
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import Config

# %%


class ProductConceptDataset(Dataset):
    def __init__(self, name: str):
        self.concepts = read_csv(join(name, "product_concepts.csv"), dtype="category")

        self.config = Config(
            np.array(self.concepts.columns),
            np.array(
                [
                    self.concepts[column].cat.categories
                    for column in self.concepts.columns
                ]
            ),
            3,
        )

        for column in self.concepts.columns:
            self.concepts[column] = self.concepts[column].cat.codes

        self.concepts = t.tensor(self.concepts.values, dtype=t.long)

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.instances = imread_collection(
            [join(name, f"{i}.png") for i in range(len(self.concepts))],
            conserve_memory=False,
        )
        self.instances = t.stack([self.transform(image) for image in self.instances])

    def __len__(self) -> int:
        return len(self.concepts)

    def __getitem__(
        self, i
    ) -> Tuple[Float[Tensor, "color height width"], Int[Tensor, "domain"]]:
        return self.instances[i], self.concepts[i]


class ProductConceptDataModule(l.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.num_workers = 4
        self.pin_memory = True

        self.train = ProductConceptDataset(self.data_dir + "/train")
        self.val = ProductConceptDataset(self.data_dir + "/val")
        self.test = ProductConceptDataset(self.data_dir + "/test")

        self.config = self.train.config

    def train_dataloader(self):
        return DataLoader(
            self.train,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            len(self.val),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            len(self.test),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=len(self.test),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def on_after_batch_transfer(
        self,
        batch: Tuple[
            Float[Tensor, "batch color height width"], Int[Tensor, "batch domain"]
        ],
        dataloader_idx: int,
    ) -> Tuple[
        Float[Tensor, "batch color height width"],
        Int[Tensor, "batch domain"],
        Float[Tensor, "batch label"],
    ]:
        instance, concept = batch
        
        if self.trainer.predicting:
            return instance, concept, t.ones(instance.shape[0], dtype=t.double)
          
        false_concept = t.randint_like(
            concept,
            0,
            self.config.num_properties - 1,  # -1 to avoid the true concept
        )
        false_concept[false_concept >= concept] += 1

        concept = self.config.add_offset(concept)
        false_concept = self.config.add_offset(false_concept)

        instance = t.cat([instance, instance])
        concept = t.cat([concept, false_concept])
        label = t.cat(
            [
                t.ones(instance.shape[0] // 2, dtype=t.double, device=instance.device),
                t.zeros(instance.shape[0] // 2, dtype=t.double, device=instance.device),
            ]
        )
        return instance, concept, label
