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

from einops import repeat, pack

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
            "product_concept",
            3,
        )

        for column in self.concepts.columns:
            self.concepts[column] = self.concepts[column].cat.codes

        self.concepts = t.tensor(self.concepts.values, dtype=t.long)

        transform = transforms.Compose([transforms.ToTensor()])

        instances = imread_collection(
            [join(name, f"{i}.png") for i in range(len(self.concepts))],
            conserve_memory=False,
        )
        self.instances = pack([transform(image) for image in instances], "* color height width")[0]

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
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=len(self.val),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=len(self.test),
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

        if self.trainer.predicting:  # type: ignore
            return instance, concept, t.ones(instance.shape[0], dtype=t.double)

        num_negatives = 1

        true_concept = repeat(
            concept, "batch domain -> (copies batch) domain", copies=num_negatives
        )

        false_concept = t.randint_like(
            true_concept,
            0,
            self.config.num_properties - 1,  # -1 to avoid the true concept
        )
        false_concept[false_concept >= true_concept] += 1

        new_instance = repeat(
            instance,
            "batch color height width -> (copies batch) color height width",
            copies=num_negatives + 1,
        )
        new_concept = pack([concept, false_concept], "* domain")[0]
        new_concept = self.config.add_offset(new_concept)
        new_label = pack(
            [
                t.ones(concept.shape[0], dtype=t.double, device=instance.device),
                t.zeros(false_concept.shape[0], dtype=t.double, device=instance.device),
            ],
            "*",
        )[0]
        return new_instance, new_concept, new_label
