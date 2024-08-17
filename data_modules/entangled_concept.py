# %%
from os.path import join
from typing import Tuple

import lightning as l
import numpy as np
from sklearn.calibration import column_or_1d
import torch as t
from einops import pack, rearrange
from jaxtyping import Float, Int
from pandas import read_csv
from skimage.io import imread_collection
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import Config

# %%


class EntangledConceptDataset(Dataset):
    def __init__(self, name: str, concept_name: str):
        match concept_name:
            case "distribute_three" | "progression":
                concepts = read_csv(join(name, concept_name + ".csv"), dtype="bool")
                self.config = Config(
                    np.array(
                        [
                            "color_0",
                            "position_0",
                            "color_1",
                            "position_1",
                            "color_2",
                            "position_2",
                        ]
                    ),
                    np.array([["0"], ["0"], ["0"], ["0"], ["0"], ["0"]]),
                    "general",
                )
                self.config.images_per_instance = 3
            case "blackbird":
                concepts = read_csv(join(name, "blackbird.csv"), dtype={"correct": "bool"})
                self.config = Config(
                    np.array([column for column in concepts.columns if column != "correct"]),
                    np.array(
                        [["0"] for column in concepts.columns if column != "correct"]
                    ),
                    "general",
                )
                self.config.images_per_instance = 9
            case _:
                concepts = read_csv(
                    join(name, "product_concepts.csv"), dtype="category"
                )
                self.config = Config(
                    np.array(concepts.columns),
                    np.array(
                        [concepts[column].cat.categories for column in concepts.columns]
                    ),
                    "domain_only",
                )

        match concept_name:
            case "correlated":
                concepts = (
                    (concepts["color"] == "red") & (concepts["shape"] == "circle")
                ) | ((concepts["color"] == "blue") & (concepts["shape"] == "square"))
                domains = ["color", "shape"]
            case "red":
                concepts = concepts["color"] == "red"
                domains = ["color"]
            case "red_and_circle":
                concepts = (concepts["color"] == "red") & (
                    concepts["shape"] == "circle"
                )
                domains = ["color", "shape"]
            case "red_or_blue":
                concepts = (concepts["color"] == "red") | (concepts["color"] == "blue")
                domains = ["color"]
            case "red_or_circle":
                concepts = (concepts["color"] == "red") | (
                    concepts["shape"] == "circle"
                )
                domains = ["color", "shape"]
            case "progression":
                concepts = concepts["correct"]
                domains = ["position_0", "position_1", "position_2"]
            case "distribute_three":
                concepts = concepts["correct"]
                domains = ["color_0", "color_1", "color_2"]
            case "blackbird":
                concepts = concepts["correct"]
                domains = self.config.instance_domains

        self.concepts = t.tensor(concepts, dtype=t.double)
        self.config.concept_domains = np.array(domains)

        print(
            f"Balance of {name}-{concept_name} dataset: {self.concepts.float().mean().item()} true"
        )

        transform = transforms.Compose([transforms.ToTensor()])
        
        match concept_name:
            case "distribute_three" | "progression":
                images_per_instance = 3
            case "blackbird":
                images_per_instance = 9
            case _:
                images_per_instance = 1

        self.instances = imread_collection(
            [join(name, f"{i}.png") for i in range(len(self.concepts) * images_per_instance)],
            conserve_memory=False,
        )
        self.instances = pack(
            [transform(image) for image in self.instances], "* color height width"
        )[0]

        if concept_name == "progression":  # Make puzzles column-major
            self.instances = rearrange(
                self.instances,
                "(puzzle row column) color height width -> (puzzle column) row color height width",
                row=3,
                column=3,
            )
        elif concept_name == "distribute_three":
            self.instances = rearrange(
                self.instances,
                "(puzzle row column) color height width -> (puzzle row) column color height width",
                row=3,
                column=3,
            )
        elif concept_name == "blackbird":
            self.instances = rearrange(
                self.instances,
                "(puzzle image) color height width -> puzzle image color height width",
                image=9,
            )

    def __len__(self) -> int:
        return len(self.concepts)

    def __getitem__(
        self, i: int
    ) -> Tuple[Float[Tensor, "batch color height width"], Int[Tensor, "batch"]]:
        return self.instances[i], self.concepts[i]


# %%

if __name__ == "__main__":
    # correlated

    # dataset = EntangledConceptDataset("blackbird/data/shapes/val", "correlated")
    # print(len(dataset))

    # rows

    dataset = EntangledConceptDataset("blackbird/data/balanced/val", "progression")
    print(len(dataset))

    x, y = dataset[0]
    print(y)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)
    for i in range(3):
        ax[i].imshow(x[i].permute(1, 2, 0))
        ax[i].axis("off")
    plt.show()

    # blackbird

    # dataset = EntangledConceptDataset("blackbird/data/balanced/val", "blackbird")
    # print(len(dataset))

    # x, y = dataset[0]
    # print(y)

    # fig, ax = plt.subplots(3, 3)
    # for i in range(3):
    #     for j in range(3):
    #         ax[i, j].imshow(x[3 * i + j].permute(1, 2, 0))
    # plt.show()

#%%

class EntangledConceptDataModule(l.LightningDataModule):
    def __init__(self, data_dir: str, type: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.type = type
        self.batch_size = batch_size

        self.num_workers = 4
        self.pin_memory = True

        self.train = EntangledConceptDataset(self.data_dir + "/train", self.type)
        self.val = EntangledConceptDataset(self.data_dir + "/val", self.type)
        self.test = EntangledConceptDataset(self.data_dir + "/test", self.type)

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
        None,
        Float[Tensor, "batch label"],
    ]:
        instance, concept = batch
        return instance, None, concept

# %%
