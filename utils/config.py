from ast import Str
from dataclasses import dataclass

import lightning as l
import numpy as np
import torch as t
from jaxtyping import Int
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from torch import Tensor
from tqdm import tqdm


@dataclass
class Config:
    domains: np.ndarray
    properties: np.ndarray

    concept_type: str
    concept_domains: np.ndarray

    decoder_multiplier: float = 0
    layers: int = 1
    
    multiple_images: bool = False

    def __init__(
        self,
        domains: np.ndarray,
        properties: np.ndarray,
        concept_type: str,
    ):
        self.domains = domains
        self.properties = properties

        self.concept_type = concept_type
        self.concept_domains = domains

        self.offsets = t.tensor(
            [i * self.num_properties for i in range(self.num_domains)],
            dtype=t.int,
        )

    @property
    def concept_embedding_dim(self) -> int:
        if self.is_product_concept:
            return 3
        elif self.is_general_concept:
            return self.num_domains * 6 * self.layers

        return self.num_concept_domains * 3 * self.layers

    @property
    def num_concept_domains(self) -> int:
        return len(self.concept_domains)

    @property
    def num_domains(self) -> int:
        return len(self.domains)

    @property
    def num_wires(self) -> int:
        if self.is_general_concept:
            return self.num_domains * 2
        return self.num_domains

    @property
    def num_properties(self) -> int:
        return self.properties.shape[1]

    @property
    def num_concepts(self) -> int:
        return self.num_domains * self.num_properties if self.is_product_concept else 1

    @property
    def is_product_concept(self) -> bool:
        return self.concept_type == "product"

    @property
    def is_entangled_concept(self) -> bool:
        return self.concept_type == "domain_only" or self.concept_type == "general"

    @property
    def is_general_concept(self) -> bool:
        return self.concept_type == "general"

    @property
    def concept_domain_indices(self) -> list[int]:
        return [
            i for i, domain in enumerate(self.domains) if domain in self.concept_domains
        ]

    def add_offset(
        self, concepts: Int[Tensor, "batch domain"]
    ) -> Int[Tensor, "batch domain"]:
        return concepts + self.offsets.to(concepts.device)

    def remove_offset(
        self, concepts: Int[Tensor, "batch domain"]
    ) -> Int[Tensor, "batch domain"]:
        return concepts - self.offsets.to(concepts.device)

    def decode_concept(self, concepts: Int[Tensor, "batch domain"]) -> np.ndarray:
        return self.properties.reshape(-1)[self.add_offset(concepts).cpu().numpy()]

    def decode_domain(self, domain: int) -> str:
        return self.domains[domain]

    def trainer(self, name: str, max_epochs: int = 100) -> l.Trainer:
        class NoValidationBar(TQDMProgressBar):
            def init_validation_tqdm(self):
                bar = tqdm(disable=True)
                return bar

        return l.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                (
                    ModelCheckpoint(
                        monitor="val_accuracy",
                        mode="max",
                        save_top_k=1,
                        filename=name + "-accuracy-{epoch:02d}",
                    )
                    if self.is_entangled_concept
                    else ModelCheckpoint(
                        monitor="val_loss",
                        mode="min",
                        save_top_k=1,
                        filename=name + "-loss-{epoch:02d}",
                    )
                ),
                NoValidationBar(),
            ],
        )
