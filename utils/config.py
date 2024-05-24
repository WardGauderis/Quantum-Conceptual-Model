from ast import Str
from dataclasses import dataclass
import torch as t
from torch import Tensor
from jaxtyping import Int
import numpy as np


@dataclass
class Config:
    domains: np.ndarray
    properties: np.ndarray

    concept_embedding_dim: int

    decoder_multiplier: float = 0

    def __init__(
        self, domains: np.ndarray, properties: np.ndarray, concept_embedding_dim: int
    ):
        self.domains = domains
        self.properties = properties

        self.concept_embedding_dim = concept_embedding_dim

        self.offsets = t.tensor(
            [i * self.num_properties for i in range(self.num_domains)],
            dtype=t.int,
        )

    @property
    def num_domains(self) -> int:
        return self.properties.shape[0]

    @property
    def num_properties(self) -> int:
        return self.properties.shape[1]

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
