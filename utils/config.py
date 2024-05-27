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

    concept_type: str
    concept_domains: np.ndarray
    concept_embedding_dim: int

    decoder_multiplier: float = 0

    def __init__(
        self,
        domains: np.ndarray,
        properties: np.ndarray,
        concept_type: str,
        concept_embedding_dim: int,
    ):
        self.domains = domains
        self.properties = properties

        self.concept_type = concept_type
        self.concept_domains = domains
        self.concept_embedding_dim = concept_embedding_dim

        self.offsets = t.tensor(
            [i * self.num_properties for i in range(self.num_instance_domains)],
            dtype=t.int,
        )

    @property
    def num_concept_domains(self) -> int:
        return len(self.concept_domains)

    @property
    def num_instance_domains(self) -> int:
        return self.properties.shape[0]

    @property
    def num_properties(self) -> int:
        return self.properties.shape[1]

    @property
    def num_concepts(self) -> int:
        return (
            self.num_instance_domains * self.num_properties
            if self.is_product_concept
            else 1
        )

    @property
    def is_product_concept(self) -> bool:
        return self.concept_type == "product_concept"

    @property
    def is_entangled_concept(self) -> bool:
        return self.concept_type == "entangled_concept"

    @property
    def concept_domain_indices(self) -> list[int]:
        return [
            i for i, domain in enumerate(self.domains) if domain in self.concept_domains
        ]

    @property
    def instance_domain_indices(self) -> list[int]:
        return list(range(self.num_instance_domains))

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
