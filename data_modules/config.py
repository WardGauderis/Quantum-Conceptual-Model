from dataclasses import dataclass
import torch as t
from torch import Tensor
from jaxtyping import Int


@dataclass
class Config:
    num_domains: int
    num_properties: int
    concept_embedding_dim: int
    
    decoder_multiplier: float = 0

    def __init__(self, num_domains: int, num_properties: int, concept_embedding_dim: int):
        self.num_domains = num_domains
        self.num_properties = num_properties
        self.concept_embedding_dim = concept_embedding_dim

        self.offsets = t.tensor(
            [i * self.num_properties for i in range(self.num_domains)],
            dtype=t.int,
        )

    def add_offset(
        self, concepts: Int[Tensor, "batch domain"]
    ) -> Int[Tensor, "batch domain"]:
        return concepts + self.offsets.to(concepts.device)

    def remove_offset(
        self, concepts: Int[Tensor, "batch domain"]
    ) -> Int[Tensor, "batch domain"]:
        return concepts - self.offsets.to(concepts.device)