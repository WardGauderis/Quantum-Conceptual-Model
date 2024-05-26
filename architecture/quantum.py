# %%

from functools import partial

import pennylane as qml
import torch as t
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from torch import Tensor, nn

from utils import Config, create_circuit

# %%


class VQC(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.concept_weights = nn.Embedding(
            config.num_domains * config.num_properties,
            config.concept_embedding_dim,
            scale_grad_by_freq=True,
        )
        nn.init.uniform_(self.concept_weights.weight, 0, 2 * t.pi)

        dev = qml.device("default.qubit", wires=config.num_domains)
        self.circuit = qml.QNode(
            create_circuit(config.concept_type, config.concept_domain_indices),
            dev,
            interface="torch",
            diff_method="backprop",
            cachesize=40000,
        )
        # self.circuit = qml.compile(self.circuit)
        self.num_concept_domains = config.num_concept_domains

    def forward(
        self,
        instance: Float[Tensor, "batch domain weights"],
        concept_index: Int[Tensor, ""],
    ) -> Float[Tensor, "batch domain"]:
        if concept_index is None:
            concept = rearrange(
                self.concept_weights.weight,
                "(repeat domain weights) -> repeat domain weights",
                domain=self.num_concept_domains,
                weights=3,
            )
        else:
            concept = rearrange(
                self.concept_weights(concept_index),
                "batch domain weights -> domain weights batch",
            )
            
        instance = rearrange(instance, "batch domain weights -> domain weights batch")

        probabilities = t.stack(self.circuit(instance, concept)) / 2 + 0.5
        return rearrange(probabilities, "domain batch -> batch domain")


# %%
