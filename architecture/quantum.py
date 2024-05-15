# %%

from functools import partial

import pennylane as qml
import torch as t
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from torch import Tensor, nn

from data_modules import Config

# %%

class VQC(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.concept_weights = nn.Embedding(
            config.num_domains * config.num_properties, config.concept_embedding_dim, scale_grad_by_freq=True
        )
        nn.init.uniform_(self.concept_weights.weight, 0, 2 * t.pi)

        dev = qml.device("default.qubit", wires=config.num_domains)
        self.circuit = qml.QNode(partial(circuit, config.num_domains), dev, interface="torch")
        # self.circuit = qml.compile(self.circuit)

    def forward(
        self, instance: Float[Tensor, "batch encoding"], concept_index: Int[Tensor, ""]
    ) -> Float[Tensor, "batch domain"]:
        concept = rearrange(
            self.concept_weights(concept_index),
            "batch domain weights -> domain weights batch",
            weights=3,
        )

        instance = rearrange(
            instance, "batch (domain weights) -> domain weights batch", weights=3
        )

        probabilities = t.stack(self.circuit(instance, concept)) / 2 + 0.5
        return rearrange(probabilities, "domain batch -> batch domain")


# %%
