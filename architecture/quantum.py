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
            config.num_concepts,
            config.concept_embedding_dim,
            scale_grad_by_freq=True,
        )
        if config.is_product_concept:
            nn.init.uniform_(self.concept_weights.weight, 0, 2 * t.pi)
        else:
            nn.init.uniform_(self.concept_weights.weight, 0, 1) # TODO: check

        dev = qml.device("default.qubit", wires=config.num_wires)
        self.circuit = qml.QNode(
            create_circuit(
                config.concept_type,
                config.num_domains,
                config.concept_domain_indices,
            ),
            dev,
            interface="torch",
            diff_method="backprop",
            cachesize=40000,
        )
        # self.circuit = qml.compile(self.circuit)

        self.config = config

    def plot(self):
        instance = t.zeros(self.config.num_domains, 3)

        if self.config.is_product_concept:
            concept = rearrange(
                self.concept_weights(
                    t.zeros(self.config.num_domains, dtype=t.long)
                ),
                "domain weights -> domain weights",
            )
        else:
            concept = rearrange(
                self.concept_weights.weight,
                "none (layer domain weights) -> (none layer) domain weights",
                layer = self.config.layers,
                weights=3,
            )

        qml.draw_mpl(self.circuit, style="black_white", expansion_strategy="device")(
            instance, concept
        )

    def forward(
        self,
        instance: Float[Tensor, "batch domain weights"],
        concept_index: Int[Tensor, "batch domain"],
    ) -> Float[Tensor, "batch domain"]:
        if self.config.is_product_concept:
            concept = rearrange(
                self.concept_weights(concept_index),
                "batch domain weights -> domain weights batch",
            )
        else:
            concept = rearrange(
                self.concept_weights.weight,
                "none (layer domain weights) -> (none layer) domain weights",
                layer=self.config.layers,
                weights=3,
            )

        instance = rearrange(instance, "batch domain weights -> domain weights batch")

        probabilities = t.stack(self.circuit(instance, concept)) / 2 + 0.5
        return rearrange(probabilities, "domain batch -> batch domain")


# %%
