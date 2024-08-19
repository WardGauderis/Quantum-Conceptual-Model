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
                config.num_instance_domains,
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
        instance = t.zeros(self.config.num_instance_domains, 3)

        if self.config.is_product_concept:
            concept = rearrange(
                self.concept_weights(
                    t.zeros(self.config.num_instance_domains, dtype=t.long)
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
        
        # TODO: switch to qml.probs instead of qml.expval
        # This requires removing the exp -> probs hack and domain probability multiplication with on/off switch for per domain probabilities
        
        # probabilities = self.circuit(instance, concept)[:, 0]
        # return rearrange(probabilities, "domain batch -> batch domain")
        # color_probs = rearrange(
        #     color_probs,
        #     "(puzzle row color) -> puzzle row color",
        #     row=3,
        #     color=len(color_encodings),
        # )

        probabilities = t.stack(self.circuit(instance, concept)) / 2 + 0.5
        return rearrange(probabilities, "domain batch -> batch domain")


# %%
