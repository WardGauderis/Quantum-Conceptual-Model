# %%

import torch as t
from torch import nn, Tensor
import pennylane as qml
import torch
from jaxtyping import Float, Int
from einops import rearrange, reduce, repeat
from functools import partial

# %%


def instance_circuit(num_domains: int, instance: Float[Tensor, "domain weights batch"]):
    qml.broadcast(
        qml.Rot, wires=range(num_domains), pattern="single", parameters=instance
    )


def concept_circuit(num_domains: int, concept: Float[Tensor, "domain weights batch"]):
    qml.broadcast(
        qml.adjoint(qml.Rot),
        wires=range(num_domains),
        pattern="single",
        parameters=concept,
    )
    # qml.StronglyEntanglingLayers(concept, wires=range(num_domains), imprimitive=qml.CZ)


def circuit(
    num_domains: int,
    instance: Float[Tensor, "domain weights batch"],
    concept: Float[Tensor, "domain weights batch"],
):
    instance_circuit(num_domains, instance)
    concept_circuit(num_domains, concept)

    return [qml.expval(qml.PauliZ(w)) for w in range(num_domains)]

    # return [qml.probs(wires=w) for w in range(num_domains)]
    # return qml.probs(wires=range(num_domains))


# dev = qml.device("default.qubit", wires=4)

# circuit = partial(circuit, 4)

# # c = qml.compile()(circuit)
# c = qml.QNode(circuit, dev, interface="torch", diff_method="backprop", cachesize=40000)

# # c = qml.QNode(circuit, dev, interface="torch")
# # c = qml.compile(c)

# # print(qml.draw(c)(torch.randn(4, 3), torch.randn(4, 4, 3)))
# qml.draw_mpl(c, style="black_white", expansion_strategy="device")(
#     torch.randn(4, 3), torch.randn(4, 3)
# )

# params = torch.randn(4, 3, 10, device="cuda")
# # params = repeat(params, "domain weights -> domain weights batch", batch=10)
# params2 = torch.randn(4, 3, 10, device="cuda")

# torch.stack(c(params, params2)) / 2 + 0.5

# %%


class VQC(nn.Module):
    def __init__(self):
        super().__init__()

        num_concepts = 12
        num_domains = 4
        embedding_dim = 3

        self.concept_weights = nn.Embedding(
            num_concepts, embedding_dim, scale_grad_by_freq=True
        )
        nn.init.uniform_(self.concept_weights.weight, 0, 2 * torch.pi)

        dev = qml.device("default.qubit", wires=num_domains)
        self.circuit = qml.QNode(partial(circuit, num_domains), dev, interface="torch")
        # self.circuit = qml.compile(self.circuit)

    def forward(
        self, instance: Float[Tensor, "batch encoding"], concept_index: Int[Tensor, ""]
    ) -> Float[Tensor, "batch label"]:
        concept = rearrange(
            self.concept_weights(concept_index),
            "batch domain weights -> domain weights batch",
            weights=3,
        )

        instance = rearrange(
            instance, "batch (domain weights) -> domain weights batch", weights=3
        )

        probabilities = torch.stack(self.circuit(instance, concept)) / 2 + 0.5
        return reduce(probabilities, "probability batch -> batch", "prod")


# %%
