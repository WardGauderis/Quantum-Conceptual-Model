# %%

from ast import List
from functools import partial
from re import T

from matplotlib import table
import pennylane as qml
import torch as t
from jaxtyping import Float
from torch import Tensor


def instance_circuit(
    domains: list[int], instance: Float[Tensor, "domain weights batch"]
):
    qml.broadcast(qml.Rot, wires=domains, pattern="single", parameters=instance)


def product_concept_circuit(
    domains: list[int], concept: Float[Tensor, "domain weights batch"]
):
    qml.broadcast(
        qml.adjoint(qml.Rot),
        wires=domains,
        pattern="single",
        parameters=concept,
    )


def entangled_concept_circuit(
    domains: list[int], concept: Float[Tensor, "repeat domain weights"]
):
    qml.StronglyEntanglingLayers(concept, wires=domains, imprimitive=qml.CZ)


def create_circuit(
    type: str,
    domains: list[int],
):
    match type:
        case "product_concept":

            def circuit(
                instance: Float[Tensor, "domain weights batch"],
                concept: Float[Tensor, "domain weights batch"],
            ):
                instance_circuit(domains, instance)
                product_concept_circuit(domains, concept)

                return [qml.expval(qml.PauliZ(w)) for w in domains]

        case "entangled_concept":

            def circuit(
                instance: Float[Tensor, "domain weights batch"],
                concept: Float[Tensor, "repeat domain batch"],
            ):
                instance_circuit(domains, instance)
                entangled_concept_circuit(domains, concept)

                return [qml.expval(qml.PauliZ(w)) for w in domains]

    return circuit


# %%

# # return [qml.probs(wires=w) for w in range(num_domains)]
# # return qml.probs(wires=range(num_domains))

# dev = qml.device("default.qubit", wires=4)

# c = create_circuit("entangled_concept", 4)

# # c = qml.compile()(circuit)
# c = qml.QNode(c, dev, interface="torch", diff_method="backprop", cachesize=40000)

# # c = qml.QNode(circuit, dev, interface="torch")
# # c = qml.compile(c)

# # print(qml.draw(c)(torch.randn(4, 3), torch.randn(4, 4, 3)))
# qml.draw_mpl(c, style="black_white", expansion_strategy="device")(
#     t.randn(4, 3), t.randn(5, 4, 3)
# )

# params = t.randn(4, 3, 10, device="cuda")
# # params = repeat(params, "domain weights -> domain weights batch", batch=10)
# params2 = t.randn(4, 3, 10, device="cuda")
# params2 = t.randn(5, 4, 3, device="cuda")

# t.stack(c(params, params2)) / 2 + 0.5

# %%
