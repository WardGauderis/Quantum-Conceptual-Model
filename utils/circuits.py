# %%

from functools import partial
from re import T

import pennylane as qml
import torch as t
from jaxtyping import Float
from matplotlib import table
from pennylane.operation import AnyWires, Operation
from torch import Tensor


# qml.StronglyEntanglingLayers but edited to take into account the edge case of two wires
class StronglyEntanglingLayers(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, ranges=None, imprimitive=None, id=None):
        shape = qml.math.shape(weights)[-3:]

        if shape[1] != len(wires):
            raise ValueError(
                f"Weights tensor must have second dimension of length {len(wires)}; got {shape[1]}"
            )

        if shape[2] != 3:
            raise ValueError(
                f"Weights tensor must have third dimension of length 3; got {shape[2]}"
            )

        if ranges is None:
            if len(wires) > 1:
                # tile ranges with iterations of range(1, n_wires)
                ranges = tuple((l % (len(wires) - 1)) + 1 for l in range(shape[0]))
            else:
                ranges = (0,) * shape[0]
        else:
            ranges = tuple(ranges)
            if len(ranges) != shape[0]:
                raise ValueError(
                    f"Range sequence must be of length {shape[0]}; got {len(ranges)}"
                )
            for r in ranges:
                if r % len(wires) == 0:
                    raise ValueError(
                        f"Ranges must not be zero nor divisible by the number of wires; got {r}"
                    )

        self._hyperparameters = {
            "ranges": ranges,
            "imprimitive": imprimitive or qml.CNOT,
        }

        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(weights, wires, ranges, imprimitive):
        n_layers = qml.math.shape(weights)[0]
        wires = qml.wires.Wires(wires)
        op_list = []

        for l in range(n_layers):
            for i in range(len(wires)):
                op_list.append(
                    qml.Rot(
                        weights[..., l, i, 0],
                        weights[..., l, i, 1],
                        weights[..., l, i, 2],
                        wires=wires[i],
                    )
                )

            # Fix to avoid repeating gates when there are only two wires
            if len(wires) == 2:
                act_on = wires.subset([0, 1])
                op_list.append(imprimitive(wires=act_on))
            elif len(wires) > 1:
                for i in range(len(wires)):
                    act_on = wires.subset([i, i + ranges[l]], periodic_boundary=True)
                    op_list.append(imprimitive(wires=act_on))

        return op_list

    @staticmethod
    def shape(n_layers, n_wires):
        return n_layers, n_wires, 3


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
    domains: list[int], concept: Float[Tensor, "layer domain weights"]
):
    StronglyEntanglingLayers(concept, wires=domains, imprimitive=qml.CZ)


def create_circuit(
    type: str,
    num_domains: int,
    concept_domains: list[int],
):
    instance_domains = list(range(num_domains))

    match type:
        case "product":

            def circuit(
                instance: Float[Tensor, "domain weights batch"],
                concept: Float[Tensor, "domain weights batch"],
            ):
                instance_circuit(instance_domains, instance)
                product_concept_circuit(concept_domains, concept)

                return [qml.expval(qml.PauliZ(w)) for w in concept_domains]

        case "domain_only":

            def circuit(
                instance: Float[Tensor, "domain weights batch"],
                concept: Float[Tensor, "layer domain batch"],
            ):
                instance_circuit(instance_domains, instance)
                entangled_concept_circuit(concept_domains, concept)

                return [qml.expval(qml.PauliZ(w)) for w in concept_domains]

        case "general":

            def circuit(
                instance: Float[Tensor, "domain weights batch"],
                concept: Float[Tensor, "layer domain weights batch"],
            ):
                instance_circuit(instance_domains, instance)
                entangled_concept_circuit(
                    concept_domains + list(range(num_domains, num_domains + len(concept_domains))),
                    concept,
                )

                # TODO: clean up
                return [
                    qml.expval(qml.PauliZ(w + num_domains)) for w in range(len(concept_domains))
                ]

    return circuit


# %%

# # return [qml.probs(wires=w) for w in range(num_domains)]
# # return qml.probs(wires=range(num_domains))

# dev = qml.device("default.qubit", wires=4)

# c = create_circuit("domain_only", 4)

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
