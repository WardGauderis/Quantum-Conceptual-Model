import pennylane as qml
from jaxtyping import Float
from torch import Tensor

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