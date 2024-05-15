from functools import partial

import pennylane as qml
import qutip as qt
import torch as t
from circuits import concept_circuit, instance_circuit
from jaxtyping import Float
from torch import Tensor

from einops import rearrange

dev = qml.device("default.qubit", wires=1)


def instance_state(instance: Float[Tensor, "weights batch"]):
    instance_circuit(1, instance)
    return qml.state()


def concept_state(concept: Float[Tensor, "weights batch"]):
    concept_circuit(1, concept)
    return qml.state()


instance_state = qml.QNode(instance_state, dev, interface="torch")
concept_state = qml.QNode(concept_state, dev, interface="torch")

color_map = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "olive",
    "cyan",
    "magenta",
]


def plot_concepts(
    b: qt.Bloch, concepts: Float[Tensor, "batch weights"], names: list[str]
):
    concepts = rearrange(concepts, "batch weights -> weights batch")

    states = [qt.Qobj(s) for s in concept_state(concepts)]
    b.add_states(states)

    for state, label, color in zip(states, names, color_map):
        state = (
            t.Tensor(
                [
                    qt.expect(qt.sigmax(), state),
                    qt.expect(qt.sigmay(), state),
                    qt.expect(qt.sigmaz(), state),
                ]
            )
            * 1.2
        )
        b.add_annotation(state, label, color=color, fontsize=12)


def plot_instances(
    b: qt.Bloch,
    instances: Float[Tensor, "batch weights"],
    concepts: Float[Tensor, "batch"],
):
    instances = rearrange(instances, "batch weights -> weights batch")

    b.point_marker = ["o"]
    b.point_color = color_map[concepts]
    b.point_size = [7]

    states = [qt.Qobj(s) for s in instance_state(instances)]
    b.add_states(states, kind="point", alpha=0.3)


def plot_representations(
    concepts: Float[Tensor, "batch weights"],
    instances: Float[Tensor, "batch weights"],
    instance_concepts: Float[Tensor, "batch"],
    concept_names: list[str],
):
    b = qt.Bloch(view=(-60, 30), figsize=(4, 4))
    b.font_size = 12
    b.vector_color = color_map

    plot_concepts(b, concepts, concept_names)
    plot_instances(b, instances, instance_concepts)

    b.make_sphere()
    b.render()
    b.show()
