# %%

from random import randint

import numpy as np
import pennylane as qml
import torch as t
from einops import rearrange, reduce, repeat
from qiskit.result import marginal_counts
from qiskit.visualization import plot_distribution
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider.fake_provider import FakeKyiv

from architecture import Hybrid
from architecture.quantum import VQC
from data_modules import EntangledConceptDataModule
from utils.circuits import create_circuit
import matplotlib.pyplot as plt

t.set_grad_enabled(False)

# %% BLACKBIRD DATASET AND MODELS

blackbird = EntangledConceptDataModule("blackbird/data/balanced", "blackbird", 2**6)
config = blackbird.config

blackbird_model = Hybrid.load_from_checkpoint(
    "lightning_logs/blackbird/checkpoints/blackbird-selection-epoch=44.ckpt"
)
distribute_three_model = Hybrid.load_from_checkpoint(
    "lightning_logs/distribute_three/checkpoints/distribute_three-selection-epoch=42.ckpt"
)
progression_model = Hybrid.load_from_checkpoint(
    "lightning_logs/progression/checkpoints/progression-selection-epoch=98.ckpt"
)
# %%
dataloader = blackbird.test_dataloader()
instances, labels = next(iter(dataloader))
instances, labels = instances.to(blackbird_model.device), labels.to(
    blackbird_model.device
)
property_labels = np.array(dataloader.dataset.properties)
property_labels = rearrange(
    property_labels, "puzzle (row column type) -> puzzle type row column", row=3, type=2
)
property_labels = property_labels[labels.cpu().numpy() == 1]

encoder = blackbird_model.encoder
distribute_three = distribute_three_model.vqc
progression = progression_model.vqc

# %%

encodings = encoder(instances, config.images_per_instance)
encodings = rearrange(
    encodings,
    "puzzle (row column domain) encoding -> puzzle row column domain encoding",
    row=3,
    column=3,
    domain=2,
)

row = rearrange(
    encodings,
    "puzzle row column domain encoding -> (puzzle row) (column domain) encoding",
)
column = rearrange(
    encodings,
    "puzzle row column domain encoding -> (puzzle column) (row domain) encoding",
)

row_probs = distribute_three(row, None)
row_probs = reduce(row_probs, "(puzzle row) domain -> puzzle", "prod", row=3)

column_probs = progression(column, None)
column_probs = reduce(
    column_probs, "(puzzle column) domain -> puzzle", "prod", column=3
)

probs = t.stack([row_probs, column_probs], dim=1)
probs = reduce(probs, "puzzle orientation -> puzzle", "prod")


# TODO: plot instead
def print_stats(probs, labels):
    plt.hist(probs.cpu().detach().flatten(), bins=50)
    plt.xlabel('Property Probability')
    plt.show()
    
    print("Positive")
    print(f"min: {probs[labels == 1].min()}", end=" ")
    print(f"mean: {probs[labels == 1].mean()}", end=" ")
    print(f"std: {probs[labels == 1].std()}")
    print("Negative")
    print(f"max: {probs[labels == 0].max()}", end=" ")
    print(f"mean: {probs[labels == 0].mean()}", end=" ")
    print(f"std: {probs[labels == 0].std()}\n")


print("Row")
print_stats(row_probs, labels)
print("Column")
print_stats(column_probs, labels)
print("Puzzle")
print_stats(probs, labels)

preds = probs > 0.5**6
accuracy = (preds == labels).float().mean()
print(f"Blackbird detection accuracy: {accuracy}")

# %%

properties = blackbird_model.config.properties
offsets = blackbird_model.config.offsets

property_encodings = blackbird_model.vqc.concept_weights.weight.detach()
color_encodings = property_encodings[offsets[0] : offsets[0] + len(properties[0])]
position_encodings = property_encodings[offsets[1] : offsets[1] + len(properties[1])]

# %% COLOR AND POSITION CIRCUITS


# influences position circuit and selects color constraint
missing_row = 2 #randint(0, 2)
# influences color circuit and selects position constraint
missing_column = 0 #randint(0, 2)

print(f"Missing panel at ({missing_row}, {missing_column})")

# %%  COLOR AND POSITION PREDICTIONS


def create_and_visualise_circuit(
    vqc: VQC, dev: qml.Device, row_wise: bool
) -> qml.QNode:
    circuit = qml.QNode(
        create_circuit(
            "generative",
            vqc.config.num_instance_domains,
            vqc.config.concept_domain_indices,
            missing_domain_index=missing_column if row_wise else missing_row,
        ),
        dev,
        interface="torch",
    )

    instance = t.zeros(vqc.config.num_instance_domains, 3)
    concept = rearrange(
        vqc.concept_weights.weight.detach(),
        "none (layer domain weights) -> (none layer) domain weights",
        layer=vqc.config.layers,
        weights=3,
    )
    output_property = t.zeros(1, 3)

    fig, axes = qml.draw_mpl(circuit, style="black_white", expansion_strategy="device")(
        instance, concept, output_property
    )
    fig.show()

    return circuit


def make_property_predictions(
    vqc: VQC,
    circuit: qml.QNode,
    instance_encodings: t.Tensor,
    property_encodings: t.Tensor,
    row_wise: bool,
) -> tuple[t.Tensor, t.Tensor]:
    if row_wise:
        row = repeat(
            instance_encodings[:, missing_row, :],
            "puzzle column domain encoding -> (column domain) encoding (puzzle property)",
            property=len(property_encodings),
        )
    else:  # if column-wise, replace row with column
        row = repeat(
            instance_encodings[:, :, missing_column],
            "puzzle row domain encoding -> (row domain) encoding (puzzle property)",
            property=len(property_encodings),
        )

    concept = rearrange(
        vqc.concept_weights.weight.detach(),
        "none (layer domain weights) -> (none layer) domain weights",
        layer=vqc.config.layers,
        weights=3,
    )

    # property is encoded as qml.Rot=RzRyRz, so we need to negate to get the complex conjugate
    conjugated_properties = -property_encodings.clone()
    conjugated_properties[:, 1] *= -1
    properties = repeat(
        conjugated_properties,
        "property (encoding) -> domain encoding (puzzle property)",
        puzzle=len(instance_encodings),
        domain=1,
    )

    # print(row.shape, concept.shape, properties.shape)

    full_probs = circuit(row, concept, properties)
    property_probs = rearrange(
        full_probs[:, 0],
        "(puzzle property) -> puzzle property",
        property=len(property_encodings),
    )

    plt.hist(property_probs.cpu().detach().flatten(), bins=50)
    plt.show()

    property_preds = t.argmax(property_probs, dim=1)
    return property_preds, full_probs


def check_color(colors: np.ndarray):
    valid_colors = ["red", "green", "blue", "yellow"]
    return np.unique(colors).size == 3 and all(
        color in valid_colors for color in colors
    )


def check_position(positions: np.ndarray):
    return (
        (positions == np.array(["bottom_left", "top_left", "top_right"])).all()
        or (positions == np.array(["top_left", "top_right", "bottom_right"])).all()
        or (positions == np.array(["top_right", "bottom_right", "bottom_left"])).all()
        or (positions == np.array(["bottom_right", "bottom_left", "top_left"])).all()
    )


def check_puzzle(puzzle: np.ndarray):
    color_accuracy = np.apply_along_axis(check_color, 1, puzzle[0]).all()
    position_accuracy = np.apply_along_axis(check_position, 0, puzzle[1]).all()
    return position_accuracy


def check_predictions(color_preds, position_preds, property_labels):
    preds = property_labels.copy()
    preds[:, :, missing_row, missing_column] = "MISSING"

    preds[:, 0, missing_row, missing_column] = properties[0, color_preds.cpu()]
    preds[:, 1, missing_row, missing_column] = properties[1, position_preds.cpu()]

    correct = np.apply_along_axis(
        lambda puzzle: check_puzzle(
            rearrange(
                puzzle, "(property row column) -> property row column", row=3, column=3
            )
        ),
        1,
        rearrange(preds, "puzzle property row column -> puzzle (property row column)"),
    )

    print(f"Blackbird solving accuracy: {correct.mean()} ({len(correct)} puzzles)")

    new_solution = property_labels[correct] != preds[correct]
    print(
        f"Found {new_solution.sum()} new solutions out of {len(new_solution)} solved puzzles"
    )
    
# %%

dev = qml.device("default.qubit", wires=10)

color_circuit = create_and_visualise_circuit(distribute_three, dev, row_wise=True)
position_circuit = create_and_visualise_circuit(progression, dev, row_wise=False)

color_preds, full_color_probs = make_property_predictions(
    distribute_three,
    color_circuit,
    encodings[labels == 1],
    color_encodings,
    row_wise=True,
)
position_preds, full_position_probs = make_property_predictions(
    progression,
    position_circuit,
    encodings[labels == 1],
    position_encodings,
    row_wise=False,
)

check_predictions(color_preds, position_preds, property_labels)

# %% ############################################################################

service = QiskitRuntimeService(channel="ibm_quantum", instance="ibm-q/open/main")

backend = FakeKyiv()
print(f"{backend.name}: {backend.status().pending_jobs} pending jobs")


# backend = service.least_busy(operational=True, simulator=False, min_num_qubits=10)
# backend = service.backend("ibm_kyiv")
# print(f"{backend.name}: {backend.status().pending_jobs} pending jobs")

backend = FakeKyiv()
print(f"{backend.name}: {backend.status().pending_jobs} pending jobs")

dev = qml.device("qiskit.remote", wires=10, backend=backend, shots=2**14,)

# dev = qml.device("default.qubit", wires=10)

color_circuit = create_and_visualise_circuit(distribute_three, dev, row_wise=True)
position_circuit = create_and_visualise_circuit(progression, dev, row_wise=False)

# job = service.jobs()[0]
# result = job.result()

# plot_distribution(result.get_counts(0), number_to_keep=20)

# marginal = marginal_counts(result, [0])
# plot_distribution(marginal.get_counts(0), number_to_keep=20)

# %%
%%time

# SOLUTIONS

# less wires
# remove bell state

# better simulator <- nothing

# check flags

# more shots <- nothing

amount = 5 #len(labels[labels == 1])
batch = 2

# 0: 12, 0, 0 -> 0
# 1 color: 12, 2, 2 botched
# 1: 13, 2, 2 -> 1 (color 4)
# 2: 14, 2, 2

r = range(batch*amount, (batch+1)*amount)

color_preds, full_color_probs = make_property_predictions(
    distribute_three,
    color_circuit,
    encodings[labels == 1][r],
    color_encodings,
    row_wise=True,
)
if dev.name == "Qiskit PennyLane plugin" and backend.name != "fake_kyiv":
    t.save(color_preds, f"color_preds_batch{batch}.pt")
    t.save(full_color_probs, f"full_color_probs_batch{batch}.pt")

position_preds, full_position_probs = make_property_predictions(
    progression,
    position_circuit,
    encodings[labels == 1][r],
    position_encodings,
    row_wise=False,
)
if dev.name == "Qiskit PennyLane plugin" and backend.name != "fake_kyiv":
    t.save(position_preds, f"position_preds_batch{batch}.pt")
    t.save(full_position_probs, f"full_position_probs_batch{batch}.pt")

check_predictions(color_preds, position_preds, property_labels[r])



#%%

# position = t.load(f"position_preds_batch{0}.pt")
# color = t.load(f"color_preds_batch{0}.pt")

# check_predictions(color, position, property_labels[0:5])

# result = service.job("cv31gqtfkm5g0084z6qg").result()
# result = marginal_counts(result, [6, 7, 8, 9])

# # %%
# plot_distribution(result.get_counts(0), number_to_keep=20)
# # %%
# for i in range(8, 12):
#     print(result.get_counts(i)["0000"] / result.get_counts(i)["0001"])
#     print(result.get_counts(i)["0000"])

# # %% [2, 0, 2, 1, 3]
