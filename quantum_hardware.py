# %%

from random import randint
import pennylane as qml
import torch as t
from einops import rearrange, reduce, repeat

from architecture import Hybrid
from architecture.quantum import VQC
from data_modules import EntangledConceptDataModule
from utils.circuits import create_circuit

import numpy as np

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


def print_stats(probs, labels):
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

property_encodings = blackbird_model.vqc.concept_weights.weight
color_encodings = property_encodings[offsets[0] : offsets[0] + len(properties[0])]
position_encodings = property_encodings[offsets[1] : offsets[1] + len(properties[1])]

# %% COLOR AND POSITION CIRCUITS

dev = qml.device("default.qubit", wires=10)

# influences position circuit and selects color constraint
missing_row = randint(0, 2)
# influences color circuit and selects position constraint
missing_column = randint(0, 2)

print(f"Missing panel at ({missing_row}, {missing_column})")


def create_and_visualise_circuit(vqc: VQC, row_wise: bool) -> qml.QNode:
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
        vqc.concept_weights.weight,
        "none (layer domain weights) -> (none layer) domain weights",
        layer=vqc.config.layers,
        weights=3,
    )
    output_property = t.zeros(1, 3)

    qml.draw_mpl(circuit, style="black_white", expansion_strategy="device")(
        instance, concept, output_property
    )

    return circuit


color_circuit = create_and_visualise_circuit(distribute_three, row_wise=True)
position_circuit = create_and_visualise_circuit(progression, row_wise=False)

# %%  COLOR AND POSITION PREDICTIONS


def make_property_predictions(
    vqc: VQC, circuit: qml.QNode, property_encodings: t.Tensor, row_wise: bool
) -> t.Tensor:
    if row_wise:
        row = repeat(
            encodings[labels == 1, missing_row, :],
            "puzzle column domain encoding -> (column domain) encoding (puzzle property)",
            property=len(property_encodings),
        )
    else:  # if column-wise, replace row with column
        row = repeat(
            encodings[labels == 1, :, missing_column],
            "puzzle row domain encoding -> (row domain) encoding (puzzle property)",
            property=len(property_encodings),
        )

    concept = rearrange(
        vqc.concept_weights.weight,
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
        puzzle=len(encodings[labels == 1]),
        domain=1,
    )

    property_probs = circuit(row, concept, properties)[:, 0]
    property_probs = rearrange(
        property_probs,
        "(puzzle property) -> puzzle property",
        property=len(property_encodings),
    )

    property_preds = t.argmax(property_probs, dim=1)
    return property_preds


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
    return color_accuracy and position_accuracy


# %%

color_preds = make_property_predictions(
    distribute_three, color_circuit, color_encodings, row_wise=True
)
position_preds = make_property_predictions(
    progression, position_circuit, position_encodings, row_wise=False
)


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

print(f"Blackbird solving accuracy: {correct.mean()}")

new_solution = property_labels != preds
print(f"Found {new_solution.sum()} new solutions out of {len(new_solution)} solvable puzzles")
# %%
