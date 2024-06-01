# %%

# TODO: remove if name == main

import copy

import lightning as l
import matplotlib.pyplot as plt
import torch as t

from architecture import Hybrid
from data_modules import ProductConceptDataModule
from data_modules.entangled_concept import EntangledConceptDataModule
from utils import plot_model_representations

t.set_float32_matmul_precision("high")


# %% SHAPES MODEL

shapes = ProductConceptDataModule("blackbird/data/shapes", 2**4)
shapes_model = Hybrid(shapes.config)
shapes_trainer = shapes.config.trainer("shapes")

shapes_model.vqc.plot()

# %%
plot_model_representations(shapes, shapes_model, shapes_trainer)

shapes_trainer.fit(shapes_model, shapes)

try:
    shapes_model = Hybrid.load_from_checkpoint(
        shapes_trainer.checkpoint_callback.best_model_path  # type: ignore
    )
except IsADirectoryError:
    shapes_model = Hybrid.load_from_checkpoint(
        "lightning_logs/shapes/checkpoints/shapes-loss-epoch=77.ckpt"
    )

shapes_trainer.validate(shapes_model, shapes)
shapes_trainer.test(shapes_model, shapes)

plot_model_representations(shapes, shapes_model, shapes_trainer)

# %% RAINBOW MODEL

rainbow = ProductConceptDataModule("blackbird/data/rainbow", 2**6)
rainbow_model = Hybrid(rainbow.config)
rainbow_trainer = rainbow.config.trainer("rainbow")

# %%

rainbow_trainer.fit(rainbow_model, rainbow)

# %%

rainbow_model = Hybrid.load_from_checkpoint(
    rainbow_trainer.checkpoint_callback.best_model_path  # type: ignore
)

rainbow_trainer.validate(rainbow_model, rainbow)
rainbow_trainer.test(rainbow_model, rainbow)
plot_model_representations(rainbow, rainbow_model, rainbow_trainer)

# %% DECODER MODEL

decoder = shapes
decoder.config.decoder_multiplier = 1500
decoder_model = Hybrid(decoder.config)
decoder_trainer = decoder.config.trainer("decoder")

# %%

decoder_trainer.fit(decoder_model, decoder)

# %%

decoder_model = Hybrid.load_from_checkpoint(
    decoder_trainer.checkpoint_callback.best_model_path  # type: ignore
)

decoder_trainer.validate(decoder_model, decoder)
decoder_trainer.test(decoder_model, decoder)
plot_model_representations(decoder, decoder_model, decoder_trainer)

prediction = decoder_trainer.predict(decoder_model, decoder)
if prediction is not None:
    x_pred, y_pred, _ = prediction[0]

    plt.imshow(x_pred[0].permute(1, 2, 0))
    plt.show()

    x = decoder.predict_dataloader().dataset[0]
    plt.imshow(x[0].permute(1, 2, 0))

# %% CORRELATED CONCEPT

correlated = EntangledConceptDataModule("blackbird/data/shapes", "correlated", 2**6)
correlated.config.layers = 2
correlated_model = Hybrid(correlated.config)
correlated_model.encoder = copy.deepcopy(shapes_model.encoder)
correlated_model.encoder.requires_grad_(False)
correlated_trainer = correlated.config.trainer("correlated")

correlated_model.vqc.plot()

correlated_trainer.fit(correlated_model, correlated)

correlated_model = Hybrid.load_from_checkpoint(
    correlated_trainer.checkpoint_callback.best_model_path  # type: ignore
)

correlated_trainer.validate(correlated_model, correlated)
correlated_trainer.test(correlated_model, correlated)

# %% GENERAL CONCEPT

general = EntangledConceptDataModule("blackbird/data/shapes", "red", 2**6)
general.config.concept_type = "general"
general.config.layers = 2
general_model = Hybrid(general.config)
general_model.encoder = copy.deepcopy(shapes_model.encoder)
general_model.encoder.requires_grad_(False)
general_trainer = general.config.trainer("general")

general_model.vqc.plot()

general_trainer.fit(general_model, general)

general_model = Hybrid.load_from_checkpoint(
    general_trainer.checkpoint_callback.best_model_path  # type: ignore
)

general_trainer.validate(general_model, general)
general_trainer.test(general_model, general)

# %% LOGIC OPERATOR CONCEPT

conjunction = EntangledConceptDataModule(
    "blackbird/data/shapes", "red_and_circle", 2**6
)
conjunction.config.concept_type = "general"
conjunction.config.layers = 3
conjunction_model = Hybrid(conjunction.config)
conjunction_model.encoder = copy.deepcopy(shapes_model.encoder)
conjunction_model.encoder.requires_grad_(False)
conjunction_trainer = conjunction.config.trainer("conjunction")

conjunction_model.vqc.plot()

conjunction_trainer.fit(conjunction_model, conjunction)

conjunction_model = Hybrid.load_from_checkpoint(
    conjunction_trainer.checkpoint_callback.best_model_path  # type: ignore
)

conjunction_trainer.validate(conjunction_model, conjunction)
conjunction_trainer.test(conjunction_model, conjunction)

# %%

disjunction = EntangledConceptDataModule("blackbird/data/shapes", "red_or_blue", 2**6)
disjunction.config.concept_type = "general"
disjunction.config.layers = 2
disjunction_model = Hybrid(disjunction.config)
disjunction_model.encoder = copy.deepcopy(shapes_model.encoder)
disjunction_model.encoder.requires_grad_(False)
disjunction_trainer = disjunction.config.trainer("disjunction")

disjunction_model.vqc.plot()

disjunction_trainer.fit(disjunction_model, disjunction)

disjunction_model = Hybrid.load_from_checkpoint(
    disjunction_trainer.checkpoint_callback.best_model_path  # type: ignore
)

disjunction_trainer.validate(disjunction_model, disjunction)
disjunction_trainer.test(disjunction_model, disjunction)

# %%

disjunction_within = EntangledConceptDataModule(
    "blackbird/data/shapes", "red_or_circle", 2**6
)
disjunction_within.config.concept_type = "general"
disjunction_within.config.layers = 3
disjunction_within_model = Hybrid(disjunction_within.config)
disjunction_within_model.encoder = copy.deepcopy(shapes_model.encoder)
disjunction_within_model.encoder.requires_grad_(False)
disjunction_within_trainer = disjunction_within.config.trainer("disjunction_within")

disjunction_within_model.vqc.plot()

disjunction_within_trainer.fit(disjunction_within_model, disjunction_within)

disjunction_within_model = Hybrid.load_from_checkpoint(
    disjunction_within_trainer.checkpoint_callback.best_model_path  # type: ignore
)

disjunction_within_trainer.validate(disjunction_within_model, disjunction_within)
disjunction_within_trainer.test(disjunction_within_model, disjunction_within)

# %%
