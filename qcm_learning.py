# %%

# TODO: remove if name == main

import copy
import lightning as l
import matplotlib.pyplot as plt
import torch as t
from einops import rearrange
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from tqdm import tqdm

from architecture import Hybrid
from data_modules import ProductConceptDataModule
from data_modules.entangled_concept import EntangledConceptDataModule
from utils import plot_representations

t.set_float32_matmul_precision("high")


class NoValidationBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar


def trainer(name: str, max_epochs: int = 100) -> l.Trainer:
    return l.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename=name + "-loss-{epoch:02d}",
            ),
            # ModelCheckpoint(
            #     monitor="val_accuracy",
            #     mode="max",
            #     save_top_k=1,
            #     filename=name + "-accuracy-{epoch:02d}",
            # ),
            NoValidationBar(),
        ],
    )


def plot_model_representations(
    data: ProductConceptDataModule, model: l.LightningModule, trainer: l.Trainer
):
    prediction = trainer.predict(model, data)
    if prediction is not None:
        concepts = rearrange(
            model.vqc.concept_weights.weight.detach(),
            "(domain property) weights -> domain property weights",
            domain=data.config.num_instance_domains,
        )

        instances = prediction[0][2]
        instance_concepts = next(iter(data.predict_dataloader()))[1]

        for domain in range(data.config.num_instance_domains):
            plot_representations(
                concepts[domain],
                instances[:, domain],
                instance_concepts[:, domain],
                data.config.properties[domain],
            )


# %% SHAPES MODEL

shapes = ProductConceptDataModule("blackbird/data/shapes", 2**6)
shapes_model = Hybrid(shapes.config)
shapes_trainer = trainer("shapes")

shapes_model.vqc.plot()

# %%

shapes_trainer.fit(shapes_model, shapes)

# %%

try:
    shapes_model = Hybrid.load_from_checkpoint(
        shapes_trainer.checkpoint_callback.best_model_path  # type: ignore
    )
except IsADirectoryError:
    shapes_model = Hybrid.load_from_checkpoint(
        "lightning_logs/shapes/checkpoints/shapes-loss-epoch=80.ckpt"
    )

shapes_trainer.validate(shapes_model, shapes)
shapes_trainer.test(shapes_model, shapes)
plot_model_representations(shapes, shapes_model, shapes_trainer)


# %% RAINBOW MODEL

rainbow = ProductConceptDataModule("blackbird/data/rainbow", 2**6)
rainbow_model = Hybrid(rainbow.config)
rainbow_trainer = trainer("rainbow")

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
decoder_trainer = trainer("decoder")

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

correlated = EntangledConceptDataModule("blackbird/data/shapes", "correlated", 2**6, 2)
correlated_model = Hybrid(correlated.config)
correlated_model.encoder = copy.deepcopy(shapes_model.encoder)
correlated_model.encoder.requires_grad_(False)
correlated_trainer = trainer("correlated")

correlated_model.vqc.plot()

correlated_trainer.fit(correlated_model, correlated)

correlated_model = Hybrid.load_from_checkpoint(
    correlated_trainer.checkpoint_callback.best_model_path  # type: ignore
)

correlated_trainer.validate(correlated_model, correlated)
correlated_trainer.test(correlated_model, correlated)

# %% GENERAL CONCEPT

general = EntangledConceptDataModule("blackbird/data/shapes", "red", 2**6, 2)
general_model = Hybrid(general.config)
general_model.encoder = copy.deepcopy(shapes_model.encoder)
general_model.encoder.requires_grad_(False)
general_trainer = trainer("general")

general_model.vqc.plot()

general_trainer.fit(general_model, general)

general_model = Hybrid.load_from_checkpoint(
    general_trainer.checkpoint_callback.best_model_path  # type: ignore
)

general_trainer.validate(general_model, general)
general_trainer.test(general_model, general)

# %% LOGIC OPERATOR CONCEPT

conjunction = EntangledConceptDataModule("blackbird/data/shapes", "red_and_circle", 2**6, 3)
conjunction_model = Hybrid(conjunction.config)
conjunction_model.encoder = copy.deepcopy(shapes_model.encoder)
conjunction_model.encoder.requires_grad_(False)
conjunction_trainer = trainer("conjunction")

conjunction_model.vqc.plot()

conjunction_trainer.fit(conjunction_model, conjunction)

conjunction_model = Hybrid.load_from_checkpoint(
    conjunction_trainer.checkpoint_callback.best_model_path  # type: ignore
)

conjunction_trainer.validate(conjunction_model, conjunction)
conjunction_trainer.test(conjunction_model, conjunction)

# %%

disjunction = EntangledConceptDataModule("blackbird/data/shapes", "red_or_blue", 2**6, 3)
disjunction_model = Hybrid(disjunction.config)
disjunction_model.encoder = copy.deepcopy(shapes_model.encoder)
disjunction_model.encoder.requires_grad_(False)
disjunction_trainer = trainer("disjunction")

disjunction_model.vqc.plot()

disjunction_trainer.fit(disjunction_model, disjunction)

disjunction_model = Hybrid.load_from_checkpoint(
    disjunction_trainer.checkpoint_callback.best_model_path  # type: ignore
)  

disjunction_trainer.validate(disjunction_model, disjunction)
disjunction_trainer.test(disjunction_model, disjunction)

# %%

disjunction_within = EntangledConceptDataModule(
    "blackbird/data/shapes", "red_or_circle", 2**6, 3
)
disjunction_within_model = Hybrid(disjunction_within.config)
disjunction_within_model.encoder = copy.deepcopy(shapes_model.encoder)
disjunction_within_model.encoder.requires_grad_(False)
disjunction_within_trainer = trainer("disjunction_within")

disjunction_within_model.vqc.plot()

disjunction_within_trainer.fit(disjunction_within_model, disjunction_within)

disjunction_within_model = Hybrid.load_from_checkpoint(
    disjunction_within_trainer.checkpoint_callback.best_model_path  # type: ignore
)

disjunction_within_trainer.validate(disjunction_within_model, disjunction_within)
disjunction_within_trainer.test(disjunction_within_model, disjunction_within)

# %%
