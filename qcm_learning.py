# %%

import lightning as l
import matplotlib.pyplot as plt
import torch as t
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
)

from architecture import Hybrid
from data_modules import ProductConceptDataModule
from tqdm import tqdm

t.set_float32_matmul_precision("high")


class NoValidationBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar


def trainer():
    return l.Trainer(
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="hybrid-{epoch:02d}",
            ),
            NoValidationBar(),
        ],
    )

# %%

shapes = ProductConceptDataModule("blackbird/data/shapes", 2**6)
shapes_model = Hybrid(shapes.config)
shapes_trainer = trainer()

# %%

shapes_trainer.fit(shapes_model, shapes)

# %%

shapes_trainer.validate(shapes_model, shapes)
shapes_trainer.test(shapes_model, shapes)
# TODO: visualise the results

# %%

rainbow = ProductConceptDataModule("blackbird/data/rainbow", 2**6)
rainbow_model = Hybrid(rainbow.config)
rainbow_trainer = trainer()

# %%

rainbow_trainer.fit(rainbow_model, rainbow)

# %%

rainbow_trainer.validate(rainbow_model, rainbow)
rainbow_trainer.test(rainbow_model, rainbow)
# TODO: visualise the results

# %%

decoder_config = shapes.config
decoder_config.decoder_multiplier = 1500
decoder_model = Hybrid(shapes.config)
decoder_trainer = trainer()

# %%

decoder_trainer.fit(decoder_model, shapes)

# %%

decoder_trainer.validate(decoder_model, shapes)
decoder_trainer.test(decoder_model, shapes)
# TODO: visualise the results

prediction = decoder_trainer.predict(shapes_model, shapes)
if prediction is not None:
    x_pred, y_pred = prediction[0]

    plt.imshow(x_pred[0].permute(1, 2, 0))
    plt.show()

    x = shapes.predict_dataloader().dataset[0]
    plt.imshow(x[0].permute(1, 2, 0))

# %%


# %%
