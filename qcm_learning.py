# %%

import lightning as l
import matplotlib.pyplot as plt
import torch as t
from lightning.pytorch.callbacks import ModelCheckpoint
from sympy import N

from architecture import Hybrid
from data_modules import ProductConceptDataModule

t.set_float32_matmul_precision("high")

# %%

shapes = ProductConceptDataModule("blackbird/data/shapes", 2**6)

model = Hybrid()

checkpoint = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    filename="hybrid-{epoch:02d}",
)

trainer = l.Trainer(max_epochs=100, log_every_n_steps=25, callbacks=[checkpoint])
# %%

trainer.fit(model, shapes)

trainer.test(model, shapes)

# %%

result = trainer.predict(model, shapes)

x_pred, y_pred = result[0]

plt.imshow(x_pred[0].permute(1, 2, 0))

x = shapes.predict_dataloader().dataset[0]

plt.imshow(x[0].permute(1, 2, 0))