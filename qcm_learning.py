# %%
from architecture import Hybrid
from data_modules import ProductConceptDataModule
import torch as t
import lightning as l

t.set_float32_matmul_precision("high")

# %%

shapes = ProductConceptDataModule("blackbird/data/shapes", 2 ** 6)

model = Hybrid()

trainer = l.Trainer(max_epochs=100)
# %%

trainer.fit(model, shapes)
# %%
