# %%

import copy
from architecture import Hybrid
from data_modules import ProductConceptDataModule, EntangledConceptDataModule
from utils import plot_model_representations

# %% BLACKBIRD MODEL

blackbird = ProductConceptDataModule("blackbird/data/balanced", 2**6)
blackbird_model = Hybrid(blackbird.config)
blackbird_trainer = blackbird.config.trainer("blackbird")

blackbird_model.vqc.plot()

# %%
# plot_model_representations(blackbird, blackbird_model, blackbird_trainer, batch_size=3000)

blackbird_trainer.fit(blackbird_model, blackbird)

# %%

try:
    blackbird_model = Hybrid.load_from_checkpoint(
        blackbird_trainer.checkpoint_callback.best_model_path  # type: ignore
    )
except Exception:
    blackbird_model = Hybrid.load_from_checkpoint(
        "lightning_logs/blackbird/checkpoints/blackbird-loss-epoch=85.ckpt"
    )

# blackbird_trainer.validate(blackbird_model, blackbird)
# blackbird_trainer.test(blackbird_model, blackbird)

# plot_model_representations(blackbird, blackbird_model, blackbird_trainer, batch_size=3000)

# %% DISTRIBUTE_TREE CONCEPT

distribute_three = EntangledConceptDataModule("blackbird/data/balanced", "distribute_three", 2**6)
distribute_three.config.layers = 8
distribute_three_model = Hybrid(distribute_three.config)
distribute_three_model.encoder = copy.deepcopy(blackbird_model.encoder)
distribute_three_model.encoder.requires_grad_(False)
distribute_three_trainer = distribute_three.config.trainer("distribute_three")

distribute_three_model.vqc.plot()

distribute_three_trainer.fit(distribute_three_model, distribute_three)

distribution_three_model = Hybrid.load_from_checkpoint(
    distribute_three_trainer.checkpoint_callback.best_model_path  # type: ignore
)

distribute_three_trainer.validate(distribution_three_model, distribute_three)
distribute_three_trainer.test(distribution_three_model, distribute_three)

# %%
