# %%

import dis
from architecture import Hybrid
from data_modules import EntangledConceptDataModule, ProductConceptDataModule

from einops import rearrange

# %% BLACKBIRD DATASET AND MODELS

blackbird = EntangledConceptDataModule("blackbird/data/balanced", "blackbird", 2**6)
config = blackbird.config

blackbird_model = Hybrid.load_from_checkpoint("lightning_logs/blackbird/checkpoints/blackbird-selection-epoch=44.ckpt")
distribute_three_model = Hybrid.load_from_checkpoint("lightning_logs/distribute_three/checkpoints/progression-selection-epoch=99.ckpt")
progression_model = Hybrid.load_from_checkpoint("lightning_logs/progression/checkpoints/progression-selection-epoch=99.ckpt")

instances, labels = next(iter(blackbird.test_dataloader()))
instances, labels = instances.to(blackbird_model.device), labels.to(blackbird_model.device)

encoder = blackbird_model.encoder
distribute_three = distribute_three_model.vqc
progression = progression_model.vqc

#%%

encodings = encoder(instances, config.images_per_instance)

#%%

distribute_three.plot()

# %%
plot_model_representations(blackbird, blackbird_model, blackbird_trainer, batch_size=1000)

blackbird_trainer.fit(blackbird_model, blackbird)

# %%

try:
    blackbird_model = Hybrid.load_from_checkpoint(
        blackbird_trainer.checkpoint_callback.best_model_path  # type: ignore
    )
except Exception:
    blackbird_model = Hybrid.load_from_checkpoint(
    "lightning_logs/blackbird/checkpoints/blackbird-loss-epoch=66.ckpt"
    )

blackbird_trainer.validate(blackbird_model, blackbird)
blackbird_trainer.test(blackbird_model, blackbird)

plot_model_representations(blackbird, blackbird_model, blackbird_trainer, batch_size=1000)

# %% DISTRIBUTE_TREE CONCEPT

distribute_three = EntangledConceptDataModule("blackbird/data/balanced", "distribute_three", 2**6)
distribute_three.config.layers = 8
distribute_three_model = Hybrid(distribute_three.config)

blackbird_model = Hybrid.load_from_checkpoint(
    "lightning_logs/blackbird/checkpoints/blackbird-selection-epoch=44.ckpt"
)
distribute_three_model.encoder = copy.deepcopy(blackbird_model.encoder)
distribute_three_model.encoder.requires_grad_(False)
distribute_three_trainer = distribute_three.config.trainer("distribute_three")

distribute_three_model.vqc.plot()

distribute_three_trainer.fit(distribute_three_model, distribute_three)

distribute_three_trainer.validate(distribute_three_model, distribute_three, ckpt_path="best")
distribute_three_trainer.test(distribute_three_model, distribute_three, ckpt_path="best")

# %% PROGRESSION CONCEPT

progression = EntangledConceptDataModule("blackbird/data/balanced", "progression", 2**6)
progression.config.layers = 4
progression_model = Hybrid(progression.config)

blackbird_model = Hybrid.load_from_checkpoint(
    "lightning_logs/blackbird/checkpoints/blackbird-selection-epoch=44.ckpt"
)
progression_model.encoder = copy.deepcopy(blackbird_model.encoder)
progression_model.encoder.requires_grad_(False)
progression_trainer = progression.config.trainer("progression")

progression_model.vqc.plot()

progression_trainer.fit(progression_model, progression)

progression_trainer.validate(progression_model, progression, ckpt_path="best")
progression_trainer.test(progression_model, progression, ckpt_path="best")

# %%
