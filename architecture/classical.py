# %%

# TODO: wandb + lightning

# TODO: einops


from dataclasses import dataclass

import torch as t
from jaxtyping import Float, Int
from rich import print
from rich.table import Table
from torch import Tensor, nn
from torchinfo import summary
from utils import Config

from einops import rearrange

# %%


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2),
            nn.ReLU(True),
        )

        self.dense = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, config.num_domains // config.images_per_instance * 3),
            # nn.ReLU(True),
        )

        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_uniform_(weight)
            else:
                nn.init.zeros_(weight)

    def forward(
        self,
        x: Float[Tensor, "batch color height width"],
        images_per_instance: int = 1,
    ) -> Float[Tensor, "batch domain weights"]:
        if images_per_instance > 1:
            x = rearrange(
                x, "batch image color height width -> (batch image) color height width"
            )
            
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = rearrange(x, "batch (domain weights) -> batch domain weights", weights=3)
        
        if images_per_instance > 1:
            x = rearrange(
                x, "(batch image) domain weights -> batch (image domain) weights", image=images_per_instance
            )
        return x


# %%


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.dense = nn.Sequential(
            nn.Linear(config.num_domains * 3, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2),
        )

        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_uniform_(weight)
            else:
                nn.init.zeros_(weight)

    def forward(
        self, x: Float[Tensor, "batch domain weights"]
    ) -> Float[Tensor, "batch color height width"]:
        x = rearrange(x, "batch domain weights -> batch (domain weights)")
        x = self.dense(x)
        x = x.view(x.size(0), 64, 2, 2)
        x = self.conv(x)
        return x
