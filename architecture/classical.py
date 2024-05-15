#%%

# TODO: wandb + lightning

# TODO: einops


from dataclasses import dataclass

import torch as t
from jaxtyping import Float, Int
from rich import print
from rich.table import Table
from torch import Tensor, nn
from torchinfo import summary
from data_modules import Config

#%%

    

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
			nn.Linear(256, config.num_domains * 3),
			# nn.ReLU(True),
		)

		for weight in self.parameters():
			if len(weight.shape) > 1:
				nn.init.xavier_uniform_(weight)
			else:
				nn.init.zeros_(weight)

	def forward(
		self, x: Float[Tensor, "batch color height width"]
	) -> Float[Tensor, "batch encoding"]:
		x = self.conv(x)
		x = x.view(x.size(0), -1)
		x = self.dense(x)
		return x

#%%

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

	def forward(self, x: Float[Tensor, "batch encoding"]) -> Float[Tensor, "batch color height width"]:
		x = self.dense(x)
		x = x.view(x.size(0), 64, 2, 2)
		x = self.conv(x)
		return x


#%%

if __name__ == "__main__":
	model = Encoder(12)
	model.to("cuda:0")
	print(summary(model, (1, 3, 64, 64)))

# %%
