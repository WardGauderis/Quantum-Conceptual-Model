import torch
from torch import nn

from torchsummary import summary


class Encoder(nn.Module):
	def __init__(self, output: int):
		super().__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(3, 64, 4, 2),
			nn.ReLU(True),
			nn.Conv2d(64, 64, 4, 2),
			nn.ReLU(True),
			nn.Conv2d(64, 64, 4, 2),
			nn.ReLU(True),
			nn.Conv2d(64, 64, 4, 2),
			nn.ReLU(True)
		)

		self.dense = nn.Sequential(
			nn.Linear(256, 256),
			nn.ReLU(True),
			nn.Linear(256, output),
			# nn.ReLU(True),
		)

		for weight in self.parameters():
			if len(weight.shape) > 1:
				nn.init.xavier_uniform_(weight)
			else:
				nn.init.zeros_(weight)


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.conv(x)
		x = x.view(x.size(0), -1)
		x = self.dense(x)
		return x


class Decoder(nn.Module):
	def __init__(self, output: int):
		super().__init__()

		self.dense = nn.Sequential(
			nn.Linear(output, 256),
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

		self.loss = nn.MSELoss()

		for weight in self.parameters():
			if len(weight.shape) > 1:
				nn.init.xavier_uniform_(weight)
			else:
				nn.init.zeros_(weight)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.dense(x)
		x = x.view(x.size(0), 64, 2, 2)
		x = self.conv(x)
		return x

	def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		x = self.backward(x)
		return self.loss(x, y)



if __name__ == "__main__":
	model = Encoder(12)
	model.to("cuda:0")
	summary(model, (3, 64, 64))