from os.path import join
from typing import Tuple

import numpy as np
import torch as t
from pandas import read_csv
from skimage.io import imread_collection
from torch.utils.data import Dataset
from torchvision import transforms


class EntangledConceptDataset(Dataset):
	def __init__(self, name: str, concept_name: str, device: t.device):
		self.name = name

		match concept_name:
			case "distribute_three":
				filename = "distribute_three.csv"
			case "progression":
				filename = "progression.csv"
			case "blackbird":
				filename = "blackbird.csv"
			case _:
				filename = "product_concepts.csv"

		self.concepts = read_csv(join(name, filename), dtype="category")

		match concept_name:
			case "twike":
				self.concepts = (
					(self.concepts["color"] == "red")
					& (self.concepts["shape"] == "circle")
				) | (
					(self.concepts["color"] == "blue")
					& (self.concepts["shape"] == "square")
				)
			case "red":
				self.concepts = self.concepts["color"] == "red"
			case "red_and_circle":
				self.concepts = (self.concepts["color"] == "red") & (
					self.concepts["shape"] == "circle"
				)
			case "red__or_blue":
				self.concepts = (self.concepts["color"] == "red") | (
					self.concepts["color"] == "blue"
				)
			case "red_or_circle":
				self.concepts = (self.concepts["color"] == "red") | (
					self.concepts["shape"] == "circle"
				)
			case "blackbird":
				self.concepts = self.concepts["correct"]

		self.concepts = t.tensor(
			self.concepts, dtype=t.double, device=device
		)

		print(
			f"Balance of {self.name}-{concept_name} dataset: {self.concepts.mean().item()} true"
		)

		self.transform = transforms.Compose([transforms.ToTensor()])

		self.instances = imread_collection(
			[join(self.name, f"{i}.png") for i in range(len(self.concepts))],
			conserve_memory=False,
		)
		self.instances = t.stack(
			[self.transform(image) for image in self.instances]
		).to(device)

		if concept_name == "progression":  # Make puzzles column-major
			self.instances = self.instances.reshape(-1, 3, 3, 3, 64, 64).transpose(2, 1)
		if concept_name == "distribute_three" or concept_name == "progression":
			self.instances = self.instances.reshape(len(self.concepts), 3, 3, 64, 64)
		elif concept_name == "blackbird":
			self.instances = self.instances.reshape(len(self.concepts), 9, 3, 64, 64)

	def __len__(self) -> int:
		return len(self.concepts)

	def preprocess(
		self, x: t.Tensor, concept: t.Tensor
	) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
		return x, t.zeros_like(concept, dtype=t.long), concept

	def __getitem__(self, i) -> Tuple[t.Tensor, t.Tensor]:
		return self.instances[i], self.concepts[i]


if __name__ == "__main__":
	# twike
    
	dataset = EntangledConceptDataset("data/shapes/val", "twike", t.device("cuda:0"))
	print(len(dataset))
 
	# rows

	dataset = EntangledConceptDataset(
		"data/blackbird/val", "distribute_three", t.device("cuda:0")
	)
	print(len(dataset))

	x, y = dataset[3]
	print(x.shape, y.shape)

	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(1, 3)
	for i in range(3):
		ax[i].imshow(x[i].permute(1, 2, 0))
		ax[i].axis("off")
	plt.show() 
 
	# blackbird
 
	dataset = EntangledConceptDataset("data/blackbird/val", "blackbird", t.device("cuda:0"))
	print(len(dataset))

	x, y = dataset[0]
	print(y)

	fig, ax = plt.subplots(3, 3)
	for i in range(3):
		for j in range(3):
			ax[i, j].imshow(x[3 * i + j].permute(1, 2, 0))
	plt.show()
