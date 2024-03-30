from os.path import join

import numpy as np
import torch
from pandas import read_csv
from skimage.io import imread_collection
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple


class ProductConceptDataset(Dataset):
	def __init__(self, name: str, device: torch.device):
		self.name = name

		self.concepts = read_csv(join(name, "labels.csv"), dtype="category")
		self.domains = self.concepts.columns
		self.domain_properties = np.array(
			[self.concepts[column].cat.categories for column in self.domains]
		)

		self.num_properties = self.domain_properties.shape[1]
		self.offsets = torch.tensor(
			[i * self.num_properties for i in range(len(self.domain_properties))],
			dtype=torch.int,
			device=device,
		)

		for column in self.concepts.columns:
			self.concepts[column] = self.concepts[column].cat.codes

		self.concepts = torch.tensor(
			self.concepts.values, dtype=torch.long, device=device
		)

		self.transform = transforms.Compose([transforms.ToTensor()])

		self.instances = imread_collection(
			[join(self.name, f"{i}.png") for i in range(len(self.concepts))],
			conserve_memory=False,
		)
		self.instances = torch.stack([self.transform(image) for image in self.instances])
		self.instances = self.instances.to(device)

	def __len__(self) -> int:
		return len(self.concepts)

	def add_offset(self, concepts: torch.Tensor) -> torch.Tensor:
		return concepts + self.offsets

	def remove_offset(self, concepts: torch.Tensor) -> torch.Tensor:
		return concepts - self.offsets

	def decode_concept(self, concepts: torch.Tensor) -> np.ndarray:
		return self.domain_properties.reshape(-1)[concepts.cpu().numpy()]

	def decode_domain(self, domain: int) -> str:
		return self.domains[domain]

	def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.add_offset(self.concepts[i]), self.instances[i]

	def preprocess(
		self, x: torch.Tensor, concept: torch.Tensor
		) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		false_concept = torch.randint_like(
			concept, 0, self.num_properties - 1, device=concept.device, dtype=torch.long
		)
		false_concept[false_concept == self.remove_offset(concept)] += 1
		false_concept = self.add_offset(false_concept)

		x = torch.cat([x, x])
		concept = torch.cat([concept, false_concept])
		y = torch.cat(
			[
				torch.ones(x.shape[0] // 2, dtype=torch.double, device=x.device),
				torch.zeros(x.shape[0] // 2, dtype=torch.double, device=x.device),
			]
		)

		return x, concept, y


if __name__ == "__main__":
	dataset = ProductConceptDataset("data/shapes/val", "cuda:0")
	print(dataset.domain_properties)
