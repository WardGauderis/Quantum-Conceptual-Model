from os.path import exists, join
from typing import Tuple

import numpy as np
import torch as t
from pandas import read_csv
from skimage.io import imread_collection
from torch.utils.data import Dataset
from torchvision import transforms


class ProductConceptDataset(Dataset):
	def __init__(self, name: str, device: t.device):
		self.name = name
  
		self.concepts = read_csv(join(name, "product_concepts.csv"), dtype="category")

   
		self.domains = self.concepts.columns
		self.domain_properties = np.array(
			[self.concepts[column].cat.categories for column in self.domains]
		)

		self.num_properties = self.domain_properties.shape[1]
		self.offsets = t.tensor(
			[i * self.num_properties for i in range(len(self.domain_properties))],
			dtype=t.int,
			device=device,
		)

		for column in self.concepts.columns:
			self.concepts[column] = self.concepts[column].cat.codes

		self.concepts = t.tensor(
			self.concepts.values, dtype=t.long, device=device
		)

		self.transform = transforms.Compose([transforms.ToTensor()])

		self.instances = imread_collection(
			[join(self.name, f"{i}.png") for i in range(len(self.concepts))],
			conserve_memory=False,
		)
		self.instances = t.stack([self.transform(image) for image in self.instances]).to(device)

	def __len__(self) -> int:
		return len(self.concepts)

	def add_offset(self, concepts: t.Tensor) -> t.Tensor:
		return concepts + self.offsets

	def remove_offset(self, concepts: t.Tensor) -> t.Tensor:
		return concepts - self.offsets

	def decode_concept(self, concepts: t.Tensor) -> np.ndarray:
		return self.domain_properties.reshape(-1)[concepts.cpu().numpy()]

	def decode_domain(self, domain: int) -> str:
		return self.domains[domain]

	def __getitem__(self, i) -> Tuple[t.Tensor, t.Tensor]:
		return self.add_offset(self.concepts[i]), self.instances[i]

	def preprocess(
		self, x: t.Tensor, concept: t.Tensor
		) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
		false_concept = t.randint_like(
			concept, 0, self.num_properties - 1, device=concept.device, dtype=t.long # -1 to avoid the true concept
		)
		false_concept[false_concept >= self.remove_offset(concept)] += 1
		false_concept = self.add_offset(false_concept)

		x = t.cat([x, x])
		concept = t.cat([concept, false_concept])
		y = t.cat(
			[
				t.ones(x.shape[0] // 2, dtype=t.double, device=x.device),
				t.zeros(x.shape[0] // 2, dtype=t.double, device=x.device),
			]
		)

		return x, concept, y


if __name__ == "__main__":
	dataset = ProductConceptDataset("data/shapes/val", t.device("cuda:0"))
	print(dataset.domain_properties)
