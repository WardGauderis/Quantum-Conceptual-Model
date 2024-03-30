from os.path import join

import numpy as np
import torch
from pandas import read_csv
from skimage.io import imread_collection
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple

class EntangledConceptDataset(Dataset):
	def __init__(self, name: str, concept_name: str, device: torch.device):
		self.name = name

		self.concepts = read_csv(join(name, "labels.csv"), dtype="category")

		self.concepts["twike"] = ((self.concepts["color"] == "red") & (self.concepts["shape"] == "circle")) | (
				(self.concepts["color"] == "blue") & (self.concepts["shape"] == "square"))
		self.concepts["red"] = self.concepts["color"] == "red"
		self.concepts["red_circle"] = (self.concepts["color"] == "red") & (self.concepts["shape"] == "circle")
		self.concepts["red_blue"] = (self.concepts["color"] == "red") | (self.concepts["color"] == "blue")
		self.concepts["red_or_circle"] = (self.concepts["color"] == "red") | (self.concepts["shape"] == "circle")

		self.concepts = torch.tensor(self.concepts[concept_name], dtype=torch.double, device=device)
		print(f"Balance of {self.name}-{concept_name} dataset: {self.concepts.mean().item()} true")

		self.transform = transforms.Compose([
			transforms.ToTensor()
		])

		self.instances = imread_collection([join(self.name, f"{i}.png") for i in range(len(self.concepts))],
										conserve_memory=False)
		self.instances = torch.stack([self.transform(image) for image in self.instances])
		self.instances = self.instances.to(device)

	def __len__(self) -> int:
		return len(self.concepts)

	def preprocess(self, x: torch.Tensor, concept: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		return x, torch.zeros_like(concept, dtype=torch.long), concept

	def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.instances[i], self.concepts[i]


if __name__ == "__main__":
	dataset = EntangledConceptDataset("data/shapes/val", "twike", "cuda:0")
	print(len(dataset))