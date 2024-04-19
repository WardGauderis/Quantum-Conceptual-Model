#%%

import torch as t
from torch import nn, Tensor
import pennylane as qml
import torch 
from jaxtyping import Float, Int


#%%

class VQC(nn.Module):
    def __init__(self):
        super().__init__()
        
        num_concepts = 12
        embedding_dim = 12
        
        self.weights = nn.Embedding(num_concepts, embedding_dim, scale_grad_by_freq=True)
        nn.init.uniform_(self.weights.weights, 0, 2 * torch.pi)
        
    def forward(instance: Float[Tensor, "batch encoding"], concept: Int[Tensor, ""]) -> Float[Tensor, "batch label"]:
        return self.weights(instance)