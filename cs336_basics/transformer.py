import torch
import torch.nn as nn

from einops import einsum, reduce
from math import sqrt
from torch import Tensor
from jaxtyping import Bool, Float, Int


class Linear(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        W = torch.empty((d_out, d_in), dtype=dtype, device=device)
        std = sqrt(2 / (d_in + d_out))
        trunc = 3 * std
        nn.init.trunc_normal_(tensor=W, std=std, a=-trunc, b=trunc)
        self.W = nn.Parameter(W)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        W = torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        std = 1
        trunc = 3 * std
        nn.init.trunc_normal_(tensor=W, std=std, a=-trunc, b=trunc)
        self.W = nn.Parameter(W)

    def forward(self, x: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_out"]:
        return self.W[x]


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        G = torch.ones((d_model), dtype=dtype, device=device)
        self.G = nn.Parameter(G)

    def forward(
        self, x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        x = x.to(torch.float32)
        squared = x * x
        summed = reduce(squared, "... d_model -> ... 1", "mean")
        rms = torch.sqrt(summed + self.eps)
        return (x / rms * self.G).to(x.dtype)
