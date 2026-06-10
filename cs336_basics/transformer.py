import torch
import torch.nn as nn

from einops import einsum, reduce, rearrange, repeat
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
        result = x / rms * self.G
        return result.to(x.dtype)


def swiglu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x * torch.sigmoid(x)


class Feedforward(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.gate_proj = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)
        self.up_proj = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)
        self.down_proj = Linear(d_in=d_ff, d_out=d_model, device=device, dtype=dtype)

    def forward(
        self, x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        return self.down_proj(swiglu(self.gate_proj(x)) * self.up_proj(x))


class RoPE(nn.Module):

    def __init__(
        self,
        d_keys: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        - there is one R_pos of size d_keys * d_keys, for each sequence position pos
        angle(pos, dim) = pos / {theta ^ {2i / d_keys}}
        """
        inverse_frequency = theta ** (torch.arange(0, d_keys / 2, 2) / d_keys)
        sequence_positions = torch.arange(max_seq_len)
        # angles = torch.outer(sequence_positions, inverse_frequency)
        angles = einsum(
            sequence_positions,
            inverse_frequency,
            "seq_len, d_keys_half -> seq_len d_keys_half",
        )
        angles = repeat(angles, "... d_keys_half -> ... (d_keys_half two)", two=2)
        self.register_buffer("sins", angles.sin(), persistent=False)
        self.register_buffer("coss", angles.cos(), persistent=False)

    def forward(
        self,
        in_query_or_key: Float[Tensor, " ... sequence_length d_keys"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ):
        raise NotImplementedError


def softmax(x: Float[Tensor, " ... "], dim: int):
    max = torch.amax(x, dim=dim, keepdim=True)
    z = (x - max).exp()
    return z / z.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    QK = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    d_k = K.shape[-1]
    scores = QK / sqrt(d_k)
    scores = scores.masked_fill(~mask, float("-inf"))
    scores = softmax(scores, dim=-1)
    return einsum(scores, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiheadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.q_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)

    def forward(
        self, x: Float[Tensor, " ... sequence_length d_model"]
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        Q = rearrange(
            self.q_proj(x),
            " ... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k",
        )
        K = rearrange(
            self.k_proj(x),
            " ... sequence_length ( d_k) -> ... num_heads sequence_length d_k",
        )
        V = self.v_proj(x)
