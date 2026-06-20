import torch
import torch.nn as nn

from einops import einsum, reduce, rearrange, repeat
from math import sqrt
from torch import Tensor
from jaxtyping import Bool, Float, Int
from typing import Optional
from collections.abc import Callable, Iterable


class Linear(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        weight = torch.empty((d_out, d_in), dtype=dtype, device=device)
        std = sqrt(2 / (d_in + d_out))
        trunc = 3 * std
        nn.init.trunc_normal_(tensor=weight, std=std, a=-trunc, b=trunc)
        self.weight = nn.Parameter(weight)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        weight = torch.empty(
            (num_embeddings, embedding_dim), dtype=dtype, device=device
        )
        std = 1
        trunc = 3 * std
        nn.init.trunc_normal_(tensor=weight, std=std, a=-trunc, b=trunc)
        self.weight = nn.Parameter(weight)

    def forward(self, x: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_out"]:
        return self.weight[x]


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
        self.weight = nn.Parameter(G)

    def forward(
        self, x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        x = x.to(torch.float32)
        squared = x * x
        summed = reduce(squared, "... d_model -> ... 1", "mean")
        rms = torch.sqrt(summed + self.eps)
        result = x / rms * self.weight
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

        self.w1 = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_in=d_ff, d_out=d_model, device=device, dtype=dtype)

    def forward(
        self, x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        return self.w2(swiglu(self.w1(x)) * self.w3(x))


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
        super().__init__()

        inverse_frequency = theta ** -(torch.arange(0, d_keys, 2) / d_keys)
        sequence_positions = torch.arange(max_seq_len)
        # Same as angles = torch.outer(sequence_positions, inverse_frequency)
        angles = einsum(
            sequence_positions,
            inverse_frequency,
            "seq_len, d_keys_half -> seq_len d_keys_half",
        )
        angles = repeat(angles, "... d_keys_half -> ... (two d_keys_half)", two=2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

        assert self.cos.shape[-1] == d_keys

    def forward(
        self,
        keys: Float[Tensor, " ... sequence_length d_keys"],
        token_positions: Int[Tensor, " ... sequence_length"],
        rotate_adjacent_dims: False,
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        def rotate_half(x):
            l, r = torch.chunk(x, 2, dim=-1)
            return torch.cat([-r, l], dim=-1)

        if rotate_adjacent_dims:
            # Convert from split-half RoPE to adjacent-dim version
            keys = rearrange(keys, "... (half two) -> ... (two half)", two=2)

        rotated = rotate_half(keys)
        out = cos * keys + sin * rotated

        if rotate_adjacent_dims:
            # Undo from adjacent-dim to split-half
            out = rearrange(out, "... (two half) -> ... (half two)", two=2)

        return out


def softmax(x: Float[Tensor, " ... "], dim: int):
    max = torch.amax(x, dim=dim, keepdim=True)
    x = x - max
    z = x.exp()
    return z / z.sum(dim=dim, keepdim=True)


def cross_entropy(
    logits: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    batch_size = targets.shape[-1]

    max = torch.amax(logits, dim=-1, keepdim=True)
    logits = logits - max

    ce = logits.exp().sum(dim=-1).log() - logits[torch.arange(batch_size), targets]

    return ce.mean()


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    attend_mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    QK = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    d_k = K.shape[-1]
    scores = QK / sqrt(d_k)

    scores = scores.masked_fill(~attend_mask, float("-inf"))

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
        super().__init__()

        self.num_heads = num_heads
        self.q_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.output_proj = Linear(
            d_in=d_model, d_out=d_model, device=device, dtype=dtype
        )
        self.device = device

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, "... seq_len"] = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        def partition_heads(x):
            return rearrange(
                x,
                " ... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
                num_heads=self.num_heads,
            )

        def concat_heads(x):
            return rearrange(
                x,
                " ... num_heads seq_len d_k -> ... seq_len (num_heads d_k)",
                num_heads=self.num_heads,
            )

        Q, K, V = partition_heads(Q), partition_heads(K), partition_heads(V)

        Q, K = self._rotate_qk(Q, token_positions), self._rotate_qk(K, token_positions)

        seq_len = x.shape[-2]
        attend_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool)
        )

        att = scaled_dot_product_attention(Q, K, V, attend_mask)
        att = concat_heads(att)
        return self.output_proj(att)

    def _rotate_qk(
        self,
        keys: Float[Tensor, " ... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ):
        return keys


class MultiheadSelfAttentionWithRope(MultiheadSelfAttention):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )
        self.rope = RoPE(
            d_keys=d_model // num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    def _rotate_qk(
        self,
        keys: Float[Tensor, " ... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ):
        return self.rope(keys, token_positions, rotate_adjacent_dims=True)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model=d_model)
        self.attn = MultiheadSelfAttentionWithRope(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = Feedforward(
            d_model=d_model,
            d_ff=d_ff,
        )

    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]):
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len)

        x = self.attn(self.ln1(x), token_positions=token_positions) + x
        x = self.ffn(self.ln2(x)) + x
        return x


class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )
        self.layers = nn.Sequential(
            *(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
                for _ in range(num_layers)
            )
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(d_in=d_model, d_out=vocab_size)

    def forward(
        self, x: Float[Tensor, " batch sequence_length d_model"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.lm_head(self.ln_final(x))
        return x


class SGDOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                t = state.get("t", 0)

                param.data -= lr / sqrt(t + 1) * param.grad.data

                state["t"] = t + 1
