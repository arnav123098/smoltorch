from .module import Module
from .linear import Linear
import torch
import torch.nn.functional as F

class CausalSelfAttn(Module):
    def __init__(
        self,
        n_emb: int,
        n_head: int
    ):
        super().__init__()
        self.c_attn = Linear(n_emb, 3 * n_emb)
        self.proj = Linear(n_emb, n_emb)
        self.n_emb = n_emb
        self.n_head = n_head
        
    def __call__(
            self,
            input: torch.Tensor
        ) -> torch.Tensor:
        B, T, C = input.size()
        q, k, v = self.c_attn(input).split(self.n_emb, dim=-1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        self.out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        self.out = self.out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out
