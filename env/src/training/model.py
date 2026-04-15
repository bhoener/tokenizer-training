import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class RoPE(nn.Module):
    def __init__(
        self,
        d: int,
        base: float = 10000,
        device: torch.device | None = None,
        cache_len: int = 1024,
    ):
        super().__init__()
        self.d = d
        self.base = base
        self.device = device
        self.thetas = base ** (-2.0 * (torch.arange(0, d / 2).repeat_interleave(2)) / d)
        if device is not None:
            self.thetas = self.thetas.to(device)

        self.register_buffer(
            "angles", torch.einsum("i, j -> ij", torch.arange(cache_len), self.thetas)
        )

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        odds = x[..., 1::2]
        evens = -x[..., ::2]
        return torch.cat((evens, odds), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, d = x.size()

        return x * torch.cos(self.angles[:L, :]) + self.rotate(x) * torch.sin(
            self.angles[:L, :]
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0

        self.d_head = d_model // n_heads

        self.rope = RoPE(self.d_head)

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        Q = norm(self.wq(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2))
        K = norm(self.wk(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2))
        Q, K = self.rope(Q), self.rope(K)
        V = self.wv(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        return self.wo(attn_scores.permute(0, 2, 1, 3).contiguous().view(B, L, D))


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out

        self.W = nn.Linear(d_in, d_h, bias=False)
        self.V = nn.Linear(d_in, d_h, bias=False)

        self.W2 = nn.Linear(d_h, d_out, bias=False)

    def swish(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return x * F.sigmoid(beta * x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.swish(self.W(x)) * self.V(x))


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_res: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.mlp = SwiGLU(d_in=d_model, d_h=d_model * 4, d_out=d_model)

        if attn_res:
            self.q_attn = nn.Parameter(torch.randn(d_model))
            self.q_mlp = nn.Parameter(torch.randn(d_model))

    def compute_res(self, hidden_states: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        k = norm(hidden_states.permute(1, 2, 0, 3))  # (N, B, L, D) -> (B, L, N, D)
        alphas = F.softmax(torch.einsum("b l n d, d -> b n l", k, q), dim=1)
        out = torch.einsum("n b l d, b n l -> b l d", hidden_states, alphas)
        return out

    def forward(self, x: torch.Tensor | None) -> torch.Tensor:
        if x.ndim > 3:
            res_out_mha = self.compute_res(x, self.q_attn)
            res_out_mha = res_out_mha + self.mha(norm(res_out_mha))
            x = torch.cat((x, res_out_mha.unsqueeze(0)))
            res_out_mlp = self.compute_res(x, self.q_mlp)
            res_out_mlp = res_out_mlp + self.mlp(norm(res_out_mlp))
            x = torch.cat((x, res_out_mlp.unsqueeze(0)))
            return x
        else:
            x = x + self.mha(norm(x))
            x = x + self.mlp(norm(x))
            return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        attn_res: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attn_res = attn_res

        self.emb = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(d_model=d_model, n_heads=n_heads, attn_res=attn_res)
                for _ in range(n_layers)
            ]
        )

        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, x: torch.Tensor, ys: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor]:
        x = self.emb(x)

        if self.attn_res:
            x = x.unsqueeze(0)

        for layer in self.layers:
            x = layer(x)

        if self.attn_res:
            x = x[-1]

        logits = self.out_proj(x)

        if ys is None:
            return logits

        loss = F.cross_entropy(logits.view(-1, self.vocab_size), ys.view(-1))

        return logits, loss
