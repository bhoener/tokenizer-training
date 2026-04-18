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
    def __init__(self, d_model: int, n_heads: int, xsa: bool = False):
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
        
        self.xsa = xsa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        Q = norm(self.wq(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2))
        K = norm(self.wk(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2))
        Q, K = self.rope(Q), self.rope(K)
        V = self.wv(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        
        # https://arxiv.org/abs/2603.09078
        if self.xsa:
            Vn = F.normalize(V, dim=-1) 
            attn_scores = attn_scores - (attn_scores * Vn).sum(dim=-1, keepdim=True) * Vn 

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
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_res: bool = False,
        layer_number: int | None = None,
        block_size: int = 8,
        xsa: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.attn_res = attn_res
        self.layer_number = layer_number
        self.block_size = block_size

        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, xsa=xsa)
        self.mlp = SwiGLU(d_in=d_model, d_h=d_model * 4, d_out=d_model)

        if attn_res:
            self.q_attn = nn.Linear(d_model, 1, bias=False)
            self.q_mlp = nn.Linear(d_model, 1, bias=False)

    # see https://arxiv.org/abs/2603.15031
    def compute_res_block(
        self, blocks: list[torch.Tensor], partial_block: torch.Tensor, q: nn.Linear
    ) -> torch.Tensor:
        V = torch.stack(blocks + [partial_block])

        K = norm(V)
        alphas = torch.einsum("n b l d, d -> n b l", K, q.weight.squeeze()).softmax(0)
        out = torch.einsum("n b l d, n b l -> b l d", V, alphas)
        return out

    def forward(
        self,
        x: list[torch.Tensor] | torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if self.attn_res:
            partial_block = hidden_states
            h = self.compute_res_block(x, partial_block, self.q_attn)

            if self.layer_number % (self.block_size // 2) == 0:
                x.append(partial_block)
                partial_block = None

            attn_out = self.mha(norm(h))

            partial_block = (
                partial_block + attn_out if partial_block is not None else attn_out
            )

            h = self.compute_res_block(x, partial_block, self.q_mlp)
            mlp_out = self.mlp(norm(h))

            partial_block = partial_block + mlp_out

            return x, partial_block
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
        block_size: int = 8,
        xsa: bool = False,
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
                DecoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    attn_res=attn_res,
                    layer_number=i,
                    block_size=block_size,
                    xsa=xsa,
                )
                for i in range(n_layers)
            ]
        )

        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, x: torch.Tensor, ys: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor]:
        x = self.emb(x)

        if self.attn_res:
            blocks = [x]
            hidden_states = x
            for layer in self.layers:
                blocks, hidden_states = layer(blocks, hidden_states)
            x = hidden_states
        else:
            for layer in self.layers:
                x = layer(x)

        logits = self.out_proj(x)

        if ys is None:
            return logits

        loss = F.cross_entropy(logits.view(-1, self.vocab_size), ys.view(-1))

        return logits, loss
