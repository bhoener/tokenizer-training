import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import Tokenizer
from sympy import isprime


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class CompressedTokenizer:
    def __init__(self, source_tokenizer: Tokenizer):
        self.source_tokenizer = source_tokenizer

        self.vocab = {
            token_id: self.simplify(source_tokenizer.decode_single(token_id)) for token_id in range(len(source_tokenizer.vocab.keys()) + 255)
        }

        self.lut = torch.empty(len(self.vocab.keys()) + 256)
        for i in range(256):
            self.lut[i] = i
        for i, token in enumerate(self.vocab.keys()):
            self.lut[token] = i + 256
    def simplify(self, token: str) -> str:
        token = token.lower().strip().replace("'", "")

    @torch.no_grad()
    def encode(self, ids) -> torch.Tensor:
        return self.lut[ids].long()


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor):
        B, L, D = x.size()

        return self.conv(F.pad(x.reshape(D, -1), (self.kernel_size - 1, 0))).reshape(
            B, L, D
        )


class EngramHash(nn.Module):
    def __init__(self, N: int, K: int, M: list[int], d: int, tokenizer: Tokenizer):
        super().__init__()
        self.compressed_tok = CompressedTokenizer(tokenizer)
        self.max_ngram = N
        self.num_ngrams = N - 2
        self.heads_per_ngram = K
        self.vocab_per_ngram = M
        self.d = d

        self.head_sizes = self.get_prime_head_sizes()

        self.register_buffer("n_mults", torch.randint(0, 2**16 - 1, (self.max_ngram,)))

    @torch.no_grad()
    def hash_sequence(self, x: torch.Tensor) -> torch.Tensor:
        compressed = self.compressed_tok.encode(x)
        # (N, B, L)
        ngram_shifts = [self.pad(compressed, n) for n in range(self.max_ngram, 0, -1)]
        # [0, 0, 1, 2]
        # [0, 1, 2, 3]
        # [1, 2, 3, 4]

        # if n = 2, we take [-2:]
        # [0, 1, 2, 3]
        # [1, 2, 3, 4]
        # -> [0, 1]
        out = []
        for n in range(2, self.max_ngram + 1):
            ngram_out = []
            shifted = ngram_shifts[-n:]
            h = shifted[0] * self.n_mults[0]
            for i in range(1, n):
                h = torch.bitwise_xor(self.n_mults[n - 1] * shifted[i], h)

            for k in range(self.heads_per_ngram):
                ngram_out.append(torch.remainder(h, self.head_sizes[n - 2][k]))

            out.append(ngram_out)

        return torch.stack([torch.stack(n, axis=-1) for n in out])

    def get_next_prime(self, start: int) -> int:
        return start if isprime(start) else self.get_next_prime(start + start % 2 + 1)

    def get_prime_head_sizes(self) -> list[list[int]]:
        out = []

        for vocab_size in self.vocab_per_ngram:
            ngram_out = []

            start = vocab_size - 1

            for _ in range(self.heads_per_ngram):
                prime = self.get_next_prime(start)
                ngram_out.append(prime)
                start = prime + 1
            out.append(ngram_out)
        return out

    def pad(self, ins: torch.Tensor, n: int) -> torch.Tensor:
        return F.pad(ins, (n, 0, 0, 0))[:, :-n]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hash_sequence(x)


class EngramEmbedding(nn.Module):
    def __init__(self, head_sizes: list[list[int]], d: int):
        super().__init__()
        self.head_sizes = head_sizes
        self.d = d

        self.total_size = sum(sum(ngram) for ngram in head_sizes)

        self.emb = nn.Embedding(self.total_size, d)

        # if we get a hash of
        # [
        #   [[[1, 2], [3, 4]], [[4, 5], [6, 7]]],
        #   [[[3, 6], [9, 8]], [[7, 4], [2, 1]]],
        # ] (2 x 2 x 2 x 2) (N x B x L x K)
        # offset table should be (N x 1 x 1 x K)?

        offsets = []

        for ngram in head_sizes:
            total = 0
            ngram_offsets = []
            for size in ngram:
                ngram_offsets.append(total)
                total += size

            offsets.append(ngram_offsets)

        self.register_buffer(
            "offsets", torch.tensor(offsets).long().unsqueeze(1).unsqueeze(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x + self.offsets)


class Engram(nn.Module):
    def __init__(
        self, N, K, M, d, d_model, tokenizer: Tokenizer, conv_kernel_size: int = 4
    ):
        super().__init__()
        self.hash = EngramHash(N=N, K=K, M=M, d=d, tokenizer=tokenizer)

        self.d = d
        self.d_model = d_model
        self.d_per_head = d // (K * (N - 1))

        self.emb = EngramEmbedding(self.hash.head_sizes, self.d_per_head)

        self.conv = CausalConv1d(d_model, d_model, kernel_size=conv_kernel_size)

        self.wk = nn.Linear(self.d, self.d_model)
        self.wv = nn.Linear(self.d, self.d_model)

    def forward(self, x: torch.Tensor, indices) -> torch.Tensor:
        B, L, D = x.size()
        idx = self.hash(indices)
        embeddings = self.emb(idx).view(B, L, -1)

        k = self.wk(embeddings)
        v = self.wv(embeddings)

        alphas = F.sigmoid((norm(x) * norm(k)).sum(dim=-1) / D**0.5)

        v_gated = alphas.unsqueeze(-1) * v

        Y = F.silu(self.conv(v_gated)) + v_gated

        return Y


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
            attn_scores = (
                attn_scores - (attn_scores * Vn).sum(dim=-1, keepdim=True) * Vn
            )

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
        engram: Engram | None = None,
    ):
        super().__init__()
        assert not attn_res or engram is None, "engram with attn_res not supported"
        self.d_model = d_model
        self.n_heads = n_heads

        self.attn_res = attn_res
        self.layer_number = layer_number
        self.block_size = block_size

        self.engram = engram
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
        hidden_states_or_indices: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if self.attn_res:
            partial_block = hidden_states_or_indices
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
            if self.engram is not None:
                x = x + self.engram(norm(x), hidden_states_or_indices)
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
        attn_res: bool = False,
        block_size: int = 8,
        xsa: bool = False,
        engram: bool = False,
        engram_max_n: int = 3,
        engram_heads: int = 8,
        engram_vocab_sizes: list[int] = [512, 2048],
        engram_d: int | None = None,
        engram_tokenizer: Tokenizer = None,
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
                    engram=Engram(
                        N=engram_max_n,
                        K=engram_heads,
                        M=engram_vocab_sizes,
                        d=engram_d if engram_d is not None else d_model,
                        d_model=d_model,
                        tokenizer=engram_tokenizer,
                    )
                    if engram and i == 2
                    else None,
                )
                for i in range(n_layers)
            ]
        )

        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, x: torch.Tensor, ys: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor]:
        idx = x
        x = self.emb(x)

        if self.attn_res:
            blocks = [x]
            hidden_states = x
            for layer in self.layers:
                blocks, hidden_states = layer(blocks, hidden_states)
            x = hidden_states
        else:
            for layer in self.layers:
                x = layer(x, idx)

        logits = self.out_proj(x)

        if ys is None:
            return logits

        loss = F.cross_entropy(logits.view(-1, self.vocab_size), ys.view(-1))

        return logits, loss
