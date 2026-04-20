# Engram Notes

Important paper: https://arxiv.org/abs/1709.03933

> we employ $K$ distinct hash heads for each $N$-gram order $N$. Each head $k$ maps the compressed context to an index within an embedding table $E_{n, k}$ (of prime size $M_{n, k}$) via a deterministic function $\phi_{n, k}$

$\phi_{n, k} : \mathbb{R}^{n \times d} \rightarrow \mathbb{Z}_{\ge 0}$ is the $k$-th hash function for $N$-gram order $n$ which maps the $n$ embedding vectors to indices in a hash table.

It seems that they retrieve $n \times k$ vectors of length $d_{model}$ per token and concatenate them all before projection back down to $d_{model}$.

Maybe we choose a fixed $d_{mem}$, $n$, and $k$, and figure out the dimensions of the embeddings based on that?

We cannot do:
```python
token_cache = {}

for i in vocab:
    for j in vocab:
        token_cache[(i, j)] = some_vector
```

So instead we have a fixed size table of embeddings and use multiple hash heads so that even if collisions occur they are not detrimental:
```python
class Engram(nn.Module):
  def __init__(self, N: int, K: int, M: int, d_per_n: int):
    super().__init__()
    self.N = N
    self.K = K
    self.M = M
    self.d = d_per_n // K

    self.emb = nn.Parameter(torch.randn(N - 2, K, M, d))

  def hash(self, x: tuple[int], k: int) -> int:
    h = 14695981039346656037

    for b in bytes(x):
      h = h * 1099511628211
      h = h * k
      h = h ^ b
    
    return h % self.M

  def forward(self, ngrams: list[tuple[int]]) -> torch.Tensor:
    idx = [[self.hash(ngram, k) for k in range(self.K)] for ngram in ngrams]
    return self.emb[len(ngrams[0]) - 2, :, idx].view(len(ngrams), -1)
```

It seems like each hash head has a different (prime) size, so the lookup cannot be a big $n \times k \times M \times d$ lookup table.