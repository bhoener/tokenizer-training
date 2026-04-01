from model import RoPE
import matplotlib.pyplot as plt

rope = RoPE(16)

x = torch.randn(8, 16)

out = rope(x)

plt.imshow(out)