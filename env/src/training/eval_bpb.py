import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT

from dataloader import DataLoader
import math
from tokenizer import Tokenizer


@torch.no_grad()
def calc_bpb(model: GPT, dl: DataLoader, enc: Tokenizer, steps: int = 10) -> float:
    loss_sum = 0.0
    bytes_sum = 0.0

    for _ in range(steps):
        xs, ys = dl.next()

        logits = model(xs)

        per_token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), ys.view(-1), reduction="none"
        ).view(-1)

        token_bytes = torch.tensor(
            [
                len(enc.decode_single(token).encode("utf-8"))
                for token in xs.view(-1).cpu().numpy()
            ]
        )

        loss_sum += per_token_loss.sum().item()
        bytes_sum += token_bytes.sum().item()

    return (loss_sum / bytes_sum) / math.log(2)


def main() -> None:
    model = GPT(
        vocab_size=32768,
        d_model=768,
        n_heads=12,
        n_layers=12,
    )

    model.load_state_dict(
        {
            k.replace("_orig_mod.", ""): v
            for k, v in torch.load("saved_models/sweet-fog-133.pth").items()
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = DataLoader("data/outputs/code/", 4, 512, device=device)
    
    model = model.to(device)

    enc = Tokenizer("src/saved_tokenizers/updated/vocab.txt")

    print("BPB:", calc_bpb(model, dl, enc))


if __name__ == "__main__":
    main()
