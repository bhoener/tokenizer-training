import torch
import torch.nn.functional as F
from model import GPT
import argparse
import yaml
from tokenizer import Tokenizer


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file", type=str, default="src/training/configs/sample_inference.yaml"
    )

    args = parser.parse_args()

    with open(args.config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device(config["system"]["device"])

    torch.set_default_device(device)
    torch.set_float32_matmul_precision("high")

    model = GPT(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
    )

    model.load_state_dict(
        {
            k.replace("_orig_mod.", ""): v
            for k, v in torch.load(config["filepath"]).items()
        }
    )

    enc = Tokenizer(config["tokenizer"]["filepath"])

    while (prompt := input("Enter a prompt: ")).lower() not in {"q", "quit", "exit"}:
        freqs = {}

        prompt_tokens = [3] + enc.encode(prompt)

        while (len(prompt_tokens)) < config["sampling"]["seq_len"] - 1:
            x = torch.tensor(prompt_tokens, device=device).unsqueeze(0)
            
            logits = model(x)

            logits = logits / config["sampling"]["temperature"]
            
            logits = logits[0, -1]

            if config["sampling"]["top_k"] > 1:
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask[torch.topk(logits, config["sampling"]["top_k"]).indices] = False
                logits[mask] = float("-inf")

            for token, freq in freqs.items():
                logits[token] -= config["sampling"]["freq_penalty"] * freq
                freq *= config["sampling"]["freq_decay"]

            probs = F.softmax(logits, dim=-1)

            idx = torch.multinomial(probs, num_samples=1).item()

            freqs[idx] = 1 if idx not in freqs else freqs[idx] + 1

            if idx == 3:
                break

            prompt_tokens.append(idx)

            print(enc.decode_single(idx), end="")

        print()

if __name__ == "__main__":
    main()
