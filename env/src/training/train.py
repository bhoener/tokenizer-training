import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import math
import os
import yaml
from model import GPT
from dataloader import DataLoader
from tokenizer import Tokenizer


@torch.no_grad()
def calc_bpb(model: GPT, dl: DataLoader, enc: Tokenizer, steps: int = 10):
    bpb_tot = 0
    
    for _ in range(steps):
        xs, ys = dl.next()

        logits = model(xs)

        per_token_loss = (
            F.cross_entropy(
                logits.view(-1, logits.size(-1)), ys.view(-1), reduction="none"
            )
            .view(-1)
        )

        token_bytes = torch.tensor([len(enc.decode_single(token).encode("utf-8")) for token in xs.view(-1).cpu().numpy()])
        
        tokens_per_byte = xs.numel() / token_bytes.sum()

        loss = per_token_loss.mean()
        
        bpb = tokens_per_byte * loss / math.log(2)
        
        bpb_tot += bpb
    return bpb_tot / steps


def train() -> None:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_file", type=str, default=None)

    parser.add_argument("--train_time_minutes", type=int, default=60)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)

    parser.add_argument("--data_dir", type=str, default="data/outputs/fineweb/")

    parser.add_argument("--adam_lr", type=float, default=3e-4)
    parser.add_argument("--muon_lr", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--cooldown_frac", type=float, default=0.1)

    parser.add_argument("--vocab_size", type=int, default=32768)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    
    parser.add_argument("--attention_residuals", type=bool, default=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", type=bool, default=True)

    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="ImprovedTransformer")
    parser.add_argument("--log_every", type=int, default=10)
    
    parser.add_argument("--save_dir", type=str, default="saved_models")

    args = parser.parse_args()
    
    if args.config_file is not None:
        with open(args.config_file, "r", encoding="utf-8") as f:
            for k, v in yaml.load(f, Loader=yaml.SafeLoader).items():
                setattr(args, k, v)
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    train_start = time.time()

    device = torch.device(args.device)

    torch.set_default_device(device)

    torch.set_float32_matmul_precision("high")

    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, config=args)

    def get_lr(it: int, time: float, max_lr: float) -> float:
        min_lr = max_lr * 0.01
        prog = (time / 60) / (args.train_time_minutes)
        if it <= args.warmup_steps:
            return min_lr + (it / args.warmup_steps) * max_lr
        if prog > 1 - args.cooldown_frac:
            return max_lr + (min_lr - max_lr) * (
                (prog - (1 - args.cooldown_frac)) / args.cooldown_frac
            )
        return max_lr

    model = GPT(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        attn_res=args.attn_res,
    ).to(device)

    for param in model.parameters():
        if param.ndim == 2 and param.size(0) == param.size(1):
            nn.init.orthogonal_(param)
        elif param.ndim == 2 and param.size(1) == args.vocab_size:
            nn.init.kaiming_normal_(param, gain=0.01)

    if args.compile:
        model = torch.compile(model)

    dl = DataLoader(
        args.data_dir,
        args.micro_batch_size,
        args.seq_len,
        device=device,
    )

    optimizers = {
        torch.optim.Muon(
            param for param in model.parameters() if param.ndim == 2
        ): args.muon_lr,
        torch.optim.AdamW(
            param for param in model.parameters() if param.ndim != 2
        ): args.adam_lr,
    }

    step = 0
    while (train_time := time.time() - train_start) / 60 < args.train_time_minutes:
        step_t0 = time.time()
        loss_accum = 0.0

        for micro_step in range(args.grad_accum_steps):
            xs, ys = dl.next()

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(xs, ys)

            loss = loss / args.grad_accum_steps

            loss.backward()

            loss_accum += loss.detach().item()

        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        for optim, max_lr in optimizers.items():
            for param_group in optim.param_groups:
                param_group["lr"] = get_lr(step, train_time, max_lr)
            optim.step()
            optim.zero_grad()

        tok_s = (
            args.micro_batch_size
            * args.seq_len
            * args.grad_accum_steps
            / (time.time() - step_t0)
        )
        
        if args.wandb:
            run.log(
                {
                    "step": step,
                    "loss": loss_accum,
                    "time": train_time,
                    "norm": norm.item(),
                    "tok/s": tok_s,
                    "lr": get_lr(step, train_time, args.muon_lr),
                }
            )

        if step % args.log_every == 0:
            print(
                f"step: {step:8d} | loss: {loss_accum:8.4f} | norm: {norm.item():8.4f} | time: {train_time:8.2f} | tok/s: {tok_s:8f} | lr: {get_lr(step, train_time, args.muon_lr):8.6f}"
            )

        step += 1

    enc = Tokenizer("src/saved_tokenizers/main/vocab.txt")

    print("Final BPB:", calc_bpb(model, dl, enc))
    
    torch.save(model.state_dict(), os.path.join(args.save_dir, (run.name if args.wandb else "model") + ".pth"))


if __name__ == "__main__":
    train()
