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

        per_token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), ys.view(-1), reduction="none"
        ).view(-1)

        token_bytes = torch.tensor(
            [
                len(enc.decode_single(token).encode("utf-8"))
                for token in xs.view(-1).cpu().numpy()
            ]
        )

        tokens_per_byte = xs.numel() / token_bytes.sum()

        loss = per_token_loss.mean()

        bpb = tokens_per_byte * loss / math.log(2)

        bpb_tot += bpb
    return bpb_tot / steps


# from https://github.com/karpathy/nanoGPT/
def estimate_mfu(total_params: int, cfg: argparse.Namespace, dt: float):
    # first estimate the number of flops we do per iteration.
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    N = total_params
    L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.d_model // cfg.n_heads, cfg.seq_len
    flops_per_token = 6 * N + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * cfg.grad_accum_steps * cfg.micro_batch_size

    flops_achieved = flops_per_iter * (1.0 / dt)  # per second
    flops_promised = 44e12  # rtx4060ti 16gb peak bf16 (might be wrong)
    mfu = flops_achieved / flops_promised
    return mfu


def save_checkpoint(
    path: str,
    model: GPT,
    args: argparse.Namespace,
    optimizers: list[torch.optim.Optimizer],
    step: int,
    train_time: float,
    wandb_run=None,
) -> None:
    model_state_dict = model.state_dict()
    optim_state_dicts = [optim.state_dict() for optim in optimizers]
    args_dict = vars(args)
    args_dict["resume_step"] = step
    args_dict["resume_time"] = train_time
    args_dict["wandb_run_id"] = wandb_run.id

    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model_state_dict, os.path.join(path, "model_state.pth"))
    for optim, state in zip(optimizers, optim_state_dicts):
        torch.save(state, os.path.join(path, f"{optim.__class__.__name__}_state.pth"))
    with open(os.path.join(path, "train_args.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml.dump(args_dict))


def get_namespace_from_yaml(filename: str) -> argparse.Namespace:
    out = argparse.Namespace()
    with open(filename, "r", encoding="utf-8") as f:
        for k, v in yaml.load(f, Loader=yaml.SafeLoader).items():
            setattr(out, k, v)
    return out


def train() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, default=None)

    parser.add_argument("--resume_from_dir", type=str, default=None)
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--resume_time", type=float, default=0)

    parser.add_argument("--finetune_from", type=str, default=None)

    parser.add_argument("--train_time_minutes", type=int, default=1 * 60)
    parser.add_argument("--micro_batch_size", type=int, default=16)
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

    parser.add_argument("--attn_res", type=bool, default=False)
    parser.add_argument("--attn_res_block_size", type=int, default=8)

    parser.add_argument("--xsa", type=bool, default=False)

    parser.add_argument("--engram", type=bool, default=False)
    parser.add_argument("--engram_max_n", type=int, default=3)
    parser.add_argument("--engram_heads", type=int, default=8)
    parser.add_argument(
        "--engram_vocab_sizes", type=list[int], default=[131072, 262144]
    )
    parser.add_argument("--engram_d", type=int, default=None)
    parser.add_argument(
        "--engram_tokenizer_dir",
        type=str,
        default="src/saved_tokenizers/updated/vocab.txt",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", type=bool, default=True)

    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="ImprovedTransformer")
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--checkpoint_mode", type=str, default="overwrite")

    args = parser.parse_args()

    if args.resume_from_dir is not None:
        resume_from = args.resume_from_dir
        args = get_namespace_from_yaml(
            os.path.join(args.resume_from_dir, "train_args.yaml")
        )
        args.resume_from_dir = resume_from
    elif args.config_file is not None:
        args = get_namespace_from_yaml(args.config_file)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    device = torch.device(args.device)

    torch.set_default_device(device)

    torch.set_float32_matmul_precision("high")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            config=args,
            resume="allow",
            id=args.wandb_run_id,
        )

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
        block_size=args.attn_res_block_size,
        xsa=args.xsa,
        engram=args.engram,
        engram_max_n=args.engram_max_n,
        engram_heads=args.engram_heads,
        engram_vocab_sizes=args.engram_vocab_sizes,
        engram_d=args.engram_d,
        engram_tokenizer=Tokenizer(args.engram_tokenizer_dir),
    ).to(device)

    if args.compile:
        model = torch.compile(model)

    if args.resume_from_dir is not None:
        model.load_state_dict(
            torch.load(os.path.join(args.resume_from_dir, "model_state.pth"))
        )
    elif args.finetune_from is not None:
        model.load_state_dict(
            torch.load(args.finetune_from)
        )

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model Parameters: {total_params / 1e6:.1f}M")

    if args.resume_from_dir is None and args.finetune_from is None:
        for param in model.parameters():
            if param.ndim == 2 and param.size(0) == param.size(1):
                nn.init.orthogonal_(param)
            elif param.ndim == 2 and param.size(1) == args.vocab_size:
                nn.init.kaiming_normal_(param, gain=0.01)

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

    if args.resume_from_dir is not None:
        for optim in optimizers.keys():
            optim.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume_from_dir, f"{optim.__class__.__name__}_state.pth"
                    )
                )
            )

    step = args.resume_step
    train_time = args.resume_time

    for _ in range(step * args.grad_accum_steps):
        dl.next()

    while train_time / 60 < args.train_time_minutes:
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

        dt = time.time() - step_t0
        tok_s = args.micro_batch_size * args.seq_len * args.grad_accum_steps / dt

        mfu = estimate_mfu(total_params, args, dt)

        if args.wandb:
            run.log(
                {
                    "step": step,
                    "loss": loss_accum,
                    "time": train_time,
                    "norm": norm.item(),
                    "tok/s": tok_s,
                    "lr": get_lr(step, train_time, args.muon_lr),
                    "mfu": mfu * 100,
                }
            )

        if step % args.log_every == 0:
            print(
                f"step: {step:8d} | loss: {loss_accum:8.4f} | norm: {norm.item():8.4f} | time: {train_time:8.2f} | tok/s: {tok_s:8.1f} | lr: {get_lr(step, train_time, args.muon_lr):8.6f} | mfu: {mfu * 100:8.2f}%"
            )

        if (
            args.save_every is not None
            and step % args.save_every == 0
            and args.save_dir is not None
            and step != args.resume_step
        ):
            save_checkpoint(
                os.path.join(
                    args.save_dir,
                    "checkpoints",
                    (run.name if args.wandb else "default")
                    + (
                        ""
                        if args.checkpoint_mode == "overwrite"
                        else f"-{step // args.save_every:04d}"
                    ),
                ),
                model,
                args,
                optimizers,
                step,
                train_time,
                run,
            )
            print("Model saved to", args.save_dir)

        step += 1

        train_time += time.time() - step_t0

    enc = Tokenizer("src/saved_tokenizers/main/vocab.txt")

    print("Final BPB:", calc_bpb(model, dl, enc))

    torch.save(
        model.state_dict(),
        os.path.join(args.save_dir, (run.name if args.wandb else "model") + ".pth"),
    )


if __name__ == "__main__":
    train()
