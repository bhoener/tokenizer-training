import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import math
from src.training.model import GPT
from src.training.dataloader import DataLoader

def train() -> None:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_time_minutes", type=int, default=60)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=512)
    
    parser.add_argument("--data_dir", type=str, default="data/outputs/fineweb/")
    
    parser.add_argument("--adam_lr", type=float, default=3e-4)
    parser.add_argument("--muon_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    
    parser.add_argument("--vocab_size", type=int, default=32768)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", type=bool, default=False)
    
    args = parser.parse_args()
    
    train_start = time.time()
    
    device = torch.device(args.device)
    
    torch.set_default_device(device)
    
    torch.set_float32_matmul_precision("high")
    
    def get_lr(it: int, max_lr: float) -> float:
        min_lr = max_lr * 0.01
        if it <= args.warmup_steps:
            return min_lr + (it / args.warmup_steps) * it
        return max_lr
    
    model = GPT(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)
    
    if args.compile:
        model = torch.compile(model)
    
    dl = DataLoader(
        args.data_dir,
        args.micro_batch_size,
        args.seq_len,
        device=device,
    )
    
    optimizers = [
        torch.optim.Muon(param for param in model.parameters() if param.ndim() == 2),
        torch.optim.AdamW(param for param in model.parameters() if param.ndim() != 2),
    ]
    
    step = 0
    while (time.time() - train_start) / 60 < args.train_time_minutes:
        loss_accum = 0.0
        for micro_step in range(args.grad_accum_steps):
            xs, ys = dl.next()

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, loss = model(xs, ys)


            loss = loss / args.grad_accum_steps
            
            loss.backward()
            
            loss_accum += loss.detach().item()
        
        for optim in optimizers:
            for param_group in optim.param_groups():
                param_group["lr"] = get_lr(step)
    
        step += 1
    