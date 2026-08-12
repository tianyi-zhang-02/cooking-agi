"""Train the hand-rolled Transformer.

Two tasks:

  copy  (default) -- synthetic in-context recall: the model sees
                     [BOS] p_1..p_L [SEP] p_1..p_L with p random every batch, and is
                     scored only on the second copy. The pattern is fresh noise, so
                     nothing is memorisable in the weights: solving it *requires*
                     attention (an induction head reading back across [SEP]).
                     A bigram/MLP baseline is pinned at ln(n_sym).

  text            -- char-level LM on any utf-8 file: --task text --data <path>

Usage:
    python train.py
    python train.py --task text --data some.txt --steps 2000
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from model import Config, Transformer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class CopyTask:
    """[BOS] pattern [SEP] pattern -- supervise only the repeat."""

    def __init__(self, length: int = 32, n_sym: int = 48):
        self.L, self.n_sym = length, n_sym
        self.bos, self.sep = n_sym, n_sym + 1
        self.vocab_size = n_sym + 2
        self.seq_len = 2 * length + 2

    def batch(self, bs: int, device):
        pat = torch.randint(0, self.n_sym, (bs, self.L), device=device)
        col = lambda tok: torch.full((bs, 1), tok, device=device, dtype=torch.long)
        seq = torch.cat([col(self.bos), pat, col(self.sep), pat], dim=1)
        x, y = seq[:, :-1], seq[:, 1:].clone()
        y[:, : self.L + 1] = -100  # ignore the prompt half
        return x, y

    def prompt(self, bs: int, device):
        """Just [BOS] pattern [SEP]; the model must emit the pattern back."""
        pat = torch.randint(0, self.n_sym, (bs, self.L), device=device)
        col = lambda tok: torch.full((bs, 1), tok, device=device, dtype=torch.long)
        return torch.cat([col(self.bos), pat, col(self.sep)], dim=1), pat


class CharText:
    def __init__(self, path: str, seq_len: int = 256):
        text = open(path, encoding="utf-8").read()
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.seq_len = seq_len
        stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in stoi.items()}
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        self.train, self.val = data[:n], data[n:]

    def batch(self, bs: int, device, split: str = "train"):
        d = self.train if split == "train" else self.val
        i = torch.randint(0, len(d) - self.seq_len - 1, (bs,))
        x = torch.stack([d[j:j + self.seq_len] for j in i]).to(device)
        y = torch.stack([d[j + 1:j + 1 + self.seq_len] for j in i]).to(device)
        return x, y

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# --------------------------------------------------------------------------- #
# optim
# --------------------------------------------------------------------------- #
def make_optimizer(model, lr: float, weight_decay: float = 0.1):
    # decay matmul weights, not norms/embeddings-as-vectors
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8,
    )


def lr_at(step: int, total: int, base_lr: float, warmup: int = 100) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * p))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["copy", "text"], default="copy")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-kv-head", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()

    if args.task == "copy":
        task = CopyTask()
        chance = math.log(task.n_sym)
    else:
        assert args.data, "--task text needs --data <file>"
        task = CharText(args.data, args.seq_len)
        chance = math.log(task.vocab_size)

    cfg = Config(
        vocab_size=task.vocab_size, n_layer=args.n_layer, n_head=args.n_head,
        n_kv_head=args.n_kv_head, d_model=args.d_model, max_seq_len=task.seq_len + 8,
    )
    model = Transformer(cfg).to(device)
    opt = make_optimizer(model, args.lr)

    print(f"device={device}  task={args.task}  vocab={cfg.vocab_size}  seq_len={task.seq_len}")
    print(f"model: {cfg.n_layer}L x {cfg.d_model}d, {cfg.n_head}q/{cfg.n_kv_head}kv heads, "
          f"{model.n_params():,} params ({model.n_params(True):,} non-emb)")
    print(f"chance loss (uniform) = {chance:.4f}\n")

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr)

        x, y = task.batch(args.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:5d} | loss {loss.item():7.4f} | "
                  f"lr {opt.param_groups[0]['lr']:.2e} | gnorm {gnorm:6.3f} | "
                  f"{time.time() - t0:5.1f}s")

    # ------------------------------------------------------------------ eval
    print()
    model.eval()
    if args.task == "copy":
        prompt, pat = task.prompt(32, device)
        out = model.generate(prompt, max_new_tokens=task.L, temperature=0.0)
        pred = out[:, prompt.shape[1]:]
        tok_acc = (pred == pat).float().mean().item()
        seq_acc = (pred == pat).all(dim=1).float().mean().item()
        print(f"greedy decode via KV cache on 32 unseen patterns:")
        print(f"  token accuracy    {tok_acc:.1%}")
        print(f"  sequence accuracy {seq_acc:.1%}   (chance = {1 / task.n_sym:.1%} per token)")
        print(f"  target  {pat[0, :12].tolist()} ...")
        print(f"  decoded {pred[0, :12].tolist()} ...")
    else:
        with torch.no_grad():
            vl = torch.stack([model(*task.batch(args.batch_size, device, "val"))[1]
                              for _ in range(20)]).mean()
        print(f"val loss {vl.item():.4f}  (ppl {math.exp(vl.item()):.2f})")
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
        sample = model.generate(start, max_new_tokens=400, temperature=0.8, top_k=40)
        print("-" * 60)
        print(task.decode(sample[0].tolist()))
        print("-" * 60)


if __name__ == "__main__":
    main()
