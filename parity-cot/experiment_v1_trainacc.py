#!/usr/bin/env python
"""Empirical replication of the k-parity CoT theory paper (parity_cot).

We train a small one-layer causal transformer on k-parity (n-bit input,
y = prod_{i in S} x_i for a hidden subset S of k positions) in two regimes:

  (A) direct        — the model receives the n input bits as tokens and is
                      trained to predict y (answer token only).
  (B) cot_teacher   — the model receives all n input bits AS PLUS all k-1
                      teacher-forced intermediate 2-parity values (the two
                      product values the model needs at each internal node)
                      as additional tokens, and is trained to predict each
                      intermediate 2-parity AND the final answer.

The model is a genuine one-layer transformer (single self-attention + FFN,
causal mask, LayerNorm, positional embeddings) — we do NOT decompose each
parity head into independent linear terms.

Setup from the paper (Section 2.2, 3.2):
  * input bits x_i ∈ {±1}^n
  * hidden parity over a random subset S of size k
  * 1-layer transformer with causal attention
  * teacher forcing feeds in the true intermediate values so each
    position can be trained independently
  * loss over intermediate states + final answer

Hyperparameters are chosen so each (n,k,seed) run stays under a few minutes
on a single GPU.
"""

import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Data + tree over k target positions
# ----------------------------------------------------------------------

def build_binary_tree(k: int, subset):
    """Balanced binary tree whose leaves are the k positions in `subset`.

    Returns a list `chain` of length k-1 in post-order.  Each entry is the
    list of input-bit indices in that node's subtree.  The final entry
    (the root) covers all k bits, so its value equals the true answer y.
    """
    tok = []          # tokens: 'L<i>' leaf, 'N<idx>' node referencing chain[<idx>]
    chain = []        # (left_tok, right_tok)

    def go(lo, hi):
        span = hi - lo
        if span == 1:
            return "L%d" % lo
        mid = lo + span // 2
        l = go(lo, mid)
        r = go(mid, hi)
        idx = len(chain)
        chain.append((l, r))
        return "N%d" % idx

    go(0, k)

    def leaves(tok):
        if tok.startswith("L"):
            return [int(tok[1:])]
        i = int(tok[1:])
        l, r = chain[i]
        return leaves(l) + leaves(r)

    final = [list(subset[j] for j in leaves_l)
             for leaves_l in (leaves(l) + leaves(r) for l, r in chain)]
    return final


def make_dataset(n: int, k: int, size: int, seed: int):
    """Returns bits (size,n), y (size,), chain (list of (b1,b2) tuples, len k-1).

    For each internal node in the chain, (b1,b2) are the two positions whose
    bits' product defines the student's 2-parity at that node. In the balanced
    tree, the root is the product of the two halves — its children are each a
    k/2 parity; but our model can't compute a k/2-parity in one step, so we
    flatten the tree recursively: each internal node in a FULL binary tree over
    k positions is a product of two leaves or two internal nodes. To keep this
    experiment tractable, we use a PAIRWISE tree: each internal node is the
    product of its two immediate children, and children are always leaves
    (bits) for a first pass — this handles k=2,4,8 cleanly but not k=6.

    For non-power-of-2 k we fall back to a simple flat pair: every internal
    node computes prod of its two child bits (or child parities if depth>1).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx_pool = torch.randperm(n, generator=g).to(torch.int64)
    subset = idx_pool[:k].tolist()
    x = torch.randint(0, 2, (size, n), generator=g).to(torch.int64) * 2 - 1
    bits = x.to(torch.float32)
    y = bits[:, subset].prod(-1)
    chain = build_binary_tree(k, subset)
    return bits, y, chain


# ----------------------------------------------------------------------
# Transformer (one causal layer, LayerNorm, positional embedding)
# ----------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, total: int, d: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(total, d))

    def forward(self):
        return self.pe


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.d = d

    def forward(self, x, mask):
        b, s, _ = x.shape
        # Single-head attention: keep batch axis, no extra head dim.
        q = self.q(x)               # b, s, d
        k = self.k(x)               # b, s, d
        v = self.v(x)               # b, s, d
        scores = torch.einsum('bmd,bnd->bmn', q, k) / math.sqrt(self.d)  # b, s, s
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        w = F.softmax(scores, dim=-1)
        return torch.einsum('bmn,bnd->bmd', w, v)   # b, s, d


def make_causal_mask(total: int):
    return torch.triu(torch.ones(total, total, dtype=torch.bool, device=DEVICE), diagonal=1)


class OneLayerTransformer(nn.Module):
    """One-layer causal transformer.

    Tokens:
      0 .. n-1        : data tokens (each encodes one input bit).
      n .. n+T-1      : teacher tokens (T = number of teacher-forced
                        intermediate values — for cot_teacher mode this is the
                        k-1 chain values; for direct mode T=0 and the answer
                        lives at the final position).
      n+T             : answer token.

    Heads:
      * direct mode: one head at the answer position predicts y.
      * cot mode: T heads at the T teacher positions, each predicts the
        product of that node's two children (the student's 2-parity at
        that node), plus one head at the answer position predicts y.
    """
    def __init__(self, n: int, T: int, d: int):
        super().__init__()
        self.n = n
        self.T = T
        self.d = d
        total = n + T + 1
        self.pe = nn.Parameter(torch.zeros(total, d))
        # data embedding (one row per data position)
        self.data_embed = nn.Parameter(torch.randn(n, d) * 0.1)
        # teacher token embedding (one row per teacher position)
        if T > 0:
            self.teach_embed = nn.Parameter(torch.randn(T, d) * 0.1)
        else:
            self.teach_embed = None
        # answer token embedding
        self.answer_embed = nn.Parameter(torch.zeros(d))
        # single attention head + FFN
        self.attn = CausalSelfAttention(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        # heads
        self.heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(T)])
        self.head_ans = nn.Linear(d, 1)
        self.mask = make_causal_mask(total)

    def forward(self, bits, teacher_vals=None):
        b = bits.size(0)
        data = torch.einsum('bn,nd->bnd', bits, self.data_embed)        # b,n,d
        if self.T > 0:
            teach = torch.einsum('bt,td->btd', teacher_vals, self.teach_embed)  # b,T,d
        else:
            teach = None
        ans = self.answer_embed.unsqueeze(0).expand(b, 1, -1)          # b,1,d
        if teach is not None:
            tokens = torch.cat([data, teach, ans], dim=1)
        else:
            tokens = torch.cat([data, ans], dim=1)
        tokens = tokens + self.pe
        h = self.ln1(tokens)
        h = self.attn(h, self.mask)
        tokens = tokens + h
        tokens = tokens + self.ffn(self.ln2(tokens))
        if self.T > 0:
            int_outs = [self.heads[j](tokens[:, self.n + j]).squeeze(-1)
                        for j in range(self.T)]
            int_outs = torch.stack(int_outs, dim=1)                    # b,T
            ans_out = self.head_ans(tokens[:, -1]).squeeze(-1)         # b
            return int_outs, ans_out
        return self.head_ans(tokens[:, -1]).squeeze(-1)


# ----------------------------------------------------------------------
# Regime runners
# ----------------------------------------------------------------------

def train_direct(bits, y, n, d, steps, lr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = OneLayerTransformer(n, T=0, d=d).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    log = []
    bits_dev = bits.to(DEVICE).float()
    y_dev = y.to(DEVICE).float()
    for step in range(1, steps + 1):
        opt.zero_grad()
        logit = model(bits_dev)
        loss = F.binary_cross_entropy_with_logits(logit, (y_dev + 1) / 2)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            acc = ((logit > 0).float() == ((y_dev + 1) / 2).float()).float().mean().item()
            log.append({"step": step, "loss": loss.item(), "acc": acc})
    return log


def train_cot_teacher(bits, y, chain, n, d, steps, lr, seed):
    """Chain-of-thought with teacher forcing.

    `chain` is a list of k-1 entries (post-order over a balanced binary tree
    of leaves = the k target bit positions). Each entry is a tuple of input
    indices; the teacher value = product of the bits at those indices.  The
    model must predict each entry's product, plus the final answer.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = len(chain)
    model = OneLayerTransformer(n, T=T, d=d).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bits_dev = bits.to(DEVICE).float()
    y_dev = y.to(DEVICE).float()
    # Precompute teacher values for each chain entry
    teacher_list = []
    for entry in chain:
        t = bits_dev[:, entry[0]]
        for idx in entry[1:]:
            t = t * bits_dev[:, idx]
        teacher_list.append(t)
    teacher = torch.stack(teacher_list, dim=1)   # b, T
    log = []
    for step in range(1, steps + 1):
        opt.zero_grad()
        int_logits, ans_logit = model(bits_dev, teacher)
        int_acc = ((int_logits > 0).float() == ((teacher + 1) / 2).float()).float().mean().item()
        int_loss = F.binary_cross_entropy_with_logits(int_logits, (teacher + 1) / 2)
        ans_loss = F.binary_cross_entropy_with_logits(ans_logit, (y_dev + 1) / 2)
        loss = int_loss + ans_loss
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            ans_acc = ((ans_logit > 0).float() == ((y_dev + 1) / 2).float()).float().mean().item()
            log.append({"step": step, "loss": loss.item(),
                        "acc_int": int_acc, "acc_ans": ans_acc,
                        "acc": (int_acc + ans_acc) / 2})
    return log


# ----------------------------------------------------------------------
# Experiment matrix + report
# ----------------------------------------------------------------------

def main():
    n_list = [16, 32, 64]
    k_list = [4, 6]
    d = 64
    steps = 300
    lr = 2e-3
    train_size = 4096
    seeds = [0, 1, 2]

    hparams = dict(n_list=n_list, k_list=k_list, d=d, steps=steps,
                   lr=lr, train_size=train_size, seeds=seeds,
                   device=str(DEVICE))
    print(f"[hparams] {hparams}")

    runs = []
    t0 = time.time()
    failures = 0
    for n in n_list:
        for k in k_list:
            for regime in ["direct", "cot_teacher"]:
                for seed in seeds:
                    bits, y, chain = make_dataset(n, k, train_size, seed)
                    if regime == "direct":
                        try:
                            log = train_direct(bits, y, n, d, steps, lr, seed)
                            runs.append(dict(n=n, k=k, regime=regime, seed=seed,
                                             chain_len=len(chain), curves=log))
                            print(f"[ok ] n={n:2d} k={k} {regime:12s} seed={seed} "
                                  f"acc  0: {log[0]['acc']:.3f} -> {log[-1]['acc']:.3f} "
                                  f"({time.time()-t0:.0f}s total)")
                        except Exception as e:
                            failures += 1
                            runs.append(dict(n=n, k=k, regime=regime, seed=seed, error=str(e)))
                            print(f"[ERR] n={n} k={k} {regime} seed={seed}: {e}")
                    else:
                        try:
                            log = train_cot_teacher(bits, y, chain, n, d, steps, lr, seed)
                            runs.append(dict(n=n, k=k, regime=regime, seed=seed,
                                             chain_len=len(chain), curves=log))
                            print(f"[ok ] n={n:2d} k={k} {regime:12s} seed={seed} "
                                  f"acc_ans 0: {log[0]['acc_ans']:.3f} -> {log[-1]['acc_ans']:.3f} "
                                  f"({time.time()-t0:.0f}s total)")
                        except Exception as e:
                            failures += 1
                            runs.append(dict(n=n, k=k, regime=regime, seed=seed, error=str(e)))
                            print(f"[ERR] n={n} k={k} {regime} seed={seed}: {e}")

    results = {"hparams": hparams, "runs": runs,
               "failures": failures, "total_seconds": time.time() - t0}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] saved results.json ({time.time()-t0:.1f}s) failures={failures}")

    # Plots
    for n, k in [(n, k) for n in n_list for k in k_list]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for regime in ["direct", "cot_teacher"]:
            sub = [r for r in runs
                   if r.get("n") == n and r.get("k") == k and r.get("regime") == regime
                   and "curves" in r]
            if not sub:
                continue
            acc_arrays = np.stack([
                np.array([c[("acc" if regime == "direct" else "acc_ans")] for c in r["curves"]])
                for r in sub
            ])
            steps_arr = np.array([c["step"] for c in sub[0]["curves"]])
            ax.plot(steps_arr, acc_arrays.mean(axis=0),
                    label=f"{regime} (mean of {len(sub)} seeds)")
            ax.fill_between(steps_arr, acc_arrays.min(axis=0), acc_arrays.max(axis=0), alpha=0.15)
        ax.set_title(f"k-parity  n={n}, k={k}")
        ax.set_xlabel("training steps")
        ax.set_ylabel("accuracy on answer")
        ax.set_ylim(0.85, 1.02)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"results_n{n}_k{k}.png", dpi=120)
        plt.close(fig)

    # Combined summary plot
    fig, axes = plt.subplots(1, len(k_list), figsize=(6 * len(k_list), 4.5), sharey=True)
    for ax, k in zip(axes, k_list):
        for regime in ["direct", "cot_teacher"]:
            for n in n_list:
                sub = [r for r in runs
                       if r.get("n") == n and r.get("k") == k and r.get("regime") == regime
                       and "curves" in r]
                if not sub:
                    continue
                acc_arrays = np.stack([
                    np.array([c[("acc" if regime == "direct" else "acc_ans")] for c in r["curves"]])
                    for r in sub
                ])
                steps_arr = np.array([c["step"] for c in sub[0]["curves"]])
                style = "-" if n == 16 else ("--" if n == 32 else ":")
                color = "#d62728" if regime == "direct" else "#1f77b4"
                ax.plot(steps_arr, acc_arrays.mean(axis=0),
                        linestyle=style, color=color,
                        label=f"{regime} n={n}")
        ax.set_title(f"k = {k} (solid=16, dashed=32, dotted=64)")
        ax.set_xlabel("training steps")
        ax.set_ylabel("answer accuracy")
        ax.set_ylim(0.85, 1.02)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results.png", dpi=120)
    plt.close(fig)
    print(f"[done] saved results.png and per-setting PNGs; total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
