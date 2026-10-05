import torch, torch.nn as nn, torch.nn.functional as F, math, json, sys, time
dev = 'cuda'
V = 10; SEP = 10; PAD = 11; NV = 12; MAXL = 130
def sample(task, B, lo, hi, g):
    L = int(torch.randint(lo, hi + 1, (1,), generator=g))
    x = torch.randint(0, V, (B, L), generator=g)
    y = {'copy': x, 'reverse': x.flip(1), 'sort': x.sort(1).values}[task]
    seq = torch.cat([x, torch.full((B, 1), SEP), y], 1)
    return seq.to(dev), L          # loss/eval only on positions after SEP
def rope(x, pos):
    d = x.size(-1); inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device) / d))
    a = pos[:, None] * inv[None]; c, s = a.cos(), a.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)
class Block(nn.Module):
    def __init__(s, d, h, pe):
        super().__init__(); s.h = h; s.pe = pe
        s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, D // s.h).unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        if s.pe == 'rope':
            p = torch.arange(T, device=x.device).float(); q, k = rope(q, p), rope(k, p)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + s.o(a); return x + s.ff(s.n2(x))
class LM(nn.Module):
    def __init__(s, pe, d=128, h=4, L=4):
        super().__init__(); s.pe = pe
        s.emb = nn.Embedding(NV, d)
        s.pos = nn.Embedding(MAXL, d) if pe == 'learned' else None
        s.blocks = nn.ModuleList([Block(d, h, pe) for _ in range(L)])
        s.n = nn.LayerNorm(d); s.out = nn.Linear(d, NV)
    def forward(s, x):
        h = s.emb(x)
        if s.pos is not None: h = h + s.pos(torch.arange(x.size(1), device=x.device))
        for b in s.blocks: h = b(h)
        return s.out(s.n(h))
def evaluate(m, task, L, g, n=512):
    m.eval(); seq, _ = sample(task, n, L, L, g)
    with torch.no_grad(): pred = m(seq[:, :-1]).argmax(-1)
    ok = (pred[:, L:] == seq[:, L + 1:]).all(1).float().mean().item(); m.train(); return ok
def run(task, pe, seed, steps=6000):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); m = LM(pe).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    for t in range(steps):
        seq, L = sample(task, 64, 1, 20, g)
        lg = m(seq[:, :-1])[:, L:]; loss = F.cross_entropy(lg.reshape(-1, NV), seq[:, L + 1:].reshape(-1))
        for gr in opt.param_groups: gr['lr'] = 1e-3 * min(1, (t + 1) / 200) * (0.5 * (1 + math.cos(math.pi * t / steps)))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    gt = torch.Generator().manual_seed(10_000 + seed)    # fresh held-out samples
    return {L: evaluate(m, task, L, gt) for L in [5, 10, 15, 20, 25, 30, 40, 50, 60]}
if __name__ == '__main__':
    res = []; t0 = time.time()
    for task in ['copy', 'reverse', 'sort']:
        for pe in ['learned', 'rope', 'nope']:
            for seed in [0, 1]:
                r = run(task, pe, seed); res.append(dict(task=task, pe=pe, seed=seed, acc=r))
                print(task, pe, seed, {k: round(v, 2) for k, v in r.items()}, f'{time.time()-t0:.0f}s', flush=True)
                json.dump(res, open('results.json', 'w'))
