"""Reference (hand-written) MATCH3 experiment: pairwise vs triadic vs HOMA-style fusion, one attention layer."""
import torch, torch.nn as nn, torch.nn.functional as F, json, time, sys
dev = 'cuda'
def make_batch(B, N, M, g):
    x = torch.randint(0, M, (B, N), generator=g)
    # y_i = exists j,z (any, incl. equal indices as in the paper's definition over all positions): x_i+x_j+x_z = 0 mod M
    s = (x[:, :, None] + x[:, None, :]) % M                      # B,N,N pair sums
    need = (-x) % M                                               # B,N value needed from a pair sum
    present.scatter_(2, s.reshape(B, N * N)[:, None, :].expand(B, N, N * N)[:, :1].expand(B, N, N * N)[:, :1].expand(B, N, N * N), True) if False else None
    ps = torch.zeros(B, M, dtype=torch.bool); ps.scatter_(1, s.reshape(B, -1), True)
    y = ps.gather(1, need)                                        # B,N
    return x.to(dev), y.float().to(dev)
class Attn(nn.Module):
    def __init__(s, kind, d=64, h=4, r=16):
        super().__init__(); s.kind, s.h, s.dh = kind, h, d // h
        s.q = nn.Linear(d, d); s.k = nn.Linear(d, d); s.v = nn.Linear(d, d); s.o = nn.Linear(d, d)
        if kind != 'pairwise': s.u = nn.Linear(d, d)
        if kind == 'homa': s.fuse = nn.Sequential(nn.Linear(2 * s.dh, 2 * s.dh), nn.GELU(), nn.Linear(2 * s.dh, s.dh))
    def forward(s, x):
        B, N, D = x.shape; sp = lambda t: t.view(B, N, s.h, s.dh).transpose(1, 2)
        Q, K, V = sp(s.q(x)), sp(s.k(x)), sp(s.v(x))
        if s.kind in ('pairwise', 'homa'):
            A = F.softmax(Q @ K.transpose(-1, -2) / s.dh ** .5, -1); O2 = A @ V             # B,h,N,dh
        if s.kind in ('triadic', 'homa'):
            U = sp(s.u(x))
            S = torch.einsum('bhic,bhjc,bhzc->bhijz', Q, K, U) / s.dh ** .5
            A3 = F.softmax(S.reshape(B, s.h, N, N * N), -1).view(B, s.h, N, N, N)
            O3 = torch.einsum('bhijz,bhjc,bhzc->bhic', A3, V, V)                           # sum_jz A * (V_j * V_z)
        O = {'pairwise': lambda: O2, 'triadic': lambda: O3, 'homa': lambda: s.fuse(torch.cat([O2, O3], -1))}[s.kind]()
        return s.o(O.transpose(1, 2).reshape(B, N, D))
class Net(nn.Module):
    def __init__(s, kind, M, d=64):
        super().__init__(); s.e = nn.Embedding(M, d); s.a = Attn(kind, d); s.ln = nn.LayerNorm(d); s.out = nn.Linear(d, 1)
    def forward(s, x): h = s.e(x); h = h + s.a(h); return s.out(s.ln(h)).squeeze(-1)
def run(kind, N, seed, steps=3000):
    M = round(1.45 * N * N); torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = Net(kind, M).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for t in range(steps):
        x, y = make_batch(128, N, M, g); loss = F.binary_cross_entropy_with_logits(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    gt = torch.Generator().manual_seed(999 + seed); x, y = make_batch(2048, N, M, gt)   # fresh held-out
    with torch.no_grad(): p = (m(x) > 0).float()
    acc = (p == y).float().mean().item(); pos = y.mean().item()
    tpr = p[y == 1].mean().item() if (y == 1).any() else float('nan'); tnr = (1 - p[y == 0]).mean().item() if (y == 0).any() else float('nan')
    return dict(acc=acc, pos_rate=pos, bal_acc=(tpr + tnr) / 2, params=sum(q.numel() for q in m.parameters()))
if __name__ == '__main__':
    res = []; t0 = time.time()
    for N in [8, 12, 16, 24]:
        for kind in ['pairwise', 'triadic', 'homa']:
            for seed in [0, 1]:
                r = run(kind, N, seed); res.append(dict(N=N, kind=kind, seed=seed, **r))
                print(N, kind, seed, {k: round(v, 3) for k, v in r.items()}, f'{time.time()-t0:.0f}s', flush=True)
                json.dump(res, open('results.json', 'w'))
