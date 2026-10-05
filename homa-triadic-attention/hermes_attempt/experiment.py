"""MATCH3 replication (small scale) of HOMA paper.

Task (Sanford et al. 2024, as in paper): x ~ iid Z_M, N tokens; y_i = 1 iff
exists j,z in [N] with x_i + x_j + x_z = 0 mod M. Per-token binary classification.

Models (one attention layer + embedding + linear head, d_model=64, 4 heads, matched budget):
  pairwise : standard non-causal softmax self-attention
  triadic  : S3[i,j,z] = sum_c Q[i,c]*K[j,c]*U[z,c]/sqrt(dh); softmax over (j,z); value V[j]*V[z]
  homa     : pairwise || triadic per head, fuse with small GELU MLP, output projection

N in {8,12,16,24}, M=N, 3000 AdamW steps, 2 seeds, fresh held-out eval.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, time, os, sys

dev = 'cuda'
D_MODEL = 64
HEADS = 4
STEPS = 3000
BATCH = 128
LR = 2e-3
N_LIST = [8, 12, 16, 24]
SEEDS = [0, 1]
RESULTS = '/workspace/homa/results.json'


def make_batch(B, N, M, gen):
    x = torch.randint(0, M, (B, N), generator=gen)
    xg = x.long()
    # All ordered pair-sums: P[i, j, k] = (x[i,j] + x[i,k]) mod M   -> B,N,N
    pair = (xg.unsqueeze(2) + xg.unsqueeze(1)).remainder(M)         # B,N,N
    # y[i,t] = 1 iff exists (j,k): x[i,t] + pair[i,j,k] == 0 (mod M)
    q = xg.unsqueeze(1) % M                                          # B,1,N  (t is last)
    y = ((pair.unsqueeze(1) + q.unsqueeze(-1)) % M).any(dim=(2, 3)) # B,N
    return x.to(dev), y.to(dev).float()


class Attn(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind, self.dh = kind, D_MODEL // HEADS
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.o = nn.Linear(D_MODEL, D_MODEL)
        if kind in ('triadic', 'homa'):
            self.u = nn.Linear(D_MODEL, D_MODEL)
        if kind == 'homa':
            d = self.dh
            self.fuse = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        B, N = x.shape[0], x.shape[1]
        sp = lambda t: t.view(B, N, HEADS, self.dh).transpose(1, 2)  # B,h,N,dh
        Q, K, V = sp(self.q(x)), sp(self.k(x)), sp(self.v(x))
        if self.kind in ('pairwise', 'homa'):
            A = F.softmax((Q @ K.transpose(-1, -2)) / self.dh ** 0.5, -1)
            O2 = A @ V
        if self.kind in ('triadic', 'homa'):
            U = sp(self.u(x))
            S = torch.einsum('bhic,bhjc,bhzc->bhijz', Q, K, U) / self.dh ** 0.5
            A3 = F.softmax(S.reshape(B, HEADS, N, N * N), -1).view(B, HEADS, N, N, N)
            O3 = torch.einsum('bhijz,bhjc,bhzc->bhic', A3, V, V)
        if self.kind == 'pairwise':
            O = O2
        elif self.kind == 'triadic':
            O = O3
        else:
            O = self.fuse(torch.cat([O2, O3], -1))
        return self.o(O.transpose(1, 2).reshape(B, N, D_MODEL))


class Net(nn.Module):
    def __init__(self, kind, M):
        super().__init__()
        self.e = nn.Embedding(M, D_MODEL)
        self.a = Attn(kind)
        self.out = nn.Linear(D_MODEL, 1)

    def forward(self, x):
        h = self.a(self.e(x))
        return self.out(h).squeeze(-1)


def param_count(m):
    return sum(p.numel() for p in m.parameters())


def run(kind, N, seed, steps=STEPS):
    M = N
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = Net(kind, M).to(dev)
    pc = param_count(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    t0 = time.time()
    for _ in range(steps):
        x, y = make_batch(BATCH, N, M, g)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # fresh held-out: new generator stream, never drawn from training
    gt = torch.Generator().manual_seed(1234567 + seed)
    with torch.no_grad():
        x, y = make_batch(2048, N, M, gt)
        p = (model(x) > 0).float()
        acc = (p == y).float().mean().item()
        pos = y.mean().item()
        tpr = p[y == 1].mean().item() if bool((y == 1).any()) else float('nan')
        tnr = (1 - p[y == 0]).mean().item() if bool((y == 0).any()) else float('nan')
    return dict(N=N, kind=kind, seed=seed, M=M, params=pc, pos_rate=round(pos, 4),
                acc=round(acc, 4), tpr=round(tpr, 4), tnr=round(tnr, 4),
                bal_acc=round((tpr + tnr) / 2, 4), time_s=round(time.time() - t0, 1))


def main():
    res = []
    t_start = time.time()
    for N in N_LIST:
        for kind in ['pairwise', 'triadic', 'homa']:
            for seed in SEEDS:
                r = run(kind, N, seed)
                res.append(r)
                print(f"N={N} {kind} seed={seed} acc={r['acc']} bal={r['bal_acc']} params={r['params']} {r['time_s']}s total={time.time()-t_start:.0f}s", flush=True)
                json.dump(res, open(RESULTS, 'w'), indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
    main()
