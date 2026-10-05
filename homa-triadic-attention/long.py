import json, sys, time, math, torch, torch.nn.functional as F
import experiment as e
# longer training with cosine LR, N in {8,12}, pairwise vs triadic, 2 seeds
def run_long(kind, N, seed, steps=20000, B=256):
    M = round(1.45 * N * N); torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = e.Net(kind, M).to(e.dev); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for t in range(steps):
        for gr in opt.param_groups: gr['lr'] = 2e-3 * min(1, (t + 1) / 300) * 0.5 * (1 + math.cos(math.pi * t / steps))
        x, y = e.make_batch(B, N, M, g); loss = F.binary_cross_entropy_with_logits(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    gt = torch.Generator().manual_seed(999 + seed); x, y = e.make_batch(4096, N, M, gt)
    with torch.no_grad(): p = (m(x) > 0).float()
    tpr = p[y == 1].mean().item(); tnr = (1 - p[y == 0]).mean().item()
    return dict(acc=(p == y).float().mean().item(), pos_rate=y.mean().item(), bal_acc=(tpr + tnr) / 2, final_loss=loss.item())
res = []; t0 = time.time()
for N in [8, 12]:
    for kind in ['pairwise', 'triadic']:
        for seed in [0, 1]:
            r = run_long(kind, N, seed); res.append(dict(N=N, kind=kind, seed=seed, steps=20000, **r))
            print(N, kind, seed, {k: round(v, 3) for k, v in r.items()}, f'{time.time()-t0:.0f}s', flush=True); json.dump(res, open('results_long.json', 'w'))
