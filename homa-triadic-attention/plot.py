import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
short = json.load(open('results_3000steps.json')); long_ = json.load(open('results_long.json'))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
cols = {'pairwise': 'C0', 'triadic': 'C1', 'homa': 'C2'}
for kind in cols:
    for a, data, key, ttl in [(ax[0], short, 'bal_acc', '3,000 steps (batch 128)')]:
        Ns = sorted({r['N'] for r in data}); m = [np.mean([r[key] for r in data if r['N']==N and r['kind']==kind]) for N in Ns]
        a.plot(Ns, m, 'o-', color=cols[kind], label=kind); a.set_title(ttl)
for kind in ['pairwise', 'triadic']:
    Ns = sorted({r['N'] for r in long_}); m = [np.mean([r['bal_acc'] for r in long_ if r['N']==N and r['kind']==kind]) for N in Ns]
    ax[1].plot(Ns, m, 'o-', color=cols[kind], label=kind)
ax[1].set_title('20,000 steps (batch 256, cosine LR)')
for a in ax: a.axhline(.5, ls=':', c='gray'); a.set_xlabel('sequence length N'); a.set_ylabel('held-out balanced accuracy'); a.legend()
plt.tight_layout(); plt.savefig('results.png', dpi=130)
