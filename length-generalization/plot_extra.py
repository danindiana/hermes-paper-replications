import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r = json.load(open('results.json')); T=['copy','reverse','sort']; P=['learned','rope','nope']
Ls = list(r[0]['acc'])
# per-seed lines
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
for a, t in zip(ax, T):
    for pe, c in zip(P, ['C0','C1','C2']):
        for x in [x for x in r if x['task']==t and x['pe']==pe]:
            a.plot([int(L) for L in Ls], [x['acc'][L] for L in Ls], '-' if x['seed']==0 else '--', color=c, marker='o', ms=3, label=f"{pe} s{x['seed']}")
    a.axvline(20, ls=':', c='gray'); a.set_title(t); a.set_xlabel('test length (trained 1-20)')
ax[0].set_ylabel('exact match'); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig('fig_per_seed.png', dpi=130)
# heatmaps at L=30 and 40
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for a, L in zip(ax, ['30', '40']):
    M = np.array([[np.mean([x['acc'][L] for x in r if x['task']==t and x['pe']==pe]) for pe in P] for t in T])
    im = a.imshow(M, vmin=0, vmax=1, cmap='viridis'); a.set_xticks(range(3), P); a.set_yticks(range(3), T); a.set_title(f'mean exact match @ length {L}')
    for i in range(3):
        for j in range(3): a.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center', color='w' if M[i,j]<.6 else 'k')
plt.tight_layout(); plt.savefig('fig_heatmap.png', dpi=130)
