import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
v2 = json.load(open('results.json'))['runs']; v1 = json.load(open('results_v1_trainacc.json'))['runs']
def curve(runs, n, k, reg):
    rs = [r for r in runs if r['n']==n and r['k']==k and r['regime']==reg]
    key = 'acc' if reg=='direct' else 'acc_ans'
    return [c['step'] for c in rs[0]['curves']], np.array([[c[key] for c in r['curves']] for r in rs])
# 1: held-out acc vs steps, grid
fig, ax = plt.subplots(2, 3, figsize=(13, 6), sharey=True)
for i, k in enumerate([4, 6]):
    for j, n in enumerate([16, 32, 64]):
        a = ax[i, j]
        for reg, c, lab in [('direct','C3','direct'),('cot_teacher','C2','CoT (teacher-forced)')]:
            s, m = curve(v2, n, k, reg); a.plot(s, m.mean(0), color=c, label=lab); a.fill_between(s, m.min(0), m.max(0), color=c, alpha=.2)
        a.set_title(f'n={n}, k={k}'); a.axhline(.5, ls=':', c='gray')
        if i==1: a.set_xlabel('training step')
        if j==0: a.set_ylabel('held-out answer accuracy')
ax[0,0].legend(); plt.suptitle('Held-out accuracy: direct vs CoT (mean, min-max band over 3 seeds)'); plt.tight_layout(); plt.savefig('fig_heldout_curves.png', dpi=130)
# 2: the bug - train-acc (v1) vs held-out (v2), direct only
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for j, n in enumerate([16, 32, 64]):
    a = ax[j]
    for runs, c, lab in [(v1,'C1','v1: accuracy on TRAIN data (bug)'),(v2,'C3','v2: HELD-OUT accuracy')]:
        s, m = curve(runs, n, 6 if n>16 else 4, 'direct'); a.plot(s, m.mean(0), color=c, label=lab)
    a.set_title(f'direct, n={n}, k={6 if n>16 else 4}'); a.set_xlabel('training step'); a.axhline(.5, ls=':', c='gray')
ax[0].set_ylabel('accuracy'); ax[0].legend(fontsize=8); plt.suptitle('Memorisation masquerading as learning'); plt.tight_layout(); plt.savefig('fig_bug_train_vs_heldout.png', dpi=130)
