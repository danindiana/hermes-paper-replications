import json,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
r=json.load(open('results.json')); fig,ax=plt.subplots(1,3,figsize=(14,3.8),sharey=True)
for a,t in zip(ax,['copy','reverse','sort']):
    for pe,c in zip(['learned','rope','nope'],['C0','C1','C2']):
        rs=[x['acc'] for x in r if x['task']==t and x['pe']==pe]; Ls=list(rs[0]); m=np.mean([[x[L] for L in Ls] for x in rs],0)
        a.plot([int(L) for L in Ls],m,'o-',color=c,label=pe)
    a.axvline(20,ls='--',c='gray'); a.set_title(t); a.set_xlabel('test length (trained 1-20)')
ax[0].set_ylabel('exact match'); ax[0].legend(); plt.tight_layout(); plt.savefig('results.png',dpi=130)
