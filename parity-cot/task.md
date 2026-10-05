You are replicating a theory paper empirically. Work in /workspace/parity_cot (paper at paper.txt; read the abstract, setup, and numerical-experiments sections with grep/sed, don't dump the whole file).
Goal: using PyTorch on GPU (cuda is available), train a small 1-layer transformer on k-parity (n-bit input, parity of a hidden subset of k coordinates) in two regimes:
 (A) direct: predict the parity label only;
 (B) chain-of-thought with teacher forcing: predict the k intermediate running parities, then the answer.
Sweep n in {16,32,64} with k=4 (and k=6 if time), fixed sample budget and steps. Report test accuracy vs training steps for A vs B, 3 seeds each.
Write experiment.py, run it, save results.json and a plot results.png, and write RESULTS.md: what you found, whether it agrees with the paper's claim (CoT learns far faster than direct), caveats. Keep each run under a few minutes. If something fails, debug it. Finish with a short summary.
