Replicate, at small scale, part of the paper in /workspace/homa/paper.txt ("Higher-Order Modular Attention", HOMA). Use grep/sed on the text; do not dump it.
RULES: run every python command as `CUDA_VISIBLE_DEVICES=1 python3 ...` (GPU0 is busy; ~5GB free; tiny models only). Run training in the background with nohup and poll the log; no command should block more than 2 minutes. Evaluate ONLY on freshly sampled held-out data (never training data) and report exact numbers from results.json. Do not fabricate results; if something fails, say so in RESULTS.md.
Task MATCH3 (Sanford et al. 2024, defined in the paper): sequence x of N integers in Z_M; label y_i = 1 if there exist j,z with x_i + x_j + x_z = 0 mod M. Use M=N (or near), sample sequences so labels are roughly balanced. Per-token binary classification.
Models (one attention layer + embedding + linear head, no MLP, d_model=64, 4 heads, same budget for all):
  (a) PAIRWISE: standard softmax self-attention (non-causal).
  (b) TRIADIC: scores S3[i,j,z]=sum_c Q[i,c]*K[j,c]*U[z,c], softmax over all (j,z) pairs, value = V[j]*V[z] (elementwise), output projection.
  (c) HOMA: run (a) and (b) in parallel per head, concatenate, fuse with a small GELU MLP, output projection.
Experiment: N in {8,12,16,24}, train each model ~3000 steps AdamW, 2 seeds; report held-out per-token accuracy and balanced accuracy. Hypothesis from paper: pairwise needs more width as N grows while triadic solves it with constant width.
Write experiment.py, results.json, results.png (acc vs N per model), and RESULTS.md (findings + caveats). Finish with a short summary.
