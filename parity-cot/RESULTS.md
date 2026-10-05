# Empirical check: CoT vs direct on k-parity (Kim & Suzuki, ICLR 2025)

Setup: 1-layer causal single-head transformer (d=64), AdamW lr 2e-3, 300 full-batch steps, 4096 train samples, 4096 **held-out** samples, hidden random subset, n in {16,32,64}, k in {4,6}, 3 seeds. CoT = teacher-forced tree of 2-parities (k-1 intermediate tokens).

| setting (final step, 3 seeds) | direct (held-out acc) | CoT teacher-forced (held-out answer acc) |
|---|---|---|
| n=16,k=4 | 1.00, 1.00, **0.76** | 1.00 x3 |
| n=16,k=6 | 1.00 x3 | 1.00 x3 |
| n=32,k=4 | 1.00 x3 | 1.00 x3 |
| n=32,k=6 | 0.58, 0.49, 0.50 (chance) | 1.00 x3 |
| n=64,k=4 | 0.51, 0.50, 0.50 (chance) | 1.00 x3 |
| n=64,k=6 | 0.50 x3 (chance) | 1.00 x3 |

(Values read directly from results.json; an earlier draft of this table wrongly listed n=16,k=4 direct as 1.00 for all seeds.)

Consistent with the paper: direct learning needs rapidly more resources as n^k grows; with stepwise supervision it's learned immediately.

## Process note (the interesting part)
Hermes (qwen3.8:27b, local) wrote the 430-line experiment.py autonomously, but its first-run "result" (direct = 100% everywhere, i.e. *no* gap) was **an artifact: accuracy was measured on the training set**, so direct just memorised 4096 samples. Fixing it with a held-out set produced the paper's gap. (v1 kept as experiment_v1_trainacc.py / results_v1_trainacc.json.) A second Hermes run also hit CUDA OOM because Ollama had the 27B resident on GPU 0 — rerun on GPU 1.

## Caveats
- CoT answer accuracy is teacher-forced: the final intermediate token already equals the answer, so this tests learnability of the stepwise supervision, not free-running generation (the paper's harder part, needing augmented data).
- 300 steps, one hyperparameter setting; direct might succeed with far longer training / more data (the paper's point is cost, not impossibility).
- 3 seeds only.
