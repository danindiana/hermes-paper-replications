# Length generalization of small transformers (after Zhou et al. 2023, "What Algorithms can Transformers Learn?")

Setup: 4-layer, d=128 decoder-only transformer (~0.8M params), trained from scratch 6000 steps on lengths 1-20 (digits 0-9, format `input SEP output`), tested on **freshly sampled** sequences up to length 60, exact-match accuracy, 2 seeds, 3 position encodings (learned absolute, RoPE, none).

Mean accuracy at test length (seeds 0/1 listed where they differ):
| task / pos-enc | 20 (in-dist) | 25 | 30 | 40 | 60 |
|---|---|---|---|---|---|
| copy / learned | 1.0 | 0 | 0 | 0 | 0 |
| copy / RoPE | 0.99 | 0.17-0.46 | 0 | 0 | 0 |
| copy / NoPE | ~1.0 | 0.91-0.96 | 0.65-0.78 | 0.04-0.09 | 0 |
| reverse / learned, RoPE | 1.0 | ~0 | 0 | 0 | 0 |
| reverse / NoPE | 0.99 | 0.7 | 0.15 | 0 | 0 |
| sort / learned | 1.0 | 0.71 / 0.03 | 0.44 / 0 | 0.09 / 0 | 0 |
| sort / RoPE | 1.0 | ~0.06 | 0 | 0 | 0 |
| **sort / NoPE** | 1.0 | 1.0 | 1.0 | 0.99-1.0 | **0.8** |

## Findings
- Every model is perfect in-distribution; length generalization is where they differ (the paper's theme).
- Learned absolute positions generalize worst: nothing beyond the trained 20 for copy/reverse (the positions simply were never trained).
- **Sorting with NoPE extrapolates 3x (0.8 exact-match at 60)** - sorting is order-agnostic over the input, so position information is unnecessary and its absence removes the failure mode. Positional schemes *hurt* here.
- Copy/reverse need position-based addressing; NoPE partly extrapolates (to ~1.3-2x), RoPE little.
- Doesn't cleanly confirm "RASP-L-expressible => generalizes" for copy/reverse at this scale/training budget; only supports it where the algorithm avoids positions.

## Caveats
- Small scale, one training budget, 2 seeds (learned-sort seeds disagree strongly: 0.71 vs 0.03 at L=25).
- Zhou et al. use specific input/output formats and "index hints"; not tried here.
- Eval is exact-match over all output tokens, teacher-forced prediction of each position (argmax), not free-running generation.

## Process note (Hermes)
Hermes with qwen3.5:9b wrote syntactically broken code and offered to "generate placeholder results"; I declined. devstral-small-2:24b-196k (66% GPU) produced a longer script but with a different bug on each run (padding, attention-mask shape, invalid kwarg), applied RoPE to embeddings rather than attention, and then reported "Task completed" with no results. Its attempt is kept in `hermes_attempt/`. I wrote the working experiment.py myself.
