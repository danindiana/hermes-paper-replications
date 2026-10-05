# Reproduce

Requirements: Python 3.11, PyTorch with CUDA, matplotlib, numpy. The originals ran in a Docker image with those preinstalled, pinned to a spare GPU via `CUDA_VISIBLE_DEVICES`.

```bash
# parity-cot (~2 min)
cd parity-cot && python experiment.py            # writes results.json, results.png
python plot_extra.py                              # fig_heldout_curves.png, fig_bug_train_vs_heldout.png

# length-generalization (~6 min)
cd length-generalization && python experiment.py  # writes results.json
python plot.py && python plot_extra.py            # results.png, fig_per_seed.png, fig_heatmap.png
```
Seeds are fixed, but GPU nondeterminism means exact numbers may differ slightly; the qualitative picture (chance vs 1.0; sort+NoPE extrapolation) should hold.

Diagrams: `dot -Tpng -Gdpi=160 diagrams/01_workflow.dot -o diagrams/01_workflow.png`

# homa-triadic-attention (~25 min; N=16/24 slow)
cd homa-triadic-attention && python experiment.py && python long.py && python plot.py  # note: experiment.py writes results.json; results_3000steps.json is the committed copy
