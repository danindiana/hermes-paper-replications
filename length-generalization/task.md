You are replicating a paper empirically at small scale. Work in /workspace/lengthgen (paper.txt = "What Algorithms can Transformers Learn? A Study in Length Generalization", RASP-L). Skim abstract/intro/experiments with grep/sed; don't dump the whole file.
IMPORTANT: GPU0 is occupied by another process. Run every python command as: CUDA_VISIBLE_DEVICES=1 python3 ... (about 5GB free; keep models tiny, <20M params).
Task: train small causal decoder-only transformers (e.g. 4 layers, d=128) from scratch on 3 synthetic tasks, trained on lengths 1..20, tested on lengths up to 60:
  1. copy (output the input sequence)
  2. reverse
  3. sort (ascending) of digits
Compare 2 position encodings: learned absolute vs RoPE (or NoPE if easy). Report exact-match accuracy per test length for each (task, posenc), 2 seeds. Hypothesis from paper: tasks expressible in RASP-L length-generalize; others don't.
Write experiment.py, run it (each run < ~3 min; run in background with nohup and poll a log, never block on a command longer than 2 minutes), save results.json + results.png, and RESULTS.md with findings, whether they agree with the paper, and caveats. IMPORTANT: evaluate on freshly sampled held-out data, never training data; for in-distribution lengths report that separately. Finish with a short summary.
