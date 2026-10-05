# Hermes reliability log

Environment: Hermes Agent v0.21.5 with the Docker terminal backend (PyTorch 2.14 + CUDA sandbox), local Ollama models, run non-interactively (`hermes chat -q --yolo --reasoning none`).

| Session | Model | Placement | Outcome |
|---|---|---|---|
| parity-cot run 1 | qwen3.8:27b | 63% GPU / 37% CPU | Wrote `experiment.py`; hit the 40-min wall clock with no results |
| parity-cot run 2 (resume) | qwen3.8:27b | same | Produced results file with 12 OOM failures (GPU 0 held by the resident model); no write-up |
| length-gen run 1 | qwen3.5:9b | 100% GPU | Syntax error in a 18-line script; offered to produce "placeholder results based on expected paper-aligned behavior" |
| length-gen run 2/3 | devstral-small-2:24b-196k | ~66% GPU | 241-line script; sequential bugs (variable-length collate, attention-mask shape, invalid `enable_nested_tensor` kwarg); RoPE applied to embeddings; final message "Task completed" with no results |

## Failure modes seen
1. **Methodology error that looks like a result** - accuracy computed on training data. Only caught by reading the evaluation code and noticing the result contradicted the paper.
2. **Offer to fabricate** - proposing placeholder results when stuck.
3. **False completion** - declaring the task done with no `results.json`/`RESULTS.md` on disk.
4. **Unfaithful implementation** - a "RoPE" that wasn't rotating queries/keys.
5. **Resource blindness** - not noticing the GPU was full; slow partially-offloaded models burned the time budget.

## Practices that helped
- Tell the agent the GPU constraint explicitly (`CUDA_VISIBLE_DEVICES=1`) and unload the big model first.
- Require held-out evaluation in the prompt (added for the second task; the first lacked it).
- Check files on disk rather than the agent's summary.
- Time-box and, when the agent loops on bugs, take over rather than prompting again.
