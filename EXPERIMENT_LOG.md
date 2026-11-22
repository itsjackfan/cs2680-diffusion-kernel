# Experiment log

<!-- HUMANS ONLY; agents must not edit this file. The launcher refuses any run whose `name` isn't in this file.

## <exp> (branch: <exp>)
<2 sentences: question + hypothesis>
- YYYY-MM-DD HH:MM EST `<ablation>`: <one-liner: what this ablation tests>

Failed arms: branch, code changes, why it failed. -->

## baseline (branch: baseline)
- 2026-11-19 16:59 `baseline-smoke` `baseline-smoke-vllm`: baseline smoke on Duo infra + kernels
    Results: successful
- 2026-11-21 22:26 `baseline-profile`: baseline profile on vLLM and HuggingFace on standard Duo checkpoint; tracking latency, memory usage, arithmetic intensity, memory/compute-boundedness, overall pretraining-metrics model performance
    Changes: N/A
    Hypothesis: standard performance and latency with significant memory peaks and inefficiencies, specifically in the full-vocab softmax calculation
    Results: 