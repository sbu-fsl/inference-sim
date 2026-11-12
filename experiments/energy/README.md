# Energy model study

Reproduces the held-out power-model selection and experiments E2–E5 for the energy meter
(`docs/guide/energy.md`). Python 3 stdlib only.

**Selection gate (declared before any prediction):**
- The default model has the lowest held-out power MAPE after calibration on Llama-3.1-8B.
- If two candidates are within 2 percentage points, the one with fewer fitted parameters wins.
- The winner must also beat both the TDP baseline and the constant-mean-power baseline.

```bash
go build -o blis main.go
python3 experiments/energy/run.py gate          # ~30 s on 8 cores; embedded catalog and registry
python3 experiments/energy/run.py experiments   # needs findings/blis-vllm-0.32/gate.json
```

- **Ground truth:** the ML.ENERGY leaderboard (`lm-arena-chat`, H100, vLLM, BF16) at
  commit `2bdd0070186956688e1f842dc28077980fe64013`. It is downloaded to `.data/` and
  sha256-checked against the hashes in `run.py`. It is not committed.
- **`gate`:** simulates 36 measured configurations × 3 prompt lengths (256/512/1024). Each run
  writes a per-step trace. The steady-state window is steps with batch ≥ 0.9 × the measured
  average batch. Every power model is scored offline on the same trace. The script then:
  - fits on Llama-3.1-8B only;
  - scores Qwen3-14B, Llama-3.1-70B (TP4) and Qwen3-30B-A3B as held-out;
  - applies the pre-declared gate;
  - writes `findings/blis-vllm-0.32/gate.json` and `points.csv` (the CSV repeats measured values, so it is gitignored).

  It also checks that its Python power model matches the Go meter
  (`go_python_crosscheck_max_rel_err`).
- **Two bases:** model B (`resource_share`) needs per-resource busy time, which the
  blis-vllm 0.32 engine does not expose through `engine.Engine`. The A-vs-B comparison
  (study 1) therefore ran on the earlier kernel-adapter base (inference-sim `29f783c7`,
  blis-latency-kernel v0.3.0). This base re-validates A and the baselines (study 2). Both
  are summarized in `findings.md`. The JSON outputs are gitignored and regenerate in about 30 s.
- **`experiments`:** runs the winning model with its fitted constants and writes
  `findings/blis-vllm-0.32/experiments.json`:
  - E2: workload shape;
  - E3: 1/2/4/8 instances at 20 req/s with PUE 1.2 and 400 gCO2/kWh;
  - E4: Llama-3.1-70B at TP 2/4/8.
