# Energy Consumption

BLIS can meter the GPU energy of a run. Every executed step is priced in watts by a pluggable
**power model** and integrated over the step's duration. Time when an instance runs no step is
charged at idle power. Metering is off by default. When on, it writes files only, so stdout is
unchanged (INV-6, INV-13).

```bash
./blis run --scenario <file> --scenarios <dir> --rate 10 --num-requests 500 \
  --energy-config examples/energy/h100-resource-share.yaml \
  --energy-report energy.json \
  --energy-trace energy.csv
```

| Flag | Meaning |
|---|---|
| `--energy-config` | Strict-YAML power model. Enables metering. |
| `--energy-report` | JSON: per instance and cluster busy/idle J, average W/GPU, J/output-token, J/request, GPU Wh, facility Wh (× PUE), gCO2. |
| `--energy-trace` | CSV, one row per step: batch, prefill/decode tokens, the latency kernel's per-resource busy µs, W/GPU. `start_us` is the step's start on the simulation clock. |

Both `run` and `replay` accept the flags. The meter sits in `sim.EngineDriver` and records every batch the engine prices with `StepTime`.

## Power models

| `model:` | Power per GPU during a step | Source |
|---|---|---|
| `resource_share` | `Σ_r (t_r / Σt) · W_r` over the kernel's busy time on `sm`, `hbm`, `nvlink`, `nic`, `pcie`, `copy_engine`, `host` | This work; constants from Patel et al. (ASPLOS'24). Needs an engine that exposes `sim.StepBreakdown`. The blis-vllm 0.32 engine does not yet, so it is refused there. |
| `utilization` | `P_idle + (P_max − P_idle) · min(1, MFU/MFU_sat)^γ` with `MFU = 2·N_active/TP · tokens / (peak · t)` | Vidur-Energy (Özcan et al., arXiv:2507.11417) |
| `tdp` | constant `watts` | Rated-power baseline |

Common keys: `idle_w` (per-GPU idle power), `pue` (default 1), and `carbon_g_per_kwh`.
`examples/energy/` has a zero-shot H100 file for each model. Fitted H100 constants and their
held-out accuracy against ML.ENERGY measurements are in `experiments/energy/findings.md`.
`experiments/energy/README.md` shows how to reproduce them.

## Adding a power model

1. In `sim/energy/energy.go`, define a type with `StepPower(sim.StepSample) float64`. A
   `StepSample` carries the step duration, the kernel's per-resource busy time, prefill and
   decode tokens, and the batch size.
2. Add its YAML block to `Config` and a `case` to `build`, validating its parameters.
3. Add its offline twin to `MODEL_FNS` in `experiments/energy/run.py` to score it with the
   same gate.

## Limitations

- Only GPU-board energy is counted (what NVML/Zeus measure). Host CPU, DRAM and network are
  excluded; facility overhead enters only through `pue`.
- Not yet supported, and refused: data-parallel groups (dp > 1), P/D disaggregated
  deployments and pipeline parallelism (pp > 1).
- Asynchronous KV offload transfers run outside steps, so their time is charged at idle power.
- There are no DVFS, power-cap or thermal effects.
- Accuracy is bounded by the latency kernel's own step-time error, since energy = power × time.
