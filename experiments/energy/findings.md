# Energy model findings

Ground truth: ML.ENERGY leaderboard, `lm-arena-chat`, H100 SXM, vLLM, BF16, steady state
([ml-energy/leaderboard@2bdd007](https://github.com/ml-energy/leaderboard/tree/2bdd0070186956688e1f842dc28077980fe64013/public/data/models)).
There are 36 configurations. 11 are for calibration (Llama-3.1-8B). 25 are held out
(Qwen3-14B, Llama-3.1-70B TP4, Qwen3-30B-A3B). Prompt length is 512 tokens unless stated.
`python3 experiments/energy/run.py gate` regenerates every number in this file.

- Study 1: inference-sim `29f783c7`, blis-latency-kernel v0.3.0 adapter. This base exposes
  per-resource busy time, so models A and B are both scored.
- Study 2: this branch, blis-vllm 0.32 engine. A and the baselines only (B needs a
  per-resource breakdown the engine does not expose).

## Selection (held-out)

| Regime | Model | Fitted params | Power MAPE (study 1) | Power MAPE (study 2) | J/token MAPE (study 2) | Rank ρ (study 2) |
|---|---|---|---|---|---|---|
| Zero-shot | TDP 700 W | 0 | 35.8% | 35.8% | 73.6% | 0.998 |
| Zero-shot | A utilization | 0 | 39.7% | 39.1% | 39.5% | 0.993 |
| Zero-shot | B resource-share | 0 | 22.8% | n/a | n/a | n/a |
| Calibrated | Constant mean power | 1 | 22.4% | 22.4% | 56.0% | 0.998 |
| **Calibrated** | **A utilization** | 4 | **13.1%** | **13.2%** | 46.1% | 0.998 |
| Calibrated | B resource-share | 3 | 33.3% | n/a | n/a | n/a |
| Post-hoc | B, weights bounded to 0–1000 W | 3 | 12.6% | n/a | n/a | n/a |

The gate was fixed before any prediction:
- the lowest calibrated held-out power MAPE wins;
- within 2 points, fewer parameters wins;
- the winner must beat both baselines.

Result: **A wins on both bases.** The post-hoc row reaches 12.6% only with W_sm at the 1000 W
bound, which is above the 700 W TDP, so it does not change the verdict.

## Fitted constants (model A, H100)

| Parameter | Zero-shot (Vidur-Energy) | Study 1 | Study 2 |
|---|---|---|---|
| P_floor (W) | 60 | 444 | 436 |
| P_max (W) | 700 | 689 | 689 |
| MFU_sat | 0.45 | 0.15 | 0.15 |
| γ | 0.7 | 0.80 | 0.75 |

The fitted P_floor is the power of an active GPU at near-zero MFU. Idle gaps between steps use
a separate `idle_w` of 60 W.

## Error decomposition (study 2)

| Quantity | MAPE |
|---|---|
| A power (held-out) | 13.2% |
| Simulated ITL vs measured | 24.6% |
| Simulated output throughput vs measured | 19.1% |
| A J/token (held-out) | 46.1% |

J/token = power ÷ throughput, so most of the J/token error comes from step time, not power.

| Prompt length | Power MAPE | J/token MAPE |
|---|---|---|
| 256 | 13.0% | 31.0% |
| 512 | 13.2% | 46.1% |
| 1024 | 13.4% | 80.5% |

±20% on each fitted A parameter changes mean held-out J/token by:

| Parameter | Change |
|---|---|
| P_floor | ±8.7% |
| P_max | ±11.3% |
| MFU_sat | +1.6% / −1.2% |
| γ | +2.3% / −1.7% |

## Experiments with model A (study 2)

E3, over-provisioning: Llama-3.1-8B, 4,000 chat requests at 20 req/s, PUE 1.2, 400 gCO2/kWh.

| Instances | Avg W/GPU | J/request | Facility Wh | gCO2 | Idle share | p99 TTFT |
|---|---|---|---|---|---|---|
| 1 | 645 | 33.7 | 45.0 | 18.0 | 0.0% | 688 ms |
| 2 | 564 | 58.3 | 77.7 | 31.1 | 0.0% | 277 ms |
| 4 | 507 | 104.6 | 139.5 | 55.8 | 0.1% | 214 ms |
| 8 | 474 | 194.9 | 259.9 | 103.9 | 0.2% | 210 ms |

E2, workload shape: Llama-3.1-8B, 1×H100, 2 req/s, 1,000 requests.

| Prompt / response | Avg W/GPU | J/request | J/output token | Energy in steps with prefill |
|---|---|---|---|---|
| 128 / 512 | 489 | 247.5 | 0.483 | 3.4% |
| 512 / 512 | 492 | 249.3 | 0.487 | 5.4% |
| 512 / 2048 | 574 | 302.9 | 0.148 | 5.8% |
| 2048 / 256 | 484 | 243.8 | 0.953 | 12.7% |
| 4096 / 64 | 398 | 200.0 | 3.125 | 29.6% |
| 8192 / 128 | 526 | 265.0 | 2.070 | 48.5% |

E4, parallelism: Llama-3.1-70B, burst of 1,024 requests (512 in / 384 out), max_num_seqs 256.

| Layout | Avg W/GPU | J/output token | J/request | Mean ITL | p99 TTFT |
|---|---|---|---|---|---|
| TP2 (2 GPUs) | 568 | 1.835 | 704.7 | 60.6 ms | 617 s |
| TP4 (4 GPUs) | 688 | 0.734 | 281.9 | 63.6 ms | 88 s |
| TP8 (8 GPUs) | 686 | 1.006 | 386.4 | 42.4 ms | 62 s |
