# Simulated vs measured: KIRA's four KV-tier devices in BLIS

Run 28 October 2026. BLIS simulates KIRA's hardware runs on the four ibm disk tiers and scores each simulated
value against what KIRA measured. Sources: KIRA's
[performance report](https://github.com/sbu-fsl/KIRA/blob/main/archive/docs/2026-10-26-performance.md), the
tier-compare report (`.claude/reports/tier_compare/`, E32/E33), M15 (`.claude/reports/m15_batch2.md`) and the run
log. Only KIRA's **stock vLLM** and **recompute** arms are simulated; BLIS has no model of KIRA's `enforce` policy.

| experiment | host, GPU | disk tier (fio read / write MB/s) | KIRA workload |
|---|---|---|---|
| E32 job 57 | ibm1, RTX A5000 | Intel SSDSC2KG96 SATA (561 / 512) | agent_spill: 65,536-token prompts, 22 prefixes × 2, 1/64 req/s, 32 GiB DRAM |
| E32 job 58 | ibm1, RTX A5000 | Samsung 970 EVO Plus NVMe (2568 / 3221) | same |
| E33 job 39 | ibm2, Quadro RTX 6000 | Crucial MX500 SATA (507 / 518) | chat: 1,024-token prompts, 150 prefixes × 2, 0.5 req/s, GPU capped at 4,096 blocks, 4 GiB DRAM |
| E33 job 40 | ibm2, Quadro RTX 6000 | Samsung PM953 NVMe (1017 / 764) | same |

## Reproduce

```bash
git clone --branch 0.3.0 --depth 1 https://github.com/inference-sim/blis-catalog.git ../blis-deps/blis-catalog
git clone --branch v0.2.0 --depth 1 https://github.com/inference-sim/blis-registry.git ../blis-deps/blis-registry
go build -o blis main.go                        # repo root
cd experiments/kv-offload-kira
python3 overlay/build_overlay.py                # builds ../blis-deps/overlay-kira with the committed calibration.json
python3 kira_repro.py calibrate                 # optional: refits the 4 coefficients on independent KIRA numbers
python3 kira_repro.py checks                    # memory, long prefill, knees: unfitted
python3 kira_repro.py cases                     # every scored case × 5 seeds, writes results-kira/cases.md
```

Pins: inference-sim `29f783c7` (the fork's `1813579` plus one commit touching only review tooling), blis-catalog
`0.3.0`, blis-registry `v0.2.0`, kernel v0.3.0, KIRA `a72f0b43`. Runs are deterministic. `cases` takes about
4 minutes of CPU.

## What BLIS needed (an overlay; nothing upstream changed)

[`overlay/build_overlay.py`](overlay/build_overlay.py) copies the pinned catalog and registry and adds:

- **Qwen3-4B-Instruct-2507:** vendor `config.json` at HF `cdbee75f` and `graph.yaml` from the catalog's own
  `derive_graph.py` (head_dim 128, tied embeddings).
- **Two GPUs, datasheet facts:**
  - [RTX A5000](overlay/rtx-a5000.yaml): 111.1 dense BF16 TFLOPS, 768 GB/s, 24 GiB, CC 8.6.
  - [Quadro RTX 6000](overlay/quadro-rtx-6000.yaml): 130.5 FP16 TFLOPS (no BF16 on Turing), 672 GB/s, 24 GiB, CC 7.5.
- **The four disks** as named storage devices, with KIRA's fio sequential rates. The base latency (150 µs SATA,
  100 µs NVMe) is assumed from the catalog's classes, not from fio's high-depth 34–251 ms p50s.
- **GPU coefficients** copied from L40S, except four fitted below.

The scenarios ([overlay/scenarios/](overlay/scenarios/)) follow KIRA's `benchmark/rig/serve.sh`:
- `--enforce-eager`, prefix caching, block 16, and vLLM 0.28's `serve` defaults for a <70 GiB GPU (2,048 batched
  tokens, 256 sequences).
- ibm1: util 0.9 and max-model-len 65,856 (+256, see Limits).
- ibm2: max-model-len 8,192 and the GPU held to exactly 4,096 blocks.

**Workload.** KIRA's warm pass re-sends the cold pass's exact prompts (its restores are whole 4,096-block prompts and
410-block suffix tails). BLIS replays each prompt as a two-round session with `--session-mode fixed-accumulate`:
round 0 is the cold request at its Poisson arrival, and round 1 is the warm request at the same offset in the second
pass. Within a pass, prefix siblings share their prefix as in KIRA's shuffled `prefix_repetition` dataset.

## Calibration: four coefficients, fitted only on KIRA numbers that are not scored

| GPU | coefficient (L40S donor → fitted) | fitted to (KIRA measured) | source |
|---|---|---|---|
| A5000 | `attention_prefill_work_scale` 0.46 → **0.804** | one cold 65,536-token prefill = 25.4 s | M15 E1: 3.8e8 B/s over 9.66 GB; vLLM prefill time 20–30 s |
| A5000 | `host_launch_eager_per_layer` 7 → **27.1 µs/kernel** | chat TPOT = 39.35 ms (1024/256, 0.5 req/s) | E36 jobs 65/66 |
| Quadro | `attention_prefill_work_scale` 0.46 → **0.0154** | one 1,024-token chat prefill = 372 ms | tier_compare diagnosis: vLLM `request_prefill_time`, E33 cold prompts |
| Quadro | `host_launch_eager_per_layer` 7 → **48.2 µs/kernel** | chat TPOT = 52.2 ms | E36 jobs 47/48 |

The launch-cost fits are the hosts' 2.2 GHz Xeon running vLLM eager, which is KIRA's own diagnosis of where the step
time goes. The Quadro's tiny attention scale is Triton attention on Turing (no FlashAttention).

## Unfitted checks

| check | KIRA measured | BLIS | error |
|---|--:|--:|--:|
| GPU KV cache, ibm1, util 0.9, 65,856 ctx | 95,696 tokens | 95,872 | +0.2% |
| GPU KV cache, ibm2, same settings | 96,383 tokens | 95,872 | −0.5% |
| one 65,536-token prefill on ibm2 (fitted only at 1K) | ≈520 s (130 tok/s, run log job 10) | 844 s | +62% |
| knee, ibm1 agent_hot, KIRA's rule (≥70% of offered, p99 < 5× lowest) | 1/32 req/s | 1/32 req/s | match |
| knee, ibm2 chat, same rule | 1 req/s | 1 req/s | match |

## Results: the four devices (warm pass, stock, pooled over 5 seeds)

| experiment | KIRA TTFT p50 | BLIS TTFT p50 (seed range) | error | KIRA decode / request | BLIS decode / request | error |
|---|--:|--:|--:|--:|--:|--:|
| E32 ibm1 SATA | 18.8 s | 17.7 s (17.7–17.7) | **−6%** | 8.83 s | 13.0 s | +47% |
| E32 ibm1 NVMe | 4.3 s | 4.25 s (4.25–4.25) | **−1%** | 8.69 s | 12.9 s | +49% |
| E33 ibm2 SATA | 539 ms | 544 ms (542–546) | **+1%** | 9.84 s | 11.5 s | **+17%** |
| E33 ibm2 NVMe | 384 ms | 407 ms (405–409) | **+6%** | 9.66 s | 11.5 s | **+19%** |

| device effect | KIRA | BLIS | error |
|---|--:|--:|--:|
| ibm1: SATA ÷ NVMe warm TTFT p50 | 4.37× | 4.16× | −5% |
| ibm2: SATA ÷ NVMe warm TTFT p50 | 1.40× | 1.34× | −4% |
| decode change from the device, either host | "changes neither" | <1% | match |

KIRA publishes no p99 or recompute arm for E32/E33. The external-KV share of warm prompt tokens is 92–95% on ibm1
and 79% on ibm2. ibm2's other 20% are the 257 extra tokens BLIS's warm request carries; see Limits.

One more NVMe value from KIRA's tier_compare diagnosis can be scored:

| NVMe, stock, warm pass | KIRA measured | BLIS (seed range) | error |
|---|--:|--:|--:|
| E33 ibm2 NVMe, mean prefill per request (vLLM `request_prefill_time`, scheduled → first token) | 77 ms | 82 ms (81–84) | **+7%** |

BLIS doesn't schedule a request until its offload load lands, so `ttft − scheduling_delay` measures the same span.
The diagnosis also gives ibm1 NVMe stock's `kv_offload_lookup_async_delay` (2.6 s), but that isn't scored. BLIS
reports only the whole wait before the first scheduled step (disk fetch, DRAM→GPU load and queue: 4.19 s median),
not the deferral alone.

All four TTFT medians and both device ratios are within 6%, and ibm2 decode within 20%. ibm1 decode is 47–49% too
slow. KIRA's 8.83 s over 256 tokens is a 34.5 ms step, while BLIS charges 51 ms. BLIS adds eager launch cost to GPU
time serially, but vLLM 0.28's asynchronous scheduler overlaps the two. KIRA's diagnosis says scheduler time "hides
under the 32 ms step". At chat batch sizes, where the launch cost was fitted, the two agree; at one 65K request per
step they don't.

## The same calibration on KIRA's other ibm1 runs (E1 / batch 1, SATA)

| case | KIRA p50 | BLIS p50 (seed range) | error | KIRA p99 | BLIS p99 | error |
|---|--:|--:|--:|--:|--:|--:|
| agent_spill, 32 GiB, stock (E1 job 46) | 18.8 | 17.7 (17.7–17.7) | **−6%** | 38.9 | 80.6 | +107% |
| agent_spill, 36 GiB, stock (E1 job 49) | 18.8 | 17.7 (17.7–17.7) | **−6%** | 38.5 | 54.1 | +41% |
| agent_spill, recompute (E1 jobs 46/49) | 27.45 | 26.8 (25.7–48.9) | **−2%** | 119.5 | 125 | **+5%** |
| agent_hot 4×3, 32 GiB, stock (E1 job 46) | 11.8 | 13.8 (3.2–27.8) | **+17%** | 31.7 | 65.2 | +106% |
| agent_hot 4×3, 36 GiB, stock (E1 job 49) | 4.9 | 20.3 (3.2–28.4) | +315% | 30.5 | 82.5 | +170% |
| agent_hot 4×3, recompute (E1 jobs 46/49) | 74.75 | 25.7 (25.7–31.5) | −66% | 142.0 | 136 | **−4%** |
| agent_hot 8×2, stock (batch 1, grid-16/22) | 42.15 | 17.7 (17.7–22.0) | −58% | 64.0 | 59.3 | **−7%** |
| agent_hot 8×2, recompute (batch 1, grid-16) | 65.9 | 25.7 (25.7–34.9) | −61% | n/a | 116 | n/a |

Gates (p50 and the prefill mean within 20%, p99 and decode within 30%): **14 of 24 scored values pass** across
all three tables.

- **The disk-dominated cases fit.** These are every agent_spill cell and all four device experiments.
- **The p99s run long** because each request decodes about 48% too slowly in BLIS, so requests occupy the GPU
  longer and queue more.
- **agent_hot is the open problem.** Its 12–16-request passes are dominated by what is resident in DRAM. BLIS's
  seed range spans 3–28 s for one configuration, so it is very sensitive to arrival order.
- **KIRA's agent_hot recompute arm (74–76 s) is still unexplained.** The same prompts at the same rate cost 27 s in
  agent_spill.
- **The batch-1 medians are queue-dominated** (42–66 s). BLIS doesn't reproduce that queue; those runs predate
  M15's software.

### The same stock cases on ibm1's NVMe (predictions: KIRA ran them on SATA only)

KIRA's report has no NVMe run of E1 or batch 1, so these have nothing to score against. Recompute arms don't touch
the disk and are unchanged. TTFT in seconds, warm pass, 5 seeds:

| case | BLIS SATA p50 | BLIS NVMe p50 (seed range) | BLIS SATA p99 | BLIS NVMe p99 |
|---|--:|--:|--:|--:|
| agent_spill, 32 GiB | 17.7 | 4.25 (4.25–4.25) | 80.6 | 30.9 |
| agent_spill, 36 GiB | 17.7 | 4.25 (4.25–4.25) | 54.1 | 38.4 |
| agent_hot 4×3, 32 GiB | 13.8 | 3.53 (0.80–20.3) | 65.2 | 56.3 |
| agent_hot 4×3, 36 GiB | 20.3 | 15.5 (1.83–25.8) | 82.5 | 79.6 |
| agent_hot 8×2 (batch 1) | 17.7 | 4.25 (4.25–9.89) | 59.3 | 39.9 |

agent_spill gets the same 4.2× speedup E32 measured. agent_hot gains less and stays dominated by what is resident in
DRAM.

## Limits

- **Warm prompt length.** BLIS's warm request is round 0's prompt plus its 256 output tokens plus one new token,
  which is 257 tokens more than KIRA re-sends. With prompt-only offload those tokens are recomputed unless still on
  the GPU: about +0.1 s on a 65K agent request, up to 20% of a 1K chat prompt. max-model-len is raised by 256 to
  admit it.
- **Engine version.** BLIS models vLLM 0.31.0 semantics (0.29.0 rules), and host overhead adds to GPU time.
  KIRA ran 0.28.0 with async scheduling, which overlaps them.
- **Coefficients.** Only four are calibrated. GEMM efficiency, HBM derate, attention decode and the host link are
  L40S-derived, and the disk base latencies are assumed.
- **Traces.** These are reconstructions from KIRA's class definitions; KIRA's raw request traces are gitignored, so
  there is no per-request comparison.
- **Samples.** KIRA's cells are single runs (12–300 requests), so a p99 from 12–44 requests is essentially the max.

`run_first_set.py` is the earlier Route A generalization run (Qwen2.5-7B on L40S and A5000). It is not part of
this comparison, and its outputs (`results/`, `results-a5000/`) are not committed. Raw per-run metrics, traces and
offload configs under `results-kira/` are regenerated by `kira_repro.py` and are not committed either.
