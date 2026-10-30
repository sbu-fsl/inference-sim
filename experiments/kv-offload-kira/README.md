# Simulated vs measured: KIRA's four KV-tier devices in BLIS

Run 28 October 2026; ibm1 rerun 30 October with the A5000's coefficients measured on ibm1 itself. BLIS simulates KIRA's hardware runs on the four ibm disk tiers and scores each simulated
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
python3 overlay/build_overlay.py                # builds ../blis-deps/overlay-kira with the committed measured.json + calibration.json
python3 kira_repro.py calibrate                 # optional: refits the 3 KIRA-fitted coefficients
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
- **GPU coefficients:** on the A5000, the eight that matter here are measured on ibm1 (below), and the launch cost
  is fitted to KIRA. On the Quadro they are still copied from L40S, except two fitted to KIRA, until ibm2's stage is
  free to run the same benchmark. Coefficients a TP1 dense model never reads (NCCL, MoE, sliding window) stay
  copied on both.

The scenarios ([overlay/scenarios/](overlay/scenarios/)) follow KIRA's `benchmark/rig/serve.sh`:
- `--enforce-eager`, prefix caching, block 16, and vLLM 0.28's `serve` defaults for a <70 GiB GPU (2,048 batched
  tokens, 256 sequences).
- ibm1: util 0.9 and max-model-len 65,856 (+256, see Limits).
- ibm2: max-model-len 8,192 and the GPU held to exactly 4,096 blocks.

**Workload.** KIRA's warm pass re-sends the cold pass's exact prompts (its restores are whole 4,096-block prompts and
410-block suffix tails). BLIS replays each prompt as a two-round session with `--session-mode fixed-accumulate`:
round 0 is the cold request at its Poisson arrival, and round 1 is the warm request at the same offset in the second
pass. Within a pass, prefix siblings share their prefix as in KIRA's shuffled `prefix_repetition` dataset.

## Measured on ibm1: the A5000's kernel coefficients

[`gpubench/gpubench.py`](gpubench/gpubench.py) ran as a Slurm job on ibm1 (jobs 102 and 103, 30 Oct 2026, one
A5000, vLLM 0.28.0, bf16, FlashAttention 2). It times the kernels vLLM runs for Qwen3-4B. Calls under 1 ms are
replayed from a CUDA graph, because BLIS charges launch cost separately and eager timing measured ~97 µs of Python
per attention call. [`gpubench/fit_gpubench.py`](gpubench/fit_gpubench.py) fits the rows with blis-registry's own
fitters (the ones that made the L40S values) and writes [`overlay/measured.json`](overlay/measured.json).

| coefficient | was (L40S copy or datasheet) | **measured on ibm1** | fit |
|---|--:|--:|---|
| `gemm_eps_max_bf16` / `gemm_m_half_bf16` | 0.807 / 146 tokens | **0.847 / 64 tokens** | rms 0.050, 24 token counts |
| `hbm_derate` | 0.80 | **0.889** (683 of 768 GB/s) | same in both jobs |
| `host_link_bandwidth` | 32,000 B/µs (PCIe 4 x16) | **12,319 B/µs** (12.3 GB/s) | same in both jobs |
| `attention_decode_floor` / `_rate` | 11.0 µs / 645,120 B/µs | **16.5 µs / 691,200 B/µs** | geometric error 1.05×, 221 shapes |
| `attention_prefill_floor` / `_work_scale` | 12.5 µs / 0.46 | **31.0 µs / 0.64** | geometric error 1.18×, 30 shapes |

The host link is the biggest change: ibm1's GPUs copy to and from host memory at 12.3 GB/s, 2.6× slower than the
datasheet link, so a DRAM-tier restore takes 2.6× longer than the copied value modelled.

## Calibration: three coefficients, fitted only on KIRA numbers that are not scored

| GPU | coefficient (donor → fitted) | fitted to (KIRA measured) | source |
|---|---|---|---|
| A5000 | `host_launch_eager_per_layer` 7 → **57.9 µs/kernel** | chat TPOT = 39.35 ms (1024/256, 0.5 req/s) | E36 jobs 65/66 |
| Quadro | `attention_prefill_work_scale` 0.46 → **0.0154** | one 1,024-token chat prefill = 372 ms | tier_compare diagnosis: vLLM `request_prefill_time`, E33 cold prompts |
| Quadro | `host_launch_eager_per_layer` 7 → **48.2 µs/kernel** | chat TPOT = 52.2 ms | E36 jobs 47/48 |

The launch-cost fits are the hosts' 2.2 GHz Xeon running vLLM eager, which is KIRA's own diagnosis of where the step
time goes. No GPU benchmark measures it, since it is CPU dispatch. With the A5000 measured, the GPU explains less of
KIRA's chat TPOT, so the fit charges more to the CPU (57.9 µs against 27.1 µs on the L40S copies). The Quadro's tiny
attention scale is Triton attention on Turing (no FlashAttention). The A5000's prefill scale is now measured, so the
25.4 s 65K prefill it used to be fitted to is an unfitted check below.

## Unfitted checks

| check | KIRA measured | BLIS | error |
|---|--:|--:|--:|
| GPU KV cache, ibm1, util 0.9, 65,856 ctx | 95,696 tokens | 95,872 | +0.2% |
| GPU KV cache, ibm2, same settings | 96,383 tokens | 95,872 | −0.5% |
| one cold 65,536-token prefill on ibm1 (M15 E1: 3.8e8 B/s over 9.66 GB) | 25.4 s | 28.0 s | +10% |
| one 65,536-token prefill on ibm2 (fitted only at 1K) | ≈520 s (130 tok/s, run log job 10) | 844 s | +62% |
| knee, ibm1 agent_hot, KIRA's rule (≥70% of offered, p99 < 5× lowest) | 1/32 req/s | 1/32 req/s | match |
| knee, ibm2 chat, same rule | 1 req/s | 1 req/s | match |

## Results: the four devices (warm pass, stock, pooled over 5 seeds)

| experiment | KIRA TTFT p50 | BLIS TTFT p50 (seed range) | error | KIRA decode / request | BLIS decode / request | error |
|---|--:|--:|--:|--:|--:|--:|
| E32 ibm1 SATA | 18.8 s | 18.1 s (18.1–18.2) | **−3%** | 8.83 s | 12.8 s | +45% |
| E32 ibm1 NVMe | 4.3 s | 4.69 s (4.69–4.69) | **+9%** | 8.69 s | 12.7 s | +46% |
| E33 ibm2 SATA | 539 ms | 544 ms (542–546) | **+1%** | 9.84 s | 11.5 s | **+17%** |
| E33 ibm2 NVMe | 384 ms | 407 ms (405–409) | **+6%** | 9.66 s | 11.5 s | **+19%** |

| device effect | KIRA | BLIS | error |
|---|--:|--:|--:|
| ibm1: SATA ÷ NVMe warm TTFT p50 | 4.37× | 3.86× | −12% |
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

All four TTFT medians are within 9% and both device ratios within 12%, and ibm2 decode within 20%. ibm1's NVMe
median moved from −1% to +9%, and its device ratio from −5% to −12%, because each restore now crosses the measured
12.3 GB/s host link instead of the assumed 32 GB/s.

ibm1 decode stays 45–46% too slow (47–49% before), so the copied L40S coefficients were not its cause. KIRA's
8.83 s over 256 tokens is a 34.5 ms step, while BLIS charges 50 ms. A rough split from the measured rates: the GPU
part of one step at 65K context is about 26 ms (8 GB of weights at 683 GB/s plus 9.7 GB of KV at 691 GB/s), so
about 24 ms is launch cost. BLIS adds the two (≈50 ms). Full overlap would give ≈26 ms, and KIRA measured 34.5 ms,
in between: vLLM 0.28's asynchronous scheduler hides much of the CPU time behind the GPU step, which KIRA's diagnosis
also says ("hides under the 32 ms step"). At chat batch sizes, where the launch cost is fitted, the two models agree;
at one 65K request per step they don't.

## The same calibration on KIRA's other ibm1 runs (E1 / batch 1, SATA)

| case | KIRA p50 | BLIS p50 (seed range) | error | KIRA p99 | BLIS p99 | error |
|---|--:|--:|--:|--:|--:|--:|
| agent_spill, 32 GiB, stock (E1 job 46) | 18.8 | 18.1 (18.1–18.2) | **−3%** | 38.9 | 85.7 | +120% |
| agent_spill, 36 GiB, stock (E1 job 49) | 18.8 | 18.1 (18.1–18.2) | **−3%** | 38.5 | 57.3 | +49% |
| agent_spill, recompute (E1 jobs 46/49) | 27.45 | 34.7 (28.3–57.0) | +27% | 119.5 | 145 | **+22%** |
| agent_hot 4×3, 32 GiB, stock (E1 job 46) | 11.8 | 15.1 (3.5–30.2) | +28% | 31.7 | 71.3 | +125% |
| agent_hot 4×3, 36 GiB, stock (E1 job 49) | 4.9 | 22.9 (3.5–31.1) | +367% | 30.5 | 90.3 | +196% |
| agent_hot 4×3, recompute (E1 jobs 46/49) | 74.75 | 28.3 (28.3–36.6) | −62% | 142.0 | 153 | **+8%** |
| agent_hot 8×2, stock (batch 1, grid-16/22) | 42.15 | 18.1 (18.1–23.6) | −57% | 64.0 | 63.3 | **−1%** |
| agent_hot 8×2, recompute (batch 1, grid-16) | 65.9 | 28.4 (28.3–41.8) | −57% | n/a | 134 | n/a |

Gates (p50 and the prefill mean within 20%, p99 and decode within 30%): **12 of 24 scored values pass** across
all three tables, against 14 of 24 with the copied coefficients. The two lost are agent_spill recompute (−2% →
+27%) and agent_hot 32 GiB (+17% → +28%). Both follow from prefill: the 65K prefill was fitted to 25.4 s before and
is predicted at 28.0 s now. The copied values matched KIRA partly because a fitted coefficient absorbed their
errors; the measured ones are right for the GPU, and what is left over is in the engine model.

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
| agent_spill, 32 GiB | 18.1 | 4.69 (4.69–4.69) | 85.7 | 33.3 |
| agent_spill, 36 GiB | 18.1 | 4.69 (4.69–4.69) | 57.3 | 41.1 |
| agent_hot 4×3, 32 GiB | 15.1 | 3.88 (1.16–22.7) | 71.3 | 62.4 |
| agent_hot 4×3, 36 GiB | 22.9 | 17.7 (2.19–28.6) | 90.3 | 87.4 |
| agent_hot 8×2 (batch 1) | 18.1 | 4.69 (4.69–12.2) | 63.3 | 42.7 |

agent_spill gets the same 3.9× speedup as BLIS's E32 (KIRA measured 4.4× there). agent_hot gains less and stays dominated by what is resident in
DRAM.

## Limits

- **Warm prompt length.** BLIS's warm request is round 0's prompt plus its 256 output tokens plus one new token,
  which is 257 tokens more than KIRA re-sends. With prompt-only offload those tokens are recomputed unless still on
  the GPU: about +0.1 s on a 65K agent request, up to 20% of a 1K chat prompt. max-model-len is raised by 256 to
  admit it.
- **Engine version.** BLIS models vLLM 0.31.0 semantics (0.29.0 rules), and host overhead adds to GPU time.
  KIRA ran 0.28.0 with async scheduling, which overlaps them.
- **Coefficients.** The A5000's are measured on ibm1. The Quadro's GEMM efficiency, HBM derate, attention decode and
  host link are still L40S-derived until ibm2's run. Launch cost is fitted to KIRA on both, and the disk base
  latencies are assumed.
- **Traces.** These are reconstructions from KIRA's class definitions; KIRA's raw request traces are gitignored, so
  there is no per-request comparison.
- **Samples.** KIRA's cells are single runs (12–300 requests), so a p99 from 12–44 requests is essentially the max.

`run_first_set.py` is the earlier Route A generalization run (Qwen2.5-7B on L40S and A5000). It is not part of
this comparison, and its outputs (`results/`, `results-a5000/`) are not committed. Raw per-run metrics, traces and
offload configs under `results-kira/` are regenerated by `kira_repro.py` and are not committed either.
