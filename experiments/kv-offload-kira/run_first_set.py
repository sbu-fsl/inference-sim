#!/usr/bin/env python3
"""First SATA batch of the KIRA KV-offload comparison (Route A, docs/kv-offload-blocker-research.md).

Builds KIRA-shaped two-pass traces (cold pass, gap, warm pass; every prefix seen exactly twice a
pass, open-loop Poisson arrivals), replays each through `blis replay` under three arms on the
same engine, and writes results/summary.json + results/summary.md.

    python3 experiments/kv-offload-kira/run_first_set.py knee     # stock-arm knee sweep
    python3 experiments/kv-offload-kira/run_first_set.py matrix   # workloads x arms x seeds
    KIRA_HW=a5000 python3 ...                                      # same, RTX A5000 overlay (overlay/)
"""
import json, os, random, statistics, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEPS = REPO.parent / "blis-deps"
HW = os.environ.get("KIRA_HW", "l40s")
if HW == "a5000":  # L40S-donor coefficients, uncalibrated (overlay/build_overlay.py)
    DEPS, SCEN_DIR, OUT = DEPS / "overlay-kira", HERE / "overlay" / "scenarios", HERE / "results-a5000"
    SCENARIO = "qwen2.5-7b-instruct-rtx-a5000-bf16-vllm-tp1.yaml"
else:
    SCEN_DIR, OUT = HERE / "scenarios", HERE / "results"
    SCENARIO = "qwen2.5-7b-instruct-l40s-bf16-vllm-tp1.yaml"
CATALOG, REGISTRY = DEPS / "blis-catalog", DEPS / "blis-registry"

# KIRA agent classes scaled from 65,536 to 16,384 input tokens, same 90/10 prefix/suffix split.
PREFIX, SUFFIX, OUTPUT = 14746, 1638, 256
DISK = "sata_ssd"
KV_BYTES_PER_TOKEN = 2 * 28 * 4 * 128 * 2  # Qwen2.5-7B BF16: 56 KiB
# GPU KV budget = 1.45 x max context on both GPUs (KIRA: 95,696 / 65,856), set by gpu_memory_utilization.
# CPU tier holds the same fraction of the agent_hot pool as KIRA's 32 GiB did of 8 x 58,982 x 144 KiB.
CPU_BYTES = round(32 * 2**30 * (8 * PREFIX * KV_BYTES_PER_TOKEN) / (8 * 58982 * 144 * 1024))
WORKLOADS = {"agent_hot": 8, "agent_spill": 22, "cpu_resident": 3}  # prefix pool sizes
ARMS = ["recompute", "cpu", "sata"]  # GPU prefix cache only / + CPU tier / + CPU tier + SATA fs tier
PASS_GAP_S = 600
SEEDS = [1, 2, 3, 4, 5]


def offload_yaml(arm, cpu_bytes=None):
    if arm == "recompute":
        return None
    cpu_bytes = cpu_bytes or CPU_BYTES
    y = f"kv_offload:\n  cpu_bytes_to_use: {cpu_bytes}\n  block_size: 16\n  eviction_policy: lru\n  offload_prompt_only: true\n"
    if arm == "sata":
        y += (f"  secondary_tiers:\n    - type: fs\n      root_dir: /mnt/kv-cache\n      n_read_threads: 16\n"
              f"      n_write_threads: 16\n      direct_io: true\n      device_class: {DISK}\n")
    p = OUT / "configs" / f"offload-{arm}-{cpu_bytes}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(y)
    return p


def make_trace(name, pool, rate, seed, passes=2, schedule=None, lengths=None, sightings=2):
    """Each pass sends every prefix twice in shuffled order at Poisson(rate) arrivals,
    unless an explicit [(prefix index, arrival s)] schedule is given."""
    rng = random.Random(seed)
    rows, t, bounds = [], 0.0, []
    for p in range(0 if schedule else passes):
        bounds.append(t)
        order = [i for i in range(pool) for _ in range(sightings)]
        rng.shuffle(order)
        for i in order:
            t += rng.expovariate(rate)
            rows.append((len(rows), f"{name}-p{i}", int(t * 1e6), p))
        t += PASS_GAP_S
    for i, at in schedule or []:
        rows.append((len(rows), f"{name}-p{i}", int(at * 1e6), 0))
    d = OUT / "traces"
    d.mkdir(parents=True, exist_ok=True)
    stem = d / f"{name}-r{rate:.5f}-s{seed}"
    Path(f"{stem}.yaml").write_text(
        "trace_version: 3\ntime_unit: microseconds\nmode: real\nwarm_up_requests: 0\n"
        f"workload_seed: {seed}\n")
    cols = ("request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,"
            "streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,"
            "reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,"
            "first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason")
    pre, suf, out = lengths or (PREFIX, SUFFIX, OUTPUT)
    lines = [cols] + [f"{rid},{g},,,,0,{g},{pre},false,{suf},{out},0,0,0,0,0,,0,0,{us},{us},0,0,0,ok,,"
                      for rid, g, us, _ in rows]
    Path(f"{stem}.csv").write_text("\n".join(lines) + "\n")
    return stem, bounds


def replay(stem, arm, tag, cpu_bytes=None):
    m = OUT / "raw" / f"{tag}.json"
    m.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(REPO / "blis"), "replay", "--trace-header", f"{stem}.yaml", "--trace-data", f"{stem}.csv",
           "--scenario", SCENARIO, "--scenarios", str(SCEN_DIR), "--registry", str(REGISTRY),
           "--metrics-path", str(m), "--server-series-interval-s", "1"]
    if (cfg := offload_yaml(arm, cpu_bytes)):
        cmd += ["--kv-offload-config", str(cfg)]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "BLIS_CATALOG": str(CATALOG)})
    if r.returncode:
        sys.exit(f"{tag}: blis replay failed\n{r.stderr[-2000:]}")
    return json.loads(m.read_text())


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q / 100 * len(xs) + 0.5) - 1))]  # nearest rank


def per_pass(metrics, bounds):
    """Split per-request records and server-series token sources by pass (arrival time)."""
    edges = bounds[1:] + [float("inf")]
    eng = metrics["server_series"]["engines"][0]
    out = []
    for k, (lo, hi) in enumerate(zip(bounds, edges)):
        reqs = [r for r in metrics["requests"] if lo <= r["arrived_at"] < hi]
        ttft = [r["ttft_ms"] / 1e3 for r in reqs]
        # 1 s bins from t=0: attribute token-source counters to the pass whose window holds the bin.
        b0, b1 = int(lo), (eng["bins"] if hi == float("inf") else int(hi))
        src = {k2: sum(eng[k2][b0:b1]) for k2 in ("prompt_tokens", "prompt_tokens_local_compute",
                                                  "prompt_tokens_local_cache_hit",
                                                  "prompt_tokens_external_kv_transfer")}
        last_done = max(r["arrived_at"] + r["e2e_ms"] / 1e3 for r in reqs)
        out.append({
            "phase": ["cold", "warm"][k], "n": len(reqs), "ttft_s": ttft,
            "ttft_p50_s": statistics.median(ttft), "ttft_p95_s": pct(ttft, 95), "ttft_p99_s": pct(ttft, 99),
            "e2e_p50_s": statistics.median(r["e2e_ms"] / 1e3 for r in reqs),
            "tpot_mean_ms": statistics.mean(r["itl_ms"] for r in reqs),
            "throughput_rps": len(reqs) / (last_done - min(r["arrived_at"] for r in reqs)),
            "drained_before_next_pass": last_done < hi,
            "gpu_hit_frac": src["prompt_tokens_local_cache_hit"] / src["prompt_tokens"],
            "external_hit_frac": src["prompt_tokens_external_kv_transfer"] / src["prompt_tokens"],
            "recomputed_frac": src["prompt_tokens_local_compute"] / src["prompt_tokens"],
        })
    return out


def knee():
    """Stock (sata) arm, agent_hot, three passes. Knee = highest rate whose TTFT p50 stays within 2x of the
    lowest rate's (queueing onset); the pass gaps make an achieved/offered ratio meaningless here."""
    rows = []
    for rate in [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2]:
        stem, b = make_trace("agent_hot", 8, rate, seed=1, passes=3)
        ttft = [r["ttft_ms"] / 1e3 for r in replay(stem, "sata", f"knee-{rate:.5f}")["requests"]]
        rows.append({"rate": rate, "ttft_p50_s": statistics.median(ttft), "ttft_p95_s": pct(ttft, 95)})
        rows[-1]["sustained"] = rows[-1]["ttft_p50_s"] <= 2 * rows[0]["ttft_p50_s"]
        print(json.dumps(rows[-1]))
    (OUT / "knee.json").write_text(json.dumps(rows, indent=1))


def paths():
    """Unloaded single requests, 60 s apart: full miss, GPU hit, CPU hit, then a prefix pushed out of
    GPU and CPU (five other prompts > the CPU tier) and reused: disk hit under sata, miss under cpu."""
    seq = [(0, "full miss"), (0, "GPU hit"), (1, "full miss"), (0, "CPU hit")] + \
          [(i, "full miss") for i in range(2, 7)] + [(0, "disk hit (sata) / miss (cpu)")]
    stem, _ = make_trace("paths", 0, 1, 0, schedule=[(i, 60 * k) for k, (i, _) in enumerate(seq)])
    rows = []
    for arm in ARMS:
        reqs = sorted(replay(stem, arm, f"paths-{arm}")["requests"], key=lambda r: r["arrived_at"])
        rows += [{"arm": arm, "step": k, "prefix": i, "expected": what, "ttft_s": r["ttft_ms"] / 1e3}
                 for k, ((i, what), r) in enumerate(zip(seq, reqs))]
    for r in rows:
        print(json.dumps(r))
    (OUT / "paths.json").write_text(json.dumps(rows, indent=1))


def matrix(rate):
    res = []
    for w, pool in WORKLOADS.items():
        for seed in SEEDS:
            stem, b = make_trace(w, pool, rate, seed)
            for arm in ARMS:
                for ph in per_pass(replay(stem, arm, f"{w}-{arm}-s{seed}"), b):
                    res.append({"workload": w, "arm": arm, "seed": seed, **ph})
                    print(json.dumps(res[-1]))
    (OUT / f"summary-1per{round(1 / rate)}s.json").write_text(
        json.dumps({"rate": rate, "cpu_bytes": CPU_BYTES, "runs": res}, indent=1))


def table(path):
    """Pool every seed's requests per (workload, phase, arm); seed range shown for the p50."""
    d = json.loads(Path(path).read_text())
    cells = {}
    for r in d["runs"]:
        cells.setdefault((r["workload"], r["phase"], r["arm"]), []).append(r)
    print(f"rate {d['rate']:.5f} req/s\n| workload | phase | arm | n | TTFT p50 (seed range) | p95 | p99 | E2E p50 | GPU hit | external | recomputed |")
    print("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for (w, ph, arm), rs in cells.items():
        t = [x for r in rs for x in r["ttft_s"]]
        p50s = [r["ttft_p50_s"] for r in rs]
        m = lambda f: statistics.mean(r[f] for r in rs)
        print(f"| {w} | {ph} | {arm} | {len(t)} | {statistics.median(t):.2f} ({min(p50s):.2f}-{max(p50s):.2f}) | "
              f"{pct(t, 95):.2f} | {pct(t, 99):.2f} | {m('e2e_p50_s'):.2f} | {m('gpu_hit_frac'):.0%} | "
              f"{m('external_hit_frac'):.0%} | {m('recomputed_frac'):.0%} |")
    assert all(r["drained_before_next_pass"] for r in d["runs"]), "a cold pass did not drain before warm"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    {"knee": knee, "paths": paths, "matrix": lambda: matrix(float(sys.argv[2])),
     "table": lambda: table(sys.argv[2])}[sys.argv[1]]()
