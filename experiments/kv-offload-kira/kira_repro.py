#!/usr/bin/env python3
"""Simulate KIRA's ibm1/ibm2 KV-offload runs in BLIS and score them against what KIRA measured.

    python3 experiments/kv-offload-kira/overlay/build_overlay.py   # once (calibrate rebuilds it)
    python3 experiments/kv-offload-kira/kira_repro.py calibrate    # fits 2 launch costs on independent KIRA numbers
    python3 experiments/kv-offload-kira/kira_repro.py checks       # memory, prefill, knee: unfitted checks
    python3 experiments/kv-offload-kira/kira_repro.py cases        # every scored KIRA case, 5 seeds

Workload (KIRA experiments/common/classes.py, run log section 2): every prompt is a shared prefix plus a unique
suffix; a pass sends each prefix `sightings` times in shuffled order at Poisson arrivals; the warm pass re-sends
the cold pass's prompts. BLIS replays that as one session per prompt (`--session-mode fixed-accumulate`): round 0
is the cold request, round 1 the warm one at the same offset in the second pass. Round 1 carries round 0's prompt
AND its 256 output tokens plus one new token, so a warm prompt is 257 tokens longer than KIRA's.
"""
import json, os, random, statistics, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OVERLAY = REPO.parent / "blis-deps" / "overlay-kira"
SCEN_DIR, OUT, CAL = HERE / "overlay" / "scenarios", HERE / "results-kira", HERE / "overlay" / "calibration.json"
GIB = 2**30
SEEDS = [1, 2, 3, 4, 5]
PASS_GAP_S = 600
CLASSES = {"agent": (58982, 6554, 256), "chat": (102, 922, 256)}  # prefix, suffix, output tokens
HOSTS = {  # scenario, GPU
    "ibm1": ("qwen3-4b-instruct-2507-rtx-a5000-agent.yaml", "rtx-a5000"),
    "ibm2": ("qwen3-4b-instruct-2507-quadro-rtx-6000-chat.yaml", "quadro-rtx-6000"),
}
DISKS = {"ibm1 SATA": "ibm1_intel_ssdsc2kg96_sata", "ibm1 NVMe": "ibm1_samsung_970_evo_plus_nvme",
         "ibm2 SATA": "ibm2_crucial_mx500_sata", "ibm2 NVMe": "ibm2_samsung_pm953_nvme"}

# Scored KIRA cases: warm pass, stock = released vLLM 0.28 OffloadingConnector (DRAM + disk), recompute = no
# connector. KIRA targets: TTFT p50/p99 (s) and mean decode time a request (s); None = not published.
# (label, host, class, prefixes, sightings, req/s, DRAM GiB, disk or None for recompute, (p50, p99, decode), source)
CASES = [
    ("E32 agent_spill, ibm1 SATA", "ibm1", "agent", 22, 2, 1 / 64, 32, "ibm1 SATA", (18.8, None, 8.83), "tier_compare E32 job 57"),
    ("E32 agent_spill, ibm1 NVMe", "ibm1", "agent", 22, 2, 1 / 64, 32, "ibm1 NVMe", (4.3, None, 8.69), "tier_compare E32 job 58"),
    ("E33 chat, ibm2 SATA", "ibm2", "chat", 150, 2, 0.5, 4, "ibm2 SATA", (0.539, None, 9.84), "tier_compare E33 job 39"),
    ("E33 chat, ibm2 NVMe", "ibm2", "chat", 150, 2, 0.5, 4, "ibm2 NVMe", (0.384, None, 9.66), "tier_compare E33 job 40"),
    ("E1 agent_spill, 32 GiB, stock", "ibm1", "agent", 22, 2, 1 / 64, 32, "ibm1 SATA", (18.8, 38.9, None), "m15_batch2 E1 job 46"),
    ("E1 agent_spill, 36 GiB, stock", "ibm1", "agent", 22, 2, 1 / 64, 36, "ibm1 SATA", (18.8, 38.5, None), "m15_batch2 E1 job 49"),
    ("E1 agent_hot, 32 GiB, stock", "ibm1", "agent", 4, 3, 1 / 64, 32, "ibm1 SATA", (11.8, 31.7, None), "m15_batch2 E1 job 46"),
    ("E1 agent_hot, 36 GiB, stock", "ibm1", "agent", 4, 3, 1 / 64, 36, "ibm1 SATA", (4.9, 30.5, None), "m15_batch2 E1 job 49"),
    ("E1 agent_spill, recompute", "ibm1", "agent", 22, 2, 1 / 64, 0, None, (27.45, 119.5, None), "m15_batch2 E1 jobs 46/49 (27.4/27.5, 120.1/118.9)"),
    ("E1 agent_hot, recompute", "ibm1", "agent", 4, 3, 1 / 64, 0, None, (74.75, 142.0, None), "m15_batch2 E1 jobs 46/49 (74.0/75.5, 140.3/143.7)"),
    ("batch 1 agent_hot, stock", "ibm1", "agent", 8, 2, 1 / 64, 32, "ibm1 SATA", (42.15, 64.0, None), "run log grid-16 (41.8) / grid-22 (42.5, p99 64)"),
    ("batch 1 agent_hot, recompute", "ibm1", "agent", 8, 2, 1 / 64, 0, None, (65.9, None, None), "run log grid-16"),
]
# KIRA ran E1 and batch 1 on ibm1's SATA only; the same stock cases on ibm1's NVMe are predictions (no KIRA value)
CASES += [(f"{c[0]} (NVMe)", *c[1:7], "ibm1 NVMe", (None, None, None), "prediction: KIRA ran this on SATA only")
          for c in CASES if c[7] == "ibm1 SATA" and not c[0].startswith("E32")]
# KIRA tier_compare/diagnosis.md, stock arms on the NVMe tiers (warm pass means): (case, warm_stats key, KIRA, what).
# Its ibm1 NVMe 2.6 s kv_offload_lookup_async_delay is not scored: BLIS reports only the whole pre-schedule wait
# (disk fetch + DRAM->GPU load + queue), not the deferral alone.
NVME_DIAG = [
    ("E33 chat, ibm2 NVMe", "prefill_s", 0.077, "s, scheduled to first token (vLLM request_prefill_time)"),
]


def offload_yaml(cpu_gib, disk):
    if not cpu_gib:
        return None
    y = (f"kv_offload:\n  cpu_bytes_to_use: {cpu_gib * GIB}\n  block_size: 16\n  eviction_policy: lru\n"
         "  offload_prompt_only: true\n")
    if disk:
        y += ("  secondary_tiers:\n    - type: fs\n      root_dir: /mnt/kv-cache\n      n_read_threads: 16\n"
              f"      n_write_threads: 16\n      direct_io: true\n      device_class: {DISKS[disk]}\n")
    p = OUT / "configs" / f"offload-{cpu_gib}g-{DISKS.get(disk, 'dram')}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(y)
    return p


def write_trace(stem, rows, accumulate):
    """rows: (session, round, prefix group, prefix len, input, output, arrival s)."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{stem}.yaml").write_text("trace_version: 3\ntime_unit: microseconds\nmode: real\nwarm_up_requests: 0\n"
                                    "workload_seed: 1\n" + ("session_context_growth: accumulate\n" if accumulate else ""))
    cols = ("request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,"
            "streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,"
            "reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,"
            "first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason")
    lines = [cols] + [f"{rid},c,,,{s},{rnd},{g},{pre},false,{inp},{out},0,0,0,0,0,,0,0,{int(t * 1e6)},{int(t * 1e6)},0,0,0,ok,,"
                      for rid, (s, rnd, g, pre, inp, out, t) in enumerate(rows)]
    Path(f"{stem}.csv").write_text("\n".join(lines) + "\n")
    return stem


def case_trace(name, cls, pool, sightings, rate, seed):
    """Cold pass = shuffled prefix repetition at Poisson arrivals; warm pass = the same prompts at the same offsets."""
    pre, suf, out = CLASSES[cls]
    rng = random.Random(seed)
    order = [i for i in range(pool) for _ in range(sightings)]
    rng.shuffle(order)
    t, cold = 0.0, []
    for i in order:
        t += rng.expovariate(rate)
        cold.append((i, t))
    warm0 = t + PASS_GAP_S
    rows = [(f"s{k}", 0, f"{name}-p{i}", pre, suf, out, at) for k, (i, at) in enumerate(cold)]
    rows += [(f"s{k}", 1, f"{name}-p{i}", pre, 1, out, warm0 + at) for k, (i, at) in enumerate(cold)]
    return write_trace(OUT / "traces" / f"{name}-r{rate:.5f}-s{seed}", rows, True), warm0


def replay(host, stem, cpu_gib, disk, tag, scenario=None, accumulate=True):
    m = OUT / "raw" / f"{tag}.json"
    m.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(REPO / "blis"), "replay", "--trace-header", f"{stem}.yaml", "--trace-data", f"{stem}.csv",
           "--scenario", scenario or HOSTS[host][0], "--scenarios", str(SCEN_DIR),
           "--registry", str(OVERLAY / "blis-registry"), "--metrics-path", str(m), "--server-series-interval-s", "1"]
    if accumulate:
        cmd += ["--session-mode", "fixed-accumulate"]
    else:  # replay's default horizon follows the trace; a lone long prefill can outlast it
        cmd += ["--horizon", str(10**13)]
    if (cfg := offload_yaml(cpu_gib, disk)):
        cmd += ["--kv-offload-config", str(cfg)]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "BLIS_CATALOG": str(OVERLAY / "blis-catalog")})
    if r.returncode:
        sys.exit(f"{tag}: blis replay failed\n{r.stderr[-2000:]}")
    return json.loads(m.read_text())


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q / 100 * len(xs) + 0.5) - 1))]  # nearest rank


def warm_stats(m, warm0):
    reqs = [r for r in m["requests"] if r["arrived_at"] >= warm0]
    cold = [r for r in m["requests"] if r["arrived_at"] < warm0]
    assert m["completed_requests"] == m["injected_requests"], "requests did not all complete"
    eng = m["server_series"]["engines"][0]
    src = {k: sum(eng[k][int(warm0):]) for k in ("prompt_tokens", "prompt_tokens_local_compute",
                                                 "prompt_tokens_local_cache_hit", "prompt_tokens_external_kv_transfer")}
    return {"n": len(reqs), "ttft_s": [r["ttft_ms"] / 1e3 for r in reqs],
            "decode_s": statistics.mean((r["e2e_ms"] - r["ttft_ms"]) / 1e3 for r in reqs),
            "tpot_ms": statistics.mean(r["itl_ms"] for r in reqs),
            # a request whose offload load is in flight is not scheduled, so the delay includes the disk wait
            "sched_s": statistics.mean(r["scheduling_delay_ms"] / 1e3 for r in reqs),
            "prefill_s": statistics.mean((r["ttft_ms"] - r["scheduling_delay_ms"]) / 1e3 for r in reqs),
            "cold_drained": max(r["arrived_at"] + r["e2e_ms"] / 1e3 for r in cold) < warm0,
            **{k.replace("prompt_tokens_", "") + "_frac": v / src["prompt_tokens"] for k, v in src.items() if k != "prompt_tokens"}}


# --- calibration: observations that are not scored cases -------------------------------------------------------

def single_prefill(host, tokens, scenario=None):
    stem = write_trace(OUT / "traces" / f"prefill-{host}-{tokens}", [("", 0, "", 0, tokens, 1, 0.0)], False)
    return replay(host, stem, 0, None, f"prefill-{host}-{tokens}", scenario, False)["requests"][0]["ttft_ms"] / 1e3


def chat_tpot(host, disk, scenario=None):
    """KIRA E36 shape: chat 1024/256 at 0.5 req/s, one cold pass of 300 requests over the 4,518-prefix pool."""
    rng, t, rows = random.Random(1), 0.0, []
    for _ in range(300):
        t += rng.expovariate(0.5)
        rows.append(("", 0, f"chat-p{rng.randrange(4518)}", 102, 922, 256, t))
    stem = write_trace(OUT / "traces" / f"chat-tpot-{host}", rows, False)
    reqs = replay(host, stem, 4, disk, f"chat-tpot-{host}", scenario, False)["requests"]
    return statistics.mean(r["itl_ms"] for r in reqs)  # ms


# gpu: [(coefficient, target, measure, lo, hi, why)], fitted in order by bisection (each observable falls as its
# coefficient's value rises for the scale, and rises with it for the launch cost; `sign` says which)
FITS = {
    # both GPUs' attention_prefill_work_scale is measured (gpubench, overlay/measured.json), so the prefills it was
    # fitted to (ibm1's 25.4 s 65K, ibm2's 372 ms 1K) are now unfitted checks; the launch cost is CPU dispatch,
    # which no kernel timing measures
    "rtx-a5000": [
        ("host_launch_eager_per_layer", 39.35, lambda: chat_tpot("ibm1", "ibm1 NVMe"), 0.0, 1000.0, +1,
         "Fitted so KIRA's chat class (1024/256, 0.5 req/s, 300 requests) has a 39.35 ms mean TPOT on ibm1 (KIRA performance.md, E36 jobs 65/66, stock: 39.2 and 39.5 ms): vLLM eager dispatch on a 2.2 GHz Xeon Silver 4114."),
    ],
    "quadro-rtx-6000": [
        ("host_launch_eager_per_layer", 52.2, lambda: chat_tpot("ibm2", "ibm2 NVMe"), 0.0, 1000.0, +1,
         "Fitted so KIRA's chat class (1024/256, 0.5 req/s, 300 requests) has a 52.2 ms mean TPOT on ibm2 (KIRA performance.md, E36 jobs 47/48, stock: 52.3 and 52.1 ms): vLLM eager dispatch on a 2.2 GHz Xeon Silver 4114."),
    ],
}


def rebuild(cal):
    CAL.write_text(json.dumps(cal, indent=1))
    subprocess.run([sys.executable, str(HERE / "overlay" / "build_overlay.py")], check=True, capture_output=True)


def calibrate():
    cal = {gpu: {} for gpu in FITS}
    for _ in range(2):  # the terms interact a little (launch cost lands in prefill steps too); a second round settles them
        for gpu, fits in FITS.items():
            for name, target, measure, lo, hi, sign, why in fits:
                for _ in range(20):
                    cal[gpu][name] = [round((lo + hi) / 2, 5), why]
                    rebuild(cal)
                    too_slow = measure() > target
                    lo, hi = (cal[gpu][name][0], hi) if too_slow == (sign < 0) else (lo, cal[gpu][name][0])
    rows = [{"gpu": g, "coefficient": n, "value": cal[g][n][0], "target": t, "fitted_to": m()}
            for g, f in FITS.items() for n, t, m, *_ in f]
    (OUT / "calibration_fit.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps(r))


# --- unfitted checks ------------------------------------------------------------------------------------------

def kira_knee(host, cls, pool, disk, cpu_gib, rates, duration):
    """KIRA's rule (run log section 2): stock arm, one pass per rate, highest rate keeping >= 70% of the offered rate
    with p99 TTFT under 5x the lowest rate's."""
    pre, suf, out = CLASSES[cls]
    rows = []
    for rate in rates:
        rng, t, trace = random.Random(1), 0.0, []
        n = max(2 * pool, round(duration * rate)) if cls == "agent" else round(duration * rate)
        order = [i % pool for i in range(n)]
        rng.shuffle(order)
        for i in order:
            t += rng.expovariate(rate)
            trace.append(("", 0, f"knee{cls}-p{i}", pre, suf, out, t))
        stem = write_trace(OUT / "traces" / f"knee-{host}-{rate:.5f}", trace, False)
        reqs = replay(host, stem, cpu_gib, disk, f"knee-{host}-{rate:.5f}", accumulate=False)["requests"]
        ttft = [r["ttft_ms"] / 1e3 for r in reqs]
        achieved = len(reqs) / max(r["arrived_at"] + r["e2e_ms"] / 1e3 for r in reqs)
        rows.append({"host": host, "rate": rate, "achieved_over_offered": round(achieved / rate, 3),
                     "ttft_p99_s": round(pct(ttft, 99), 2)})
        rows[-1]["sustained"] = rows[-1]["achieved_over_offered"] >= 0.7 and rows[-1]["ttft_p99_s"] < 5 * rows[0]["ttft_p99_s"]
    return rows


def checks():
    def kv(scenario, host):
        stem = write_trace(OUT / "traces" / "kv", [("", 0, "", 0, 16, 1, 0.0)], False)
        return replay(host, stem, 0, None, "kv-" + scenario, scenario, False)["kv_capacity_tokens_per_replica"]
    res = {
        "kv_tokens_ibm1": [kv(HOSTS["ibm1"][0], "ibm1"), 95696],
        "kv_tokens_ibm2": [kv("qwen3-4b-instruct-2507-quadro-rtx-6000-memcheck.yaml", "ibm2"), 96383],
        # m15_batch2.md E1: a 64K prefill at 3.8e8 B/s end to end over 9.66 GB (vLLM request_prefill_time 20-30 s)
        "prefill_65536_ibm1_s": [single_prefill("ibm1", 65536), 25.4],
        # tier_compare/diagnosis.md: vLLM request_prefill_time for E33's cold 1K chat prompts on ibm2
        "prefill_1024_ibm2_s": [single_prefill("ibm2", 1024), 0.372],
        # run log batch 1, ibm2 job 10: "Triton prefill of 59K tokens runs at 130 tok/s, 520 s a request"
        "prefill_65536_ibm2_s": [single_prefill("ibm2", 65536, "qwen3-4b-instruct-2507-quadro-rtx-6000-memcheck.yaml"), 520],
        "knee_ibm1_agent_hot": [kira_knee("ibm1", "agent", 8, "ibm1 SATA", 32, [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8], 600), "1/32"],
        "knee_ibm2_chat": [kira_knee("ibm2", "chat", 4518, "ibm2 SATA", 4, [0.5, 1, 2, 4, 8], 180), "1"],
    }
    (OUT / "checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# --- scored cases ----------------------------------------------------------------------------------------------

def cases():
    res = []
    for label, host, cls, pool, sightings, rate, gib, disk, target, src in CASES:
        name = f"{cls}{pool}x{sightings}"
        for seed in SEEDS:
            stem, warm0 = case_trace(name, cls, pool, sightings, rate, seed)
            tag = f"{label}-s{seed}".replace(" ", "_").replace(",", "")
            res.append({"case": label, "seed": seed, **warm_stats(replay(host, stem, gib, disk, tag), warm0)})
    (OUT / "cases.json").write_text(json.dumps(res, indent=1))
    table(res)


def table(res=None):
    res = res or json.loads((OUT / "cases.json").read_text())
    err = lambda sim, obs: f"{(sim - obs) / obs:+.0%}" if obs else "n/a"
    lines = ["| case | KIRA TTFT p50 | BLIS p50 (seed range) | error | KIRA p99 | BLIS p99 | error | KIRA decode/req | BLIS decode/req | error | external-KV tokens | n |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for label, *_, (k50, k99, kdec), src in CASES:
        rs = [x for x in res if x["case"] == label]
        t = [v for x in rs for v in x["ttft_s"]]
        p50s = [statistics.median(x["ttft_s"]) for x in rs]
        b50, b99, bdec = statistics.median(t), pct(t, 99), statistics.mean(x["decode_s"] for x in rs)
        assert all(x["cold_drained"] for x in rs), f"{label}: cold pass did not drain"
        lines.append(f"| {label} | {k50 or 'n/a'} | {b50:.3g} ({min(p50s):.3g}-{max(p50s):.3g}) | {err(b50, k50)} | {k99 or 'n/a'} | "
                     f"{b99:.3g} | {err(b99, k99)} | {kdec or 'n/a'} | {bdec:.3g} | {err(bdec, kdec)} | "
                     f"{statistics.mean(x['external_kv_transfer_frac'] for x in rs):.0%} | {len(t)} |")
    lines += ["", "| NVMe stock, warm pass | KIRA | BLIS (seed range) | error | quantity |", "|---|--:|--:|--:|---|"]
    for label, key, k, what in NVME_DIAG:
        v = [x[key] for x in res if x["case"] == label]
        lines.append(f"| {label} | {k} | {statistics.mean(v):.3g} ({min(v):.3g}-{max(v):.3g}) | {err(statistics.mean(v), k)} | {what} |")
    (OUT / "cases.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    {"calibrate": calibrate, "checks": checks, "cases": cases, "table": table}[sys.argv[1]]()
