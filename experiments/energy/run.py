#!/usr/bin/env python3
"""BLIS energy-model study: held-out selection gate against ML.ENERGY measurements, plus E2-E5.

Stdlib only. Usage (from the repo root, after `go build -o blis main.go`):

    python3 experiments/energy/run.py gate          # embedded catalog and registry
    python3 experiments/energy/run.py experiments

The pre-declared selection gate is stated in experiments/energy/README.md.
Ground-truth data is downloaded at run time from a pinned commit and sha256-checked; it is
never committed (third-party data rule).
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import json

import os
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WORK = HERE / ".work"            # generated scenarios, workloads, traces (gitignored)
DATA = HERE / ".data"            # downloaded ML.ENERGY files (gitignored)
RESULTS = HERE / "findings" / "blis-vllm-0.32"  # this base; findings/kernel-v0.3.0 holds study 1

# ML.ENERGY leaderboard data, pinned (github.com/ml-energy/leaderboard, public/data/models).
LB_COMMIT = "2bdd0070186956688e1f842dc28077980fe64013"
LB_FILES = {
    "meta-llama__Llama-3.1-8B-Instruct__lm-arena-chat": "3db897caac1af5057fe2915a85c27cccaa5d734362f4c745d6f499cc308c0fe3",
    "Qwen__Qwen3-14B__lm-arena-chat": "a257c7c10b397d2271ce7c88ce39f2444bd987fbe95c6554d4049a6a3b7b09b8",
    "meta-llama__Llama-3.1-70B-Instruct__lm-arena-chat": "bafc501076c3ad0c28640d0be931983074e8c5f6da7ed93178cd8a9edfe2fc90",
    "Qwen__Qwen3-30B-A3B-Instruct-2507__lm-arena-chat": "3b8a304cf408f747f47c77203f147852b5d1fdd07e10607f2c9deeaa4e79d65a",
}
# leaderboard file -> (BLIS catalog model, role in the gate). H100 only; TP-only parallelism
# (the 2-GPU Qwen3-30B-A3B points use expert parallelism over DP, which metering refuses).
MODELS = {
    "meta-llama__Llama-3.1-8B-Instruct__lm-arena-chat": ("llama-3.1-8b-instruct", "calibration"),
    "Qwen__Qwen3-14B__lm-arena-chat": ("qwen3-14b", "held-out"),
    "meta-llama__Llama-3.1-70B-Instruct__lm-arena-chat": ("llama-3.1-70b-instruct", "held-out"),
    "Qwen__Qwen3-30B-A3B-Instruct-2507__lm-arena-chat": ("qwen3-30b-a3b", "held-out"),
}

RES = ["sm", "hbm", "nvlink", "nic", "pcie", "copy_engine", "host"]
H100 = {"tdp": 700.0, "peak_tflops": 989.5}  # NVIDIA datasheet; catalog hardware/h100.yaml
# Zero-shot constants, fixed before any prediction (spec §4.3).
ZERO_SHOT = {
    "tdp": {"watts": 700.0},
    "utilization": {"idle_w": 60.0, "max_w": 700.0, "mfu_sat": 0.45, "gamma": 0.7},
    "resource_share": {"sm": 595.0, "hbm": 350.0, "nvlink": 350.0, "nic": 350.0,
                       "pcie": 60.0, "copy_engine": 60.0, "host": 60.0},
}
SEED = 42


# ---------------------------------------------------------------- data

def fetch():
    """Download the pinned leaderboard files, verify sha256, return {name: json}."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, sha in LB_FILES.items():
        p = DATA / f"{name}.json"
        if not p.exists():
            url = f"https://raw.githubusercontent.com/ml-energy/leaderboard/{LB_COMMIT}/public/data/models/{name}.json"
            with urllib.request.urlopen(url, timeout=60) as r:
                p.write_bytes(r.read())
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != sha:
            sys.exit(f"{p}: sha256 {got} != pinned {sha}")
        out[name] = json.loads(p.read_text())
    return out


def points(data):
    """Measured H100 TP-only configurations, one dict per point."""
    pts = []
    for name, d in data.items():
        model, role = MODELS[name]
        for c in d["configurations"]:
            par = c["parallelization"]
            if c["gpu_model"] != "H100" or par["expert_parallel"] != 1 or par["data_parallel"] != 1:
                continue
            dist = c["output_length_distribution"]
            pts.append({
                "lb": name, "model": model, "role": role, "gpus": c["num_gpus"],
                "tp": par["tensor_parallel"], "max_num_seqs": c["max_num_seqs"],
                "active_params_b": d["activated_params_billions"],
                "meas_w_per_gpu": c["avg_power_watts"] / c["num_gpus"],
                "meas_j_per_tok": c["energy_per_token_joules"],
                "meas_itl_ms": c["median_itl_ms"], "meas_tput": c["output_throughput_tokens_per_sec"],
                "meas_batch": c["avg_batch_size"], "bins": dist["bins"], "counts": dist["counts"],
            })
    return sorted(pts, key=lambda p: (p["role"] != "calibration", p["model"], p["max_num_seqs"]))


# ---------------------------------------------------------------- simulation

def scenario(name, model, tp, max_num_seqs, mbt=8192, mml=20480, gpus_per_node=None):
    return f"""kind: Scenario
name: {name}
engine: {{kind: vllm, version: "0.32.0"}}
model: {model}
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory]
cluster:
  hardware: h100
  nodes: 1
  gpus_per_node: {gpus_per_node or tp}
---
kind: Deployment
name: {name}
pools:
  - role: colocated
    nodes: 1
    parallel: {{tp: {tp}, pp: 1, dp: 1, enable_expert_parallel: false}}
    engine:
      block_size: 16
      max_num_batched_tokens: {mbt}
      max_num_seqs: {max_num_seqs}
      max_model_len: {mml}
      gpu_memory_utilization: 0.9
"""


def workload(n, rate, in_dist, out_dist):
    return json.dumps({"version": "2", "seed": SEED, "aggregate_rate": rate, "num_requests": n,
                       "clients": [{"id": "c", "rate_fraction": 1.0, "arrival": {"process": "poisson"},
                                    "input_distribution": in_dist, "output_distribution": out_dist}]})


def const(v):
    return {"type": "constant", "params": {"value": v}}


def empirical(bins, counts):
    tot = sum(counts)
    pdf = {}
    for lo, hi, c in zip(bins, bins[1:], counts):
        if c:
            k = str(int(round((lo + hi) / 2)))
            pdf[k] = pdf.get(k, 0) + c / tot
    return {"type": "empirical", "params": pdf}


def blis(tag, scen_yaml, wl_json, energy_cfg, registry, extra=(), trace=True):
    """Run blis; return (stdout metrics dict, energy report dict, trace path or None)."""
    d = WORK / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "scen.yaml").write_text(scen_yaml)
    (d / "wl.json").write_text(wl_json)
    (d / "energy.yaml").write_text(energy_cfg)
    cmd = [str(REPO / "blis"), "run", "--scenario", "scen.yaml", "--scenarios", str(d),
           *(["--registry", registry] if registry else []),
           "--workload-spec", str(d / "wl.json"), "--seed", str(SEED),
           "--energy-config", str(d / "energy.yaml"), "--energy-report", str(d / "energy.json"), *extra]
    if trace:
        cmd += ["--energy-trace", str(d / "trace.csv")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{tag}: blis failed\n{r.stderr[-1500:]}")
    # stdout is the Result document; flatten its metrics to name -> value.
    metrics = {k: v["value"] for k, v in json.loads(r.stdout)["metrics"].items()
               if isinstance(v, dict) and "value" in v}
    return metrics, json.loads((d / "energy.json").read_text()), (d / "trace.csv" if trace else None)


def yaml_energy(model="resource_share", p=None, idle=60.0, pue=1.0, carbon=0.0, active_b=8.0):
    p = p or ZERO_SHOT[model]
    if model == "resource_share":
        body = "resource_share: {" + ", ".join(f"{r}_w: {p[r]}" for r in RES) + "}"
    elif model == "utilization":
        # The curve's floor (fitted P_idle) is an active-GPU power; between-step idle stays `idle`.
        body = (f"utilization: {{floor_w: {p['idle_w']}, max_w: {p['max_w']}, mfu_sat: {p['mfu_sat']}, "
                f"gamma: {p['gamma']}, peak_tflops: {H100['peak_tflops']}, active_params_b: {active_b}}}")
    else:
        body = f"tdp: {{watts: {p['watts']}}}"
    return f"model: {model}\nidle_w: {idle}\npue: {pue}\ncarbon_g_per_kwh: {carbon}\n{body}\n"


# ---------------------------------------------------------------- offline power models

def p_tdp(step, prm, _):
    return prm["watts"]


def p_util(step, prm, flops_tok_gpu):
    sec = step["dur"] / 1e6
    mfu = flops_tok_gpu * step["tok"] / (H100["peak_tflops"] * 1e12 * sec) if sec > 0 else 0.0
    x = min(1.0, mfu / prm["mfu_sat"])
    return prm["idle_w"] + (prm["max_w"] - prm["idle_w"]) * x ** prm["gamma"]


def p_rs(step, prm, _):
    tot = sum(step["busy"])
    if tot <= 0:
        return prm["host"]
    return sum(b * prm[r] for b, r in zip(step["busy"], RES)) / tot


MODEL_FNS = {"tdp": p_tdp, "utilization": p_util, "resource_share": p_rs}


def read_trace(path, min_batch):
    """Steady-state steps: running batch >= min_batch (ML.ENERGY's saturated window)."""
    steps = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["batch"]) >= min_batch:
                steps.append({"dur": int(row["dur_us"]), "busy": [int(row[r + "_us"]) for r in RES],
                              "tok": int(row["prefill_tokens"]) + int(row["decode_tokens"]),
                              "dec": int(row["decode_tokens"]), "w_go": float(row["watts_per_gpu"])})
    return steps


def avg_power(steps, fn, prm, ftg):
    t = sum(s["dur"] for s in steps)
    return sum(fn(s, prm, ftg) * s["dur"] for s in steps) / t


def simulate_point(p, registry, input_len):
    tag = f"gate/{p['model']}-tp{p['tp']}-b{p['max_num_seqs']}-in{input_len}"
    n = max(4 * p["max_num_seqs"], 1000)
    scen = scenario("pt", p["model"], p["tp"], p["max_num_seqs"])
    wl = workload(n, 1e6, const(input_len), empirical(p["bins"], p["counts"]))
    metrics, rep, trace = blis(tag, scen, wl, yaml_energy("utilization", active_b=p["active_params_b"]), registry)
    steps = read_trace(trace, 0.9 * p["meas_batch"])
    if len(steps) < 20:  # KV-limited configs never reach the measured batch; use their own peak
        peak = max(s for s in (int(r["batch"]) for r in csv.DictReader(open(trace))))
        steps = read_trace(trace, 0.9 * peak)
    window_s = sum(s["dur"] for s in steps) / 1e6
    dec = sum(s["dec"] for s in steps)
    go_w = avg_power(steps, lambda s, *_: s["w_go"], None, None)  # Go meter, utilization zero-shot
    return {**{k: v for k, v in p.items() if k not in ("bins", "counts")}, "input_len": input_len,
            "steps": steps, "window_s": window_s, "sim_tput": dec / window_s,
            "sim_itl_ms": statistics.median(s["dur"] for s in steps) / 1e3, "go_rs_w": go_w,
            "ftg": 2 * p["active_params_b"] * 1e9 / p["tp"]}


# ---------------------------------------------------------------- fitting

def lstsq(rows, ys):
    """Ordinary least squares via normal equations (small k)."""
    k = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(k)]
    for i in range(k):  # Gauss-Jordan
        piv = max(range(i, k), key=lambda r: abs(a[r][i]))
        a[i], a[piv], b[i], b[piv] = a[piv], a[i], b[piv], b[i]
        for r in range(k):
            if r != i and a[i][i]:
                f = a[r][i] / a[i][i]
                a[r] = [x - f * y for x, y in zip(a[r], a[i])]
                b[r] -= f * b[i]
    return [b[i] / a[i][i] for i in range(k)]


def rs_features(steps):
    t = sum(s["dur"] for s in steps)
    x = [0.0] * len(RES)
    for s in steps:
        tot = sum(s["busy"]) or 1
        for i, bsy in enumerate(s["busy"]):
            x[i] += s["dur"] * bsy / tot
    return [v / t for v in x]


def fit_rs(cal):
    """W_sm, W_hbm, W_host by least squares; others held at zero-shot (not identifiable at TP1)."""
    free = ["sm", "hbm", "host"]
    rows, ys = [], []
    for pt in cal:
        x = dict(zip(RES, rs_features(pt["steps"])))
        rows.append([x[r] for r in free])
        ys.append(pt["meas_w_per_gpu"] - sum(x[r] * ZERO_SHOT["resource_share"][r] for r in RES if r not in free))
    sol = lstsq(rows, ys)
    return {**ZERO_SHOT["resource_share"], **dict(zip(free, sol))}


def fit_rs_bounded(cal, lo=0.0, hi=1000.0, iters=20000):
    """POST-HOC (not part of the declared gate): fit_rs with 0 <= W <= 1000 W, by projected
    coordinate descent on the same least-squares objective."""
    free = ["sm", "hbm", "host"]
    rows, ys = [], []
    for pt in cal:
        x = dict(zip(RES, rs_features(pt["steps"])))
        rows.append([x[r] for r in free])
        ys.append(pt["meas_w_per_gpu"] - sum(x[r] * ZERO_SHOT["resource_share"][r] for r in RES if r not in free))
    w = [ZERO_SHOT["resource_share"][r] for r in free]
    for _ in range(iters):
        for j in range(len(w)):
            num = sum(r[j] * (y - sum(r[k] * w[k] for k in range(len(w)) if k != j)) for r, y in zip(rows, ys))
            den = sum(r[j] ** 2 for r in rows)
            w[j] = min(hi, max(lo, num / den)) if den else w[j]
    return {**ZERO_SHOT["resource_share"], **dict(zip(free, w))}


def fit_util(cal):
    """Grid over (gamma, mfu_sat); closed-form least squares for (idle_w, max_w)."""
    best = None
    for gi in range(1, 31):
        gamma = gi * 0.05
        for si in range(1, 21):
            sat = si * 0.05
            rows, ys = [], []
            for pt in cal:
                t = sum(s["dur"] for s in pt["steps"])
                f = sum(min(1.0, (pt["ftg"] * s["tok"] / (H100["peak_tflops"] * 1e12 * s["dur"] / 1e6)) / sat) ** gamma
                        * s["dur"] for s in pt["steps"]) / t
                rows.append([1 - f, f])
                ys.append(pt["meas_w_per_gpu"])
            idle, mx = lstsq(rows, ys)
            sse = sum((idle * r[0] + mx * r[1] - y) ** 2 for r, y in zip(rows, ys))
            if best is None or sse < best[0]:
                best = (sse, {"idle_w": idle, "max_w": mx, "mfu_sat": sat, "gamma": gamma})
    return best[1]


# ---------------------------------------------------------------- scoring

def mape(pairs):
    return 100 * statistics.mean(abs(s - m) / m for s, m in pairs)


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1)) if n > 2 else float("nan")


def score(pts, model, prm):
    fn = MODEL_FNS[model]
    rows = []
    for pt in pts:
        w = avg_power(pt["steps"], fn, prm, pt["ftg"])
        rows.append({"model": pt["model"], "b": pt["max_num_seqs"], "role": pt["role"],
                     "sim_w": w, "meas_w": pt["meas_w_per_gpu"],
                     "sim_jpt": w * pt["gpus"] / pt["sim_tput"], "meas_jpt": pt["meas_j_per_tok"]})
    rhos = []
    for m in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == m]
        rhos.append(spearman([r["sim_jpt"] for r in sub], [r["meas_jpt"] for r in sub]))
    return {"power_mape": mape([(r["sim_w"], r["meas_w"]) for r in rows]),
            "jpt_mape": mape([(r["sim_jpt"], r["meas_jpt"]) for r in rows]),
            "rank_rho": statistics.mean(rhos), "rows": rows}


# ---------------------------------------------------------------- commands

def cmd_gate(args):
    data = fetch()
    pts = points(data)
    RESULTS.mkdir(parents=True, exist_ok=True)
    sims = {}
    with cf.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futs = {ex.submit(simulate_point, p, args.registry, L): (i, L)
                for L in (256, 512, 1024) for i, p in enumerate(pts)}
        for f in cf.as_completed(futs):
            sims[futs[f]] = f.result()
            print(f"  done {futs[f]}", file=sys.stderr)
    by_len = {L: [sims[(i, L)] for i in range(len(pts))] for L in (256, 512, 1024)}
    main = by_len[512]

    # Go/Python cross-check: Python Utilization(zero-shot) must equal the Go meter's watts.
    xc = max(abs(avg_power(p["steps"], p_util, ZERO_SHOT["utilization"], p["ftg"]) - p["go_rs_w"]) / p["go_rs_w"]
             for p in main)

    cal = [p for p in main if p["role"] == "calibration"]
    held = [p for p in main if p["role"] == "held-out"]
    fitted = {"utilization": fit_util(cal),
              "tdp": {"watts": statistics.mean(p["meas_w_per_gpu"] for p in cal)}}
    n_params = {"resource_share": 3, "utilization": 4, "tdp": 1}
    out = {"ground_truth": {"source": "github.com/ml-energy/leaderboard", "commit": LB_COMMIT,
                            "sha256": LB_FILES},
           "go_python_crosscheck_max_rel_err": xc, "zero_shot": {}, "calibrated": {},
           "fitted_params": fitted, "latency": {}}
    for regime, params in (("zero_shot", ZERO_SHOT), ("calibrated", fitted)):
        for m in ("tdp", "utilization"):
            name = "constant_mean" if (regime == "calibrated" and m == "tdp") else m
            out[regime][name] = {"held_out": score(held, m, params[m]), "calibration": score(cal, m, params[m])}
    # Kernel latency fidelity on the same window (decomposes J/token error).
    out["latency"] = {"itl_mape": mape([(p["sim_itl_ms"], p["meas_itl_ms"]) for p in main]),
                      "tput_mape": mape([(p["sim_tput"], p["meas_tput"]) for p in main])}

    # resource_share (B) is not scored on this base: the blis-vllm engine exposes no per-resource
    # busy time through engine.Engine. Its comparison is study 1 (results/kernel-v0.3.0).

    # The gate (spec §6, declared before data).
    cand = {m: out["calibrated"][m]["held_out"]["power_mape"] for m in ("utilization",)}
    base = min(out["calibrated"]["constant_mean"]["held_out"]["power_mape"],
               out["zero_shot"]["tdp"]["held_out"]["power_mape"])
    winner = min(cand, key=cand.get)
    out["gate"] = {"held_out_power_mape": cand, "best_baseline_mape": base, "winner": winner,
                   "beats_baselines": cand[winner] < base}

    # E5a: prompt-length sensitivity of the winner's calibrated held-out errors.
    out["prompt_len_sensitivity"] = {
        L: {k: v for k, v in score([p for p in by_len[L] if p["role"] == "held-out"], winner,
                                   fitted[winner]).items() if k != "rows"} for L in (256, 512, 1024)}
    # E5b: ±20% one-at-a-time parameter sensitivity of held-out J/token (winner, calibrated).
    sens = {}
    base_jpt = score(held, winner, fitted[winner])
    for k, v in fitted[winner].items():
        if not isinstance(v, float) or v == 0:
            continue
        lo = score(held, winner, {**fitted[winner], k: v * 0.8})
        hi = score(held, winner, {**fitted[winner], k: v * 1.2})
        mean_j = lambda s: statistics.mean(r["sim_jpt"] for r in s["rows"])
        sens[k] = {"minus20_pct": 100 * (mean_j(lo) / mean_j(base_jpt) - 1),
                   "plus20_pct": 100 * (mean_j(hi) / mean_j(base_jpt) - 1)}
    out["param_sensitivity"] = sens

    with open(RESULTS / "points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["role", "model", "gpus", "max_num_seqs", "meas_w_per_gpu", "meas_j_per_tok", "meas_itl_ms",
                    "sim_itl_ms", "meas_tput", "sim_tput", "window_s",
                    *[f"{r}_{m}_w" for r in ("zs", "cal") for m in ("tdp", "util")],
                    *[f"{r}_{m}_jpt" for r in ("zs", "cal") for m in ("tdp", "util")]])
        for p in main:
            ws, js = [], []
            for prm in (ZERO_SHOT, fitted):
                for m in ("tdp", "utilization"):
                    pw = avg_power(p["steps"], MODEL_FNS[m], prm[m], p["ftg"])
                    ws.append(round(pw, 1))
                    js.append(round(pw * p["gpus"] / p["sim_tput"], 4))
            w.writerow([p["role"], p["model"], p["gpus"], p["max_num_seqs"], round(p["meas_w_per_gpu"], 1),
                        round(p["meas_j_per_tok"], 4), round(p["meas_itl_ms"], 2), round(p["sim_itl_ms"], 2),
                        round(p["meas_tput"]), round(p["sim_tput"]), round(p["window_s"], 2), *ws, *js])
    strip = lambda d: {k: (strip(v) if isinstance(v, dict) else v) for k, v in d.items() if k != "rows"}
    (RESULTS / "gate.json").write_text(json.dumps(strip(out), indent=2) + "\n")
    print(json.dumps(out["gate"], indent=2))


def load_fitted():
    g = json.loads((RESULTS / "gate.json").read_text())
    return g["gate"]["winner"], g["fitted_params"]


def cmd_experiments(args):
    winner, fitted = load_fitted()
    prm = fitted[winner]
    RESULTS.mkdir(exist_ok=True)
    reg = args.registry
    ecfg = lambda active_b=8.0, **kw: yaml_energy(winner, prm, idle=60.0, active_b=active_b, **kw)
    jobs = {}
    with cf.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        # E2: workload shape at fixed request rate, Llama-3.1-8B H100 TP1.
        for i, o in ((128, 512), (512, 512), (2048, 256), (4096, 64), (512, 2048), (8192, 128)):
            jobs[ex.submit(blis, f"e2/in{i}-out{o}", scenario("e2", "llama-3.1-8b-instruct", 1, 256),
                           workload(1000, 2.0, const(i), const(o)), ecfg(), reg)] = ("e2", i, o)
        # E3: over-provisioning at a fixed 20 req/s chat load.
        chat_in = {"type": "gaussian", "params": {"mean": 512, "std_dev": 256, "min": 16, "max": 4096}}
        chat_out = {"type": "exponential", "params": {"mean": 256}}
        for n in (1, 2, 4, 8):
            jobs[ex.submit(blis, f"e3/n{n}", scenario("e3", "llama-3.1-8b-instruct", 1, 256, gpus_per_node=8),
                           workload(4000, 20.0, chat_in, chat_out), ecfg(carbon=400.0, pue=1.2), reg,
                           ("--num-instances", str(n)), False)] = ("e3", n)
        # E4: Llama-3.1-70B on 2/4/8 H100 (TP), saturating burst at max_num_seqs 256.
        for tp in (2, 4, 8):
            jobs[ex.submit(blis, f"e4/tp{tp}", scenario("e4", "llama-3.1-70b-instruct", tp, 256),
                           workload(1024, 1e6, const(512), const(384)), ecfg(active_b=70.0), reg)] = ("e4", tp)
        res = {}
        for f in cf.as_completed(jobs):
            key = jobs[f]
            try:
                res[key] = f.result()
            except RuntimeError as e:
                res[key] = e
                print(f"  {key}: {str(e).splitlines()[0]}", file=sys.stderr)

    def trace_split(trace):
        e = {"prefill_only": 0.0, "mixed": 0.0, "decode_only": 0.0}
        with open(trace) as f:
            for row in csv.DictReader(f):
                pre, dec = int(row["prefill_tokens"]), int(row["decode_tokens"])
                k = "mixed" if pre and dec else ("prefill_only" if pre else "decode_only")
                e[k] += float(row["watts_per_gpu"]) * int(row["dur_us"]) / 1e6
        tot = sum(e.values()) or 1
        return {k: round(100 * v / tot, 1) for k, v in e.items()}

    out = {"model": winner, "params": prm, "e2": [], "e3": [], "e4": []}
    for key, r in sorted(res.items(), key=lambda kv: kv[0]):
        if isinstance(r, Exception):
            out[key[0]].append({"config": list(key[1:]), "error": str(r).splitlines()[1:3]})
            continue
        m, rep, trace = r
        c = rep["cluster"]
        row = {"config": list(key[1:]), "avg_busy_w_per_gpu": round(c["avg_busy_w_per_gpu"], 1),
               "avg_w_per_gpu": round(c["avg_w_per_gpu"], 1), "j_per_output_token": round(c["j_per_output_token"], 4),
               "j_per_request": round(c["j_per_request"], 1), "idle_share_pct": round(100 * c["idle_j"] / c["gpu_j"], 1),
               "ttft_p99_ms": round(m["ttft_p99_ms"], 1), "itl_mean_ms": round(m["tpot_mean_ms"], 2),
               "tokens_per_sec": round(m["output_tok_s"], 1), "facility_wh": round(c["facility_wh"], 2),
               "g_co2": round(c["g_co2"], 1)}
        if trace:
            row["energy_by_step_type_pct"] = trace_split(trace)
        out[key[0]].append(row)
    (RESULTS / "experiments.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["gate", "experiments"])
    ap.add_argument("--registry", default="", help="registry directory replacing the embedded one (optional)")
    args = ap.parse_args()
    {"gate": cmd_gate, "experiments": cmd_experiments}[args.command](args)


if __name__ == "__main__":
    main()
