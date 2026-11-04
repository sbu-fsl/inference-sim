#!/usr/bin/env python3
"""Fit BLIS registry coefficients from a gpubench.py result, with blis-registry v0.2.0's own fitters.

The rows are written as AISimulate's Parquet tables (<data>/<sku>/{gemm,attention}/vllm/<version>/), then:
  gemm_eps_max / gemm_m_half          scripts/fit_gemm_envelope.py  (envelope + fit_ramp)
  attention_decode_floor / _rate      scripts/fit_attention.py
  attention_prefill_floor / _scale    scripts/fit_attention_prefill.py's fit on wider grids (fit_prefill)
  hbm_derate                          device copy rate / catalog BwPeakTBs
  host_link_bandwidth                 slower direction of the pinned copies, bytes/us

    python fit_gpubench.py ibm1-102.json ibm1-103.json   # needs pyarrow and pyyaml; later files win per op
"""
import importlib.util, json, math, re, subprocess, sys, tempfile
from pathlib import Path

import pyarrow as pa, pyarrow.parquet as pq, yaml

HERE = Path(__file__).resolve().parent
DEPS = HERE.parents[3] / "blis-deps"
SCRIPTS = DEPS / "blis-registry" / "scripts"
CATALOG = DEPS / "overlay-kira" / "blis-catalog"  # has the two GPUs
CHIP = {"NVIDIA RTX A5000": ("rtx_a5000", "rtx-a5000"), "Quadro RTX 6000": ("quadro_rtx_6000", "quadro-rtx-6000")}
TABLES = {"gemm": ("gemm", "gemm_perf"), "generation_attention": ("attention", "generation_attention_perf"),
          "context_attention": ("attention", "context_attention_perf")}


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_prefill(rows, peak, eps, m_half):
    """fit_attention_prefill.py's form and objective, latency = floor + causal_flops / (peak * eff(tokens) * scale)
    by mean |log ratio|, on wider grids: its scale grid starts at 0.10 and its floor stops at 39.5 us, and Turing's
    Triton prefill sits near 0.02, so there the registry fitter stops on its grid edge."""
    pts = []
    for r in rows:
        tok = r["batch_size"] * r["isl"]
        flops = 2.0 * r["batch_size"] * r["num_heads"] * r["head_dim"] * r["isl"] ** 2  # causal: half of 4*b*h*d*s^2
        pts.append((flops / (peak * eps * tok / (tok + m_half)), r["latency"] * 1e-3))
    err = lambda f, s: math.exp(sum(abs(math.log((f * 1e-6 + w / s) / lat)) for w, lat in pts) / len(pts))
    return min((err(f, s), f, s) for f in [x * 0.5 for x in range(2, 401)] for s in [x * 0.001 for x in range(5, 2001)])


def main(paths):
    """Several results from one GPU: each op's rows come from the last file that has them (a rerun's section
    replaces the earlier one; a section that failed in the rerun keeps the earlier rows)."""
    by_op = {}
    for p in paths:
        for r in json.loads(Path(p).read_text())["rows"]:
            by_op.setdefault(p, {}).setdefault(r["op_name"], []).append(r)
    rows = [r for op in {o for p in paths for o in by_op[p]}
            for r in next(by_op[p][op] for p in reversed(paths) if op in by_op[p])]
    sku, chip = next(v for k, v in CHIP.items() if k in rows[0]["device"])
    version = rows[0]["version"]
    hw = yaml.safe_load((CATALOG / "hardware" / f"{chip}.yaml").read_text())
    data = Path(tempfile.mkdtemp(prefix="gpubench-"))
    for op, (family, table) in TABLES.items():
        rs = [r for r in rows if r["op_name"] == op]
        d = data / sku / family / "vllm" / version
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rs), d / f"{table}.parquet")
    out = {"sources": {op: next(Path(p).name for p in reversed(paths) if op in by_op[p]) for op in sorted({o for p in paths for o in by_op[p]})}, "device": rows[0]["device"], "vllm": version, "dtype": rows[0]["dtype"]}

    fge = load("fit_gemm_envelope")
    peak = hw["TFlopsPeak"] * 1e12  # the dense bf16 rate (fp16 on Turing, where the GEMMs run in fp16)
    env = fge.envelope(fge.read_rows(data / sku / "gemm" / "vllm" / version, "gemm_perf"), rows[0]["dtype"], peak)
    eps, m_half, rms = fge.fit_ramp(env)
    out.update(gemm_eps_max=eps, gemm_m_half=m_half, gemm_rms=round(rms, 4), gemm_token_counts=len(env),
               gemm_best_fraction_of_peak=round(max(env.values()), 3))

    dec = subprocess.run([sys.executable, str(SCRIPTS / "fit_attention.py"),
                          str(data / sku / "attention" / "vllm" / version), "--chip", chip, "--catalog", str(CATALOG)],
                         capture_output=True, text=True)
    out["decode_fit"] = (dec.stdout + dec.stderr).strip()

    pre = load("fit_attention_prefill")
    out["prefill_fit"] = pre.fit(data, CATALOG, sku, chip, (eps, m_half), f"vllm/{version}")

    by = {r["op_name"]: r for r in rows}
    copy = by["device_copy"]
    out["hbm_derate"] = round(copy["bytes_moved"] / (copy["latency"] * 1e-3) / (hw["BwPeakTBs"] * 1e12), 3)
    out["host_link_bandwidth_bytes_per_us"] = round(min(by[k]["bytes_moved"] / (by[k]["latency"] * 1e3)
                                                       for k in ("h2d_pinned", "d2h_pinned")))
    floor, frac = (float(x) for x in re.search(r"decode_floor\s+([\d.]+) us\nattention_decode_bw_fraction\s+([\d.]+)",
                                                out["decode_fit"]).groups())
    perr, pfloor, scale = fit_prefill([r for r in rows if r["op_name"] == "context_attention"], peak, eps, m_half)
    out["prefill_fit_wide_grid"] = f"floor={pfloor:.1f}us work_scale={scale:.3f} geo-err {perr:.3f}x"
    out["coefficients"] = {  # registry names and units
        "gemm_eps_max_bf16": eps, "gemm_m_half_bf16": m_half, "hbm_derate": out["hbm_derate"],
        "host_link_bandwidth": out["host_link_bandwidth_bytes_per_us"],
        "attention_decode_floor": floor, "attention_decode_rate": round(frac * hw["BwPeakTBs"] * 1e6),
        "attention_prefill_floor": pfloor, "attention_prefill_work_scale": round(scale, 3)}
    print(json.dumps(out, indent=1))
    Path(paths[-1]).with_suffix(".fit.json").write_text(json.dumps(out, indent=1) + "\n")
    # the overlay's measured values for this GPU (build_overlay.py applies them as method: measured)
    mfile = HERE.parent / "overlay" / "measured.json"
    measured = json.loads(mfile.read_text()) if mfile.exists() else {}
    measured[chip] = {"cite": f"inference-sim experiments/kv-offload-kira/gpubench on KIRA's host ({out['device']}, "
                              f"vLLM {version}, {out['dtype']}): " + ", ".join(f"{k} from {v}" for k, v in out["sources"].items()),
                      "coefficients": out["coefficients"]}
    mfile.write_text(json.dumps(measured, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
