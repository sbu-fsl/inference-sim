#!/usr/bin/env python3
"""Build the KIRA-hosts overlay of the pinned blis-catalog 0.3.0 and blis-registry v0.2.0.

Neither pinned repo knows KIRA's GPUs, model or disks. This copies both into
../blis-deps/overlay-kira/ and adds:
  catalog:  hardware/rtx-a5000.yaml (ibm1) and hardware/quadro-rtx-6000.yaml (ibm2), datasheet facts;
            models/qwen3-4b-instruct-2507 (derive_graph.py output next to this script; vendor config.json
            downloaded from Hugging Face at a pinned revision, sha256-checked);
            the four ibm disk tiers as storage devices, rates from KIRA's fio table.
  registry: both GPUs join every shared scope list that holds l40s (memory, host overheads), get
            copies of L40S's NCCL fits (required even at TP1), and their own GEMM/attention entries.

L40S is the donor for every GPU-specific value: marked `copied`, or `assumed` when scaled by bandwidth.
measured.json (../gpubench/fit_gpubench.py) replaces donor values with kernel timings on the GPU itself, and
calibration.json ({gpu: {name: [value, rationale]}}, written by ../kira_repro.py calibrate) with fits to KIRA
observations that are not the scored results.

    python3 experiments/kv-offload-kira/overlay/build_overlay.py
"""
import hashlib, json, re, shutil, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEPS = HERE.parents[3] / "blis-deps"
OUT = DEPS / "overlay-kira"
L40S_BW = 864.0
# vendor config.json is third-party data: fetched at a pinned revision and checked, never committed
QWEN_CONFIG = ("https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/cdbee75f17c01a7cc42f958dc650907174af0554/config.json",
               "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba")
# gpu: (datasheet GB/s, host link bytes/us, host link source)
GPUS = {
    "rtx-a5000": (768.0, 32000.0, "PCIe Gen4 x16, 32 GB/s per direction (NVIDIA RTX A5000 datasheet)"),
    "quadro-rtx-6000": (672.0, 16000.0, "PCIe Gen3 x16, ~16 GB/s per direction (NVIDIA Quadro RTX 6000 datasheet)"),
}

# KIRA archive/docs/2026-10-26-performance.md, fio 22 Oct 2026: 2 MiB sequential direct I/O, 4 jobs, depth 16.
# base_latency_us is ASSUMED from the catalog's sata_ssd (150) / nvme_gen3 (100) classes; the fio completion
# p50s (34-251 ms) are high-depth queueing, which BLIS's transfer station models itself.
DISKS = {
    "ibm1_intel_ssdsc2kg96_sata": ("ibm1 SATA, Intel SSDSC2KG96 960 GB (D3-S4610), mq-deadline", 561.0, 512.0, 150.0),
    "ibm1_samsung_970_evo_plus_nvme": ("ibm1 NVMe, Samsung 970 EVO Plus 500 GB, none scheduler", 2568.0, 3221.0, 100.0),
    "ibm2_crucial_mx500_sata": ("ibm2 SATA, Crucial MX500 1 TB, mq-deadline", 507.0, 518.0, 150.0),
    "ibm2_samsung_pm953_nvme": ("ibm2 NVMe, Samsung PM953 960 GB (PCIe 3.0), none scheduler", 1017.0, 764.0, 100.0),
}


def own(hw):
    """name: (value, units, method, rationale) for one GPU. Unscaled values are the L40S fit, verbatim."""
    bw, link, link_src = GPUS[hw]
    s = f"{bw:.0f}/{L40S_BW:.0f}"
    return {
        "cost-model-primitives.yaml": {
            "gemm_eps_max_bf16": (0.807, "dimensionless", "copied", "L40S fit (AISimulate l40s gemm, vLLM linear)."),
            "gemm_m_half_bf16": (146, "tokens", "copied", "L40S fit, fitted jointly with gemm_eps_max_bf16."),
            "moe_routing_imbalance_median": (0.981, "dimensionless", "copied", "L40S fit; unused by dense models."),
            "hbm_derate": (0.8, "dimensionless", "copied", "L40S value; all three parts are GDDR6."),
            "host_link_bandwidth": (link, "bytes_per_us", "vendor_spec", f"{link_src}. Nominal, not measured."),
            "nccl_communicator_bytes_2rank": (358612992, "bytes_per_rank", "copied", "L40S value; TP1 does not use it."),
            "nccl_communicator_bytes_4rank": (411041792, "bytes_per_rank", "copied", "L40S value; TP1 does not use it."),
            "nccl_communicator_bytes_8rank": (411041792, "bytes_per_rank", "copied", "L40S value; TP1 does not use it."),
            "engine_workspace_bytes": (3758096384, "bytes_per_rank", "copied", "L40S value."),
        },
        "cost-model-attention.yaml": {
            "attention_decode_floor": (11.0, "us_per_transfer", "copied", "L40S fit."),
            "attention_decode_rate": (round(725760.0 * bw / L40S_BW), "bytes_per_us", "assumed", f"L40S fit 725,760 B/us scaled by datasheet bandwidth {s}."),
            "attention_prefill_floor": (12.5, "us_per_transfer", "copied", "L40S fit."),
            "attention_prefill_work_scale": (0.46, "dimensionless", "copied", "L40S fit (A100 has the same value)."),
            "attention_decode_floor_swa": (8.5, "us_per_transfer", "copied", "L40S fit; unused without sliding-window layers."),
            "attention_decode_rate_swa": (round(224640.0 * bw / L40S_BW), "bytes_per_us", "assumed", f"L40S fit scaled by datasheet bandwidth {s}; unused without sliding-window layers."),
        },
        "cost-model-host-overheads.yaml": {
            "host_launch_eager_per_layer": (7.0, "us_per_layer", "copied", "The shared assumed 7 us per eager kernel launch (5-10 us convention); these hosts run vLLM eager on a 2.2 GHz Xeon Silver 4114."),
        },
    }


def block(hw, name, value, units, method, why):
    return (f"  - {name}:\n      value: {value}\n      units: {units}\n      method: {method}\n"
            + ('      copied_from: "hardware=l40s"\n' if method == "copied" else "") +
            # a measured curve fit is fitted; a measured rate (copy bandwidth) is read straight off the run
            f"      fitted: {'true' if method == 'measured' and name not in ('hbm_derate', 'host_link_bandwidth') else 'false'}\n"
            f"      scope: {{hardware: [{hw}]}}\n      sources:\n"
            + (f"        - {{kind: measured_case, cite: \"KIRA (github.com/sbu-fsl/KIRA) {hw}, Qwen3-4B-Instruct-2507, vLLM 0.28.0 eager: see rationale\", role: primary}}\n"
               "      validated: [ttft_p50, ttft_p99]\n" if method == "calibrated" else
               f"        - {{kind: measured_case, cite: \"{why}\", role: primary}}\n" if method == "measured" else
               "        - {kind: model, cite: \"inference-sim experiments/kv-offload-kira/overlay/build_overlay.py (L40S donor)\", role: primary}\n")
            + f"      rationale: >\n        {why}\n")


def clone_l40s_only(txt, hw):
    """Copies of every [l40s]-only block (the NCCL fits: PCIe peer-to-peer on all three parts)."""
    out = []
    for blk in re.split(r"\n(?=  - [A-Za-z0-9_]+:\n)", txt)[1:]:
        if "scope: {hardware: [l40s]}" in blk:
            blk = re.sub(r"_l40s:\n", f"_{hw.replace('-', '_')}:\n", blk, count=1)
            blk = re.sub(r"method: \S+\n", 'method: copied\n      copied_from: "hardware=l40s"\n', blk, count=1)
            blk = blk.replace("fitted: true", "fitted: false", 1)
            out.append(blk.replace("scope: {hardware: [l40s]}", f"scope: {{hardware: [{hw}]}}").rstrip("\n") + "\n")
    return "".join(out)


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(DEPS / "blis-catalog", OUT / "blis-catalog")
    shutil.copytree(DEPS / "blis-registry", OUT / "blis-registry")
    for hw in GPUS:
        shutil.copy(HERE / f"{hw}.yaml", OUT / "blis-catalog" / "hardware" / f"{hw}.yaml")
    # Qwen3-4B-Instruct-2507: vendor config at a pinned revision + graph from the catalog's derive_graph.py
    shutil.copytree(HERE / "models", OUT / "blis-catalog" / "models", dirs_exist_ok=True)
    cfg = HERE / "models" / "qwen3-4b-instruct-2507" / "config.json"
    if not cfg.exists():
        cfg.write_bytes(urllib.request.urlopen(QWEN_CONFIG[0]).read())
    if hashlib.sha256(cfg.read_bytes()).hexdigest() != QWEN_CONFIG[1]:
        cfg.unlink()
        raise SystemExit(f"{cfg}: sha256 mismatch with {QWEN_CONFIG[0]}")
    shutil.copy(cfg, OUT / "blis-catalog" / "models" / "qwen3-4b-instruct-2507" / "config.json")
    with open(OUT / "blis-catalog" / "devices" / "storage.yaml", "a") as f:
        for name, (what, rd, wr, lat) in DISKS.items():
            f.write(f"{name}:\n  _comment: \"KIRA {what}. read/write = fio 2 MiB sequential direct I/O, 4 jobs, depth 16, "
                    f"22 Oct 2026 (KIRA archive/docs/2026-10-26-performance.md). base_latency_us ASSUMED from the "
                    f"catalog class.\"\n  read_bandwidth_mb_s: {rd}\n  write_bandwidth_mb_s: {wr}\n  base_latency_us: {lat}\n")
    sets = {hw: own(hw) for hw in GPUS}
    # measured.json ({gpu: {cite, coefficients}}, written by ../gpubench/fit_gpubench.py): kernel timings on the
    # GPU itself replace the donor values; calibration.json is applied after, for what no benchmark measures
    meas = HERE / "measured.json"
    for hw, m in (json.loads(meas.read_text()) if meas.exists() else {}).items():
        for name, value in m["coefficients"].items():
            for entries in sets[hw].values():
                if name in entries:
                    entries[name] = (value, entries[name][1], "measured", m["cite"])
    cal = HERE / "calibration.json"
    for hw, fits in (json.loads(cal.read_text()) if cal.exists() else {}).items():
        for name, (value, why) in fits.items():
            for entries in sets[hw].values():
                if name in entries:
                    entries[name] = (value, entries[name][1], "calibrated", why)
    for f in sorted((OUT / "blis-registry" / "coefficients").glob("*.yaml")):
        txt = f.read_text()
        # shared lists only (in v0.2.0 l40s is always last); an [l40s]-only entry is that part's own fit, and a
        # name these GPUs define themselves (own()) stays out of the shared list so one scope answers
        mine = {n for hw in GPUS for n in sets[hw].get(f.name, {})}
        parts = re.split(r"\n(?=  - [A-Za-z0-9_]+:\n)", txt)
        txt = "\n".join(b if b.lstrip(" -").split(":", 1)[0] in mine else b.replace(", l40s]", ", l40s, " + ", ".join(GPUS) + "]")
                        for b in parts)
        extra = "".join(clone_l40s_only(txt, hw) for hw in GPUS) if f.name == "cost-model-collectives.yaml" else ""
        extra += "".join(block(hw, n, *v) for hw in GPUS for n, v in sets[hw].get(f.name, {}).items())
        f.write_text(txt.rstrip("\n") + "\n" + extra)
    print(f"overlay at {OUT}")


if __name__ == "__main__":
    main()
