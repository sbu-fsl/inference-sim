#!/usr/bin/env python3
"""Kernel timings on one GPU of KIRA's hosts, for the BLIS registry coefficients the overlay copies from L40S.

Times the kernels vLLM 0.28 runs for Qwen3-4B, in the model's dtype (bf16 on Ampere, fp16 on Turing):
  gemm       torch.nn.functional.linear, vLLM's unquantized linear      -> gemm_eps_max / gemm_m_half
  decode     paged attention, one query token per request                -> attention_decode_floor / _rate
  prefill    paged causal attention over the whole prompt                -> attention_prefill_floor / _work_scale
  membw      device-to-device copy                                       -> hbm_derate
  hostlink   pinned host<->device copy                                    -> host_link_bandwidth
Attention runs the kernel vLLM's backend calls: vllm_flash_attn (FlashAttention) on CC >= 8.0, Triton
unified_attention below that (KIRA's Quadro). Rows use AISimulate's column names and latency in MILLISECONDS,
so blis-registry's own fitters read them unchanged after conversion to Parquet.

    python gpubench.py --out bench.json [--smoke]
"""
import argparse, collections, json, platform, socket, statistics, subprocess, sys, time, traceback

import torch

QWEN3_4B = {"num_heads": 32, "num_kv_heads": 8, "head_dim": 128}
# Qwen3-4B linear layers as (N, K): qkv, o, gate_up, down, lm_head; plus wide shapes so the envelope reaches
# the asymptote AISimulate's broader sweeps reach
GEMM_NK = [(6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728), (151936, 2560), (4096, 4096), (8192, 8192)]
GEMM_M = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144,
          8192, 12288, 16384]
DECODE_BATCH = [1, 2, 4, 8, 16, 32, 64, 128]
DECODE_CTX = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DECODE_KV_HEADS = [4, 8, 16]  # Qwen3-4B is 8; the others widen the fit as AISimulate's head sweep does
PREFILL_BATCH = [1, 2, 4, 8]
PREFILL_ISL = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
BLOCK = 16  # KIRA serves with --block-size 16
MAX_BYTES = 6 << 30  # skip shapes whose tensors would not fit beside the others


TIMING = collections.Counter()  # how each row was timed, for the result file


def timed(fn, inner):
    """Median GPU ms of one call. BLIS charges launch cost separately (host_launch_eager_per_layer), so this must
    be kernel time only: a call under 1 ms is replayed from a CUDA graph, since on the hosts' 2.2 GHz Xeon the
    Python and launch path of one attention call (~97 us, job 102) outlasts a small kernel and back-to-back
    launches would time the host. Longer calls run eagerly; the launch is negligible beside them."""
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
    once = time.perf_counter() - t0
    reps = 3 if once > 0.5 else 10  # a long prefill on Turing takes seconds; keep the job bounded
    inner = 1 if once > 0.05 else inner
    def call():  # a loop, not a list: keeping 20 lm_head outputs (1.16 GiB each at M=4096) ran jobs 103 and 95 out of memory
        for _ in range(inner):
            fn()
    mode = "eager"
    if once < 1e-3:
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):  # capture needs a warm-up on a side stream
                fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(inner):
                    fn()
            call, mode = g.replay, "cuda_graph"
        except Exception:  # a kernel that cannot be captured is timed eagerly, and the row says so
            torch.cuda.synchronize()
    TIMING[mode] += 1
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        call()
        b.record(); b.synchronize()
        out.append(a.elapsed_time(b) / inner)
    if mode == "cuda_graph":  # a graph holds its outputs' memory until reset
        g.reset()
        torch.cuda.empty_cache()
    return statistics.median(out)


def bench_gemm(meta, dtype, smoke):
    rows = []
    ms = GEMM_M[::6] if smoke else GEMM_M
    for n, k in (GEMM_NK[:2] if smoke else GEMM_NK):
        w = torch.randn(n, k, dtype=dtype, device="cuda")
        for m in ms:
            if 2 * (m * k + m * n) > MAX_BYTES:
                continue
            x = torch.randn(m, k, dtype=dtype, device="cuda")
            lat = timed(lambda: torch.nn.functional.linear(x, w), 20)
            rows.append({**meta, "op_name": "gemm", "kernel_source": "torch.nn.functional.linear",
                         "gemm_dtype": meta["dtype"], "m": m, "n": n, "k": k, "latency": lat})
            del x
        del w
        torch.cuda.empty_cache()
    return rows


def paged(batch, q_len, kv_len, kv_heads, dtype):
    """Query tokens, a KV cache of exactly the blocks needed (scattered, as a running engine's are), and metadata."""
    nq, hd = kv_heads * QWEN3_4B["num_heads"] // QWEN3_4B["num_kv_heads"], QWEN3_4B["head_dim"]  # Qwen3's 4:1 GQA
    per = -(-kv_len // BLOCK)
    blocks = batch * per + 1
    q = torch.randn(batch * q_len, nq, hd, dtype=dtype, device="cuda")
    kc = torch.randn(blocks, BLOCK, kv_heads, hd, dtype=dtype, device="cuda")
    vc = torch.randn_like(kc)
    table = torch.randperm(blocks - 1, device="cuda", dtype=torch.int32)[: batch * per].view(batch, per)
    cu_q = torch.arange(0, batch + 1, device="cuda", dtype=torch.int32) * q_len
    used = torch.full((batch,), kv_len, device="cuda", dtype=torch.int32)
    return q, kc, vc, table, cu_q, used, nq, hd


def attention_kernel(cc):
    """The kernel vLLM 0.28's backend calls on this GPU, and its kernel_source label."""
    if cc >= (8, 0):
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        fa = get_flash_attn_version()

        def run(q, kc, vc, out, cu_q, max_q, used, max_k, table, nq, hd):
            flash_attn_varlen_func(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu_q, max_seqlen_q=max_q, seqused_k=used,
                                   max_seqlen_k=max_k, softmax_scale=hd ** -0.5, causal=True, window_size=(-1, -1),
                                   block_table=table, softcap=0.0, fa_version=fa)
        return run, f"vllm_flash_attn_fa{fa}"
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.utils.math_utils import next_power_of_2
    segs, bufs = 16, {}  # NUM_PAR_SOFTMAX_SEGMENTS; the 3D path's buffers, allocated once as the backend does

    def run(q, kc, vc, out, cu_q, max_q, used, max_k, table, nq, hd):
        thr = 128 // kc.shape[2]  # MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads, as TritonAttentionMetadataBuilder sets it
        if (thr, nq) not in bufs:
            f32 = dict(dtype=torch.float32, device="cuda")
            bufs[thr, nq] = (torch.empty((thr, nq, segs, next_power_of_2(hd)), **f32),
                             torch.empty((thr, nq, segs), **f32), torch.empty((thr, nq, segs), **f32))
        so, sm, se = bufs[thr, nq]
        unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu_q, max_seqlen_q=max_q, seqused_k=used,
                          max_seqlen_k=max_k, softmax_scale=hd ** -0.5, causal=True, window_size=(-1, -1),
                          block_table=table, softcap=0, q_descale=None, k_descale=None, v_descale=None,
                          seq_threshold_3D=thr, num_par_softmax_segments=segs,
                          softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se)
    return run, "vllm_triton_attn"


def bench_attention(meta, dtype, cc, smoke):
    run, src = attention_kernel(cc)
    rows = []
    decode = [(b, c, h) for h in DECODE_KV_HEADS for b in DECODE_BATCH for c in DECODE_CTX]
    prefill = [(b, s) for b in PREFILL_BATCH for s in PREFILL_ISL if b * s <= 32768]
    if smoke:
        decode, prefill = [(1, 1024, 8), (8, 4096, 8)], [(1, 1024)]
    for b, ctx, kvh in decode:
        if 2 * b * ctx * kvh * 128 * 2 > MAX_BYTES:
            continue
        q, kc, vc, table, cu_q, used, nq, hd = paged(b, 1, ctx + 1, kvh, dtype)  # seq_lens include the new token
        out = torch.empty_like(q)
        lat = timed(lambda: run(q, kc, vc, out, cu_q, 1, used, ctx + 1, table, nq, hd), 20)
        rows.append({**meta, "op_name": "generation_attention", "kernel_source": src, "batch_size": b, "isl": 1,
                     "step": ctx, "num_heads": nq, "num_key_value_heads": kvh, "head_dim": hd, "window_size": 0,
                     "kv_cache_dtype": meta["dtype"], "latency": lat})
        del q, kc, vc, out
        torch.cuda.empty_cache()
    for b, isl in prefill:
        q, kc, vc, table, cu_q, used, nq, hd = paged(b, isl, isl, 8, dtype)
        out = torch.empty_like(q)
        lat = timed(lambda: run(q, kc, vc, out, cu_q, isl, used, isl, table, nq, hd), 5)
        rows.append({**meta, "op_name": "context_attention", "kernel_source": src, "batch_size": b, "isl": isl,
                     "step": 0, "num_heads": nq, "num_key_value_heads": 8, "head_dim": hd, "window_size": 0,
                     "kv_cache_dtype": meta["dtype"], "latency": lat})
        del q, kc, vc, out
        torch.cuda.empty_cache()
    return rows


def bench_membw(meta, smoke):
    n = (256 if smoke else 1024) << 20
    a, b = torch.empty(n, dtype=torch.uint8, device="cuda"), torch.empty(n, dtype=torch.uint8, device="cuda")
    lat = timed(lambda: b.copy_(a), 5)
    return [{**meta, "op_name": "device_copy", "bytes_moved": 2 * n, "latency": lat}]  # read + write


def bench_hostlink(meta, smoke):
    n = (64 if smoke else 512) << 20
    h = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(n, dtype=torch.uint8, device="cuda")
    return [{**meta, "op_name": "h2d_pinned", "bytes_moved": n, "latency": timed(lambda: d.copy_(h, non_blocking=True), 3)},
            {**meta, "op_name": "d2h_pinned", "bytes_moved": n, "latency": timed(lambda: h.copy_(d, non_blocking=True), 3)}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="a few shapes per section, to check the job runs")
    args = ap.parse_args()
    torch.manual_seed(0)
    cc = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if cc >= (8, 0) else torch.float16  # vLLM serves the bf16 checkpoint in fp16 on Turing
    try:
        import vllm
        vllm_version = vllm.__version__
    except Exception:
        vllm_version = "unavailable"
    meta = {"framework": "VLLM", "version": vllm_version, "device": torch.cuda.get_device_name(),
            "dtype": "bfloat16" if dtype == torch.bfloat16 else "float16"}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,clocks.sm,clocks.max.sm,clocks.mem,power.limit,"
                          "temperature.gpu,utilization.gpu", "--format=csv"], capture_output=True, text=True).stdout
    res = {"host": socket.gethostname(), "python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "capability": list(cc), "nvidia_smi_before": smi, "rows": [], "errors": {}}
    for name, fn in [("membw", lambda: bench_membw(meta, args.smoke)),
                     ("hostlink", lambda: bench_hostlink(meta, args.smoke)),
                     ("gemm", lambda: bench_gemm(meta, dtype, args.smoke)),
                     ("attention", lambda: bench_attention(meta, dtype, cc, args.smoke))]:
        t0 = time.time()
        try:  # one section failing (e.g. a kernel import that moved) must not lose the others
            rows = fn()
            res["rows"] += rows
            print(f"{name}: {len(rows)} rows in {time.time() - t0:.0f} s", flush=True)
        except Exception:
            res["errors"][name] = traceback.format_exc()
            print(f"{name}: FAILED\n{res['errors'][name]}", file=sys.stderr, flush=True)
        torch.cuda.empty_cache()
    res["timing"] = dict(TIMING)
    res["nvidia_smi_after"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {args.out}: {len(res['rows'])} rows, {len(res['errors'])} failed sections")
    sys.exit(1 if res["errors"] else 0)


if __name__ == "__main__":
    main()
