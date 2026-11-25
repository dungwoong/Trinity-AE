"""Plain GEMM O = X @ W through Trinity's backend, run the way backend/run_eval_do_bench.py runs a case, next to cuBLAS and
a persistent Triton GEMM.

Trinity side, step by step (each step mirrors run_eval.py / run_eval_do_bench.py):
  1. IR      new_demo/cases/gemm.txt: the GEMM in Trinity's IR, with symbolic bounds M, N, K (run_eval.py reads
             results/{target}/{target}_{model}_case{n}.txt the same way). A single matmul gives Trinity's optimizer
             nothing to rewrite, so the IR is written by hand: a k loop accumulating O = (@ X W) + 1 * O, in the form
             of the matmuls in Trinity's extracted expressions (results/ffn/ffn_llama_case2248.txt). The code
             generator keeps O in registers through the k loop and stores it once after it.
  2. codegen convert_ir_to_triton(ir, tensor_shapes, constants): shapes named symbolically ('M', 'K'), sizes in
             `constants`, as run_eval.py does. Written to new_demo/results/gemm/gemm_m<M>_n<N>_k<K>_benchmark.py.
  3. load    importlib loads that file; `forward`, TENSOR_PARAMS and BLOCK_PARAMS are read from it, and the argument
             list is built by name from a tensors dict plus zeros for the block params, as run_eval.py does.
  4. time    triton.testing.do_bench(lambda: forward(*args)), then time.sleep(2), as run_eval_do_bench.py does. The
             generated kernel carries Trinity's own @triton.autotune (6 tile configs); its first call autotunes.
  dtype      float16, as run_eval.py (dtype = torch.float16; Trinity's codegen also keeps intermediates in fp16).

Baselines, same inputs, same timer (do_bench + sleep):
  cublas             torch.mm(X, W)
  triton_persistent  new_demo/triton_gemm.py: a persistent Triton GEMM with TMA loads, at its tuned schedule and
                     config for this shape (triton_gemm.CONFIGS). It takes W as (N, K) K-major, so it gets W^T
                     contiguous; Trinity's IR indexes W (k, n), so it gets W.
Correctness: each output against an fp64 GEMM of the same fp16 inputs (rmse and max abs, in fp64).

Run from Trinity-AE/backend, on a GPU node:
    python new_demo/run_gemm.py [shape idx ...] [--csv=new_demo/results/gemm/gemm.csv]
Shapes: SHAPES below (the keys of triton_gemm.CONFIGS), indexed from 0.
"""
import csv
import importlib.util
import pathlib
import sys
import time

import torch
from triton.testing import do_bench

HERE = pathlib.Path(__file__).resolve().parent          # backend/new_demo
sys.path.insert(0, str(HERE.parent))                     # backend/: Trinity's codegen
sys.path.insert(0, str(HERE))                            # new_demo/: the persistent Triton GEMM baseline

from codegen.convert_module import convert_ir_to_triton  # noqa: E402
import triton_gemm  # noqa: E402

SHAPES = list(triton_gemm.CONFIGS)                       # (M, N, K)

CASE = HERE / "cases" / "gemm.txt"
RESULTS = HERE / "results" / "gemm"
FIELDS = ("shape", "m", "n", "k", "method", "ms", "tflops", "rmse", "max_abs", "config")


# ---- Trinity: steps 1-3 ------------------------------------------------------------------------------------------------
def trinity_generate(M, N, K):
    """Steps 1 + 2: read the IR case file, run Trinity's code generator, write the benchmark module."""
    ir = CASE.read_text().strip()
    tensor_shapes = {"X": ("M", "K"), "W": ("K", "N"), "O": ("M", "N")}
    constants = {"M": M, "N": N, "K": K}
    triton_code = convert_ir_to_triton(ir, tensor_shapes, constants)
    out = RESULTS / f"gemm_m{M}_n{N}_k{K}_benchmark.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(triton_code)
    return out


def trinity_load(path, tensors):
    """Step 3: load the module and build forward's arguments from TENSOR_PARAMS / BLOCK_PARAMS, as run_eval.py."""
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    forward = getattr(module, "forward")
    args = [tensors[p] for p in getattr(module, "TENSOR_PARAMS")]
    args += [0 for _ in getattr(module, "BLOCK_PARAMS")]           # run_eval.py passes 0 for every block param
    return module, (lambda: forward(*args))


# ---- baselines ---------------------------------------------------------------------------------------------------------
def bench(f):
    """Step 4, the same for every method: do_bench, then a 2 s pause (run_eval_do_bench.py)."""
    ms = do_bench(f)
    time.sleep(2)
    return ms


def main():
    idxs = [int(a) for a in sys.argv[1:] if not a.startswith("--")] or range(len(SHAPES))
    out_csv = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--csv=")), None)
    dtype = torch.float16
    rows = []
    for i in idxs:
        M, N, K = SHAPES[i]
        torch.manual_seed(42)                                       # run_eval.py's seed
        X = torch.randn(M, K, device="cuda", dtype=dtype)
        W = (torch.randn(K, N, device="cuda") / K ** 0.5).to(dtype)   # (K, N), as Trinity's IR indexes it
        Wt = W.t().contiguous()                                     # (N, K) K-major, for the persistent Triton GEMM
        ref = X.double() @ W.double()
        O = torch.empty(M, N, device="cuda", dtype=dtype)

        module, trinity = trinity_load(trinity_generate(M, N, K), {"X": X, "W": W, "O": O})
        schedule, cfg = triton_gemm.CONFIGS[(M, N, K)]
        methods = {
            "cublas": (lambda: torch.mm(X, W, out=O), ""),
            "triton_persistent": (lambda: triton_gemm.gemm(X, Wt, O, schedule, cfg),
                                  f"{schedule} BM;BN;GROUP_M;stages;warps=" + ";".join(map(str, cfg))),
            "trinity": (trinity, None),
        }
        print(f"=== shape {i}: M={M} N={N} K={K}", flush=True)
        t = {}
        for name, (f, config) in methods.items():
            O.fill_(float("nan"))
            f()                                                     # Trinity: its @triton.autotune runs here
            torch.cuda.synchronize()
            err = O.double() - ref
            rmse, max_abs = err.pow(2).mean().sqrt().item(), err.abs().max().item()
            ms = bench(f)
            t[name] = ms
            if config is None:
                config = "autotuned " + str(module.kernel_0.best_config).replace(",", ";")
            tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
            print(f"  {name:18s} {ms * 1e3:8.1f} us  {tflops:6.0f} TFLOP/s  rmse {rmse:.2e}  max abs {max_abs:.2e}"
                  f"{'  NaN!' if torch.isnan(O).any() else ''}   {config}", flush=True)
            rows.append(dict(shape=i, m=M, n=N, k=K, method=name, ms=f"{ms:.5f}", tflops=f"{tflops:.1f}",
                             rmse=f"{rmse:.3e}", max_abs=f"{max_abs:.3e}", config=config))
        print(f"  Trinity time / persistent Triton time = {t['trinity'] / t['triton_persistent']:.2f}", flush=True)
    if out_csv:
        pathlib.Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
