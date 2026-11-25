"""Trinity's fused candidates for RMSNorm + Linear, SwiGLU and LoRA (one shape each): generate, check and time every
candidate the optimizer produced, as backend/profile/benchmark.py does, plus a correctness check.

Pipeline:
  1. optimizer   optimizer/tests/new_demo.rs runs Trinity's optimizer on the unfused program (the same calls as
                 tests/ffn.rs) and writes new_demo/cases/<workload>_candidates.txt, one "<id>: <IR>" per line.
                     cd optimizer && cargo test --release --test new_demo -- --nocapture --test-threads=1
  2. codegen     for each candidate: convert_ir_to_triton(ir, tensor_shapes, None) with concrete shapes, as
                 profile/benchmark.py (generate_kernel_code); written to results/<workload>/cand<id>.py.
  3. tensors     as profile/benchmark.py: inputs randn.clamp(-1, 1) * 0.01 in fp16; outputs and intermediates zeros,
                 fp32 for the names the module lists in FP32_TENSOR_PARAMS; args built from TENSOR_PARAMS.
  4. check       one call on zeroed outputs / intermediates, output vs an fp64 reference of the same inputs
                 (rmse, max abs; rel = max abs / max |ref|). profile/benchmark.py does not check; we flag a candidate
                 WRONG if rel > 5e-2 or the output has NaN / inf, and pick the fastest correct one.
  5. time        triton.testing.do_bench(lambda: forward(*args)), then time.sleep(2) (run_eval_do_bench.py).
                 profile/benchmark.py instead replays a CUDA graph 100 times; we use do_bench for every method.
                 Two passes: candidates whose generated code is identical are run once; every distinct candidate is
                 screened with one do_bench (no pause), then the 5 fastest correct ones and the baselines are timed
                 again with do_bench + sleep(2). The csv keeps both times (ms_screen, ms).
Baselines on the same inputs, same timer: PyTorch eager and torch.compile of the unfused program (fp16, cuBLAS GEMMs).

From Trinity-AE/backend, on a GPU node:
    python new_demo/run_fusion.py WORKLOAD [--csv=new_demo/results/WORKLOAD/WORKLOAD.csv]
WORKLOAD: rmsnorm_linear | swiglu | lora
"""
import csv
import importlib.util
import pathlib
import sys
import time
import traceback

import torch
from triton.testing import do_bench

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from codegen.convert_module import convert_ir_to_triton  # noqa: E402

# Concrete shapes and tensor types (as profile/shapes/*.json); must match optimizer/tests/new_demo.rs.
WORKLOADS = {
    "rmsnorm_linear": dict(
        shapes={"X": (4096, 4096), "S": (4096,), "Xn": (4096, 4096), "W": (4096, 1024), "O": (4096, 1024)},
        inputs=("X", "W"), output="O",
        # S = rowsum(X^2); O = (X / sqrt(S / K)) @ W   (no eps and no weight, as written in Trinity's IR)
        ref=lambda t: (t["X"] / (t["X"].pow(2).sum(1, keepdim=True) / t["X"].shape[1]).sqrt()) @ t["W"],
        torch_fn=lambda X, W: (X / X.float().pow(2).mean(1, keepdim=True).sqrt().to(X.dtype)) @ W),
    "swiglu": dict(
        shapes={"X": (4096, 4096), "Wg": (4096, 14336), "Wu": (4096, 14336), "G": (4096, 14336), "U": (4096, 14336),
                "O": (4096, 14336)},
        inputs=("X", "Wg", "Wu"), output="O",
        ref=lambda t: torch.nn.functional.silu(t["X"] @ t["Wg"]) * (t["X"] @ t["Wu"]),
        torch_fn=lambda X, Wg, Wu: torch.nn.functional.silu(X @ Wg) * (X @ Wu)),
    "lora": dict(
        shapes={"X": (4096, 4096), "A": (4096, 16), "B": (16, 4096), "W": (4096, 4096), "T": (4096, 16),
                "Y": (4096, 4096), "O": (4096, 4096)},
        inputs=("X", "A", "B", "W"), output="O",
        ref=lambda t: t["X"] @ t["W"] + (t["X"] @ t["A"]) @ t["B"],
        torch_fn=lambda X, A, B, W: X @ W + (X @ A) @ B),
}
FIELDS = ("workload", "method", "ms", "ms_screen", "kernels", "rmse", "max_abs", "rel", "status", "note")
FINAL = 5


def candidates(name):
    """(id, IR) from cases/<name>_candidates.txt, as profile/benchmark.py parse_ir_file."""
    out = []
    for line in (HERE / "cases" / f"{name}_candidates.txt").read_text().splitlines():
        if ":" in line:
            i, ir = line.split(":", 1)
            if "dummydata" not in ir:
                out.append((int(i), ir.strip()))
    return out


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem + "_" + path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bench(f):
    ms = do_bench(f)
    time.sleep(2)
    return ms


def main():
    name = sys.argv[1]
    out_csv = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--csv=")), None)
    w = WORKLOADS[name]
    outdir = HERE / "results" / name
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    inputs = {n: (torch.randn(w["shapes"][n], device="cuda", dtype=torch.float16).clamp(-1, 1) * 0.01)
              for n in w["inputs"]}
    ref = w["ref"]({n: t.double() for n, t in inputs.items()})
    scale = ref.abs().max().item()

    def check(o):
        err = o.double() - ref
        bad = not torch.isfinite(o).all().item()
        mx = err.abs().max().item()
        return err.pow(2).mean().sqrt().item(), mx, mx / scale, bad

    rows, runners, seen = [], {}, {}
    # ---- Trinity's candidates: screening pass ----
    for cid, ir in candidates(name):
        row = dict(workload=name, method=f"trinity_cand{cid}", note="", ms="")
        try:
            code = convert_ir_to_triton(ir, w["shapes"], None)
            key = "".join(l.split("#")[0] for l in code.splitlines())
            if key in seen:                          # same generated code as an earlier candidate
                row.update(status="duplicate", note=f"same kernel as {seen[key]}")
                rows.append(row)
                continue
            seen[key] = row["method"]
            path = outdir / f"cand{cid}.py"
            path.write_text(code)
            mod = load(path)
            row["kernels"] = code.count("@triton.jit")
            fp32 = set(getattr(mod, "FP32_TENSOR_PARAMS", []))
            tensors = dict(inputs)
            for n in mod.TENSOR_PARAMS:
                if n not in tensors:
                    tensors[n] = torch.zeros(w["shapes"][n], device="cuda",
                                             dtype=torch.float32 if n in fp32 else torch.float16)
            args = [tensors[n] for n in mod.TENSOR_PARAMS] + [0 for _ in getattr(mod, "BLOCK_PARAMS", [])]
            mod.forward(*args)                       # first call: Trinity's autotune
            for n in mod.TENSOR_PARAMS:              # zero outputs / intermediates, one clean call to check
                if n not in inputs:
                    tensors[n].zero_()
            mod.forward(*args)
            torch.cuda.synchronize()
            rmse, mx, rel, bad = check(tensors[w["output"]])
            row.update(rmse=f"{rmse:.3e}", max_abs=f"{mx:.3e}", rel=f"{rel:.3e}",
                       status="WRONG" if bad or rel > 5e-2 else "ok")
            row["ms_screen"] = f"{do_bench(lambda: mod.forward(*args)):.5f}"
            runners[row["method"]] = (lambda mod=mod, args=args: mod.forward(*args))
        except Exception as e:
            row.update(status="FAILED", note=f"{type(e).__name__}: {str(e)[:120]}")
            traceback.print_exc(limit=1)
        print(f"  {row['method']:18s} {row.get('ms_screen') or '-':>10s} ms  rel {row.get('rel') or '-':>10s}  "
              f"{row['status']}  {row['note']}", flush=True)
        rows.append(row)

    # ---- final pass: the FINAL fastest correct candidates, do_bench + sleep ----
    ok = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: float(r["ms_screen"]))
    for r in ok[:FINAL]:
        r["ms"] = f"{bench(runners[r['method']]):.5f}"
        print(f"  final {r['method']:18s} {r['ms']} ms", flush=True)

    # ---- baselines ----
    args = [inputs[n] for n in w["inputs"]]
    for method, f in (("torch_eager", w["torch_fn"]), ("torch_compile", torch.compile(w["torch_fn"]))):
        o = f(*args)
        torch.cuda.synchronize()
        rmse, mx, rel, bad = check(o)
        ms = bench(lambda: f(*args))
        rows.append(dict(workload=name, method=method, ms=f"{ms:.5f}", ms_screen="", kernels="", rmse=f"{rmse:.3e}",
                         max_abs=f"{mx:.3e}",
                         rel=f"{rel:.3e}", status="WRONG" if bad or rel > 5e-2 else "ok", note=""))
        print(f"  {method:18s} {ms:10.5f} ms  rel {rel:10.3e}", flush=True)

    ok = [r for r in rows if r["method"].startswith("trinity") and r["ms"]]
    if ok:
        best = min(ok, key=lambda r: float(r["ms"]))
        tc = next(float(r["ms"]) for r in rows if r["method"] == "torch_compile")
        print(f"best correct Trinity candidate: {best['method']} {float(best['ms']):.4f} ms "
              f"(torch.compile / Trinity = {tc / float(best['ms']):.3f}); "
              f"{sum(r['status'] == 'ok' for r in rows)} correct of {sum(r['status'] != 'duplicate' for r in rows)} "
              f"distinct ({len(candidates(name))} candidates)")
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
