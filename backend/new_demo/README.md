# new_demo: Trinity on GEMM-based fusion workloads

Runs Trinity's backend on new workloads and compares it with cuBLAS and a persistent Triton kernel, using Trinity's own
evaluation flow (`backend/run_eval.py`, `backend/run_eval_do_bench.py`). Everything needed is in this folder.

```
new_demo/
├── cases/               Trinity IR (input to Trinity's code generator)
│   ├── gemm.txt                         the GEMM, written by hand
│   └── <workload>_candidates.txt        Trinity's optimizer output for rmsnorm_linear / swiglu / lora
├── run_gemm.py          GEMM: generates Trinity's kernel, runs it and the baselines, checks and times them
├── run_fusion.py        RMSNorm + Linear / SwiGLU / LoRA: runs every candidate Trinity's optimizer produced
├── triton_gemm.py       baseline: persistent Triton GEMM (TMA loads) with its tuned config per shape
└── results/<workload>/  written by the run scripts (only gemm/ is checked in)
    └── gemm/
        ├── gemm_m<M>_n<N>_k<K>_benchmark.py   Trinity's generated Triton code (convert_ir_to_triton output, unedited)
        ├── gemm.csv                            one row per shape x method
        └── gemm.log
```

## GEMM (`run_gemm.py`)

`O = X @ W` at the 10 shapes in `triton_gemm.CONFIGS`.

### How Trinity is run

| step | here | what `run_eval.py` does |
|---|---|---|
| 1. IR | `cases/gemm.txt`, with symbolic bounds `M`, `N`, `K` | reads `results/{target}/{target}_{model}_case{n}.txt` |
| 2. codegen | `convert_ir_to_triton(ir, tensor_shapes, constants)`; shapes named symbolically, sizes in `constants`; writes `results/gemm/gemm_m<M>_n<N>_k<K>_benchmark.py` | same call, writes `results/{target}/{target}_{model}_benchmark{n}.py` |
| 3. load | importlib; `forward`, `TENSOR_PARAMS`, `BLOCK_PARAMS` read from the module; args built by name, `0` for each block param | same |
| 4. time | `triton.testing.do_bench(lambda: forward(*args))`, then `time.sleep(2)` | `run_eval_do_bench.py` |
| dtype | float16 | `dtype = torch.float16` |

The IR is written by hand because a single matmul gives Trinity's optimizer nothing to rewrite:

```
(ploop 0 M tile_m m
  (ploop 0 N tile_n n
    (sloop 0 K tile_k k
      (store (output O) (+ (@ (load (input X) (index (tile m) (tile k)))
                              (load (input W) (index (tile k) (tile n))))
                           (* 1 (load (output O) (index (tile m) (tile n)))))
             (index (tile m) (tile n))))))
```

The matmul has the same form as the ones in Trinity's extracted expressions (e.g. `results/ffn/ffn_llama_case2248.txt`).
The code generator keeps `O` in registers through the k loop and stores it once after it. The generated kernel carries
Trinity's own `@triton.autotune` over 6 tile configs; its first call autotunes.

### Baselines (same inputs, same timer)

- `cublas`: `torch.mm(X, W)`.
- `triton_persistent`: `triton_gemm.py`, a persistent Triton GEMM with TMA loads, at its tuned schedule and config for
  that shape. It takes `W` K-major, as `(N, K)`.

Each output is checked against an fp64 GEMM of the same fp16 inputs (rmse, max abs).

### Run

From `Trinity-AE/backend`, on a GPU node:

```bash
python new_demo/run_gemm.py --csv=new_demo/results/gemm/gemm.csv     # or: python new_demo/run_gemm.py 1 6   (shape indices)
```

### Result (H100, fp16, `results/gemm/gemm.csv`)

Trinity's generated GEMM is 1.41–1.81× slower than the persistent Triton GEMM, which is within a few percent of cuBLAS.
Its error is 3–8× higher (rmse 6.6e-4 to 1.7e-3 vs 2.1e-4) and grows with K. In the generated kernel
(`results/gemm/*_benchmark.py`):

- the accumulator is fp16, rounded after every k step: `O = (tl.dot(a, b).to(fp16) + (1 * O).to(fp16)).to(fp16)`, so the
  dot cannot accumulate in place, and rounding error grows with the number of k steps;
- one program per output tile, pointer loads (no TMA), Triton's default 4 warps / 3 stages; the autotuner picks
  128 x 128 x 128 at every shape.

### Reference code
`run_eval_do_bench.py` line 244 onwards, they open the case file, run `convert_ir_to_triton` and write to an output file. They then use importlib to load it in. We're following that. The GEMM example is correct, at least.

## Fusion: RMSNorm + Linear, SwiGLU, LoRA (`run_fusion.py`)

One shape each: RMSNorm + Linear M=4096 N=1024 K=4096 (no eps, no weight); SwiGLU M=4096 N=14336 K=4096;
LoRA M=N=K=4096, rank 16. fp16.

1. Optimizer: `optimizer/tests/new_demo.rs` writes each unfused program in Trinity's IR (as `tests/ffn.rs`, but with
   M tiled) and runs the same calls as `tests/ffn.rs`: `run_until_saturated(expr, rules(), 10)`,
   `list_expressions_with_target_cost_v3_part1(.., 6, 5)`, `list_expressions_from_semi_with_cost`, `postprocess_v2`.
   Output: `cases/<workload>_candidates.txt`.
       cd optimizer && cargo test --release --test new_demo -- --nocapture --test-threads=1
2. `run_fusion.py <workload>`: as `profile/benchmark.py`, skips candidates containing `dummydata`, generates each with
   `convert_ir_to_triton(ir, shapes, None)`, inputs `randn.clamp(-1, 1) * 0.01`, outputs / intermediates zeroed (fp32
   for `FP32_TENSOR_PARAMS`). Added: candidates with identical generated code run once; each output is checked against
   an fp64 reference (WRONG if max abs error > 5% of max |ref|); timing with `do_bench` (screen all, then the 5 fastest
   correct ones again with `do_bench` + `sleep(2)`). Baselines: PyTorch eager and torch.compile of the unfused program.

### Result (2026-11-25; `python new_demo/run_fusion.py <workload> --csv=...` regenerates the per-candidate csv)

| workload | candidates run | correct | wrong output | fail to compile | fastest correct (kernels) | torch.compile | Trinity speed vs torch.compile |
|---|---|---|---|---|---|---|---|
| RMSNorm + Linear | 87 | 18 | 52 | 16 | cand244, 3 kernels, 0.115 ms | 0.074 ms | 0.65× |
| SwiGLU | 12 | 12 | 0 | 0 | cand8, 3 kernels, 2.36 ms | 1.47 ms | 0.62× |
| LoRA | 12 | 6 | 6 | 0 | cand12, 3 kernels, 0.344 ms | 0.258 ms | 0.75× |

In all three, the fastest correct candidate is not fused: three kernels, as in the input (for RMSNorm + Linear, with the
division moved after the GEMM into a third kernel). Fused candidates make the loop over output columns (n) sequential
inside each m program, so only M / tile_m programs run; no candidate keeps (m, n) parallel with the two GEMMs (SwiGLU)
or the sum of squares and the GEMM (RMSNorm + Linear) in one k loop. Wrong candidates accumulate a row reduction (S, T)
inside the sequential n loop without resetting it.

For comparison, Trinity's published LLaMA FFN kernel (`results/ffn/ffn_llama_benchmark2248.py`, `kernel_1`, decode with
M = 16 as one full tile) does fuse RMSNorm + SwiGLU this way: one program per output-column tile, the sum of squares
recomputed in each tile inside the same k loop as both GEMMs (which share the x tile), the division applied to the
accumulators after the loop, and silu * mul before a single store. With M tiled, the fused candidates above make n
sequential inside m instead.

