# Trinity Backend

> Triton Code Generation and Evaluation

## Overview

Trinity Backend converts optimized IR expressions from the optimizer into executable Triton GPU kernels. It also provides evaluation scripts for benchmarking generated kernels against baselines.

## Directory Structure

```
backend/
├── codegen/                    # Triton code generation
│   ├── __init__.py             # Exports TritonCodeGen, convert_ir_to_triton
│   ├── convert_module.py       # Public API: convert_ir_to_triton()
│   ├── TritonGen.py            # Mediator assembling generator components
│   ├── IrParser.py             # IR expression parser
│   ├── AstNode.py              # AST node definitions
│   ├── NodeType.py             # Node type enums
│   └── triton_generator/       # Modular code generation components
│       ├── state.py            # Shared mutable state (CodeGenState)
│       ├── shape_utils.py      # Shape/constant resolution
│       ├── kernel.py           # Kernel signature & autotune
│       ├── analysis/           # AST analysis (tensors, deps, accumulators)
│       ├── codegen/            # Triton code emission (loops, memory, math, etc.)
│       └── pipeline/           # Generation pipeline (single/sequential kernels)
├── evaluation/                 # IR list & profiled results
├── figure67/                   # Data for figure6 & 7
├── scripts/                    # Evaluation scripts
│   ├── evaluate_all.sh         # Run all benchmarks
│   └── evaluate67.sh
├── profile/                    # Profile IR list & find optimal
│   ├── benchmark.py            # Unified benchmark framework
│   └── shapes/                 # Tensor shape configs (JSON per model/method)
├── results/                    # Generated benchmark results
├── run_eval.py                 # Main evaluation entry point
├── baselines.py                # Baseline implementations
└── format.py                   # Output formatting utilities
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.8+
- PyTorch 2.8+
- Triton 3.4+
- CUDA 12.x

## Usage

### Run for figure 4

```bash
# Run all benchmarks
# GPU: 5090, A100, H100
./scripts/evaluate_all.sh {GPU}
```

### Run for figure6&7
```bash
# Run all benchmarks
./scripts/evaluate67.sh
# Run single test
python run_figures67.py [options]

# Options:
#   --m        : Model type (llama, falcon)
#   --t        : Architecture (vanilla, prenorm, qknorm, keyformer, roco, ffn)
#   --n        : Case number for IR
#   --d        : CUDA device number (default: 0)
```

## IR List Profiling

Profile multiple IR expressions from a file to find the best performing kernels. Pre-generated IR expression files from the optimizer are available in `evaluation/{architecture}/` for each architecture and model. Tensor shapes for each model/method configuration are defined in `profile/shapes/{method}_{model}.json`.

### Usage
```bash
python -m profile.benchmark [options]

# Options:
#   --ir      : Path to the IR expressions file
#   --shapes  : Path to shapes.json for tensor shapes (required)
#   --start   : Start from specific test case ID (default: 0)
#   --num     : Number of expressions to benchmark (default: 10)
#   --end     : Run from start ID to the last test case
#   --output  : Path to save benchmark results (default: ./profile_result/benchmark.json)
#   --topk    : Number of top kernels to report (default: 5)
#   --all     : Profiling all cases in the IR expressions file
```

### Examples
```bash
# Profile 512 IR expressions starting from ID 0
python -m profile.benchmark --ir evaluation/vanilla/vanilla_llama_cost6_kern1.txt --shapes profile/shapes/vanilla_llama.json --num 512

# Profile from ID 500 to the end
python -m profile.benchmark --ir evaluation/ffn/falcon_ffn_cost6_kern5_wo_scheduler2.txt --shapes profile/shapes/ffn_falcon.json --start 500 --end

# Profile all IR expressions
# Please use this command to find the best IR
python -m profile.benchmark --ir evaluation/prenorm/falcon_ffn_cost6_kern5_wo_scheduler2.txt --shapes profile/shapes/prenorm_llama.json --all
```

- You can check top-k kernel result in the output JSON file (default: `./profile_result/benchmark.json`).

### Convert IR to Triton and Run Benchmark

```bash
python run_eval.py [options]

# Options:
#   --o        : 0 = Convert IR to Triton only
#                1 = Run benchmark only (requires pre-generated Triton code)
#                2 = Convert and run benchmark
#   --m        : Model type (llama, falcon)
#   --t        : Architecture (vanilla, prenorm, qknorm, keyformer, roco, ffn)
#   --n        : Case number for IR
#   --baseline : Baselines to compare (trinity, tensorrt, pytorch, inductor, flashinfer, flashtensor)
#                If not specified, all baselines are run by default
#   --d        : CUDA device number (default: 0)
#   --print_output : Print kernel output values (default: off)
```

### Examples

```bash
# Convert and benchmark llama vanilla attention with all baselines
python run_eval.py --o 2 --m llama --t vanilla --n 946

# Convert and benchmark falcon ffn (case 2248)
python run_eval.py --o 2 --m falcon --t ffn --n 2248

# Only convert IR to Triton code
python run_eval.py --o 0 --m llama --t prenorm --n 579

# Run benchmark with baseline comparison (torch inductor)
python run_eval.py --o 2 --m llama --t vanilla --n 946 --baseline inductor
```
### Input & Output
- Input IR expression file: `results/{method}/{method}_{model}_case{n}.txt`
- Output generated Triton code: `results/{method}/{method}_{model}_benchmark{n}.py`
- Benchmark results will print to stdout

## Model Configurations

| Model  | sequence length (M) | head dimension (D)  | hidden dimension (N)   | head number (H) |
|--------|----|----|------|-----|
| LLaMA  | 16 | 128 | 4096 | 32  |
| Falcon | 16 | 64  | 4544 | 71  |

You can modify tensor sizes by editing `model_configs.json`.

## Appendix

### IR Formatting Helper

`format.py` is a helper script that formats LISP-style IR expressions for better readability.

```bash
python format.py [options]

# Options:
#   --n : Case number to format
#   --m : Model type (llama, falcon)
#   --t : Architecture type (vanilla, prenorm, qknorm, keyformer, roco, ffn)
```

#### Example
```bash
# Format llama vanilla IR case 946
python format.py --n 946 --m llama --t vanilla
```

The formatted output will overwrite the original file at `results/{t}/{t}_{m}_case{n}.txt`.
