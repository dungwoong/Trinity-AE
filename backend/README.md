# Trinity Backend

> Triton Code Generation and Evaluation

## Overview

Trinity Backend converts optimized IR expressions from the optimizer into executable Triton GPU kernels. It also provides evaluation scripts for benchmarking generated kernels against baselines.

## Directory Structure

```
backend/
├── codegen/                # Triton code generation
│   ├── TritonGen.py        # Main Triton code generator
│   ├── IrParser.py         # IR expression parser
│   ├── AstNode.py          # AST node definitions
│   └── NodeType.py         # Node type enums
├── evaluation/             # Evaluation configurations
│   ├── vanilla/            # Vanilla attention configs
│   ├── prenorm/            # Pre-normalization configs
│   ├── qknorm/             # QK normalization configs
│   ├── keyformer/          # Keyformer configs
│   ├── roco/               # RoCo configs
│   └── ffn/                # FFN configs
├── scripts/                # Evaluation scripts
│   ├── evaluate_all.sh     # Run all benchmarks
│   └── evaluate_figure56.sh
├── results/                # Generated benchmark results
├── run_eval.py             # Main evaluation entry point
├── baselines.py            # Baseline implementations
└── format.py               # Output formatting utilities
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

### Convert IR to Triton and Run Benchmark

```bash
# Basic format
python run_eval.py --o {option} --m {model} --t {method} --n {case_number} --baseline {baseline} --print_output

# Options:
#   --o 0: Convert IR to Triton only
#   --o 1: Run benchmark only (requires pre-generated Triton code)
#   --o 2: Convert and run benchmark

# Models: llama, falcon
# Methods: vanilla, prenorm, qknorm, keyformer, roco, ffn
# Baseline: tensorrt, pytorch(eager), torch inductor(max-autotune-no-cudagraphs), flashinfer, flashtensor
# --print_output: If specified, prints the kernel output values (default: off)
```

### Examples

```bash
# Convert and benchmark llama vanilla attention (case 946)
python run_eval.py --o 2 --m llama --t vanilla --n 946

# Convert and benchmark falcon ffn (case 2248)
python run_eval.py --o 2 --m falcon --t ffn --n 2248

# Only convert IR to Triton code
python run_eval.py --o 0 --m llama --t prenorm --n 579

# Run benchmark with baseline comparison (torch inductor)
python run_eval.py --o 2 --m llama --t vanilla --n 946 --baseline inductor
```

### Run for figure 4

```bash
# Run all benchmarks for specific GPU (5090, A100, H100)
./scripts/evaluate_all.sh 5090
```

### Run for figure5&6
```bash
./scripts/evaluate_figure56.sh
```

## Input/Output

### Input
- IR expression file: `results/{method}/{method}_{model}_case{n}.txt`

### Output
- Generated Triton code: `results/{method}/{method}_{model}_benchmark{n}.py`
- Benchmark results printed to stdout

## Model Configurations

| Model  | sequence length (M) | head dimension (D)  | hidden dimension (N)   | head number (H) |
|--------|----|----|------|-----|
| LLaMA  | 16 | 128 | 4096 | 32  |
| Falcon | 16 | 64  | 4544 | 71  |
