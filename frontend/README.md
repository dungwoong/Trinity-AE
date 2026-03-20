# TrinityFE-TVM

Experimental tooling built on **TVM Unity (Relax)** for model lowering,
IR transformation, and execution, focused on transformer-style models.


## Environment
- Python: **3.11**
- Conda environment name: `tvm`
- TVM version: **TVM Unity (Relax, MLC AI nightly)**
- PyTorch: **installed via pip in the conda env**


## Setup
### 1. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate trinity
```

If the environment already exists:
```bash
conda activate trinity
```

### 2. Verify installation
```bash
python - <<'PY'
import tvm
from tvm import relax

print("TVM version:", tvm.__version__)
PY
```


## Running Experiments
### Run all built-in models
```bash
./run_all.sh
```
This script runs all built-in models and generates artifacts under
the `outputs/` directory, which is git-ignored.

### Run a specific built-in model
```bash
python -m model.{model_name}
```
Example:
```bash
python -m model.falcon7b
```

### Run any model module (CLI)
Use the CLI when you have a custom model module that provides
`build_model_and_inputs()`.
```bash
python cli.py --module path.to.your_model
```

## CLI Export

The CLI provides a uniform way to export any model that exposes a factory
function and supports per-run options. The default factory name is
`build_model_and_inputs`, but you can point to a different function with
`--factory`.

### Factory contract

Each model module should define `build_model_and_inputs()` and return a dict.
Example:

```python
def build_model_and_inputs():
    device = torch.device("cpu")
    dtype = torch.float32

    model = MyModel().to(device=device, dtype=dtype)
    x = torch.rand((1, 16, 64), device=device, dtype=dtype)

    return {
        "model": model,
        "example_inputs": x,
        "inline_shape_op": True,
        "inline_elementwise_op": True,
        "remove_short_loop_threshold": 16,
        "decompose_nested_op_ratio": 0.3,
        # "basename": "my_model",
        # "context": "my_model",
    }
```

Required keys

- `model`: The PyTorch module instance
- `example_inputs`: Inputs for `to_relax`/`to_tir`

Optional keys

- `inline_shape_op`: `bool`, default `True`
- `inline_elementwise_op`: `bool`, default `True`
- `remove_short_loop_threshold`: `int`, default `64`
- `decompose_nested_op_ratio`: `float`, default `0.3`
- `basename`: Output basename
- `context`: Validation context name

### Basic usage

```bash
python cli.py --module model.DecAttn
```

### With overrides

```bash
python cli.py \
  --module model.DecAttn \
  --output-dir ./outputs \
  --remove-short-loop-threshold 64 \
  --decompose-nested-op-ratio 0.3
```

### Disable inlining

```bash
python cli.py \
  --module model.DecAttn \
  --no-inline-shape \
  --no-inline-elementwise
```

### Output artifacts

The CLI writes artifacts under `outputs/` by default:

- `outputs/tvm/{basename}_tir.py`
- `outputs/trinity/{basename}/main.txt`
- `outputs/trinity/{basename}/ir.txt`
- `outputs/trinity/{basename}/shapes.json`

### Notes

- `--basename` overrides the inferred name from the module.
- `--context` overrides the validation context label.
- If a factory returns `basename`/`context`, CLI args take precedence.


## Project Structure
```graphql
.
├── core/          # Core logic and passes
├── ir/            # IR utilities and transformations
├── model/         # Model entry points
├── utils/         # Shared helper utilities
├── outputs/       # Generated artifacts (git-ignored)
├── run_all.sh
├── environment.yml
└── README.md
```


## Notes
- Targets TVM Unity (Relax), not legacy Relay.
- Generated outputs are excluded from version control by design.
- CUDA support is optional and configured per machine.


## References
TVM Unity Installation Guide
https://llm.mlc.ai/docs/install/tvm.html

TVM Relax (MLC AI)
https://github.com/mlc-ai/relax
