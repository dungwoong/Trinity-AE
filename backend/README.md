# Backend

Triton code generation and kernel benchmarking.

## Profile

Benchmarking tool for Triton kernels.

### Usage

```bash
python -m backend.profile.benchmark \
  --shapes <path/to/shapes.json> \
  --ir <path/to/ir.txt>
```

### Options

| Option | Description |
|--------|-------------|
| `--shapes` | Path to shapes.json (required) |
| `--ir` | Path to IR expressions file |
| `--output` | Path to save benchmark results |
| `--start` | Start from test case ID |
| `--num` | Number of expressions to benchmark |
| `--end` | Run from start ID to the last test case |
| `--topk` | Number of top kernels to report |
| `--all` | Run all configurations |

### Examples

```bash
# Vanilla Attention (LLaMA)
python -m backend.profile.benchmark \
  --shapes backend/profile/shapes/vanilla_llama.json \
  --ir backend/evaluation/vanilla/vanilla_llama_cost6_kern1.txt

# Keyformer (Falcon)
python -m backend.profile.benchmark \
  --shapes backend/profile/shapes/keyformer_falcon.json \
  --ir backend/evaluation/keyformer/keyformer_falcon_cost6_kern1.txt

# FFN (LLaMA)
python -m backend.profile.benchmark \
  --shapes backend/profile/shapes/ffn_llama.json \
  --ir backend/evaluation/ffn/llama_ffn_cost6_kern5_wo_scheduler2.txt
```

## shapes.json Format

Both formats are supported:

```json
// Flat format (frontend-generated)
{
  "X": {"shape": [16, 4096], "type": "input"},
  "O2": {"shape": [16, 4096], "type": "output"}
}

// Nested format (backend predefined)
{
  "config": {"M": 16, "N": 4096, "D": 128, "H": 32, "P": 1024},
  "tensors": {
    "X": {"shape": [16, 4096], "type": "input"},
    "O2": {"shape": [16, 4096], "type": "output"}
  }
}
```

### Predefined Shapes (profile/shapes/)

| File | Model | Description |
|------|-------|-------------|
| `vanilla_llama.json` | LLaMA 7B | Vanilla Attention |
| `vanilla_falcon.json` | Falcon 7B | Vanilla Attention |
| `keyformer_llama.json` | LLaMA 7B | Keyformer |
| `keyformer_falcon.json` | Falcon 7B | Keyformer |
| `roco_llama.json` | LLaMA 7B | RoCo |
| `roco_falcon.json` | Falcon 7B | RoCo |
| `qknorm_llama.json` | LLaMA 7B | QK Normalization |
| `qknorm_falcon.json` | Falcon 7B | QK Normalization |
| `prenorm_llama.json` | LLaMA 7B | Pre-normalization |
| `prenorm_falcon.json` | Falcon 7B | Pre-normalization |
| `ffn_llama.json` | LLaMA 7B | FFN Layer |
| `ffn_falcon.json` | Falcon 7B | FFN Layer |

### Tensor Types

- `input`: Input tensors (random initialization)
- `intermediate`: Intermediate tensors (zero initialization)
- `output`: Output tensors (zero initialization)
