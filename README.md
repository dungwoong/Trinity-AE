# Trinity Project \[ASPLOS 2026\] (To appear)

> Three-Dimensional Tensor Program Optimization via Tile-level Equality Saturation

Trinity is the first tensor program optimizer that achieves scalable joint optimization through tile-level equality saturation. Trinity’s IR can capture the essence of all three optimization axes (algebraic equivalence, memory I/O, compute orchestration). By leveraging equality saturation, Trinity enables scalable joint optimization across the entire graph. 

## Prerequisites

### frontend

```bash
cd frontend
conda env create -f environment.yml
conda activate trinity
cd ..
```

### backend

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### optimizer

```bash
sudo apt install build-essential \
    clang \
    libclang-dev \
    llvm-dev \
    libz3-dev \
    pkg-config
```

## How to use

## Quickstart

Create a demo script under `scripts/`, for example `scripts/DecAttn.py` or `scripts/roco.py`. In that script, define your PyTorch module and call `trinity.optimize(...)` with example inputs. Trinity converts the module into Trinity IR and exports the lowered IR and related artifacts.

You can run the demo script as:

```bash
python scripts/DecAttn.py
```

> Note: For attention-layer workloads, the optimizer typically spends about 10-15 minutes in equality saturation.

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "frontend"))

import torch
import torch.nn as nn
import trinity


class RocoAttn(nn.Module):
    def __init__(self, M, H, D, P, cache_K, cache_V, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.H = H
        self.D = D
        self.P = P
        self.N = H * D
        self.device = device
        self.dtype = dtype

        self.q_proj = nn.Linear(self.N, self.N, bias=False)
        self.k_proj = nn.Linear(self.N, self.N, bias=False)
        self.v_proj = nn.Linear(self.N, self.N, bias=False)

        self.register_buffer("cache_K", cache_K.to(device))
        self.register_buffer("cache_V", cache_V.to(device))

    def forward(self, X):
        q1 = self.q_proj(X)
        k1 = self.k_proj(X)
        v1 = self.v_proj(X)

        q2 = q1.view(self.M, self.H, self.D)
        k2 = k1.view(self.M, self.H, self.D)
        v2 = v1.view(self.M, self.H, self.D)

        q = q2.permute(1, 0, 2)
        k = k2.permute(1, 0, 2)
        v = v2.permute(1, 0, 2)

        end = self.cache_K.size(1)
        start = end - k.size(1)
        self.cache_K[:, start:end, :] = k
        self.cache_V[:, start:end, :] = v

        c = torch.matmul(q, self.cache_K.permute(0, 2, 1))
        c_exp = torch.exp(c)
        c_sum = c_exp.sum(dim=2)
        c_div = c_exp / c_sum.unsqueeze(-1)

        o = torch.matmul(c_div, self.cache_V)
        o1 = o.permute(1, 0, 2)
        o2 = o1.contiguous().view(self.M, self.N)
        return o2


if __name__ == "__main__":
    M, H, D, P = 16, 32, 128, 528
    N = H * D

    X = torch.randn((M, N))
    K_cache = torch.randn((H, P, D))
    V_cache = torch.randn((H, P, D))

    model = RocoAttn(M, H, D, P, K_cache, V_cache)
    result = trinity.optimize(model, X, basename="roco")
```

## What `trinity.optimize(...)` Does

`trinity.optimize(...)` takes a PyTorch `nn.Module` and example inputs, lowers the model into Trinity IR, applies frontend optimization passes, and exports the generated artifacts.

With `basename="roco"`, Trinity exports files such as:

- `output_dir/tvm/roco_tir.py`
- `output_dir/trinity/roco/main.txt`
- `output_dir/trinity/roco/ir.txt`
- `output_dir/trinity/roco/shapes.json`

## Model Requirements

- The input model must be a valid PyTorch `nn.Module`.
- `forward(...)` must execute successfully with the provided example inputs.
- Trinity supports common tensor operations used in attention-style workloads, such as reshape, permute, matmul, reductions, and buffer updates.
- `torch.topk` and `torch.split` are currently not supported.
