# Trinity Project \[ASPLOS 2026\] (To appear)

> Three-Dimensional Tensor Program Optimization via Tile-level Equality Saturation

Trinity is the first tensor program optimizer that achieves scalable joint optimization through tile-level equality saturation. Trinity’s IR can capture the essence of all three optimization axes (algebraic equivalence, memory I/O, compute orchestration). By leveraging equality saturation, Trinity enables scalable joint optimization across the entire graph. 

## Prerequisites

### optimizer

```bash
sudo apt install build-essential \
    clang \
    libclang-dev \
    llvm-dev \
    libz3-dev \
    pkg-config
```
