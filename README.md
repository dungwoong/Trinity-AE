# Trinity Project \[ASPLOS 2026\] (To appear)

> Three-Dimensional Tensor Program Optimization via Tile-level Equality Saturation

Trinity is the first tensor program optimizer that achieves scalable joint optimization through tile-level equality saturation. Trinity's IR can capture the essence of all three optimization axes (algebraic equivalence, memory I/O, compute orchestration). By leveraging equality saturation, Trinity enables scalable joint optimization across the entire graph.

## Setup

Run the repository setup script from the project root:

```bash
chmod +x setup.sh
./setup.sh
```

What it does:
- creates the `trinity` conda environment from `frontend/environment.yml` if it does not already exist
- installs backend Python dependencies from `backend/requirements.txt`
- installs optimizer system dependencies with `apt`

The script uses `sudo` for the optimizer packages, so it may prompt for your password.

## Project Structure

```
Trinity-AE/
├── frontend/       # PyTorch model → Trinity IR conversion
├── optimizer/      # Tile-level equality saturation (Rust)
├── backend/        # IR → Triton kernel generation & profiling
├── trinity/        # End-to-end pipeline automation
└── scripts/        # Model definitions & run example pipeline
```

For detailed documentation, see the README in each directory:
- [`scripts/`](scripts/README.md) — Usage guide, API reference, and config options
- [`frontend/`](frontend/README.md) — IR conversion details
- [`optimizer/`](optimizer/README.md) — Equality saturation engine
- [`backend/`](backend/README.md) — Triton code generation and profiling
