# Triton Generator Architecture

## Goal

`triton_generator` is organized around the current implementation reality:

- It generates Triton code strings directly from the AST.
- It does not have a separate lowered IR.
- Components collaborate through a shared `CodeGenState` and explicit cross-component calls via `self.gen`.

Because of that, the package is split by responsibility, not by an idealized
compiler pipeline that does not exist in this codebase. Thin package
initializers assemble focused helper modules into public components.

## Top-Level Layout

- `state.py`
  - Generator state initialization, counters, caches, debug state.
- `shape_utils.py`
  - Shape resolution, constant resolution, padding helpers.
- `analysis/`
  - Read-only AST/tensor analysis and planning.
  - `tensor_usage.py`
  - `dependencies.py`
  - `accumulators.py`
  - `allocations.py`
  - `__init__.py` assembles the public `Analyzer`.
- `codegen/`
  - Direct Triton string generation from AST nodes.
- `kernel.py`
  - Kernel signature and autotune helpers.
- `pipeline/`
  - High-level orchestration for single-kernel, seq-kernel, and wrapper generation.
  - `entrypoint.py`
  - `single_kernel.py`
  - `seq_kernels.py`
  - `wrapper.py`
  - `__init__.py` assembles the public `Pipeline`.

## Boundary Rules

### `analysis/`

Put code here when it answers questions such as:

- Which tensors are used?
- Which tensors cross kernel or loop boundaries?
- Which tensors are accumulators?
- What allocations or memory behavior are required?

`analysis/` should avoid directly emitting Triton code except for planning
snippets tightly coupled to analysis results, such as accumulator/allocation
initialization helpers.

### `codegen/`

Put code here when it directly emits Triton code strings from AST nodes or
generator state.

Examples:

- dispatch
- loops
- indexing
- memory ops
- math ops
- matmul ops
- transforms
- masking

`codegen/` is intentionally a single stage because this codebase does not have
a real `lowering -> emission` split today.

### `kernel.py`

Keep only kernel-level Triton concerns here:

- kernel signature/header
- autotune decorator generation

Do not place wrapper orchestration here.

### `pipeline/`

Put code here when it coordinates larger generation flows:

- entrypoint `generate()`
- single-kernel generation
- seq-kernel generation
- wrapper generation

`pipeline/` owns orchestration, not low-level AST emission details.

## Design Principles

1. Prefer direct names over compiler-theory names when the implementation is direct string emission.
2. Keep helper files small enough that a maintainer can understand one file in one sitting.
3. Shared state and shape helpers should stay out of `analysis/` and `codegen/` when possible.
4. New helpers should be placed by responsibility, not by caller convenience.
5. If a helper is not referenced internally and has no intended external use, remove it.
6. Keep the generated Triton code stable for representative benchmark cases.
7. Keep package `__init__.py` files thin: exports and component assembly only.

## Validation Rule

After structural changes, at minimum:

- import `backend.codegen.convert_module.convert_ir_to_triton`
- generate Triton for representative IR cases from `backend/results/`
- run at least one end-to-end caller such as `backend/run_eval.py`
