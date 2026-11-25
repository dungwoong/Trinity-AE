"""Baseline: a persistent Triton GEMM with TMA loads, out = x @ W^T with x (M, K) and W (N, K) (both K-major).

A fixed grid of #SMs programs walks the output tiles in one of two persistent schedules:
    chunk   each program takes a contiguous run of tiles
    stride  tile_id = pid, pid + #SMs, pid + 2 #SMs, ...   (Triton's persistent-matmul tutorial)
Tiles are visited in grouped order (GROUP_M rows of tiles at a time) for L2 reuse. Accumulation is fp32; the result is
stored in the output tensor's dtype.

CONFIGS: per shape, the schedule and (BLOCK_M, BLOCK_N, GROUP_M, num_stages, num_warps) that ran fastest in a sweep over
BLOCK_M in {64, 128}, BLOCK_N in {128, 256}, stages in {3, 4, 5}, warps in {4, 8} (BLOCK_K = 64, GROUP_M = 8) on an H100.
"""
import torch
import triton
import triton.language as tl

# (M, N, K) -> (schedule, (BLOCK_M, BLOCK_N, GROUP_M, num_stages, num_warps))
CONFIGS = {
    (2048, 2048, 2048): ("stride", (128, 256, 8, 4, 8)),
    (4096, 4096, 4096): ("chunk", (128, 256, 8, 4, 8)),
    (8192, 8192, 8192): ("stride", (128, 256, 8, 4, 8)),
    (1024, 14336, 4096): ("stride", (128, 128, 8, 5, 4)),
    (1024, 28672, 8192): ("stride", (128, 256, 8, 3, 8)),
    (1024, 16384, 2048): ("stride", (128, 256, 8, 4, 8)),
    (4096, 1024, 4096): ("stride", (128, 256, 8, 4, 8)),
    (4096, 1024, 8192): ("chunk", (128, 256, 8, 4, 8)),
    (4096, 1536, 7168): ("stride", (64, 256, 8, 5, 8)),
    (4096, 1024, 16384): ("chunk", (128, 256, 8, 4, 8)),
}


def _alloc(size, alignment, stream):   # scratch for device-side TMA descriptors
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc)


@triton.jit
def _tile(tile_id, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    group = GROUP_M * num_pid_n
    first_m = tile_id // group * GROUP_M
    size_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    return first_m + tile_id % group % size_m, tile_id % group // size_m


@triton.jit
def _do_tile(x_desc, w_desc, out_ptr, pid_m, pid_n, M, N, K,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K):
        acc = tl.dot(x_desc.load([off_m, k]), tl.permute(w_desc.load([off_n, k]), [1, 0]), acc)
    rm = off_m + tl.arange(0, BLOCK_M)
    rn = off_n + tl.arange(0, BLOCK_N)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def gemm_kernel(x_ptr, w_ptr, out_ptr, M, N, K, NUM_SMS: tl.constexpr, STRIDE: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(x_ptr, [M, K], [K, 1], [BLOCK_M, BLOCK_K])
    w_desc = tl.make_tensor_descriptor(w_ptr, [N, K], [K, 1], [BLOCK_N, BLOCK_K])
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total = num_pid_m * num_pid_n
    if STRIDE:
        for tile_id in tl.range(tl.program_id(0), total, NUM_SMS):
            pid_m, pid_n = _tile(tile_id, num_pid_m, num_pid_n, GROUP_M)
            _do_tile(x_desc, w_desc, out_ptr, pid_m, pid_n, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    else:
        per = tl.cdiv(total, NUM_SMS)
        start = tl.program_id(0) * per
        for tile_id in tl.range(start, tl.minimum(start + per, total)):
            pid_m, pid_n = _tile(tile_id, num_pid_m, num_pid_n, GROUP_M)
            _do_tile(x_desc, w_desc, out_ptr, pid_m, pid_n, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)


SMS = torch.cuda.get_device_properties(0).multi_processor_count


def gemm(x, W, out, schedule, cfg):
    """out = x @ W^T. x (M, K), W (N, K), out (M, N), all contiguous; cfg = (BM, BN, GROUP_M, stages, warps)."""
    bm, bn, gm, stages, warps = cfg
    (M, K), N = x.shape, W.shape[0]
    grid = min(SMS, triton.cdiv(M, bm) * triton.cdiv(N, bn))
    gemm_kernel[(grid,)](x, W, out, M, N, K, grid, schedule == "stride", BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=64,
                         GROUP_M=gm, num_warps=warps, num_stages=stages)
    return out
