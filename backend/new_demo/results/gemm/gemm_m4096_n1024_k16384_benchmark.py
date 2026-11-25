import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128})
    ], key=[]
)
@triton.jit
def kernel_0(
    O_ptr,
    O_stride0: tl.constexpr,
    O_stride1: tl.constexpr,
    W_ptr,
    W_stride0: tl.constexpr,
    W_stride1: tl.constexpr,
    X_ptr,
    X_stride0: tl.constexpr,
    X_stride1: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    # Initialize kernel accumulators
    O = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    # Parallel loop m from 0 to O_dim0 with tile size BLOCK_M
    # Executed across grid dimension 0
    m = 0 + tl.program_id(0) * BLOCK_M
    
    # Parallel loop n from 0 to O_dim1 with tile size BLOCK_N
    # Executed across grid dimension 1
    n = 0 + tl.program_id(1) * BLOCK_N
    
    # Sequential loop k from 0 to 16384 with tile size BLOCK_K
    for k in range(0, 16384, BLOCK_K):
        offset_0 = (m + tl.arange(0, BLOCK_M))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        m_indices = m + tl.arange(0, BLOCK_M)
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_0 = (m_indices < M)[:, None] & (k_indices < K)[None, :]
        temp_0 = tl.load(X_ptr + offset_0, mask=mask_0, other=0.0)
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * W_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * W_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_1 = (k_indices < K)[:, None] & (n_indices < N)[None, :]
        temp_1 = tl.load(W_ptr + offset_1, mask=mask_1, other=0.0)
        O = (tl.dot(temp_0, temp_1).to(tl.float16) + (1 * O).to(tl.float16)).to(tl.float16)
    # Store kernel accumulators
    offset_2 = (m + tl.arange(0, BLOCK_M))[:, None] * O_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * O_stride1
    m_indices = m + tl.arange(0, BLOCK_M)
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_2 = (m_indices < M)[:, None] & (n_indices < N)[None, :]
    tl.store(O_ptr + offset_2, O, mask=mask_2)


# Metadata for benchmark.py
TENSOR_PARAMS = ['O', 'W', 'X']
FP32_TENSOR_PARAMS = []
BLOCK_PARAMS = ['block_k', 'block_m', 'block_n']

def forward(O, W, X, block_k=16, block_m=16, block_n=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[lambda meta: ((4096 - 0 + meta["BLOCK_M"] - 1) // meta["BLOCK_M"], (1024 - 0 + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],)](
        O,
        O.stride(0),
        O.stride(1),
        W,
        W.stride(0),
        W.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        # BLOCK_K, BLOCK_M, BLOCK_N are provided by autotune,
        # BLOCK_M is automatically set by autotune,
        # BLOCK_N is automatically set by autotune,
        # BLOCK_K is automatically set by autotune,
        K=16384,
        M=4096,
        N=1024
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
