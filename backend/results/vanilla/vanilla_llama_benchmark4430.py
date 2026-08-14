import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128})
    ], key=[]
)
@triton.jit
def kernel_0(
    K1_ptr,
    K1_stride0: tl.constexpr,
    K1_stride1: tl.constexpr,
    Q1_ptr,
    Q1_stride0: tl.constexpr,
    Q1_stride1: tl.constexpr,
    V1_ptr,
    V1_stride0: tl.constexpr,
    V1_stride1: tl.constexpr,
    WK_ptr,
    WK_stride0: tl.constexpr,
    WK_stride1: tl.constexpr,
    WQ_ptr,
    WQ_stride0: tl.constexpr,
    WQ_stride1: tl.constexpr,
    WV_ptr,
    WV_stride0: tl.constexpr,
    WV_stride1: tl.constexpr,
    X_ptr,
    X_stride0: tl.constexpr,
    X_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    N4: tl.constexpr,
    P: tl.constexpr
):
    # Initialize kernel accumulators
    K1 = tl.zeros((16, BLOCK_N), dtype=tl.float16)
    Q1 = tl.zeros((16, BLOCK_N), dtype=tl.float16)
    V1 = tl.zeros((16, BLOCK_N), dtype=tl.float16)
    # Parallel loop n from 0 to Q1_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop k from 0 to 4096 with tile size BLOCK_K
    for k in range(0, 4096, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_0 = (k_indices < N)[None, :]
        temp_0 = tl.load(X_ptr + offset_0, mask=mask_0, other=0.0)
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * WQ_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WQ_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_1 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_1 = tl.load(WQ_ptr + offset_1, mask=mask_1, other=0.0)
        Q1 = (tl.dot(temp_0, temp_1).to(tl.float16) + (1 * Q1).to(tl.float16)).to(tl.float16)
        offset_2 = (k + tl.arange(0, BLOCK_K))[:, None] * WK_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WK_stride1
        mask_2 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_2 = tl.load(WK_ptr + offset_2, mask=mask_2, other=0.0)
        K1 = (tl.dot(temp_0, temp_2).to(tl.float16) + (1 * K1).to(tl.float16)).to(tl.float16)
        offset_3 = (k + tl.arange(0, BLOCK_K))[:, None] * WV_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WV_stride1
        mask_3 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_3 = tl.load(WV_ptr + offset_3, mask=mask_3, other=0.0)
        V1 = (tl.dot(temp_0, temp_3).to(tl.float16) + (1 * V1).to(tl.float16)).to(tl.float16)
    # Store kernel accumulators
    offset_4 = (tl.arange(0, 16))[:, None] * K1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * K1_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_4 = (n_indices < N)[None, :]
    tl.store(K1_ptr + offset_4, K1, mask=mask_4)
    offset_5 = (tl.arange(0, 16))[:, None] * Q1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * Q1_stride1
    mask_5 = (n_indices < N)[None, :]
    tl.store(Q1_ptr + offset_5, Q1, mask=mask_5)
    offset_6 = (tl.arange(0, 16))[:, None] * V1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * V1_stride1
    mask_6 = (n_indices < N)[None, :]
    tl.store(V1_ptr + offset_6, V1, mask=mask_6)



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_P': 32}),
        triton.Config({'BLOCK_P': 64}),
        triton.Config({'BLOCK_P': 128})
    ], key=[]
)
@triton.jit
def kernel_1(
    K1_ptr,
    K1_stride0: tl.constexpr,
    K1_stride1: tl.constexpr,
    K_cache_ptr,
    K_cache_stride0: tl.constexpr,
    K_cache_stride1: tl.constexpr,
    K_cache_stride2: tl.constexpr,
    O2_ptr,
    O2_stride0: tl.constexpr,
    O2_stride1: tl.constexpr,
    Q1_ptr,
    Q1_stride0: tl.constexpr,
    Q1_stride1: tl.constexpr,
    V1_ptr,
    V1_stride0: tl.constexpr,
    V1_stride1: tl.constexpr,
    V_cache_ptr,
    V_cache_stride0: tl.constexpr,
    V_cache_stride1: tl.constexpr,
    V_cache_stride2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    N4: tl.constexpr,
    P: tl.constexpr
):
    # Allocate intermediate tensors
    C_exp = tl.zeros((1, M, BLOCK_P), dtype=tl.float32)
    C_sum = tl.zeros((1, M), dtype=tl.float32)
    O = tl.zeros((1, M, D), dtype=tl.float32)

    # Parallel loop n from 0 to Q1_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    offset_0 = (tl.arange(0, 16))[:, None] * Q1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * Q1_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_7 = (n_indices < N)[None, :]
    temp_0 = tl.load(Q1_ptr + offset_0, mask=mask_7, other=0.0)
    temp_1 = tl.expand_dims(temp_0, 1)
    Q = tl.permute(temp_1, (1, 0, 2))
    offset_1 = (tl.arange(0, 16))[:, None] * K1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * K1_stride1
    mask_8 = (n_indices < N)[None, :]
    temp_2 = tl.load(K1_ptr + offset_1, mask=mask_8, other=0.0)
    temp_3 = tl.expand_dims(temp_2, 1)
    K = tl.permute(temp_3, (1, 0, 2))
    offset_2 = (tl.arange(0, 16))[:, None] * V1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * V1_stride1
    mask_9 = (n_indices < N)[None, :]
    temp_4 = tl.load(V1_ptr + offset_2, mask=mask_9, other=0.0)
    temp_5 = tl.expand_dims(temp_4, 1)
    V = tl.permute(temp_5, (1, 0, 2))
    offset_3 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * K_cache_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * K_cache_stride1 + (tl.arange(0, 128))[None, None, :] * K_cache_stride2
    tl.store(K_cache_ptr + offset_3, K)
    offset_4 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * V_cache_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * V_cache_stride1 + (tl.arange(0, 128))[None, None, :] * V_cache_stride2
    tl.store(V_cache_ptr + offset_4, V)
    # Sequential loop p from 0 to 1024 with tile size BLOCK_P
    for p in range(0, 1024, BLOCK_P):
        offset_5 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * K_cache_stride0 + (p + tl.arange(0, BLOCK_P))[None, :, None] * K_cache_stride1 + (tl.arange(0, 128))[None, None, :] * K_cache_stride2
        p_indices = p + tl.arange(0, BLOCK_P)
        mask_10 = (p_indices < P+M)[None, :, None]
        temp_6 = tl.load(K_cache_ptr + offset_5, mask=mask_10, other=0.0)
        temp_7 = tl.permute(temp_6, (0, 2, 1))
        C_exp = tl.exp(tl.dot(Q, temp_7).to(tl.float32))
        C_sum = (tl.sum(C_exp, axis=2, dtype=tl.float32) + (1 * C_sum))
        offset_6 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * V_cache_stride0 + (p + tl.arange(0, BLOCK_P))[None, :, None] * V_cache_stride1 + (tl.arange(0, 128))[None, None, :] * V_cache_stride2
        mask_11 = (p_indices < P+M)[None, :, None]
        temp_8 = tl.load(V_cache_ptr + offset_6, mask=mask_11, other=0.0)
        O = ((O * 1) + tl.dot(C_exp, temp_8.to(tl.float32)))
    # Skipped empty sloop with dummy body
    O = (O / C_sum[:, :, None])
    temp_9 = tl.permute(O, (1, 0, 2))
    offset_7 = (tl.arange(0, 16))[:, None] * O2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * O2_stride1
    mask_12 = (n_indices < N)[None, :]
    tl.store(O2_ptr + offset_7, tl.reshape(temp_9, (16, 128)).to(tl.float16), mask=mask_12)


# Metadata for benchmark.py
TENSOR_PARAMS = ['K1', 'K_cache', 'O2', 'Q1', 'V1', 'V_cache', 'WK', 'WQ', 'WV', 'X']
FP32_TENSOR_PARAMS = []
BLOCK_PARAMS = ['block_k', 'block_n', 'block_p']

def forward(K1, K_cache, O2, Q1, V1, V_cache, WK, WQ, WV, X, block_k=16, block_n=16, block_p=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[lambda meta: ((4096 - 0 + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],)](
        K1,
        K1.stride(0),
        K1.stride(1),
        Q1,
        Q1.stride(0),
        Q1.stride(1),
        V1,
        V1.stride(0),
        V1.stride(1),
        WK,
        WK.stride(0),
        WK.stride(1),
        WQ,
        WQ.stride(0),
        WQ.stride(1),
        WV,
        WV.stride(0),
        WV.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        # BLOCK_K, BLOCK_N are provided by autotune,
        # BLOCK_N is automatically set by autotune,
        # BLOCK_K is automatically set by autotune,
        D=128,
        H=32,
        M=16,
        N=4096,
        N4=16384,
        P=1008
    )

    kernel_1[((4096 - 0 + 128 - 1) // 128,)](
        K1,
        K1.stride(0),
        K1.stride(1),
        K_cache,
        K_cache.stride(0),
        K_cache.stride(1),
        K_cache.stride(2),
        O2,
        O2.stride(0),
        O2.stride(1),
        Q1,
        Q1.stride(0),
        Q1.stride(1),
        V1,
        V1.stride(0),
        V1.stride(1),
        V_cache,
        V_cache.stride(0),
        V_cache.stride(1),
        V_cache.stride(2),
        # BLOCK_P are provided by autotune,
        BLOCK_N=128,
        # BLOCK_P is automatically set by autotune,
        D=128,
        H=32,
        M=16,
        N=4096,
        N4=16384,
        P=1008
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
