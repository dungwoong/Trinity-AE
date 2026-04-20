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
    O2_ptr,
    O2_stride0: tl.constexpr,
    O2_stride1: tl.constexpr,
    WO_ptr,
    WO_stride0: tl.constexpr,
    WO_stride1: tl.constexpr,
    X_ptr,
    X_stride0: tl.constexpr,
    X_stride1: tl.constexpr,
    attn_O2_ptr,
    attn_O2_stride0: tl.constexpr,
    attn_O2_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    N4: tl.constexpr,
    P: tl.constexpr
):
    # Allocate intermediate tensors
    attn_O1 = tl.zeros((M, BLOCK_N), dtype=tl.float16)

    # Parallel loop n from 0 to attn_O1_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop k from 0 to 4096 with tile size BLOCK_K
    for k in range(0, 4096, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * O2_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * O2_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_0 = (k_indices < N)[None, :]
        temp_0 = tl.load(O2_ptr + offset_0, mask=mask_0, other=0.0)
        
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * WO_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WO_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_1 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_1 = tl.load(WO_ptr + offset_1, mask=mask_1, other=0.0)
        
        attn_O1 = (tl.dot(temp_0, temp_1).to(tl.float16) + (1 * attn_O1).to(tl.float16)).to(tl.float16)
    offset_2 = (tl.arange(0, 16))[:, None] * X_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * X_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_2 = (n_indices < N)[None, :]
    temp_2 = tl.load(X_ptr + offset_2, mask=mask_2, other=0.0)
    offset_3 = (tl.arange(0, 16))[:, None] * attn_O2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O2_stride1
    mask_3 = (n_indices < N)[None, :]
    tl.store(attn_O2_ptr + offset_3, (temp_2 + attn_O1).to(tl.float16), mask=mask_3)



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_P': 32, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_P': 32, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_P': 32, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_P': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_P': 64, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_P': 64, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_P': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_P': 128, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_P': 128, 'BLOCK_K': 128})
    ], key=[]
)
@triton.jit
def kernel_1(
    FF1_ptr,
    FF1_stride0: tl.constexpr,
    FF1_stride1: tl.constexpr,
    WFF1a_ptr,
    WFF1a_stride0: tl.constexpr,
    WFF1a_stride1: tl.constexpr,
    WFF1b_ptr,
    WFF1b_stride0: tl.constexpr,
    WFF1b_stride1: tl.constexpr,
    attn_O2_ptr,
    attn_O2_stride0: tl.constexpr,
    attn_O2_stride1: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    N4: tl.constexpr,
    P: tl.constexpr
):
    # Allocate intermediate tensors
    FF1a = tl.zeros((M, BLOCK_P), dtype=tl.float32)
    FF1b = tl.zeros((M, BLOCK_P), dtype=tl.float32)
    attn_O3 = tl.zeros((M,), dtype=tl.float16)

    # Parallel loop p from 0 to FF1a_dim1 with tile size BLOCK_P
    # Executed across grid dimension 0
    p = 0 + tl.program_id(0) * BLOCK_P
    
    # Sequential loop k from 0 to 4096 with tile size BLOCK_K
    for k in range(0, 4096, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * attn_O2_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * attn_O2_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_4 = (k_indices < N)[None, :]
        temp_0 = tl.load(attn_O2_ptr + offset_0, mask=mask_4, other=0.0)
        # NOTE this is for RMSNorm
        attn_O3 = (tl.sum((temp_0 * temp_0).to(tl.float16), axis=1, dtype=tl.float16) + (1 * attn_O3).to(tl.float16)).to(tl.float16)
        
        # These are matmuls for swiglu
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * WFF1a_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * WFF1a_stride1
        p_indices = p + tl.arange(0, BLOCK_P)
        mask_5 = (k_indices < N)[:, None] & (p_indices < N4)[None, :]
        temp_1 = tl.load(WFF1a_ptr + offset_1, mask=mask_5, other=0.0)
        FF1a = (tl.dot(temp_0, temp_1) + (1 * FF1a))
        
        offset_2 = (k + tl.arange(0, BLOCK_K))[:, None] * WFF1b_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * WFF1b_stride1
        mask_6 = (k_indices < N)[:, None] & (p_indices < N4)[None, :]
        temp_2 = tl.load(WFF1b_ptr + offset_2, mask=mask_6, other=0.0)
        FF1b = ((1 * FF1b) + tl.dot(temp_0, temp_2))
    
    # Skipped empty sloop with dummy body
    FF1a = (FF1a / tl.sqrt((attn_O3 / 4096).to(tl.float32))[:, None])
    FF1b = (FF1b / tl.sqrt((attn_O3 / 4096).to(tl.float32))[:, None])
    offset_3 = (tl.arange(0, 16))[:, None] * FF1_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * FF1_stride1
    p_indices = p + tl.arange(0, BLOCK_P)
    mask_7 = (p_indices < N4)[None, :]
    tl.store(FF1_ptr + offset_3, ((tl.sigmoid(FF1b.to(tl.float32)) * FF1b) * FF1a), mask=mask_7)



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_N': 32, 'BLOCK_P': 32}),
        triton.Config({'BLOCK_N': 32, 'BLOCK_P': 64}),
        triton.Config({'BLOCK_N': 32, 'BLOCK_P': 128}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_P': 32}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_P': 64}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_P': 128}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_P': 32}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_P': 64}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_P': 128})
    ], key=[]
)
@triton.jit
def kernel_2(
    FF1_ptr,
    FF1_stride0: tl.constexpr,
    FF1_stride1: tl.constexpr,
    FF2_ptr,
    FF2_stride0: tl.constexpr,
    FF2_stride1: tl.constexpr,
    WFF2_ptr,
    WFF2_stride0: tl.constexpr,
    WFF2_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    N4: tl.constexpr,
    P: tl.constexpr
):
    # Initialize kernel accumulators
    FF2 = tl.zeros((16, BLOCK_N), dtype=tl.float16)
    # Parallel loop n from 0 to FF2_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop p from 0 to 16384 with tile size BLOCK_P
    for p in range(0, 16384, BLOCK_P):
        offset_0 = (tl.arange(0, 16))[:, None] * FF1_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * FF1_stride1
        p_indices = p + tl.arange(0, BLOCK_P)
        mask_8 = (p_indices < N4)[None, :]
        temp_0 = tl.load(FF1_ptr + offset_0, mask=mask_8, other=0.0)
        offset_1 = (p + tl.arange(0, BLOCK_P))[:, None] * WFF2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WFF2_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_9 = (p_indices < N4)[:, None] & (n_indices < N)[None, :]
        temp_1 = tl.load(WFF2_ptr + offset_1, mask=mask_9, other=0.0)
        FF2 = (tl.dot(temp_0, temp_1).to(tl.float16) + (FF2 * 1).to(tl.float16)).to(tl.float16)
    # Store kernel accumulators
    offset_2 = (tl.arange(0, 16))[:, None] * FF2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * FF2_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_10 = (n_indices < N)[None, :]
    tl.store(FF2_ptr + offset_2, FF2, mask=mask_10)


# Metadata for benchmark.py
TENSOR_PARAMS = ['FF1', 'FF2', 'O2', 'WFF1a', 'WFF1b', 'WFF2', 'WO', 'X', 'attn_O2']
FP32_TENSOR_PARAMS = ['FF1']
BLOCK_PARAMS = ['block_k', 'block_n', 'block_p']

def forward(FF1, FF2, O2, WFF1a, WFF1b, WFF2, WO, X, attn_O2, block_k=16, block_n=16, block_p=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[lambda meta: ((4096 - 0 + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],)](
        O2,
        O2.stride(0),
        O2.stride(1),
        WO,
        WO.stride(0),
        WO.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        attn_O2,
        attn_O2.stride(0),
        attn_O2.stride(1),
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

    kernel_1[lambda meta: ((16384 - 0 + meta["BLOCK_P"] - 1) // meta["BLOCK_P"],)](
        FF1,
        FF1.stride(0),
        FF1.stride(1),
        WFF1a,
        WFF1a.stride(0),
        WFF1a.stride(1),
        WFF1b,
        WFF1b.stride(0),
        WFF1b.stride(1),
        attn_O2,
        attn_O2.stride(0),
        attn_O2.stride(1),
        # BLOCK_K, BLOCK_P are provided by autotune,
        # BLOCK_P is automatically set by autotune,
        # BLOCK_K is automatically set by autotune,
        D=128,
        H=32,
        M=16,
        N=4096,
        N4=16384,
        P=1008
    )

    kernel_2[lambda meta: ((4096 - 0 + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],)](
        FF1,
        FF1.stride(0),
        FF1.stride(1),
        FF2,
        FF2.stride(0),
        FF2.stride(1),
        WFF2,
        WFF2.stride(0),
        WFF2.stride(1),
        # BLOCK_N, BLOCK_P are provided by autotune,
        # BLOCK_N is automatically set by autotune,
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
