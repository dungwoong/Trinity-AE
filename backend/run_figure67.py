from codegen.convert_module import convert_ir_to_triton
import argparse, torch, importlib.util, sys, json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=str, default="llama", help="Input model type")
    parser.add_argument("--t", type=str, default="vanilla", help="Benchmark type")
    parser.add_argument("--n", type=str, default="A", help="Case number for IR")
    parser.add_argument("--d", type=int, default=0, help="Type device number")
    args = parser.parse_args()
    num = args.n
    model = args.m
    target = args.t

    device = torch.device(f'cuda:{args.d}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    dtype = torch.float16

    output_file = f"./figure56/{target}/{target}_{num}.py"
    module_name = f"{target}_{model}_best"

    # Load model config from JSON
    with open('./model_configs.json', 'r') as f:
        model_configs = json.load(f)

    if model not in model_configs:
        raise ValueError(f"Unknown model: {model}. Available: {list(model_configs.keys())}")

    config = model_configs[model]
    M = config['M']
    D = config['D']
    N = config['N']
    P = config['P'] - M
    H = config['H']
    N4 = config['N4']
    num_group = config.get('num_group', 4)

    constants = {
        'M': M,
        'D': D,
        'N': N,
        'P': P,
        'H': H,
        'N4': N4,
    }
    
    tensor_shapes = {
        'X': ('M', 'N'),
        'X2': ('M',),
        'X_norm': ('M', 'N'),

        'WQ': ('N', 'N'),
        'WK': ('N', 'N'),
        'WV': ('N', 'N'),

        'Q1': ('M', 'N'),
        'K1': ('M', 'N'),
        'V1': ('M', 'N'),
        
        'Q2': ('M', 'H', 'D'),
        'K2': ('M', 'H', 'D'),
        'V2': ('M', 'H', 'D'),

        'K_cache': ('H', 'P+M', 'D'),
        'V_cache': ('H', 'P+M', 'D'),

        'Q': ('H', 'M', 'D'),
        'K': ('H', 'M', 'D'),
        'V': ('H', 'M', 'D'),

        'O': ('H', 'M', 'D'),
        'O1': ('M', 'H', 'D'),
        'O2': ('M', 'N'),

        'C': ('H', 'M', 'P+M'),
        'C_exp': ('H', 'M', 'P+M'),
        'C_div': ('H', 'M', 'P+M'),
        'C_sum': ('H', 'M'),
        'noise': ('H', 'M', 'P+M'),
        'C_perturb': ('H', 'M', 'P+M'),
        'C_exp_perturb': ('H', 'M', 'P+M'),
        'C_sum_perturb': ('H', 'M'),
        'C_div_perturb': ('H', 'M', 'P+M'),
        'C_out': ('H', 'P+M'),
        'C_out1': ('H', 'P+M'),
        'C_out2': ('H', 'P+M'),

        'Q_norm': ('H', 'M', 'D'),
        'K_norm': ('H', 'M', 'D'),

        'WO': ('N', 'N'),
        'attn_O1': ('M', 'N'),
        'attn_O2': ('M', 'N'),
        'attn_O3': ('M'),
        'attn_O_norm': ('M', 'N'),
        'WFF1a': ('N', 'N4'),
        'WFF1b': ('N', 'N4'),
        'FF1a': ('M', 'N4'),
        'FF1b': ('M', 'N4'),
        'FF1b_silu': ('M', 'N4'),
        'FF1': ('M', 'N4'),
        'FF2': ('M', 'N'),
        'WFF2': ('N4', 'N'),
    }

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # --------------- Init for Attention ---------------------
    std = 0.01
    X = torch.randn((M, N), device=device, dtype=dtype) * std
    
    WQ = torch.randn((N, N), device=device, dtype=dtype) * std
    WK = torch.randn((N, N), device=device, dtype=dtype) * std
    WV = torch.randn((N, N), device=device, dtype=dtype) * std

    K_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std
    V_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std

    K_cache_flashinfer = K_cache.clone().transpose(0, 1).contiguous()
    V_cache_flashinfer = V_cache.clone().transpose(0, 1).contiguous()

    O2 = torch.zeros((M, N), device=device, dtype=dtype) * std

    # --------------- Additional init for RoCo ---------------------
    C_exp = torch.zeros((H, M, P+M), device=device, dtype=torch.float32)
    C_out1 = torch.zeros((H, P+M), device=device, dtype=dtype) * std
    C_out2 = torch.zeros((H, P+M), device=device, dtype=dtype) * std

    # --------------- Additional init for KeyFormer ---------------------
    C = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_exp_perturb = torch.zeros((H, M, P+M), device=device, dtype=torch.float32)
    C_out = torch.zeros((H, P+M), device=device, dtype=dtype)
    noise = torch.randn((H, M, P+M), device=device, dtype=dtype) * std

    # --------------- Init for FFN ---------------------
    std = 0.001
    if target == "ffn":
        O2 = O2
    else:
        O2 = torch.randn(M, N, dtype=dtype, device=device) * std
    FF1 = torch.zeros(M, N4, dtype=dtype, device=device)
    FF2 = torch.zeros(M, N4, dtype=dtype, device=device)
    WFF1a = torch.randn(N, N4, dtype=dtype, device=device) * std
    WFF1b = torch.randn(N, N4, dtype=dtype, device=device) * std
    WFF2 = torch.randn(N4, N, dtype=dtype, device=device) * std
    WO = torch.randn(N, N, dtype=dtype, device=device) * std
    attn_O2 = torch.zeros(M, N, dtype=dtype, device=device)


    out = O2.clone()
    ITER = 1000

    # --------------- Trinity ---------------------
    print("="*50)
    print(f"Starting Trinity {target}...")

    spec = importlib.util.spec_from_file_location(module_name, output_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    forward = getattr(module, "forward")

    tensor_params = getattr(module, 'TENSOR_PARAMS')
    block_params = getattr(module, 'BLOCK_PARAMS')
    tensors = {
        'X': X, 'WQ': WQ, 'WK': WK, 'WV': WV,
        'K_cache': K_cache, 'V_cache': V_cache,
        'O2': O2, 'C': C, 'C_exp': C_exp, 'noise': noise,
        'C_exp_perturb': C_exp_perturb,
        'C_out': C_out, 'C_out1': C_out1, 'C_out2': C_out2,
        'WO': WO, 'attn_O2': attn_O2,
        'WFF1a': WFF1a, 'WFF1b': WFF1b,
        'FF1': FF1, 'FF2': FF2, 'WFF2': WFF2,
    }
    blocks = {
        'block_k': 0, 'block_n': 0, 'block_p': 0
    }
    args = []
    for param in tensor_params:
        if param in tensors:
            args.append(tensors[param])
        else:
            raise ValueError(f"Unknown tensor parameter: {param}")
    for param in block_params:
        if param in blocks:
            args.append(blocks[param])
        else:
            raise ValueError(f"Unknown block parameter: {param}")

    for _ in range(10):
        forward(*args)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(ITER):
        forward(*args)
    end_event.record()
    torch.cuda.synchronize()

    time = start_event.elapsed_time(end_event) / ITER
    print(f"Trinity: {time} ms")


if __name__ == "__main__":
    main()