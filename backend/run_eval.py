from codegen.convert_module import convert_ir_to_triton
import argparse, torch, importlib.util, sys
from baselines import device, dtype

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--o", type=int, default=0, help="0 only convert, 1 only test, 2 both convert and test")
    parser.add_argument("--m", type=str, default="llama", help="Input model type")
    parser.add_argument("--t", type=str, default="vanilla", help="Benchmark type")
    parser.add_argument("--n", type=int, default=0, help="Case number for IR")
    parser.add_argument("--baseline", nargs="*", default=[], help="List of baselines")
    parser.add_argument("--print_output", action="store_true")
    args = parser.parse_args()

    num = args.n
    option = args.o
    model = args.m
    target = args.t
    baseline = args.baseline
    print_output = args.print_output

    case_file = f"./results/{target}/{target}_{model}_case{num}.txt"
    output_file = f"./results/{target}/{target}_{model}_benchmark{num}.py"
    module_name = f"{target}_{model}_best"

    # output_file = "./evaluation/manual/manual_llama_benchmark1.py"

    if model == 'falcon':
        M = 16
        D = 64
        N = 4544
        P = 1024 - M
        H = 71
        N4 = 4*N
        num_group = 4

        constants = {
            'M': M,
            'D': D,
            'N': N,
            'P': P,
            'H': H,
            'N4': N4,
        }
    elif model == 'llama':
        M = 16
        D = 128
        N = 4096
        P = 1024 - M
        H = 32
        N4 = N*4
        num_group = 4

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

        'WQ': ('N', 'N'),
        'WK': ('N', 'N'),
        'WV': ('N', 'N'),

        'K_cache': ('H', 'P+M', 'D'),
        'V_cache': ('H', 'P+M', 'D'),

        'O2': ('M', 'N'),

        'C': ('H', 'M', 'P+M'),
        'C_exp': ('H', 'M', 'P+M'),
        'noise': ('H', 'M', 'P+M'),
        'C_exp_perturb': ('H', 'M', 'P+M'),
        'C_out': ('H', 'P+M'),
        'C_out1': ('H', 'P+M'),
        'C_out2': ('H', 'P+M'),

        'WO': ('N', 'N'),
        'attn_O2': ('M', 'N'),
        'WFF1a': ('N', 'N4'),
        'WFF1b': ('N', 'N4'),
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

    WK_gqa = torch.randn((N, N//num_group), device=device, dtype=dtype) * std
    WV_gqa = torch.randn((N, N//num_group), device=device, dtype=dtype) * std

    K_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std
    V_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std

    K_cache_flashinfer = K_cache.clone().transpose(0, 1).contiguous()
    V_cache_flashinfer = V_cache.clone().transpose(0, 1).contiguous()

    O2 = torch.zeros((M, N), device=device, dtype=dtype) * std

    # --------------- Additional init for RoCo ---------------------
    C_exp = torch.zeros((H, M, P+M), device=device, dtype=torch.float32) * std
    C_out1 = torch.zeros((H, P+M), device=device, dtype=dtype) * std
    C_out2 = torch.zeros((H, P+M), device=device, dtype=dtype) * std

    # --------------- Additional init for KeyFormer ---------------------
    C = torch.zeros((H, M, P+M), device=device, dtype=dtype) * std
    C_exp_perturb = torch.zeros((H, M, P+M), device=device, dtype=torch.float32) * std
    C_out = torch.zeros((H, P+M), device=device, dtype=dtype) * std
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

    if option == 0:
        with open(case_file, "r") as f:
            ir = f.read().strip()
            triton_code = convert_ir_to_triton(ir, tensor_shapes, constants)

            with open(output_file, "w") as f:
                f.write(triton_code)
            
            print("="*50)
            print("Triton kernel generated successfully!")
        return

    match target:
        case "vanilla":
            from baselines import Vanilla, TensorRT_Vanilla, FlashInfer_Vanilla
            trt = TensorRT_Vanilla(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            ti = Vanilla(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_Vanilla(M, N, D, P, K_cache_flashinfer.clone(), V_cache_flashinfer.clone(), WQ, WK, WV)
            from flashtensor.h100_vanilla import bench_vanilla
            ft = bench_vanilla
        case "prenorm":
            from baselines import PreNorm, TensorRT_PreNorm, FlashInfer_PreNorm
            trt = TensorRT_PreNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            ti = PreNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_PreNorm(M, N, D, P, K_cache_flashinfer.clone(), V_cache_flashinfer.clone(), WQ, WK, WV)
            from flashtensor.h100_prenorm import bench_prenorm
            ft = bench_prenorm
        case "keyformer":
            from baselines import KeyFormer, TensorRT_KeyFormer
            trt = TensorRT_KeyFormer(M, N, D, H, K_cache.clone(), V_cache.clone(), P, noise, WQ, WK, WV)
            ti = KeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = None
            from flashtensor.h100_kf import bench_kf
            ft = bench_kf
        case "qknorm":
            from baselines import QKNorm, TensorRT_QKNorm, FlashInfer_QKNorm
            trt = TensorRT_QKNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            ti = QKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_QKNorm(M, N, D, P, K_cache_flashinfer.clone(), V_cache_flashinfer.clone(), WQ, WK, WV)
            from flashtensor.h100_qknorm import bench_qknorm
            ft = bench_qknorm
        case "roco":
            from baselines import RoCo, TensorRT_RoCo, FlashInfer_RoCo
            trt = TensorRT_RoCo(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            ti = RoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = None
            from flashtensor.h100_roco import bench_roco
            ft = bench_roco
        case "gqa":
            from baselines import Vanilla_GQA, TensorRT_Vanilla_GQA
            trt = TensorRT_Vanilla_GQA(M, N, D, H, N//num_group, K_cache.clone(), V_cache.clone(), P, WQ, WK_gqa, WV_gqa)
            ti = Vanilla_GQA(M, N, D, P, N//num_group, K_cache.clone(), V_cache.clone(), WQ, WK_gqa, WV_gqa)
            fi = None
            ft = None
        case "ffn":
            from baselines import FFN, TensorRT_FFN
            trt = TensorRT_FFN(M, N, N4, WO=WO, WFF1a=WFF1a, WFF1b=WFF1b, WFF2=WFF2)
            ti = FFN(M, N, N4, WO=WO, WFF1a=WFF1a, WFF1b=WFF1b, WFF2=WFF2)
            fi = None
            ft = None

    # --------------- Trinity ---------------------
    print("="*50)
    print(f"Starting Trinity {target}...")
    if len(baseline) == 0 or "trinity" in baseline:
        if option == 0 or option == 2:
            # Convert IR to Triton kernel
            with open(case_file, "r") as f:
                ir = f.read().strip()
            triton_code = convert_ir_to_triton(ir, tensor_shapes, constants)

            with open(output_file, "w") as f:
                f.write(triton_code)
            
            print("="*50)
            print("Triton kernel generated successfully!")
        if option == 0:
            return

        spec = importlib.util.spec_from_file_location(module_name, output_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        forward = getattr(module, "forward")
        # forward = getattr(module, "forward_m1_padded")

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
        print(f"Trinity without CUDA Graph: {time} ms")
        
        if print_output:
            print(O2)

    # ----------------- TensorRT ---------------------
    if len(baseline) == 0 or "tensorrt" in baseline:
        print("="*50)
        print(f"Starting TensorRT {target}...")

        if target == "ffn":
            inputs = (O2, X)
        else:
            inputs = (X,)

        trt.half()
        
        with torch.no_grad():
            for _ in range(10):
                out = trt(*inputs)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(ITER):
                _ = trt(*inputs)
            end_event.record()
            torch.cuda.synchronize()

            time = start_event.elapsed_time(end_event) / ITER
            print(f"TensorRT without CUDA Graph: {time} ms")
        
        if print_output:
            print(out)

    # ----------------- Pytorch Eager ----------------------
    if len(baseline) == 0 or "pytorch" in baseline:
        print("="*50)
        print(f"Starting Pytorch Eager {target}...")
        
        if target == "ffn":
            inputs = (O2, X)
        else:
            inputs = (X,)

        ti = ti.eval()
        
        with torch.no_grad():
            for _ in range(10):
                out = ti(*inputs)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(ITER):
                _ = ti(*inputs)
            end_event.record()
            torch.cuda.synchronize()

            time = start_event.elapsed_time(end_event) / ITER
            print(f"Pytorch Eager without CUDA Graph: {time} ms")
        
        if print_output:
            print(out)

    # ----------------- Torch Inductor ---------------------
    if len(baseline) == 0 or "inductor" in baseline:
        print("="*50)
        print(f"Starting Torch Inductor {target}...")

        if target == "ffn":
            inputs = (O2, X)
        else:
            inputs = (X,)

        mode = "max-autotune-no-cudagraphs"
        compiled_model = torch.compile(ti, backend="inductor", mode=mode, fullgraph=True)
        for _ in range(10):
            out = compiled_model(*inputs)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(ITER):
            _ = compiled_model(*inputs)
        end_event.record()
        torch.cuda.synchronize()

        time = start_event.elapsed_time(end_event) / ITER
        print(f"Torch Inductor {mode}: {time} ms")
        
        if print_output:
            print(out)
    
    # ----------------- FlashInfer ---------------------
    if not fi is None and len(baseline) == 0 or "flashinfer" in baseline:
        print("="*50)
        print(f"Starting FlashInfer {target}...")
        
        fi.half()
        fi = fi.eval()
        
        with torch.no_grad():
            for _ in range(10):
                out = fi(X)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(ITER):
                _ = fi(X)
            end_event.record()
            torch.cuda.synchronize()

            time = start_event.elapsed_time(end_event) / ITER
            print(f"FlashInfer without CUDA Graph: {time} ms")
        
        if print_output:
            print(out)
    
    # ----------------- FlashTensor ---------------------
    if not ft is None and len(baseline) == 0 or "flashtensor" in baseline:
        print("="*50)
        print(f"Starting FlashTensor {target}...")

        
        time = ft(model, M, N, P, D, H, device, dtype)
        print(f"FlashTensor without CUDA Graph: {time} ms")


if __name__ == "__main__":
    main()