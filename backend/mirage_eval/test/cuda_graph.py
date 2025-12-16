import argparse

import torch

import mirage as mi
from mirage.kernel import KNGraph

HIDDEN_DIM = 4096
SEQ_LEN = 16
CONTEXT_LEN = 1008
NUM_HEADS = 32
HEAD_DIM = 128

DEVICE = torch.device("cuda:0")


def projection():
    """
    Builds and superoptimizes a kernel graph for a linear projection operation.

    This kernel is used for QKV (Query, Key, Value) projection in attention mechanisms.

    The function creates a Mirage kernel graph that takes an input tensor `x` of shape
    (SEQ_LEN, HIDDEN_DIM) and a weight matrix `w` of shape (HIDDEN_DIM, HIDDEN_DIM),
    performs matrix multiplication, and marks the result as the output. The graph is
    then superoptimized with the "mlp" configuration.

    Returns:
        KNGraph: The superoptimized kernel graph for the projection operation.
    """
    _graph = mi.new_kernel_graph()

    x = _graph.new_input(dims=(SEQ_LEN, HIDDEN_DIM), dtype=mi.float16)
    w = _graph.new_input(dims=(HIDDEN_DIM, HIDDEN_DIM * 3), dtype=mi.float16)
    o = _graph.matmul(x, w)

    _graph.mark_output(o)
    return _graph.superoptimize(config="mlp")


def run_cycle_kernel(
    kernel: KNGraph, x: torch.Tensor, w: torch.Tensor, stream: torch.cuda.Stream
):
    return kernel(inputs=[x, w], stream=stream)[0]  # type: ignore


def run_cycle_torch(
    x: torch.Tensor,
    w: torch.Tensor,
):
    return torch.matmul(x, w)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel",
        choices=["mirage", "torch"],
        default="mirage",
        help="Choose between 'mirage' and 'torch' (default: mirage)",
    )
    args = parser.parse_args()

    X = torch.randn(SEQ_LEN, HIDDEN_DIM, dtype=torch.float16, device=DEVICE)
    w_qkv = torch.randn(HIDDEN_DIM, HIDDEN_DIM * 3, dtype=torch.float16, device=DEVICE)

    if args.kernel == "mirage":
        print("Creating and optimizing kernel graphs")
        projection_kernel = projection()  # type: ignore
        print("Kernel optimization complete")
    else:
        projection_kernel = None

    capture_stream = torch.cuda.Stream(device=DEVICE)

    print("Running warmup iterations")
    with torch.cuda.stream(capture_stream):
        if projection_kernel is not None:
            for _ in range(16):
                run_cycle_kernel(projection_kernel, X, w_qkv, capture_stream)
        else:
            for _ in range(16):
                run_cycle_torch(X, w_qkv)
        capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()

    if projection_kernel is not None:
        with torch.cuda.graph(graph, stream=capture_stream):
            run_cycle_kernel(projection_kernel, X, w_qkv, capture_stream)
    else:
        with torch.cuda.graph(graph, stream=capture_stream):
            run_cycle_torch(X, w_qkv)

    print("Running test iterations")

    torch.cuda.synchronize()
    iteration_stream = torch.cuda.Stream(device=DEVICE)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(iteration_stream):
        start.record(iteration_stream)

        for _ in range(1000):
            graph.replay()

        end.record(iteration_stream)
        end.synchronize()

    print(f"Average iteration time: {start.elapsed_time(end) / 1000} ms")


if __name__ == "__main__":
    main()
