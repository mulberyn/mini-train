# benchmark/benchmark_linear.py

import argparse
import time

import torch

from trainer.layers.linear import Linear
from utils.utils import print_hardware_info


def synchronize(device: torch.device) -> None:
    """Synchronize accelerator before measuring time."""

    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        # MPS currently does not expose an equivalent
        # synchronize API in the same way CUDA does.
        pass


def benchmark_linear(
    batch_size: int,
    seq_len: int,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int = 20,
    iterations: int = 100,
) -> None:

    print("\n" + "=" * 70)
    print("Linear Benchmark")
    print("=" * 70)

    print(f"Device          : {device}")
    print(f"Dtype           : {dtype}")
    print(f"Batch size      : {batch_size}")
    print(f"Sequence length : {seq_len}")
    print(f"In features     : {in_features}")
    print(f"Out features    : {out_features}")
    print(f"Warmup          : {warmup}")
    print(f"Iterations      : {iterations}")

    layer = Linear(
        in_features=in_features,
        out_features=out_features,
        device=device,
        dtype=dtype,
    )

    x = torch.randn(
        batch_size,
        seq_len,
        in_features,
        device=device,
        dtype=dtype,
    )

    layer.eval()

    # ------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------

    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(x)

    synchronize(device)

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            _ = layer(x)

    synchronize(device)

    end = time.perf_counter()

    total_time = end - start

    latency_ms = total_time / iterations * 1000

    # ------------------------------------------------------------
    # FLOPs
    # ------------------------------------------------------------

    # Matrix multiplication:
    #
    # [B*S, D] @ [D, H]
    #
    # FLOPs ≈ 2 * B * S * D * H

    flops_per_iteration = (
        2
        * batch_size
        * seq_len
        * in_features
        * out_features
    )

    tflops = (
        flops_per_iteration
        / (latency_ms / 1000)
        / 1e12
    )

    # ------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------

    memory_mb = None

    if device.type == "cuda":
        memory_mb = (
            torch.cuda.max_memory_allocated(device)
            / 1024**2
        )

    print("\nResults")
    print("-" * 70)

    print(f"Latency         : {latency_ms:.3f} ms")
    print(f"Throughput      : {1 / (latency_ms / 1000):.2f} iterations/s")
    print(f"TFLOPS          : {tflops:.3f}")

    if memory_mb is not None:
        print(f"Peak GPU memory : {memory_mb:.2f} MB")

    print("=" * 70)


def parse_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype not in mapping:
        raise ValueError(
            f"Unsupported dtype: {dtype}. "
            f"Choose from {list(mapping)}"
        )

    return mapping[dtype]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=[
            "float32",
            "float64",
            "float16",
            "bfloat16",
        ],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
            "mps",
        ],
    )

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)

    args = parser.parse_args()

    print_hardware_info()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")

        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")

        else:
            device = torch.device("cpu")

    else:
        device = torch.device(args.device)

    dtype = parse_dtype(args.dtype)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    benchmark_linear(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        in_features=args.in_features,
        out_features=args.out_features,
        dtype=dtype,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()