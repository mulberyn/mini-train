import argparse
import time
import torch
from utils import print_hardware_info
from trainer.layers.rmsnorm import RMSNorm


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def parse_dtype(dtype: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float64": torch.float64, "float16": torch.float16, "bfloat16": torch.bfloat16}
    return mapping[dtype]


def benchmark_rmsnorm(batch_size: int, seq_len: int, d_model: int, eps: float, dtype: torch.dtype, device: torch.device, warmup: int, iterations: int):
    print("\n" + "=" * 70)
    print("RMSNorm Benchmark")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Dtype           : {dtype}")
    print(f"Batch size      : {batch_size}")
    print(f"Sequence length : {seq_len}")
    print(f"Hidden d_model  : {d_model}")
    print(f"Epsilon         : {eps}")
    print(f"Warmup          : {warmup}")
    print(f"Iterations      : {iterations}")

    layer = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    layer.eval()

    with torch.no_grad():
        for _ in range(warmup):
            layer(x)
    synchronize(device)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            layer(x)
    synchronize(device)
    end = time.perf_counter()
    total_time = end - start
    latency_ms = total_time / iterations * 1000

    element_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = batch_size * seq_len * d_model
    input_bytes = num_elements * element_size
    output_bytes = num_elements * element_size
    weight_bytes = d_model * element_size
    estimated_bytes = input_bytes + output_bytes + weight_bytes
    bandwidth_gbps = estimated_bytes / (latency_ms / 1000) / 1e9

    token_count = batch_size * seq_len
    tokens_per_second = token_count / (latency_ms / 1000)

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    print("\nResults")
    print("-" * 70)
    print(f"Latency         : {latency_ms:.3f} ms")
    print(f"Throughput      : {1 / (latency_ms / 1000):.2f} iterations/s")
    print(f"Token throughput: {tokens_per_second:.2f} tokens/s")
    print(f"Effective BW    : {bandwidth_gbps:.3f} GB/s")
    if peak_memory_mb is not None:
        print(f"Peak GPU memory : {peak_memory_mb:.2f} MB")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--d_model", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print_hardware_info()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dtype = parse_dtype(args.dtype)
    benchmark_rmsnorm(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        eps=args.eps,
        dtype=dtype,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()