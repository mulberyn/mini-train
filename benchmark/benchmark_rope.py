import argparse
import time
import torch
from utils import print_hardware_info
from trainer.layers.rope import RoPE


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def parse_dtype(dtype):
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    return mapping[dtype]


def benchmark_rope(batch_size, num_heads, seq_len, head_dim, dtype, device, warmup, iterations):
    print("\n" + "=" * 70)
    print("RoPE Benchmark")
    print("=" * 70)
    print(f"Device       : {device}")
    print(f"Dtype        : {dtype}")
    print(f"Batch size   : {batch_size}")
    print(f"Num heads    : {num_heads}")
    print(f"Seq length   : {seq_len}")
    print(f"Head dim     : {head_dim}")

    rope = RoPE(theta=10000.0, d_k=head_dim, max_seq_len=seq_len, device=device)
    x = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    positions = torch.arange(seq_len, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            rope(x, positions)
    synchronize(device)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            rope(x, positions)
    synchronize(device)
    elapsed = time.perf_counter() - start

    latency_ms = elapsed / iterations * 1000
    tokens = batch_size * seq_len
    tokens_per_second = tokens / (latency_ms / 1000)

    element_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = batch_size * num_heads * seq_len * head_dim
    bytes_per_iteration = 2 * num_elements * element_size
    bandwidth = bytes_per_iteration / (latency_ms / 1000) / 1e9

    print("\nResults")
    print("-" * 70)
    print(f"Latency      : {latency_ms:.3f} ms")
    print(f"Tokens/s     : {tokens_per_second:.2f}")
    print(f"Effective BW : {bandwidth:.3f} GB/s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="auto")
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

    dtype = parse_dtype(args.dtype)
    benchmark_rope(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()