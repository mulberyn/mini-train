import argparse
import time

import torch

from utils import print_hardware_info
from trainer.layers.embedding import Embedding


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def parse_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    return mapping[dtype]


def benchmark_embedding(
    vocab_size: int,
    embedding_dim: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iterations: int,
):
    print("\n" + "=" * 70)
    print("Embedding Benchmark")
    print("=" * 70)

    print(f"Device          : {device}")
    print(f"Dtype           : {dtype}")
    print(f"Vocab size      : {vocab_size}")
    print(f"Embedding dim   : {embedding_dim}")
    print(f"Batch size      : {batch_size}")
    print(f"Sequence length : {seq_len}")
    print(f"Warmup          : {warmup}")
    print(f"Iterations      : {iterations}")

    layer = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        device=device,
        dtype=dtype,
    )

    input_ids = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            seq_len,
        ),
        device=device,
    )

    layer.eval()

    # ------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------

    with torch.no_grad():
        for _ in range(warmup):
            layer(input_ids)

    synchronize(device)

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            layer(input_ids)

    synchronize(device)

    end = time.perf_counter()

    total_time = end - start

    latency_ms = (
        total_time
        / iterations
        * 1000
    )

    # ------------------------------------------------------------
    # Memory bandwidth
    # ------------------------------------------------------------

    #
    # Each token reads one embedding vector:
    #
    # number of vectors = B * S
    #
    # bytes per vector = embedding_dim * dtype_size
    #
    bytes_per_iteration = (
        batch_size
        * seq_len
        * embedding_dim
        * torch.tensor([], dtype=dtype).element_size()
    )

    bandwidth_gbps = (
        bytes_per_iteration
        / (latency_ms / 1000)
        / 1e9
    )

    # ------------------------------------------------------------
    # Embedding table size
    # ------------------------------------------------------------

    weight_bytes = (
        vocab_size
        * embedding_dim
        * torch.tensor([], dtype=dtype).element_size()
    )

    weight_mb = weight_bytes / 1024**2

    print("\nResults")
    print("-" * 70)

    print(f"Latency         : {latency_ms:.3f} ms")
    print(
        f"Throughput      : "
        f"{1 / (latency_ms / 1000):.2f} iterations/s"
    )
    print(
        f"Token throughput: "
        f"{batch_size * seq_len / (latency_ms / 1000):.2f} tokens/s"
    )
    print(
        f"Effective BW    : "
        f"{bandwidth_gbps:.3f} GB/s"
    )
    print(
        f"Embedding table : "
        f"{weight_mb:.2f} MB"
    )

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device)
            / 1024**2
        )

        print(
            f"Peak GPU memory : "
            f"{peak_memory_mb:.2f} MB"
        )

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
    )

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

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    # Hardware information
    print_hardware_info()

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dtype = parse_dtype(args.dtype)

    benchmark_embedding(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dtype=dtype,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()