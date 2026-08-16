import argparse

import torch
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
)

from utils import print_hardware_info
from trainer.layers.embedding import Embedding


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


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

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------

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

    print("\nProfiler configuration")
    print("-" * 70)

    print(f"Device          : {device}")
    print(f"Vocab size      : {args.vocab_size}")
    print(f"Embedding dim   : {args.embedding_dim}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Sequence length : {args.seq_len}")

    print("-" * 70)

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    layer = Embedding(
        num_embeddings=args.vocab_size,
        embedding_dim=args.embedding_dim,
        device=device,
        dtype=torch.float32,
    )

    input_ids = torch.randint(
        0,
        args.vocab_size,
        (
            args.batch_size,
            args.seq_len,
        ),
        device=device,
    )

    # ------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------

    with torch.no_grad():
        for _ in range(10):
            layer(input_ids)

    synchronize(device)

    # ------------------------------------------------------------
    # Profiler
    # ------------------------------------------------------------

    activities = [
        ProfilerActivity.CPU,
    ]

    if device.type == "cuda":
        activities.append(
            ProfilerActivity.CUDA
        )

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function(
            "embedding_forward"
        ):
            with torch.no_grad():
                for _ in range(10):
                    layer(input_ids)

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print("\nProfiler Results")
    print("=" * 100)

    sort_key = (
        "cuda_time_total"
        if device.type == "cuda"
        else "cpu_time_total"
    )

    print(
        prof.key_averages().table(
            sort_by=sort_key,
            row_limit=30,
        )
    )


if __name__ == "__main__":
    main()