# profiler/profile_linear.py

import argparse

import torch
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
)

from trainer.layers.linear import Linear
from utils.utils import print_hardware_info


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
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
    print(f"Batch size      : {args.batch_size}")
    print(f"Sequence length : {args.seq_len}")
    print(f"In features     : {args.in_features}")
    print(f"Out features    : {args.out_features}")
    print("-" * 70)

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    layer = Linear(
        args.in_features,
        args.out_features,
        device=device,
        dtype=torch.float32,
    )

    x = torch.randn(
        args.batch_size,
        args.seq_len,
        args.in_features,
        device=device,
    )

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            layer(x)

    synchronize(device)

    # ------------------------------------------------------------
    # Profiler
    # ------------------------------------------------------------

    activities = [ProfilerActivity.CPU]

    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:

        with record_function("linear_forward"):
            with torch.no_grad():
                for _ in range(10):
                    layer(x)

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print("\nProfiler Results")
    print("=" * 100)

    print(
        prof.key_averages().table(
            sort_by=(
                "cuda_time_total"
                if device.type == "cuda"
                else "cpu_time_total"
            ),
            row_limit=30,
        )
    )


if __name__ == "__main__":
    main()