import argparse
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from utils import print_hardware_info
from trainer.layers.rope import RoPE


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
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

    print("\nProfiler Configuration")
    print("-" * 70)
    print(f"Device       : {device}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Num heads    : {args.num_heads}")
    print(f"Seq length   : {args.seq_len}")
    print(f"Head dim     : {args.head_dim}")

    rope = RoPE(theta=10000.0, d_k=args.head_dim, max_seq_len=args.seq_len, device=device)
    x = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.head_dim, device=device)
    positions = torch.arange(args.seq_len, device=device)

    with torch.no_grad():
        for _ in range(10):
            rope(x, positions)
    synchronize(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with record_function("rope_forward"):
            with torch.no_grad():
                for _ in range(10):
                    rope(x, positions)

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print("\nProfiler Results")
    print("=" * 100)
    print(prof.key_averages().table(sort_by=sort_key, row_limit=30))


if __name__ == "__main__":
    main()