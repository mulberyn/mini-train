import argparse
import platform
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from trainer.attention.mha import MultiHeadAttention
from utils.utils import print_hardware_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    args = parser.parse_args()

    print_hardware_info()
    device = torch.device(args.device)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    mha = MultiHeadAttention(d_model=args.d_model, num_heads=args.num_heads, device=device, dtype=dtype)
    x = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device, dtype=dtype)
    mha.eval()

    with torch.no_grad():
        for _ in range(20):
            mha(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    print()
    print("Starting profiler...")
    print()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True) as prof:
        with record_function("MHA_forward"):
            with torch.no_grad():
                mha(x)

    print()
    print("=" * 100)
    print("Profiler Summary")
    print("=" * 100)
    print(prof.key_averages().table(sort_by=("cuda_time_total" if device.type == "cuda" else "cpu_time_total"), row_limit=30))

    if device.type == "cuda":
        print()
        print("=" * 100)
        print("CUDA Memory")
        print("=" * 100)
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Reserved : {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()