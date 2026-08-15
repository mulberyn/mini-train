import argparse
import platform
import time
import torch
from trainer.attention.mha import MultiHeadAttention
from utils.utils import print_hardware_info


def benchmark_mha(batch_size, seq_len, d_model, num_heads, device, dtype, warmup=20, iterations=100):
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, device=device, dtype=dtype)
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    mha.eval()
    with torch.no_grad():
        for _ in range(warmup):
            mha(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            mha(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
    latency_ms = (end - start) / iterations * 1000
    # Approximate FLOPs: QKV projections (3*2*B*S*D*D), attention (4*B*H*S^2*D_head), output projection (2*B*S*D*D)
    projection_flops = 3 * 2 * batch_size * seq_len * d_model * d_model
    attention_flops = 4 * batch_size * num_heads * seq_len * seq_len * (d_model // num_heads)
    output_flops = 2 * batch_size * seq_len * d_model * d_model
    total_flops = projection_flops + attention_flops + output_flops
    tflops = total_flops / (latency_ms / 1000) / 1e12
    print(f"B={batch_size:2d} S={seq_len:4d} D={d_model:4d} H={num_heads:2d} {str(dtype):18s} latency={latency_ms:8.3f} ms TFLOPS={tflops:7.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    print_hardware_info()
    device = torch.device(args.device)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    print()
    print("MHA Benchmark")
    print("-" * 80)
    for seq_len in [128, 256, 512, 1024, 2048]:
        benchmark_mha(batch_size=1, seq_len=seq_len, d_model=512, num_heads=8, device=device, dtype=dtype, warmup=args.warmup, iterations=args.iterations)


if __name__ == "__main__":
    main()