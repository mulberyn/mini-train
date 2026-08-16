import argparse
import time
import torch
from trainer.model.transformer import TransformerLM

from utils import print_hardware_info


def benchmark(model, inputs, warmup=20, iterations=100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(inputs)
        if inputs.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iterations):
            model(inputs)
        if inputs.is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
    total_time = end - start
    avg_time = total_time / iterations
    batch_size = inputs.shape[0]
    seq_len = inputs.shape[1]
    tokens_per_second = batch_size * seq_len / avg_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3 if inputs.is_cuda else None
    return avg_time, tokens_per_second, peak_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()
    if args.device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    print_hardware_info()
    vocab_size = 32000
    context_length = 2048
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000.0,
        device=device,
        dtype=torch.float32,
    )
    model.eval()
    inputs = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), dtype=torch.long, device=device)
    num_parameters = sum(p.numel() for p in model.parameters())
    print("=" * 80)
    print("Model")
    print("=" * 80)
    print(f"Device        : {device}")
    print(f"Parameters    : {num_parameters:,}")
    print(f"Layers        : {num_layers}")
    print(f"d_model       : {d_model}")
    print(f"num_heads     : {num_heads}")
    print(f"d_ff          : {d_ff}")
    print(f"vocab_size    : {vocab_size}")
    print(f"batch_size    : {args.batch_size}")
    print(f"seq_len       : {args.seq_len}")
    print()
    avg_time, tok_per_sec, memory = benchmark(model, inputs)
    print("=" * 80)
    print("Benchmark")
    print("=" * 80)
    print(f"Latency       : {avg_time * 1000:.3f} ms")
    print(f"Throughput    : {tok_per_sec:.2f} tokens/s")
    if memory is not None:
        print(f"Peak Memory   : {memory:.3f} GB")


if __name__ == "__main__":
    main()