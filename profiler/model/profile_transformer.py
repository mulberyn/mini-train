import torch
from torch.profiler import profile, record_function, ProfilerActivity
from trainer.model.transformer import TransformerLM

from utils import print_hardware_info

def main():
    print_hardware_info()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    batch_size = 2
    seq_len = 512
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True) as prof:
        with record_function("transformer_lm_forward"):
            with torch.no_grad():
                model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    print()
    print("=" * 80)
    print("Profiler")
    print("=" * 80)
    print(prof.key_averages().table(sort_by=("cuda_time_total" if device.type == "cuda" else "cpu_time_total"), row_limit=30))
    output_path = "profiler/model/transformer_trace.json"
    prof.export_chrome_trace(output_path)
    print()
    print(f"Chrome trace saved to: {output_path}")


if __name__ == "__main__":
    main()