import argparse
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainer.config import load_model_config
from trainer.tokenizer.bpe_tokenizer import BPETokenizer as Tokenizer
from inference.model_runner import ModelRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Generate TinyStories text using miniLLM-engine.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/latest.pt", help="Path to model checkpoint.")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer.json", help="Path to tokenizer.json.")
    parser.add_argument("--model-config", type=str, default="artifacts/model_config.json", help="Path to model_config.json saved by the training script.")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. 0 means greedy decoding.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling. None disables top-k.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--show-prompt", action="store_true", help="Print prompt before generated text.")
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they are generated.")
    return parser.parse_args()


def get_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer(path: str) -> Tokenizer:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    print(f"[Tokenizer] Loading: {path}")
    return Tokenizer.load(str(path))


def load_model_runner(
    checkpoint_path: str,
    tokenizer_path: str,
    model_config_path: str,
    device: torch.device,
) -> ModelRunner:
    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = Path(tokenizer_path)
    model_config_path = Path(model_config_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    print(f"[Model] Loading checkpoint: {checkpoint_path}")
    print(f"[Tokenizer] Loading: {tokenizer_path}")
    print(f"[Config] Loading: {model_config_path}")
    print(f"[Device] {device}")

    model_config = load_model_config(model_config_path)

    runner = ModelRunner.from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        model_config=model_config,
        device=device,
    )
    runner.model.eval()
    return runner


@torch.no_grad()
def generate(runner: ModelRunner, tokenizer: Tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_k: int | None):
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        raise ValueError("Prompt produces zero tokens.")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=runner.device)
    # temperature == 0 means greedy decoding (handled inside ModelRunner).
    output_ids = runner.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    # output_ids is a [1, T] tensor; ModelRunner.decode converts it to ids.
    text = runner.decode(output_ids[0])
    return text


def main():
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)

    print("=" * 70)
    print("miniLLM-engine TinyStories Generation")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Tokenizer       : {args.tokenizer}")
    print(f"Model config    : {args.model_config}")
    print(f"Prompt          : {args.prompt}")
    print(f"Max new tokens  : {args.max_new_tokens}")
    print(f"Temperature     : {args.temperature}")
    print(f"Top-k           : {args.top_k}")
    print(f"Seed            : {args.seed}")
    print(f"Streaming       : {'yes' if args.stream else 'no'}")
    if device.type == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(device)}")
    print("=" * 70)

    tokenizer = load_tokenizer(args.tokenizer)
    runner = load_model_runner(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        model_config_path=args.model_config,
        device=device,
    )

    print("\n[Generation]\n")
    if args.stream:
        token_ids = tokenizer.encode(args.prompt)
        if len(token_ids) == 0:
            raise ValueError("Prompt produces zero tokens.")
        if args.show_prompt:
            sys.stdout.write(args.prompt)
            sys.stdout.flush()
        for chunk in runner.stream_generate_text(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        ):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        generated_text = generate(
            runner=runner,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        if args.show_prompt:
            print(generated_text)
        else:
            if generated_text.startswith(args.prompt):
                print(generated_text[len(args.prompt):])
            else:
                print(generated_text)

    print("\n" + "=" * 70)
    print("Generation finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()