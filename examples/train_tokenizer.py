from pathlib import Path

from trainer.tokenizer import BPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
VALID_FILE = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"

OUTPUT = PROJECT_ROOT / "tokenizer" / "tinystories.json"


def main():
    print("=" * 80)
    print("Training TinyStories tokenizer")
    print("=" * 80)

    print(f"Train file : {TRAIN_FILE}")
    print(f"Valid file : {VALID_FILE}")
    print(f"Output     : {OUTPUT}")

    tokenizer = BPETokenizer.train(
        files=[TRAIN_FILE],
        vocab_size=16_000,
        special_tokens=[
            "<unk>",
            "<bos>",
            "<eos>",
            "<|endoftext|>",
        ],
        min_frequency=2,
    )

    tokenizer.save(OUTPUT)

    print()
    print(f"Tokenizer saved to: {OUTPUT}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    text = "Once upon a time, there was a little cat."

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print()
    print("Sanity check:")
    print("Text    :", text)
    print("Token IDs:", ids)
    print("Decoded :", decoded)


if __name__ == "__main__":
    main()