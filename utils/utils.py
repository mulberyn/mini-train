# benchmark/utils.py

import platform
import sys

import torch


def print_hardware_info() -> None:
    """Print Python, PyTorch, and accelerator information."""

    print("=" * 70)
    print("Hardware / Environment")
    print("=" * 70)

    print(f"Python          : {sys.version.split()[0]}")
    print(f"PyTorch         : {torch.__version__}")
    print(f"OS              : {platform.system()} {platform.release()}")
    print(f"Architecture    : {platform.machine()}")

    # CUDA
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version    : {torch.version.cuda}")
        print(f"GPU count       : {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)

            memory_gb = props.total_memory / 1024**3

            print(f"\nGPU {i}")
            print(f"  Name          : {props.name}")
            print(f"  Compute Cap.  : {props.major}.{props.minor}")
            print(f"  VRAM          : {memory_gb:.2f} GB")
            print(f"  SM count      : {props.multi_processor_count}")

    # MPS
    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )

    print(f"\nMPS available   : {mps_available}")

    if mps_available:
        print("  Backend       : Apple Metal Performance Shaders")

    print("=" * 70)