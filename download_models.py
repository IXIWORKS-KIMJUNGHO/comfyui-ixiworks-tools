"""
Model Download Script for ComfyUI-VideoDescription
Downloads Qwen3-VL model to ComfyUI models directory
"""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import sys
from pathlib import Path


def get_model_path():
    """Get the target model path in ComfyUI models directory"""
    # Get script directory
    script_dir = Path(__file__).resolve().parent
    # Go up to custom_nodes, then to ComfyUI root
    comfyui_root = script_dir.parent.parent
    # ComfyUI models directory
    model_path = comfyui_root / "models" / "video_description" / "Qwen3-VL-8B-Instruct"
    return model_path


def download_qwen3vl():
    """Download Qwen3-VL-8B-Instruct model and processor"""

    model_path = get_model_path()

    print("=" * 60)
    print("Qwen3-VL Model Download")
    print("=" * 60)
    print("\nModel: Qwen3-VL-8B-Instruct")
    print("Size: ~16GB (FP16)")
    print(f"Download location: {model_path}")
    print("\nThis may take 10-30 minutes depending on network speed.")
    print("=" * 60)

    # Create directory if it doesn't exist
    model_path.mkdir(parents=True, exist_ok=True)

    try:
        # Download model
        print("\n[1/2] Downloading model weights...")
        print(f"Saving to: {model_path}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=str(model_path)
        )
        print("✓ Model downloaded successfully")

        # Download processor
        print("\n[2/2] Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            cache_dir=str(model_path)
        )
        print("✓ Processor downloaded successfully")

        print("\n" + "=" * 60)
        print("✓ Download Complete!")
        print("=" * 60)
        print(f"\nModel location: {model_path}")
        print("Model is ready to use.")
        print("You can now run ComfyUI and use the Video Description node.")

        return True

    except Exception as e:
        print(f"\n✗ Error during download: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have ~20GB free disk space")
        print(f"3. Verify write permissions for: {model_path}")
        print("4. Try running: pip install --upgrade transformers")
        return False


if __name__ == "__main__":
    print("\nStarting download process...\n")
    success = download_qwen3vl()
    sys.exit(0 if success else 1)
