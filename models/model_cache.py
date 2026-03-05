"""
Model Cache for Qwen3-VL
Singleton pattern to avoid loading models multiple times
"""

import torch
import os
import warnings
from pathlib import Path
from typing import Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Singleton cache for Qwen3-VL model and processor
    Ensures models are loaded only once per session
    """

    _instance = None
    _qwen3vl_model: Optional[Any] = None
    _qwen3vl_processor: Optional[Any] = None
    _model_name = "Qwen/Qwen3-VL-8B-Instruct"

    @classmethod
    def _get_model_path(cls) -> Path:
        """
        Get the local model path in ComfyUI models directory

        Returns:
            Path to models/video_description/Qwen3-VL-8B-Instruct/
        """
        # Find ComfyUI root directory (go up from custom_nodes)
        current_file = Path(__file__).resolve()
        custom_nodes_dir = current_file.parent.parent.parent
        comfyui_root = custom_nodes_dir.parent

        # ComfyUI models directory
        model_path = comfyui_root / "models" / "video_description" / "Qwen3-VL-8B-Instruct"

        return model_path

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_qwen3vl(cls, use_4bit: bool = False) -> Tuple[Any, Any]:
        """
        Get or load Qwen3-VL model and processor

        Args:
            use_4bit: Whether to use 4-bit quantization for memory efficiency

        Returns:
            Tuple of (model, processor)
        """
        if cls._qwen3vl_model is None or cls._qwen3vl_processor is None:
            # Lazy import: transformers model classes are heavy and slow to import
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            # Suppress known deprecation warnings from transformers library
            warnings.filterwarnings('ignore', message='.*torchvision.*deprecated.*')
            warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
            # Get local model path
            model_path = cls._get_model_path()

            # Check if model exists locally in Hugging Face cache structure
            # HF downloads to: model_path/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/[hash]/
            local_model_exists = False
            if model_path.exists():
                # Look for HF cache structure
                hf_cache_dirs = list(model_path.glob("models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*"))
                if hf_cache_dirs:
                    # Use the HF cache directory
                    snapshot_dir = hf_cache_dirs[0]
                    if (snapshot_dir / "config.json").exists():
                        local_model_exists = True
                        model_source = str(snapshot_dir)
                        logger.info(f"Found model in HF cache: {snapshot_dir}")

            if not local_model_exists:
                logger.info(f"Local model not found at: {model_path}")
                logger.info(f"Will download from Hugging Face: {cls._model_name}")
                logger.info(f"Saving to: {model_path}")
                model_source = cls._model_name
                # Create directory if it doesn't exist
                model_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"4-bit quantization: {use_4bit}")

            # Determine device
            if torch.cuda.is_available():
                device = "cuda"
                device_map = "auto"
            elif torch.backends.mps.is_available():
                device = "mps"
                device_map = None  # MPS doesn't support device_map
            else:
                device = "cpu"
                device_map = None

            logger.info(f"Using device: {device}")

            try:
                if use_4bit:
                    try:
                        from transformers import BitsAndBytesConfig

                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16
                        )

                        logger.info("Loading model with 4-bit quantization...")
                        load_kwargs = {
                            "quantization_config": quantization_config,
                        }
                        if device_map:
                            load_kwargs["device_map"] = device_map
                        if not local_model_exists:
                            load_kwargs["cache_dir"] = str(model_path)

                        cls._qwen3vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
                            model_source,
                            **load_kwargs
                        )
                    except ImportError as e:
                        logger.warning(f"bitsandbytes not available: {e}")
                        logger.warning("Falling back to FP16 loading. Install bitsandbytes for 4-bit quantization:")
                        logger.warning("  pip install bitsandbytes")

                        # Fall back to FP16
                        logger.info("Loading model in FP16...")
                        load_kwargs = {"dtype": torch.float16}
                        if device_map:
                            load_kwargs["device_map"] = device_map
                        if not local_model_exists:
                            load_kwargs["cache_dir"] = str(model_path)

                        cls._qwen3vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
                            model_source,
                            **load_kwargs
                        )
                        if not device_map:
                            cls._qwen3vl_model = cls._qwen3vl_model.to(device)
                else:
                    logger.info("Loading model in FP16...")
                    load_kwargs = {"dtype": torch.float16}
                    if device_map:
                        load_kwargs["device_map"] = device_map
                    if not local_model_exists:
                        load_kwargs["cache_dir"] = str(model_path)

                    cls._qwen3vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_source,
                        **load_kwargs
                    )
                    if not device_map:
                        cls._qwen3vl_model = cls._qwen3vl_model.to(device)

                processor_kwargs = {}
                if not local_model_exists:
                    processor_kwargs["cache_dir"] = str(model_path)

                cls._qwen3vl_processor = AutoProcessor.from_pretrained(
                    model_source,
                    **processor_kwargs
                )

                logger.info("✓ Model loaded successfully")
                logger.info(f"Model device: {cls._qwen3vl_model.device}")
                logger.info(f"Model location: {model_path}")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

        return cls._qwen3vl_model, cls._qwen3vl_processor

    @classmethod
    def clear_cache(cls):
        """Clear cached models to free memory"""
        logger.info("Clearing model cache")
        cls._qwen3vl_model = None
        cls._qwen3vl_processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def is_loaded(cls) -> bool:
        """Check if model is already loaded"""
        return cls._qwen3vl_model is not None and cls._qwen3vl_processor is not None
