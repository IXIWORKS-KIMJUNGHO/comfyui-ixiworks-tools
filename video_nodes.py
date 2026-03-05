"""
Video Description Nodes for ComfyUI
All heavy imports (transformers, model classes) are lazy-loaded
to avoid blocking ComfyUI startup.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Tuple

# ComfyUI imports
import folder_paths

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _import_video_processor():
    from processing.video_processor import VideoProcessor
    return VideoProcessor


def _import_qwen3vl():
    from models.model_cache import ModelCache
    from models.qwen3vl_inference import Qwen3VLInference
    return ModelCache, Qwen3VLInference


class VideoDescriptionQwen3VL:
    """
    Video description node using Qwen3-VL-8B-Instruct model
    Generates detailed descriptions of video content
    """

    @classmethod
    def _get_comfyui_input_dir(cls) -> Path:
        """
        Get ComfyUI input directory path

        For ComfyUI Desktop App with custom user directory:
        - Uses folder_paths.get_input_directory() as fallback
        - If user_directory is custom, assumes input is at same level
        """
        input_dir = Path(folder_paths.get_input_directory())
        user_dir = Path(folder_paths.get_user_directory())

        # Check if user_directory is custom (not in ComfyUI base path)
        # If so, assume input directory is at the same level as user directory
        if user_dir.exists() and str(user_dir.parent) != str(input_dir.parent):
            alternative_input = user_dir.parent / "input"
            if alternative_input.exists():
                logger.info(f"Using custom input directory: {alternative_input}")
                return alternative_input

        return input_dir

    @classmethod
    def _get_analysis_prompt(cls, analysis_type: str, custom_prompt: str = "") -> tuple[str, int, float]:
        """
        Get optimized prompt, max_tokens, and temperature for analysis type

        Args:
            analysis_type: Type of analysis (detailed, summary, keywords)
            custom_prompt: Optional custom prompt to override default

        Returns:
            Tuple of (prompt, max_tokens, temperature)
        """
        if custom_prompt and custom_prompt.strip():
            # Use custom prompt with default settings
            return (custom_prompt.strip(), 256, 0.7)

        # Predefined analysis type configurations
        analysis_configs = {
            "detailed": {
                "prompt": (
                    "Provide a comprehensive and detailed description of this video. "
                    "Include information about:\n"
                    "- Main subjects and their actions\n"
                    "- Setting and environment\n"
                    "- Notable objects and elements\n"
                    "- Temporal progression of events\n"
                    "- Visual style and mood\n"
                    "Write complete sentences and ensure the description ends naturally."
                ),
                "max_tokens": 384,
                "temperature": 0.7
            },
            "summary": {
                "prompt": (
                    "Provide a brief, concise summary of this video in 2-3 sentences. "
                    "Focus on the most important elements and main action."
                ),
                "max_tokens": 128,
                "temperature": 0.5
            },
            "keywords": {
                "prompt": (
                    "Analyze this video and extract key information in the following format:\n"
                    "- Main subjects: [list the people, animals, or main objects]\n"
                    "- Actions: [list the key actions or activities]\n"
                    "- Setting: [describe the location/environment]\n"
                    "- Objects: [list notable objects or items]\n"
                    "- Mood/Style: [describe the overall tone or visual style]"
                ),
                "max_tokens": 256,
                "temperature": 0.3
            }
        }

        config = analysis_configs.get(analysis_type, analysis_configs["detailed"])
        return (config["prompt"], config["max_tokens"], config["temperature"])

    @classmethod
    def _resolve_video_path(cls, video_path: str) -> str:
        """
        Resolve video path with smart search in ComfyUI input directory

        Supports:
        - Absolute paths: /full/path/to/video.mp4
        - Just filename: video.mp4 → searches in input/
        - Relative path: subfolder/video.mp4 → searches in input/subfolder/

        Args:
            video_path: User-provided path (can be absolute, relative, or just filename)

        Returns:
            Resolved absolute path to video file

        Raises:
            FileNotFoundError: If video file cannot be found
        """
        video_path = video_path.strip()
        path_obj = Path(video_path)

        # Case 1: Absolute path - use as-is
        if path_obj.is_absolute():
            if path_obj.exists():
                return str(path_obj)
            raise FileNotFoundError(f"Video file not found at absolute path: {video_path}")

        # Case 2: Relative path or filename - search in ComfyUI input directory
        input_dir = cls._get_comfyui_input_dir()
        resolved_path = input_dir / video_path

        if resolved_path.exists():
            logger.info(f"Resolved '{video_path}' → '{resolved_path}'")
            return str(resolved_path)

        # Not found
        raise FileNotFoundError(
            f"Video file not found.\n"
            f"Searched: {resolved_path}\n"
            f"Tip: Place videos in ComfyUI/input/ directory or provide absolute path"
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "analysis_type": (["detailed", "summary", "keywords"], {
                    "default": "detailed"
                }),
                "fps": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 30.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "use_4bit": ("BOOLEAN", {
                    "default": False
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("description", "info")
    FUNCTION = "describe_video"
    CATEGORY = "IXIWORKS/Video"

    def describe_video(self, video_path, analysis_type, fps, custom_prompt="", use_4bit=False, temperature=0.7):
        """
        Generate video description using Qwen3-VL

        Args:
            video_path: Path to video file - supports:
                - Absolute path: /full/path/to/video.mp4
                - Filename: video.mp4 (searches in ComfyUI/input/)
                - Relative: subfolder/video.mp4 (searches in ComfyUI/input/subfolder/)
            analysis_type: Type of analysis (detailed, summary, keywords)
            fps: Frames per second for sampling (required)
            custom_prompt: Optional custom prompt (overrides analysis_type preset)
            use_4bit: Use 4-bit quantization (saves VRAM)
            temperature: Sampling temperature (overrides analysis_type preset if custom_prompt used)

        Returns:
            Tuple of (description, info)
        """
        try:
            # Validate and resolve video path
            video_path = video_path.strip()
            if not video_path:
                return ("Error: Video path is empty", "Please provide a video filename or path")

            # Resolve path (searches in ComfyUI input directory if needed)
            try:
                resolved_path = self._resolve_video_path(video_path)
            except FileNotFoundError as e:
                return (f"Error: {str(e)}", "File not found")

            # Lazy import heavy modules
            VideoProcessor = _import_video_processor()
            ModelCache, Qwen3VLInference = _import_qwen3vl()

            # Validate video file
            if not VideoProcessor.validate_video(resolved_path):
                return (f"Error: Invalid video file: {resolved_path}", "Video validation failed")

            # Get analysis configuration
            prompt, max_tokens, config_temperature = self._get_analysis_prompt(analysis_type, custom_prompt)

            # Use configured temperature if no custom prompt
            if not custom_prompt or not custom_prompt.strip():
                temperature = config_temperature

            # Get video info
            video_info = VideoProcessor.get_video_info(resolved_path)
            video_source = Path(resolved_path).name
            info_text = (
                f"Source: {video_source}\n"
                f"Type: {analysis_type}\n"
                f"Path: {video_path}\n"
                f"Duration: {video_info['duration']:.2f}s\n"
                f"Resolution: {video_info['width']}x{video_info['height']}\n"
                f"FPS: {video_info['fps']:.2f}\n"
                f"Sampling: {fps} FPS\n"
                f"Max tokens: {max_tokens}\n"
                f"Temperature: {temperature:.2f}\n"
                f"4-bit: {use_4bit}"
            )

            logger.info(f"Processing video: {video_source}")
            logger.info(f"Analysis type: {analysis_type}")
            logger.info(f"Resolved path: {resolved_path}")
            logger.info(f"Video duration: {video_info['duration']:.2f}s")

            # Load model (cached after first load)
            logger.info("Loading Qwen3-VL model...")
            model, processor = ModelCache.get_qwen3vl(use_4bit=use_4bit)

            # Create inference wrapper
            inference = Qwen3VLInference(model, processor)

            # Generate description
            description = inference.generate_description(
                video_path=resolved_path,
                prompt=prompt,
                max_new_tokens=max_tokens,
                fps=fps,
                temperature=temperature
            )

            return (description, info_text)

        except FileNotFoundError as e:
            error_msg = f"Video file not found: {video_path}"
            logger.error(error_msg)
            return (f"Error: {error_msg}", str(e))

        except Exception as e:
            error_msg = f"Error during inference: {str(e)}"
            logger.error(error_msg)
            return (f"Error: {error_msg}", f"Exception: {type(e).__name__}")


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VideoDescribe": VideoDescriptionQwen3VL,
}

# Display name mappings for ComfyUI UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoDescribe": "Video Describe",
}
