"""
Qwen3-VL Inference Wrapper
Handles video description generation with Qwen3-VL model
"""

import torch
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class Qwen3VLInference:
    """
    Wrapper for Qwen3-VL video description inference
    """

    def __init__(self, model, processor):
        """
        Initialize inference wrapper

        Args:
            model: Qwen3VLForConditionalGeneration model instance
            processor: AutoProcessor instance
        """
        self.model = model
        self.processor = processor
        self.device = model.device

    def generate_description(
        self,
        video_path: Union[str, Path],
        prompt: str = "Describe this video in detail.",
        max_new_tokens: int = 256,
        fps: float = 1.0,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate video description using Qwen3-VL

        Args:
            video_path: Path to video file
            prompt: Text prompt for description
            max_new_tokens: Maximum tokens to generate
            fps: Frames per second for video sampling
            temperature: Sampling temperature (0.0-1.0)
            top_p: Top-p sampling parameter

        Returns:
            Generated description text
        """
        logger.info(f"Generating description for: {Path(video_path).name}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Max tokens: {max_new_tokens}, FPS: {fps}")

        try:
            # Prepare conversation format
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {"type": "text", "text": prompt}
                ]
            }]

            # Apply chat template and tokenize
            logger.info("Processing video and tokenizing...")
            inputs = self.processor.apply_chat_template(
                conversation,
                fps=fps,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate
            logger.info("Generating description...")
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0
                )

            # Decode output
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]

            description = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            logger.info(f"✓ Generated description ({len(description)} chars)")

            return description

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def generate_with_timestamps(
        self,
        video_path: Union[str, Path],
        prompt: str = "Describe what happens in this video, including when events occur.",
        max_new_tokens: int = 512,
        fps: float = 1.0
    ) -> str:
        """
        Generate video description with temporal information

        Args:
            video_path: Path to video file
            prompt: Temporal-focused prompt
            max_new_tokens: Maximum tokens to generate
            fps: Frames per second for video sampling

        Returns:
            Generated description with timestamps
        """
        # Qwen3-VL supports text-timestamp alignment
        temporal_prompt = f"{prompt} Include specific timestamps or time references."

        return self.generate_description(
            video_path=video_path,
            prompt=temporal_prompt,
            max_new_tokens=max_new_tokens,
            fps=fps
        )
