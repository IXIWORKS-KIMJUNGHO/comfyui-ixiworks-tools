"""
Video processing utilities
Frame extraction and video preparation for Qwen3-VL
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Video frame extraction and processing for vision-language models
    """

    @staticmethod
    def extract_frames(
        video_path: Union[str, Path],
        fps: float = 1.0,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video at specified FPS

        Args:
            video_path: Path to video file
            fps: Frames per second to extract (e.g., 1.0 = 1 frame per second)
            max_frames: Maximum number of frames to extract (None = no limit)

        Returns:
            List of frames as numpy arrays (RGB format)
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Extracting frames from: {video_path.name}")
        logger.info(f"Target FPS: {fps}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps if video_fps > 0 else 0

            logger.info(f"Video FPS: {video_fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")

            # Calculate frame interval
            frame_interval = int(video_fps / fps) if fps > 0 else 1

            frames = []
            frame_count = 0
            extracted_count = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Extract frame at specified interval
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1

                    # Check max frames limit
                    if max_frames and extracted_count >= max_frames:
                        logger.info(f"Reached max frames limit: {max_frames}")
                        break

                frame_count += 1

            logger.info(f"✓ Extracted {len(frames)} frames")

            return frames

        finally:
            cap.release()

    @staticmethod
    def get_video_info(video_path: Union[str, Path]) -> dict:
        """
        Get video metadata

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video information
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            return {
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration": duration,
                "path": str(video_path)
            }

        finally:
            cap.release()

    @staticmethod
    def validate_video(video_path: Union[str, Path]) -> bool:
        """
        Check if video file is valid and readable

        Args:
            video_path: Path to video file

        Returns:
            True if valid, False otherwise
        """
        try:
            video_path = Path(video_path)

            if not video_path.exists():
                logger.error(f"Video file not found: {video_path}")
                return False

            cap = cv2.VideoCapture(str(video_path))
            is_valid = cap.isOpened()
            cap.release()

            return is_valid

        except Exception as e:
            logger.error(f"Error validating video: {e}")
            return False
