import torch
import numpy as np
from PIL import Image


class CropAndResizeNode:
    RATIOS = {
        "21:9": (21, 9),
        "1.85:1": (185, 100),
        "16:9": (16, 9),
        "9:16": (9, 16),
        "1:1": (1, 1),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (list(cls.RATIOS.keys()), {"default": "1:1"}),
                "long_side": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop_and_resize"
    CATEGORY = "IXIWORKS/Image"

    def crop_and_resize(self, image, aspect_ratio, long_side):
        if image is None:
            raise ValueError("ImageCropResize: image input is required")

        ratio_w, ratio_h = self.RATIOS[aspect_ratio]
        target_ratio = ratio_w / ratio_h

        try:
            results = []
            for i in range(image.shape[0]):
                img = image[i]  # (H, W, C)
                h, w = img.shape[0], img.shape[1]
                img_ratio = w / h

                # Calculate crop size that fits within the image
                if target_ratio > img_ratio:
                    # Target is wider → use full width, crop height
                    crop_w = w
                    crop_h = round(w / target_ratio)
                else:
                    # Target is taller → use full height, crop width
                    crop_h = h
                    crop_w = round(h * target_ratio)

                # Center crop
                x = (w - crop_w) // 2
                y = (h - crop_h) // 2
                cropped = img[y:y + crop_h, x:x + crop_w, :]

                # Resize so long side matches target (round to multiple of 8)
                if crop_w >= crop_h:
                    new_w = long_side
                    new_h = max(8, round(long_side * crop_h / crop_w / 8) * 8)
                else:
                    new_h = long_side
                    new_w = max(8, round(long_side * crop_w / crop_h / 8) * 8)

                # PIL resize with LANCZOS for quality
                pil_img = Image.fromarray(
                    (cropped.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                )
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                result = torch.from_numpy(
                    np.array(pil_img).astype(np.float32) / 255.0
                )
                results.append(result)
        except Exception as e:
            raise RuntimeError(f"ImageCropResize: processing failed: {e}") from e

        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "ImageCropResize": CropAndResizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropResize": "Crop & Resize",
}
