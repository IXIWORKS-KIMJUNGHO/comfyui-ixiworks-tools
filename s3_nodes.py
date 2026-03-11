from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


class S3UploadNode:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_presigned_url": ("STRING", {"default": "", "multiline": False}),
                "text_presigned_url": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "upload"
    CATEGORY = "IXIWORKS/Utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    def _put(self, url, body, content_type):
        """HTTP PUT upload to presigned URL."""
        req = Request(url, data=body, method="PUT")
        req.add_header("Content-Type", content_type)
        resp = urlopen(req, timeout=30)
        return resp.status

    def upload(self, image_presigned_url=None, text_presigned_url=None,
               image=None, text=None):
        import numpy as np
        from PIL import Image as PILImage
        from io import BytesIO

        # Flatten image list: List[Tensor[B,H,W,C]] → List[Tensor[H,W,C]]
        frames = []
        if image is not None:
            for img_tensor in image:
                for i in range(img_tensor.shape[0]):
                    frames.append(img_tensor[i])

        # Normalize URL lists
        img_urls = image_presigned_url if image_presigned_url else []
        txt_urls = text_presigned_url if text_presigned_url else []

        # Normalize text list
        texts = text if text else []

        uploaded = 0
        failed = 0

        # Upload images
        for i, frame in enumerate(frames):
            if i >= len(img_urls):
                print(f"[S3Upload] No presigned URL for image {i}, skipping.")
                break
            url = img_urls[i].strip()
            if not url:
                continue
            buf = BytesIO()
            try:
                img_np = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                PILImage.fromarray(img_np).save(buf, format="PNG")
                self._put(url, buf.getvalue(), "image/png")
                uploaded += 1
                print(f"[S3Upload] Uploaded image {i}")
            except (HTTPError, URLError) as e:
                failed += 1
                print(f"[S3Upload] Failed to upload image {i}: {e}")
            finally:
                buf.close()

        # Upload texts
        for i, txt in enumerate(texts):
            if i >= len(txt_urls):
                print(f"[S3Upload] No presigned URL for text {i}, skipping.")
                break
            url = txt_urls[i].strip()
            if not url:
                continue
            try:
                self._put(url, txt.encode("utf-8"), "text/plain; charset=utf-8")
                uploaded += 1
                print(f"[S3Upload] Uploaded text {i}")
            except (HTTPError, URLError) as e:
                failed += 1
                print(f"[S3Upload] Failed to upload text {i}: {e}")

        print(f"[S3Upload] Done. {uploaded} uploaded, {failed} failed.")
        return ()


NODE_CLASS_MAPPINGS = {
    "UtilS3Upload": S3UploadNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UtilS3Upload": "Upload to S3",
}
