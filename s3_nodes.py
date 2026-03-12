import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


class S3UploadNode:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_presigned_url": ("STRING", {"default": "", "multiline": True}),
                "text_presigned_url": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "text": ("STRING", {"forceInput": True}),
                "callback_url": ("STRING", {"default": ""}),
                "callback_job_id": ("STRING", {"default": ""}),
                "callback_scene_number": ("STRING", {"default": ""}),
                "callback_token": ("STRING", {"default": ""}),
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

    def _post_callback(self, callback_url, job_id, scene_number, token, uploaded, failed):
        """Send completion callback to backend."""
        payload = json.dumps({
            "storyboard_job_id": job_id,
            "scene_number": scene_number,
            "token": token,
            "uploaded": uploaded,
            "failed": failed,
        }).encode("utf-8")
        req = Request(callback_url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        resp = urlopen(req, timeout=30)
        return resp.status

    def upload(self, image_presigned_url="", text_presigned_url="",
               image=None, text=None,
               callback_url="", callback_job_id="", callback_scene_number="",
               callback_token=""):
        import numpy as np
        from PIL import Image as PILImage
        from io import BytesIO

        # INPUT_IS_LIST: scalar inputs arrive as single-element lists
        if isinstance(image_presigned_url, list):
            image_presigned_url = image_presigned_url[0] if image_presigned_url else ""
        if isinstance(text_presigned_url, list):
            text_presigned_url = text_presigned_url[0] if text_presigned_url else ""
        if isinstance(callback_url, list):
            callback_url = callback_url[0] if callback_url else ""
        if isinstance(callback_job_id, list):
            callback_job_id = callback_job_id[0] if callback_job_id else ""
        if isinstance(callback_scene_number, list):
            callback_scene_number = callback_scene_number[0] if callback_scene_number else ""
        if isinstance(callback_token, list):
            callback_token = callback_token[0] if callback_token else ""

        # Flatten image list: List[Tensor[B,H,W,C]] → List[Tensor[H,W,C]]
        frames = []
        if image is not None:
            if isinstance(image, list):
                for img_tensor in image:
                    for i in range(img_tensor.shape[0]):
                        frames.append(img_tensor[i])
            else:
                for i in range(image.shape[0]):
                    frames.append(image[i])

        # Parse newline-separated URLs
        img_urls = [u for u in image_presigned_url.strip().split("\n") if u.strip()] if image_presigned_url.strip() else []
        txt_urls = [u for u in text_presigned_url.strip().split("\n") if u.strip()] if text_presigned_url.strip() else []

        # Normalize text list: INPUT_IS_LIST makes text a list of strings
        texts = []
        if text is not None:
            if isinstance(text, list):
                texts = text
            else:
                texts = [text]

        total_frames = len(frames)
        total_texts = len(texts)
        print(f"[S3Upload] Processing: {total_frames} images, {total_texts} texts, "
              f"{len(img_urls)} image URLs, {len(txt_urls)} text URLs")

        # Prepare upload tasks
        tasks = []

        for i, frame in enumerate(frames):
            if i >= len(img_urls):
                print(f"[S3Upload] No presigned URL for image {i}, skipping.")
                break
            url = img_urls[i].strip()
            if not url:
                continue
            img_np = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            buf = BytesIO()
            PILImage.fromarray(img_np).save(buf, format="PNG")
            tasks.append(("image", i, url, buf.getvalue(), "image/png"))
            buf.close()

        for i, txt in enumerate(texts):
            if i >= len(txt_urls):
                print(f"[S3Upload] No presigned URL for text {i}, skipping.")
                break
            url = txt_urls[i].strip()
            if not url:
                continue
            tasks.append(("text", i, url, txt.encode("utf-8"), "text/plain; charset=utf-8"))

        # Upload all in parallel
        uploaded = 0
        failed = 0

        def _do_upload(task):
            kind, idx, url, body, content_type = task
            self._put(url, body, content_type)
            return kind, idx

        with ThreadPoolExecutor(max_workers=min(len(tasks), 8) or 1) as pool:
            futures = {pool.submit(_do_upload, t): t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                kind, idx = task[0], task[1]
                try:
                    future.result()
                    uploaded += 1
                    print(f"[S3Upload] Uploaded {kind} {idx}")
                except Exception as e:
                    failed += 1
                    print(f"[S3Upload] Failed to upload {kind} {idx}: {e}")

        print(f"[S3Upload] Done. {uploaded} uploaded, {failed} failed.")

        # Send completion callback
        if callback_url.strip():
            try:
                status = self._post_callback(
                    callback_url.strip(), callback_job_id, callback_scene_number,
                    callback_token, uploaded, failed)
                print(f"[S3Upload] Callback sent to {callback_url} (status {status})")
            except Exception as e:
                print(f"[S3Upload] Callback failed: {e}")

        return ()


NODE_CLASS_MAPPINGS = {
    "UtilS3Upload": S3UploadNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UtilS3Upload": "Upload to S3",
}
