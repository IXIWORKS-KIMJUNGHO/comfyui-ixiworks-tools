class AnyType(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


ANY = AnyType("*")


class SwitchBooleanNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "on_false": (ANY, {"lazy": True}),
                "on_true": (ANY, {"lazy": True}),
                "boolean_switch": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    CATEGORY = "IXIWORKS/Utils"

    def check_lazy_status(self, on_false, on_true, boolean_switch):
        needed = "on_true" if boolean_switch else "on_false"
        if (boolean_switch and on_true is None) or (not boolean_switch and on_false is None):
            return [needed]
        return []

    def switch(self, on_false, on_true, boolean_switch):
        return (on_true if boolean_switch else on_false,)


class StringToListNode:
    MAX_INPUTS = 8

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "count": ("INT", {"default": 4, "min": 1, "max": cls.MAX_INPUTS, "step": 1}),
            "prompt_1": ("STRING", {"default": "", "multiline": True}),
        }
        optional = {
            f"prompt_{i}": ("STRING", {"default": "", "multiline": True})
            for i in range(2, cls.MAX_INPUTS + 1)
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("strings",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert"
    CATEGORY = "IXIWORKS/Utils"

    def convert(self, count, **kwargs):
        result = []
        for i in range(1, count + 1):
            key = f"prompt_{i}"
            value = kwargs.get(key, "").strip()
            if value:
                result.append(value)
        if not result:
            result.append("")
        return (result,)


class JoinStringsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_a": ("STRING", {"forceInput": True}),
                "string_b": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "separator": ("STRING", {"default": " "}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("joined_string",)
    FUNCTION = "join"
    CATEGORY = "IXIWORKS/Utils"

    def join(self, string_a, string_b, separator=" "):
        return (f"{string_a}{separator}{string_b}",)


class SwitchCaseNode:
    MAX_INPUTS = 8

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "count": ("INT", {"default": 3, "min": 2, "max": cls.MAX_INPUTS, "step": 1}),
            "select": ("INT", {"default": 0, "min": 0, "max": cls.MAX_INPUTS - 1, "step": 1}),
        }
        optional = {
            f"input_{i}": (ANY, {"lazy": True})
            for i in range(cls.MAX_INPUTS)
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    CATEGORY = "IXIWORKS/Utils"

    def check_lazy_status(self, count, select, **kwargs):
        index = max(0, min(select, count - 1))
        key = f"input_{index}"
        if kwargs.get(key) is None:
            return [key]
        return []

    def switch(self, count, select, **kwargs):
        index = max(0, min(select, count - 1))
        key = f"input_{index}"
        return (kwargs.get(key, None),)


class SaveFileNode:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subfolder": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "file"}),
            },
            "optional": {
                "text": ("STRING", {"forceInput": True}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "IXIWORKS/Utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    def save(self, subfolder, filename_prefix, text=None, image=None):
        import os
        import re
        import numpy as np
        import folder_paths
        from PIL import Image as PILImage

        # INPUT_IS_LIST: scalar inputs arrive as single-element lists
        subfolder = (subfolder[0] if subfolder else "").strip()
        prefix = (filename_prefix[0] if filename_prefix else "").strip() or "file"

        output_dir = folder_paths.get_output_directory()

        # Subfolder path traversal guard
        if subfolder:
            save_dir = os.path.realpath(os.path.join(output_dir, subfolder))
            if not save_dir.startswith(os.path.realpath(output_dir)):
                print(f"[SaveFile] Invalid subfolder '{subfolder}', falling back to output dir")
                save_dir = output_dir
        else:
            save_dir = output_dir

        os.makedirs(save_dir, exist_ok=True)

        # Flatten image list: List[Tensor[B,H,W,C]] → List[Tensor[H,W,C]]
        frames = []
        if image is not None:
            for img_tensor in image:
                for i in range(img_tensor.shape[0]):
                    frames.append(img_tensor[i])

        total = len(frames) if frames else (len(text) if text else 0)
        if total == 0:
            return ()

        # Normalize texts: broadcast single or 1:1 map
        texts = None
        if text is not None:
            if len(text) == 1:
                texts = [text[0]] * total
            elif len(text) == total:
                texts = text
            else:
                print(f"[SaveFile] Text count ({len(text)}) != image count ({total}), broadcasting first")
                texts = [text[0]] * total

        # Find next counter — scan both .txt and .png
        pattern = re.compile(r"^" + re.escape(prefix) + r"_(\d+)\.(txt|png)$")
        max_num = 0
        for fname in os.listdir(save_dir):
            m = pattern.match(fname)
            if m:
                max_num = max(max_num, int(m.group(1)))

        saved_images = []

        for i in range(total):
            counter = max_num + 1 + i

            if frames:
                filename = f"{prefix}_{counter:04d}.png"
                img_path = os.path.join(save_dir, filename)
                try:
                    img_np = (frames[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    PILImage.fromarray(img_np).save(img_path)
                    print(f"[SaveFile] Saved image: {img_path}")
                    saved_images.append({
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": "output",
                    })
                except Exception as e:
                    print(f"[SaveFile] Error saving image '{img_path}': {e}")

            if texts is not None:
                txt_path = os.path.join(save_dir, f"{prefix}_{counter:04d}.txt")
                try:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(texts[i])
                    print(f"[SaveFile] Saved text: {txt_path}")
                except Exception as e:
                    print(f"[SaveFile] Error saving text '{txt_path}': {e}")

        return {"ui": {"images": saved_images}}


class LoadImageListNode:
    MAX_IMAGES = 20

    @classmethod
    def INPUT_TYPES(cls):
        import os
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        image_exts = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif'}
        files = sorted([
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and os.path.splitext(f)[1].lower() in image_exts
        ])
        file_list = ["[none]"] + files

        required = {
            "count": ("INT", {"default": 1, "min": 1, "max": cls.MAX_IMAGES, "step": 1}),
            "image_1": (file_list,),
        }
        optional = {
            f"image_{i}": (file_list,)
            for i in range(2, cls.MAX_IMAGES + 1)
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load"
    CATEGORY = "IXIWORKS/Utils"

    def load(self, count, image_1, **kwargs):
        import os
        import numpy as np
        import torch
        from PIL import Image
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        slots = [image_1] + [kwargs.get(f"image_{i}", "[none]") for i in range(2, count + 1)]

        images = []
        for name in slots:
            if name == "[none]":
                continue
            path = os.path.join(input_dir, name)
            try:
                img = Image.open(path).convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                images.append(img_tensor)
            except Exception as e:
                print(f"[LoadImageList] Error loading '{name}': {e}")

        if not images:
            images.append(torch.zeros(1, 64, 64, 3))

        return (images,)


class ImageToListNode:
    MAX_INPUTS = 8

    @classmethod
    def INPUT_TYPES(cls):
        # Only count is required, images are added dynamically via JS
        required = {
            "count": ("INT", {"default": 4, "min": 1, "max": cls.MAX_INPUTS, "step": 1}),
        }
        return {"required": required}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert"
    CATEGORY = "IXIWORKS/Utils"

    def convert(self, count, **kwargs):
        result = []
        for i in range(1, count + 1):
            key = f"image_{i}"
            img = kwargs.get(key)
            if img is not None:
                result.append(img)
        if not result:
            import torch
            result.append(torch.zeros(1, 64, 64, 3))
        return (result,)


class JsonToStringListNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "json_string": ("STRING", {"forceInput": True}),
                "key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("strings", "count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "load"
    CATEGORY = "IXIWORKS/Utils"

    def load(self, file_path, json_string=None, key=""):
        import json
        import os
        import folder_paths

        # Load JSON
        if json_string is not None and json_string.strip():
            data = json.loads(json_string)
        elif file_path.strip():
            path = file_path if os.path.isabs(file_path) else os.path.join(folder_paths.get_input_directory(), file_path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            return ([""], 0)

        # Navigate dot-notation key
        node = data
        if key.strip():
            for part in key.strip().split("."):
                if isinstance(node, dict):
                    node = node.get(part)
                elif isinstance(node, list):
                    try:
                        node = node[int(part)]
                    except (ValueError, IndexError):
                        node = None
                if node is None:
                    break

        # Flatten to string list
        if isinstance(node, dict):
            values = list(node.values())
        elif isinstance(node, list):
            values = node
        else:
            values = [node]

        result = []
        for v in values:
            if v is None:
                continue
            result.append(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))

        if not result:
            return ([""], 0)

        print(f"[JsonToStringList] Loaded {len(result)} items (key='{key}')")
        return (result, len(result))


ASPECT_RATIOS = {
    "21:9": (21, 9),
    "1.85:1": (1.85, 1),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
}


class EmptyLatentRatioNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ratio": (list(ASPECT_RATIOS.keys()), {"default": "16:9"}),
                "long_side": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 64,
                }),
                "channels": (["16", "4"], {"default": "16"}),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "IXIWORKS/Utils"

    def generate(self, ratio, long_side, channels, batch_size):
        import torch

        w_ratio, h_ratio = ASPECT_RATIOS[ratio]

        if w_ratio >= h_ratio:
            width = long_side
            height = int(long_side * h_ratio / w_ratio)
        else:
            height = long_side
            width = int(long_side * w_ratio / h_ratio)

        # Ensure divisible by 8 (latent space requirement)
        width = (width // 8) * 8
        height = (height // 8) * 8

        c = int(channels)
        latent = torch.zeros([batch_size, c, height // 8, width // 8])
        return ({"samples": latent}, width, height)


class BypassNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bypass": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input": (ANY,),
            }
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "IXIWORKS/Utils"

    def execute(self, bypass, **kwargs):
        return (kwargs.get("input", None),)


NODE_CLASS_MAPPINGS = {
    "UtilSwitch": SwitchBooleanNode,
    "UtilStringToList": StringToListNode,
    "UtilConcatStrings": JoinStringsNode,
    "UtilSwitchCase": SwitchCaseNode,
    "UtilSaveFile": SaveFileNode,
    "UtilLoadImageList": LoadImageListNode,
    "UtilImageToList": ImageToListNode,
    "UtilBypass": BypassNode,
    "UtilEmptyLatent": EmptyLatentRatioNode,
    "UtilJsonToList": JsonToStringListNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UtilSwitch": "Switch",
    "UtilStringToList": "String to List",
    "UtilConcatStrings": "Concat Strings",
    "UtilSwitchCase": "Switch Case",
    "UtilSaveFile": "Save Text & Image",
    "UtilLoadImageList": "Load Image List",
    "UtilImageToList": "Image to List",
    "UtilBypass": "Bypass",
    "UtilEmptyLatent": "Empty Latent",
    "UtilJsonToList": "JSON to List",
}
