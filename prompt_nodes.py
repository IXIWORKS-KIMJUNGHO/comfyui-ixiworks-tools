DEFAULT_STYLE_PREFIX = """\
Professional film storyboard artist. Rough pencil sketch, gray wash, monochrome only.
Single rectangular panel. Gestural figures, not anatomically detailed.
Always respect the prompt's explicit shot size — never pull back beyond it.
Default to medium-wide if no shot size specified.\
"""

_SCENE_THINK_SYSTEM = """\
You are a film storyboard director. Plan the composition, then verify it.

RULES:
• Single panel, single unified scale — NEVER mix large foreground face with small distant figures.
• Match the prompt's exact shot size — never pull wider than requested.
• Keep it minimal — short plans make better images.

Output exactly 3 lines:
Shot: [shot size matching prompt — what body range is visible]
Composition: [spatial arrangement, single consistent scale]
Subject: [key subject and action, 10 words max]\
"""

_CHARACTER_THINK_SYSTEM = """\
You are a character consistency specialist for storyboard production.
Extract the key visual features from a character description that must be preserved across every panel.
Output only a single line starting with "Key features to preserve:" followed by a comma-separated list of the most critical visual attributes.
No preamble, no extra text.\
"""


class ZImagePromptBuilderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene": ("STRING", {"default": "", "multiline": True}),
                "style_prefix": ("STRING", {"default": DEFAULT_STYLE_PREFIX, "multiline": True}),
                "api_key": ("STRING", {"default": "", "password": True}),
                "use_think": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "character": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_text",)
    FUNCTION = "build"
    CATEGORY = "IXIWORKS/Prompt"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    def build(self, scene, style_prefix, api_key, use_think, character=None):
        if not scene.strip():
            print("[ZImagePromptBuilder] Warning: scene is empty")
            return ("",)

        turns = []

        # 1. System turn
        if style_prefix.strip():
            turns.append(f"<|im_start|>system\n{style_prefix.strip()}<|im_end|>")

        # 2. Character turn (optional)
        if character and character.strip():
            turns.append(f"<|im_start|>user\n{character.strip()}<|im_end|>")
            if use_think:
                think = self._generate_think(
                    _CHARACTER_THINK_SYSTEM,
                    character.strip(),
                    api_key,
                    fallback=f"Key features to preserve: {character.strip()}"
                )
                turns.append(f"<|im_start|>assistant\n<think>\n{think}\n</think>\n<|im_end|>")
            else:
                turns.append("<|im_start|>assistant\n<|im_end|>")

        # 3. Scene turn
        turns.append(f"<|im_start|>user\n{scene.strip()}<|im_end|>")

        # 4. Think: single-pass composition plan
        if use_think and not (character and character.strip()):
            think = self._generate_think(
                _SCENE_THINK_SYSTEM,
                scene.strip(),
                api_key,
                fallback=scene.strip()[:120],
                model="claude-haiku-4-5-20251001"
            )
            turns.append(f"<|im_start|>assistant\n<think>\n{think}\n</think>\n<|im_end|>")
        else:
            # With character: scene assistant turn is intentionally empty.
            # Character think already primes Qwen for subject consistency.
            # Adding scene think would exceed token budget and dilute character features.
            turns.append("<|im_start|>assistant\n<|im_end|>")

        return ("\n\n".join(turns),)

    def _generate_think(self, system_prompt: str, user_content: str, api_key: str, fallback: str, model: str = "claude-sonnet-4-6") -> str:
        key = api_key.strip()
        if not key:
            print("[ZImagePromptBuilder] No API key — using fallback think")
            return fallback

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            response = client.messages.create(
                model=model,
                max_tokens=150,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}]
            )
            result = response.content[0].text.strip()
            print(f"[ZImagePromptBuilder] Think generated: {result[:80]}...")
            return result
        except ImportError:
            print("[ZImagePromptBuilder] anthropic package not installed — using fallback think")
            return fallback
        except Exception as e:
            print(f"[ZImagePromptBuilder] API error: {e} — using fallback think")
            return fallback


NODE_CLASS_MAPPINGS = {
    "ZImagePromptBuilder": ZImagePromptBuilderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImagePromptBuilder": "Z-Image Prompt Builder",
}
