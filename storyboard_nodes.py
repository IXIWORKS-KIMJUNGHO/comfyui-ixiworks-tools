"""
StoryBoard Nodes for ComfyUI
JSON parsing and prompt building for storyboard workflows
"""

import os
import logging
import json

# ComfyUI imports
import folder_paths

logger = logging.getLogger(__name__)

# Register prompt folder for JSON files
PROMPT_FOLDER = os.path.join(folder_paths.get_input_directory(), "prompt")
os.makedirs(PROMPT_FOLDER, exist_ok=True)
folder_paths.folder_names_and_paths["storyboard_prompts"] = (
    [PROMPT_FOLDER],
    {".json"},
)


class JsonParserNode:
    @classmethod
    def INPUT_TYPES(s):
        files = folder_paths.get_filename_list("storyboard_prompts")
        return {
            "required": {
                "JSON": (sorted(files) if files else ["(no files)"],),
            },
        }

    RETURN_TYPES = ("STRING",) * 14 + ("INT",)
    RETURN_NAMES = (
        # Scene level
        "time_of_day", "weather", "mood", "location",
        # Cut level
        "description", "camera_shot", "camera_angle", "composition",
        "camera_movement", "lens_type", "lighting_style", "lighting_direction",
        # Derived
        "focus_subject", "character_prompt",
        "count",
    )
    FUNCTION = "parse"
    CATEGORY = "IXIWORKS/StoryBoard"
    OUTPUT_IS_LIST = (True,) * 14 + (False,)

    @classmethod
    def IS_CHANGED(s, JSON):
        file_path = folder_paths.get_full_path("storyboard_prompts", JSON)
        if file_path and os.path.exists(file_path):
            return os.path.getmtime(file_path)
        return float("nan")

    @staticmethod
    def _build_character_prompt(character_data, focus_subject):
        """Build character prompt from character dict, focus subject first."""
        if not character_data:
            return ""

        def _format_char(name, char):
            parts = [name]
            if char.get("appearance"):
                parts.append(char["appearance"])
            if char.get("costume"):
                parts.append(char["costume"])
            return ", ".join(parts)

        segments = []
        # Focus subject first
        if focus_subject and focus_subject in character_data:
            segments.append(_format_char(focus_subject, character_data[focus_subject]))
        # Remaining characters
        for name, char in character_data.items():
            if name == focus_subject:
                continue
            segments.append(_format_char(name, char))

        return ". ".join(segments)

    OUTPUT_NODE = True

    @staticmethod
    def _build_preview(scene_data, character_data):
        """Build human-readable preview text."""
        lines = []
        for scene_key in sorted(scene_data.keys(), key=lambda x: int(x) if x.isdigit() else x):
            scene = scene_data[scene_key]
            if not isinstance(scene, dict):
                continue
            title = scene.get("title", "")
            ctx = ", ".join(filter(None, [scene.get("timeOfDay", ""), scene.get("weather", "")]))
            header = f"Scene {scene_key}"
            if title:
                header += f": {title}"
            if ctx:
                header += f"  ({ctx})"
            lines.append(header)

            cuts = scene.get("cuts", {})
            for cut_key in sorted(cuts.keys(), key=lambda x: int(x) if x.isdigit() else x):
                cut = cuts[cut_key]
                if not isinstance(cut, dict):
                    continue
                shot = cut.get("cameraShot", "")
                desc = cut.get("description", "")
                if len(desc) > 40:
                    desc = desc[:37] + "..."
                parts = filter(None, [shot, desc])
                lines.append(f"  Cut {cut_key}: {', '.join(parts)}")

        if character_data:
            names = ", ".join(character_data.keys())
            lines.append(f"Characters: {names}")

        MAX_PREVIEW_LINES = 15
        if len(lines) > MAX_PREVIEW_LINES:
            lines = lines[:MAX_PREVIEW_LINES] + [f"  ... and {len(lines) - MAX_PREVIEW_LINES} more"]

        return "\n".join(lines)

    def parse(self, JSON):
        file_path = folder_paths.get_full_path("storyboard_prompts", JSON)
        if not file_path or not os.path.exists(file_path):
            logger.error(f"[StoryBoard] JsonParserNode: File not found '{JSON}'")
            empty = [""]
            return {"ui": {"preview": ["File not found"]}, "result": (*([empty] * 14), 0)}

        logger.info(f"[StoryBoard] JsonParserNode: file path '{file_path}'")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            scene_data = data.get("scene", {})
            character_data = data.get("character", {})

            # Per-cut output lists
            descriptions = []
            times = []
            weathers = []
            moods = []
            locations = []
            camera_shots = []
            camera_angles = []
            compositions = []
            camera_movements = []
            lens_types = []
            lighting_styles = []
            lighting_directions = []
            focus_subjects = []
            character_prompts = []

            for scene_key in sorted(scene_data.keys(), key=lambda x: int(x) if x.isdigit() else x):
                scene = scene_data[scene_key]
                if not isinstance(scene, dict):
                    continue

                # Scene-level fields (shared by all cuts in this scene)
                s_time = scene.get("timeOfDay", "")
                s_weather = scene.get("weather", "")
                s_mood = scene.get("mood", "")
                s_location = scene.get("location", "")

                cuts = scene.get("cuts", {})
                for cut_key in sorted(cuts.keys(), key=lambda x: int(x) if x.isdigit() else x):
                    cut = cuts[cut_key]
                    if not isinstance(cut, dict):
                        continue

                    descriptions.append(cut.get("description", ""))
                    times.append(s_time)
                    weathers.append(s_weather)
                    moods.append(s_mood)
                    locations.append(s_location)
                    camera_shots.append(cut.get("cameraShot", ""))
                    camera_angles.append(cut.get("cameraAngle", ""))
                    compositions.append(cut.get("composition", ""))
                    camera_movements.append(cut.get("cameraMovement", ""))
                    lens_types.append(cut.get("lensType", ""))
                    lighting_styles.append(cut.get("lightingStyle", ""))
                    lighting_directions.append(cut.get("lightingDirection", ""))

                    focus = cut.get("focusSubject", "")
                    focus_subjects.append(focus)
                    character_prompts.append(
                        self._build_character_prompt(character_data, focus)
                    )

            count = len(descriptions)
            preview = self._build_preview(scene_data, character_data)

            if count == 0:
                logger.warning("[StoryBoard] JsonParserNode: No cuts found in JSON")
                empty = [""]
                return {"ui": {"preview": [preview or "No cuts found"]}, "result": (*([empty] * 14), 0)}

            logger.info(f"[StoryBoard] JsonParserNode: Parsed {count} cuts from {len(scene_data)} scenes")

            return {
                "ui": {"preview": [preview]},
                "result": (
                    # Scene level
                    times, weathers, moods, locations,
                    # Cut level
                    descriptions, camera_shots, camera_angles, compositions,
                    camera_movements, lens_types, lighting_styles, lighting_directions,
                    # Derived
                    focus_subjects, character_prompts,
                    count,
                ),
            }

        except Exception as e:
            logger.error(f"[StoryBoard] JsonParserNode: Error reading file: {e}")
            empty = [""]
            return {"ui": {"preview": [f"Error: {e}"]}, "result": (*([empty] * 14), 0)}


LORA_STYLE_PRESETS = {
    "inksketch": {
        "name": "Ink Sketch",
        "color_mode": "inksketch",  # specific handling for ink sketch
        "add_keywords": "monochrome, high contrast linework, detailed crosshatching, sketch texture",
    },
    "ink-wash": {
        "name": "Ink Wash",
        "color_mode": "inkwash",  # specific handling for ink wash
        "add_keywords": "atmospheric tonal depth, grayscale",
    },
    "pen-ink-illustration": {
        "name": "Pen & Ink Illustration",
        "color_mode": "penink",  # specific handling
        "add_keywords": "black and white, high contrast ink-illustration",
    },
    "ink-watercolor": {
        "name": "Ink and Watercolor",
        "color_mode": "inkwatercolor",  # specific handling
        "add_keywords": "warm earthy tones, soft watercolor background",
    },
    "watercolor-illustration": {
        "name": "Watercolor Illustration",
        "color_mode": "watercolorillust",  # specific handling
        "add_keywords": "vibrant warm palette, gentle wash effects with soft color bleeding",
    },
}

CONTROLNET_FILTER_PRESETS = {
    "pose": {
        "extra_removals": [
            "Body postures: standing, sitting, kneeling, lying down, crouching, leaning, bending, squatting",
            "Actions/movements: walking, running, jumping, dancing, waving, reaching, stretching",
            "Arm positions: arms crossed, arms raised, arms akimbo, hands on hips, pointing, holding, arms behind back",
            "Hand gestures: peace sign, thumbs up, fist, open palm, clasped hands",
            "Leg positions: legs crossed, one leg raised, spread legs, kicking",
            "Head/gaze direction: looking at viewer, looking away, looking up/down/left/right, head tilted, turned head",
            "Full body descriptors: full body pose, dynamic pose, action pose, relaxed pose, tense pose",
        ],
        "keep_extras": [
            "Facial expressions: smiling, serious, surprised, angry, sad",
            "Depth cues: bokeh, depth of field, shallow/deep focus, foreground/background blur",
            "Clothing and outfit descriptions",
            "Scene descriptions, lighting, colors, background",
        ],
    },
    "depth": {
        "extra_removals": [
            "Depth-of-field effects: bokeh, shallow focus, deep focus, tilt-shift, rack focus",
            "Foreground/background blur, lens blur, gaussian blur",
            "Depth cues: foreground element, background separation, layered depth",
        ],
        "keep_extras": [
            "Body postures and actions: standing, sitting, walking, dynamic pose, etc.",
            "Gaze direction: looking at viewer, looking away, etc.",
            "Facial expressions, clothing, colors, lighting",
            "Scene descriptions and background details",
        ],
    },
    "canny": {
        "extra_removals": [],
        "keep_extras": [
            "Body postures and actions: standing, sitting, walking, dynamic pose, etc.",
            "Gaze direction: looking at viewer, looking away, etc.",
            "Depth cues: bokeh, depth of field, shallow/deep focus",
            "Facial expressions, clothing, colors, lighting",
            "Scene descriptions and background details",
        ],
    },
}


class SBPromptFilter:
    """Filter prompt to remove conflicting keywords for specific LoRA styles or ControlNet types using Claude API."""

    SEPARATOR = "\n---PROMPT_SEPARATOR---\n"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "filter_mode": (["style", "controlnet"],),
                "style_type": (list(LORA_STYLE_PRESETS.keys()),),
                "controlnet_type": (list(CONTROLNET_FILTER_PRESETS.keys()),),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_prompt",)
    FUNCTION = "filter_prompt"
    CATEGORY = "IXIWORKS/StoryBoard"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    @staticmethod
    def _build_controlnet_system_prompt(preset_key):
        preset = CONTROLNET_FILTER_PRESETS[preset_key]

        # Common removals for all ControlNet types
        common_removals = [
            "Camera angles/perspective: low angle, high angle, eye level, bird's eye view, worm's eye view, dutch angle, overhead shot",
            "Camera framing/shot types: wide shot, medium shot, close-up, extreme close-up, full shot, cowboy shot, establishing shot",
            "Spatial composition: rule of thirds, centered, off-center, leading lines, symmetrical, diagonal composition, golden ratio",
            "Subject orientation: from behind, from side, profile view, back view, three-quarter view, frontal view, facing camera",
            "Number/arrangement of people: 1girl, 2boys, group, crowd, solo, duo, trio, multiple girls/boys",
        ]

        # Preset-specific extra removals
        extra_removals = preset["extra_removals"]

        # Build removal section
        all_removals = common_removals + extra_removals
        removal_lines = "\n".join(f"- {r}" for r in all_removals)

        # Build keep section
        keep_lines = "\n".join(f"- {k}" for k in preset["keep_extras"])

        return f"""You are a prompt optimizer that removes spatial/compositional keywords when using {preset_key} ControlNet.

Principle: ControlNet handles spatial information (composition, angles, arrangement). The prompt should focus on SUBJECT details (appearance, expression, clothing, atmosphere).

## REMOVE these keywords (ControlNet handles them):
{removal_lines}

## KEEP these (do NOT remove):
{keep_lines}

## Rules:
- Remove keywords smoothly; do not leave trailing commas or awkward gaps
- Keep the prompt natural and coherent after removing keywords
- Do NOT add new keywords that were not in the original prompt

## Output Format:
- Multiple prompts separated by "---PROMPT_SEPARATOR---"
- Format: [1] filtered_prompt"""

    def _call_api(self, system_prompt, user_content, prompts, key, label="filter"):
        """Common API call + response parsing logic."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)

            logger.info(f"[StoryBoard] SBPromptFilter: Filtering {len(prompts)} prompts ({label})")

            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_content}
                ]
            )

            result_text = response.content[0].text.strip()

            # Split and parse results
            filtered_results = []
            parts = result_text.split("---PROMPT_SEPARATOR---")

            for part in parts:
                part = part.strip()
                if part.startswith("["):
                    idx = part.find("]")
                    if idx != -1:
                        part = part[idx+1:].strip()
                if part:
                    filtered_results.append(part)

            if len(filtered_results) != len(prompts):
                logger.warning(f"[StoryBoard] SBPromptFilter: Output count mismatch ({len(filtered_results)} vs {len(prompts)}), returning originals")
                return None

            for i, (orig, filtered) in enumerate(zip(prompts, filtered_results)):
                orig_preview = orig[:50] + "..." if len(orig) > 50 else orig
                filtered_preview = filtered[:80] + "..." if len(filtered) > 80 else filtered
                logger.info(f"[StoryBoard] SBPromptFilter: [{i+1}] {orig_preview} → {filtered_preview}")

            logger.info(f"[StoryBoard] SBPromptFilter: Done - {len(filtered_results)} prompts filtered ({label})")
            return filtered_results

        except ImportError:
            logger.error("[StoryBoard] SBPromptFilter: anthropic package not installed")
            return None
        except Exception as e:
            logger.error(f"[StoryBoard] SBPromptFilter: API error - {e}")
            return None

    def _filter_style(self, prompts, style, key):
        style_info = LORA_STYLE_PRESETS.get(style, {})
        style_name = style_info.get("name", style)
        color_mode = style_info.get("color_mode", "full-color")

        if color_mode == "full-color":
            return (prompts,)

        add_keywords = style_info.get("add_keywords", "")
        combined_input = self.SEPARATOR.join([f"[{i+1}] {p}" for i, p in enumerate(prompts)])

        remove_realistic = """
## REMOVE photorealistic keywords (CRITICAL):
- photo realistic, photorealistic, realistic, hyper realistic, hyperrealistic
- photograph, photo, photography, RAW photo
- DSLR, 8k uhd, 4k, ultra realistic, lifelike
- cinematic photo, film grain (unless style-appropriate)
These keywords override LoRA styles and MUST be removed."""

        keep_instruction = """
## ALWAYS KEEP (never modify these):
- Composition: rule of thirds, leading lines, centered, off-center, etc.
- Camera angle: wide shot, medium shot, close-up, eye-level, low angle, high angle, etc.
- Camera position descriptions
- Subject pose and position descriptions
- Framing descriptions"""

        output_format = """
## Output Format:
- Multiple prompts separated by "---PROMPT_SEPARATOR---"
- Format: [1] filtered_prompt
- Place style-defining keywords at the START of each prompt"""

        style_transforms = {
            "inksketch": f"""You are a prompt optimizer for ink sketch style image generation.
{remove_realistic}

## Transform these elements for ink sketch style:

### Colors → REMOVE ALL:
- Remove all color adjectives: beige, olive, white, red, blue, golden, etc.
- "beige hoodie" → "hoodie", "red car" → "car"

### Lighting → Convert to contrast-based:
- "golden hour" → "low angle dramatic lighting"
- "warm glow" → "strong directional lighting"
- "bathed in sunlight" → "harsh angular shadows"
- Keep: shadows, contrast, silhouette
{keep_instruction}
{output_format}""",

            "inkwash": f"""You are a prompt optimizer for ink wash style image generation.
{remove_realistic}

## Transform these elements for ink wash style:

### Colors:
- KEEP muted colors: beige, olive, gray, brown
- REMOVE vivid colors: neon, vibrant, bright saturated
- REMOVE pure white/black adjectives from objects

### Lighting → Convert to atmospheric:
- "bright sunlight" → "morning light casting long diagonal shadows"
- Add: "wet pavement", "puddle reflections"

### Special: CONDENSE the prompt
- Remove verbose explanations, keep core descriptions
{keep_instruction}
{output_format}""",

            "penink": f"""You are a prompt optimizer for pen and ink illustration style image generation.
{remove_realistic}

## Transform these elements for pen & ink style:

### Colors:
- KEEP muted colors: beige, olive, white, gray, brown
- REMOVE vivid/neon colors

### Lighting → Convert with crosshatching:
- "bathed in glow" → "low angle afternoon light casting long dramatic shadows"
- "well-defined shadows" → "well-defined shadows with crosshatching"

### Technique → ENHANCE details:
- "brick buildings" → "detailed brick buildings"
- "cafe sign" → "hanging cafe sign"
- "city street" → "cobblestone city street"
- Add: "intricate linework on architectural details"
{keep_instruction}
{output_format}""",

            "inkwatercolor": f"""You are a prompt optimizer for ink and watercolor illustration style image generation.
{remove_realistic}

## Transform these elements for ink watercolor style:

### Colors:
- KEEP ALL warm colors: golden hour, warm glow, warm sunlight, amber
- KEEP muted colors: beige, olive, brown, earthy
- REMOVE only: neon, cold harsh colors

### Lighting → Keep warm, enhance atmosphere:
- Keep all warm lighting descriptions
- Add: "loose ink linework with watercolor washes"

### Technique → ENHANCE with charm:
- "city street" → "dusty city street"
- "brick buildings" → "charming brick buildings"
- "cafe sign" → "hand-painted cafe sign"
- Consider adding: "a stray dog nearby", "pigeons"
{keep_instruction}
{output_format}""",

            "watercolorillust": f"""You are a prompt optimizer for watercolor illustration style image generation.
{remove_realistic}

## Transform these elements for watercolor illustration style:

### Colors → KEEP ALL:
- Keep all colors including vibrant (vibrant, colorful, golden, warm)

### Lighting → Keep and enhance:
- Keep all warm/natural lighting
- Add: "dappled light filtering through plane trees"

### Technique → ENHANCE with color:
- "city street" → "sunlit cobblestone city street"
- "brick buildings" → "colorful brick buildings"
- "cafe sign" → "charming cafe sign with striped awning"
- Add: "wet-on-wet blending with soft edges"
{keep_instruction}
{output_format}""",
        }

        system_prompt = style_transforms.get(color_mode)
        if not system_prompt:
            return (prompts,)

        filtered = self._call_api(
            system_prompt,
            f"Filter these prompts:\n\n{combined_input}",
            prompts, key, label=f"style '{style_name}'"
        )
        if filtered is None:
            return (prompts,)

        if add_keywords:
            filtered = [f"{add_keywords}, {r}" for r in filtered]
            logger.info(f"[StoryBoard] SBPromptFilter: Prepended keywords: '{add_keywords}'")

        return (filtered,)

    def _filter_controlnet(self, prompts, cn_type, key):
        system_prompt = self._build_controlnet_system_prompt(cn_type)
        combined_input = self.SEPARATOR.join([f"[{i+1}] {p}" for i, p in enumerate(prompts)])

        filtered = self._call_api(
            system_prompt,
            f"Filter these prompts for {cn_type} ControlNet usage:\n\n{combined_input}",
            prompts, key, label=f"controlnet '{cn_type}'"
        )
        if filtered is None:
            return (prompts,)

        return (filtered,)

    def filter_prompt(self, prompt, filter_mode, style_type, controlnet_type, api_key):
        # Handle list inputs
        mode = filter_mode[0] if isinstance(filter_mode, list) else filter_mode
        style = style_type[0] if isinstance(style_type, list) else style_type
        cn_type = controlnet_type[0] if isinstance(controlnet_type, list) else controlnet_type
        key = api_key[0] if isinstance(api_key, list) else api_key
        prompts = prompt if isinstance(prompt, list) else [prompt]

        if not key:
            logger.warning("[StoryBoard] SBPromptFilter: No API key provided, returning original prompts")
            return (prompts,)

        if mode == "style":
            return self._filter_style(prompts, style, key)
        else:
            return self._filter_controlnet(prompts, cn_type, key)


MAX_CONCAT_INPUTS = 20


class SBConcatStrings:
    """Concatenate multiple STRING inputs with a separator. Dynamic input count via JS."""

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "num_inputs": ("INT", {"default": 2, "min": 2, "max": MAX_CONCAT_INPUTS}),
                "separator": ("STRING", {"default": ", "}),
            },
            "optional": {},
        }
        for i in range(1, MAX_CONCAT_INPUTS + 1):
            inputs["optional"][f"string_{i}"] = ("STRING", {"forceInput": True})
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "concat"
    CATEGORY = "IXIWORKS/StoryBoard"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def concat(self, num_inputs, separator, **kwargs):
        sep = separator[0] if isinstance(separator, list) else separator
        count = num_inputs[0] if isinstance(num_inputs, list) else num_inputs

        # Collect connected string lists in order
        string_lists = []
        for i in range(1, count + 1):
            val = kwargs.get(f"string_{i}")
            if val is None:
                continue
            if not isinstance(val, list):
                val = [val]
            string_lists.append(val)

        if not string_lists:
            return ([""],)

        # Element-wise concatenation (broadcast shorter lists with last element)
        max_len = max(len(sl) for sl in string_lists)
        results = []
        for idx in range(max_len):
            parts = []
            for sl in string_lists:
                s = sl[idx] if idx < len(sl) else sl[-1]
                if s.strip():
                    parts.append(s)
            results.append(sep.join(parts))

        return (results,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SBJsonParser": JsonParserNode,
    "SBPromptFilter": SBPromptFilter,
    "SBConcatStrings": SBConcatStrings,
}

# Display name mappings for ComfyUI UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "SBJsonParser": "SB JSON Parser",
    "SBPromptFilter": "SB Prompt Filter",
    "SBConcatStrings": "SB Concat Strings",
}
