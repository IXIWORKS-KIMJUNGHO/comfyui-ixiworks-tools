# IXIWORKS Tools for ComfyUI

Video description, storyboard, ControlNet, image, LoRA, and utility custom nodes for ComfyUI.

**21 nodes** across 6 categories.

---

## Node List

### IXIWORKS/Video (1)

| Key | Display Name | Description |
|-----|-------------|-------------|
| `VideoDescribe` | Video Describe | Full video analysis with Qwen3-VL-8B-Instruct |

### IXIWORKS/StoryBoard (6)

| Key | Display Name | Description |
|-----|-------------|-------------|
| `SBJsonParser` | SB JSON Parser | Parse storyboard JSON files into scene/character data |
| `SBPromptBuilder` | SB Prompt Builder | Combine scene data into prompt strings |
| `SBCharacterPrompt` | SB Character Prompt | Generate natural language character descriptions |
| `SBSelectCut` | Select Cut | Select a specific scene by index |
| `SBMergeStrings` | SB Merge Strings | Merge two string arrays with a separator |
| `SBPromptFilter` | SB Prompt Filter | Filter prompts for LoRA style or ControlNet type via Claude API |

### IXIWORKS/ControlNet (2)

| Key | Display Name | Description |
|-----|-------------|-------------|
| `CNPreprocessor` | CN Preprocessor | Run canny/depth/lineart/openpose/mlsd preprocessors |
| `CNStepControl` | CN Step Control | Apply step-range + linear fade to a ControlNet model patch |

### IXIWORKS/Image (1)

| Key | Display Name | Description |
|-----|-------------|-------------|
| `ImageCropResize` | Crop & Resize | Crop and resize image to a target aspect ratio |

### IXIWORKS/LoRA (1)

| Key | Display Name | Description |
|-----|-------------|-------------|
| `LoRAStepLoader` | LoRA Step Loader | Load LoRA with step-range scheduling via ComfyUI hooks |

### IXIWORKS/Utils (10)

| Key | Display Name | Description |
|-----|-------------|-------------|
| `UtilSwitch` | Switch | Boolean switch between two inputs (lazy eval) |
| `UtilSwitchCase` | Switch Case | Select one of N inputs by index (2–8, lazy eval) |
| `UtilStringToList` | String to List | Convert multiple text inputs to a list for batch processing |
| `UtilConcatStrings` | Concat Strings | Concatenate two strings with configurable separator |
| `UtilSaveText` | Save Text | Write a string to a file in the output directory |
| `UtilLoadImageList` | Load Image List | Load images from Input folder via dropdown selectors into a list |
| `UtilImageToList` | Image to List | Collect multiple IMAGE inputs into a list |
| `UtilBypass` | Bypass | Pass-through toggle — returns None when bypass=True |
| `UtilEmptyLatent` | Empty Latent | Generate an empty latent from a named aspect ratio |
| `UtilJsonToList` | JSON to List | Load a JSON file or string and extract items by dot-notation key |

---

## Installation

```bash
cd ComfyUI/custom_nodes/comfyui-ixiworks-tools
pip install -r requirements.txt
```

### Dependencies

- `torch >= 2.0.0`
- `transformers >= 4.50.0`
- `accelerate >= 0.20.0`
- `qwen-vl-utils`
- `opencv-python`
- `pillow`, `numpy`
- `anthropic` (required for SBPromptFilter only)

---

## Model Download

### VideoDescribe

The Qwen3-VL model downloads automatically on first use (~16GB). To pre-download:

```bash
python download_models.py
```

Models are saved to `ComfyUI/models/video_description/Qwen3-VL-8B-Instruct/`.

### CNPreprocessor

Preprocessor model weights are downloaded automatically from HuggingFace on first use and cached in the ComfyUI models directory.

---

## Node Details

### VideoDescribe

Analyzes a video file using Qwen3-VL vision-language model.

**Inputs**
- `video_path` (STRING): Filename or absolute path. Relative paths resolve against `ComfyUI/input/`.
- `analysis_type` (COMBO): `detailed` / `summary` / `keywords`
- `fps` (FLOAT, default 1.0): Frames per second to sample
- `custom_prompt` (STRING, optional): Overrides analysis_type preset
- `use_4bit` (BOOLEAN, default False): Enable 4-bit quantization (~8GB VRAM instead of ~16GB)
- `temperature` (FLOAT, default 0.7): Generation temperature (used with custom_prompt)

**Outputs**: `description` (STRING), `info` (STRING)

---

### SBJsonParser

Parses a storyboard JSON file from `ComfyUI/input/prompt/`.

**Input**: `JSON` (COMBO) — filename selector

**Outputs**: `zipped_prompt` (ZIPPED_PROMPT list), `zipped_character` (ZIPPED_PROMPT list), `count` (INT)

**JSON format**:
```json
{
  "scene": {
    "1": {
      "mainCharacter": { "koName": "", "enName": "", "description": "" },
      "subCharacter":  { "koName": "", "enName": "", "description": "" },
      "time":        { "ko": "", "en": "" },
      "weather":     { "ko": "", "en": "" },
      "cameraShot":  { "ko": "", "en": "" },
      "cameraAngle": { "ko": "", "en": "" },
      "description": { "ko": "", "en": "" },
      "composition": { "ko": "", "en": "" }
    }
  }
}
```

---

### SBPromptBuilder

Combines ZIPPED_PROMPT scene data into prompt strings.

**Input**: `zipped_prompt` (ZIPPED_PROMPT list, INPUT_IS_LIST)

**Output**: `prompt` (STRING list)

---

### SBCharacterPrompt

Generates natural language character descriptions from character data.

**Input**: `zipped_character` (ZIPPED_PROMPT list, INPUT_IS_LIST)

**Output**: `character_prompt` (STRING list)

---

### SBSelectCut

Selects a single scene by index from a ZIPPED_PROMPT list.

**Inputs**: `zipped_prompt` (ZIPPED_PROMPT list), `index` (INT)

**Output**: `selected_prompt` (ZIPPED_PROMPT)

---

### SBMergeStrings

Merges two STRING lists element-wise with a separator.

**Inputs**: `strings_a` (STRING), `strings_b` (STRING), `separator` (STRING, default `" "`)

**Output**: `merged_strings` (STRING list)

---

### SBPromptFilter

Filters prompts to remove conflicting keywords for a LoRA style or ControlNet type, using the Claude API.

**Inputs**
- `prompt` (STRING list)
- `filter_mode` (COMBO): `style` / `controlnet`
- `style_type` (COMBO): `inksketch` / `ink-wash` / `pen-ink-illustration` / `ink-watercolor` / `watercolor-illustration`
- `controlnet_type` (COMBO): `pose` / `depth` / `canny`
- `api_key` (STRING): Anthropic API key

**Output**: `filtered_prompt` (STRING list)

---

### CNPreprocessor

Runs a ControlNet preprocessor on an image batch.

**Inputs**
- `image` (IMAGE)
- `preprocessor` (COMBO): `canny` / `depth` / `lineart` / `openpose` / `mlsd`
- `resolution` (INT, default 512)
- `low_threshold` (INT, optional, canny only)
- `high_threshold` (INT, optional, canny only)

**Output**: `image` (IMAGE)

---

### CNStepControl

Applies step-range gating and linear fade to an existing ControlNet double-block patch. Designed for Flux / Z-Image architectures.

**Inputs**: `model` (MODEL), `strength_start` (FLOAT), `strength_end` (FLOAT), `start_at` (FLOAT), `end_at` (FLOAT)

**Output**: `model` (MODEL)

---

### ImageCropResize

Crops an image to a target aspect ratio and resizes to `long_side` pixels.

**Inputs**: `image` (IMAGE), `aspect_ratio` (COMBO), `long_side` (INT, default 1024)

**Output**: `image` (IMAGE)

---

### LoRAStepLoader

Loads a LoRA with per-step strength scheduling via ComfyUI hooks API.

**Inputs**
- `model` (MODEL), `clip` (CLIP)
- `lora_name` (COMBO)
- `strength_start` (FLOAT), `strength_end` (FLOAT)
- `start_at` (FLOAT), `end_at` (FLOAT)

**Outputs**: `model` (MODEL), `clip` (CLIP), `hooks` (HOOKS)

---

### UtilSwitch

Boolean switch with lazy evaluation — only evaluates the selected branch.

**Inputs**: `on_false` (ANY, lazy), `on_true` (ANY, lazy), `boolean_switch` (BOOLEAN)

**Output**: `output` (ANY)

---

### UtilSwitchCase

Selects one of N inputs by index with lazy evaluation (2–8 inputs, dynamic via JS).

**Inputs**: `count` (INT), `select` (INT), `input_0..7` (ANY, lazy, optional)

**Output**: `output` (ANY)

---

### UtilStringToList

Converts multiple text widget inputs into a STRING list for batch processing (up to 8).

**Inputs**: `count` (INT), `prompt_1` (STRING), `prompt_2..8` (STRING, optional)

**Output**: `strings` (STRING list)

---

### UtilConcatStrings

Concatenates two strings with a configurable separator.

**Inputs**: `string_a` (STRING), `string_b` (STRING), `separator` (STRING, optional)

**Output**: `joined_string` (STRING)

---

### UtilSaveText

Writes a string to a file in the ComfyUI output directory.

**Inputs**: `text` (STRING), `filename` (STRING, default `"output.txt"`)

No output (OUTPUT_NODE).

---

### UtilLoadImageList

Loads images selected from dropdown menus (from `ComfyUI/input/`) into an IMAGE list. Up to 20 images, count-controlled with dynamic JS show/hide.

**Inputs**
- `count` (INT, default 1, max 20): number of image slots to show
- `image_1` (COMBO): image file selector — Input folder contents
- `image_2..20` (COMBO, optional): additional image slots

Select `[none]` to skip a slot. Dropdown list auto-populates from `ComfyUI/input/`.

**Output**: `images` (IMAGE list)

---

### UtilImageToList

Collects up to 8 IMAGE inputs into an IMAGE list (count-controlled, dynamic via JS).

**Inputs**: `count` (INT), `image_1..8` (IMAGE, optional)

**Output**: `images` (IMAGE list)

---

### UtilBypass

Pass-through toggle. When `bypass=True`, returns None (and JS sets the upstream node to bypass mode).

**Inputs**: `bypass` (BOOLEAN), `input` (ANY, optional)

**Output**: `output` (ANY)

---

### UtilEmptyLatent

Generates an empty latent tensor from a named aspect ratio.

**Inputs**
- `ratio` (COMBO): `21:9` / `1.85:1` / `16:9` / `9:16` / `1:1`
- `long_side` (INT, default 1024)
- `channels` (COMBO): `16` (Flux/Lumina/SD3) / `4` (SD/SDXL)
- `batch_size` (INT, default 1)

**Outputs**: `latent` (LATENT), `width` (INT), `height` (INT)

---

### UtilJsonToList

Loads a JSON file or raw JSON string and extracts items as a STRING list.

**Inputs**
- `file_path` (STRING): filename relative to `ComfyUI/input/`, or absolute path
- `json_string` (STRING, optional): raw JSON string — overrides `file_path` if connected
- `key` (STRING, optional): dot-notation path into JSON, e.g. `"prompts.solo"`. Empty = root.

Dict values are always flattened into a single list.

**Outputs**: `strings` (STRING list), `count` (INT)

---

## Hardware Requirements

| Configuration | GPU | VRAM | Notes |
|--------------|-----|------|-------|
| Minimum | RTX 3060 | 12GB | VideoDescribe with 4-bit quant |
| Recommended | RTX 4090 | 24GB | Full FP16 |
| Apple Silicon | M1/M2/M3 | Unified | MPS supported |

---

## License

Apache License 2.0

---

## Changelog

### v2.1.0
- `UtilLoadImageList`: replaced `filenames` text input with count-based dropdown selectors (Input folder)
- `UtilJsonToList`: simplified to `file_path` + `key` only — removed `item_field`, `flatten_dict`, `shuffle`, `limit`

### v2.0.0
- Full node reorganization: unified IXIWORKS/* category prefix
- Added ControlNet category: CNPreprocessor (canny/depth/lineart/openpose/mlsd), CNStepControl
- Added Image category: ImageCropResize
- Added LoRA category: LoRAStepLoader
- Added Utils: UtilBypass, UtilEmptyLatent, UtilJsonToList, UtilLoadImageList, UtilImageToList
- Added StoryBoard: SBPromptFilter (Claude API style/controlnet filtering)
- Removed controlnet_aux dependency — direct preprocessor implementations
- Removed Layer Separation nodes and Sampler nodes
- JS extensions updated for all renamed nodes
- Various bug fixes (INPUT_IS_LIST chains, IS_CHANGED, error handling)

### v1.3.2
- Added Switch Case node
- Added Join Strings node, String to List node
- Added frontend JS support (WEB_DIRECTORY)

### v1.2.0
- Added StoryBoard category (JSON Parser, Build Prompt, Build Character Prompt, Select Index, Merge Strings)

### v1.1.0
- Smart video path resolution, performance optimization

### v1.0.0
- Full Qwen3-VL integration with model caching
