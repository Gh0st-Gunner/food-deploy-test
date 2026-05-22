# Vietnamese Food Classifier — Pipeline Documentation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                    │
│                   Upload / URL Image                                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     1. CLASSIFICATION                                │
│                                                                      │
│  Image ──► Resize 224×224 ──► Normalize (ImageNet stats)            │
│               │                                                      │
│               ▼                                                      │
│   ┌─────────────────────┐  ┌─────────────────────┐                  │
│   │   EfficientNet-B0    │  │     ResNet-50       │  ...            │
│   │     (~47MB)          │  │     (~47MB)          │                  │
│   └──────────┬──────────┘  └──────────┬──────────┘                  │
│              │                          │                             │
│              ▼                          ▼                             │
│          Softmax                   Softmax                            │
│              │                          │                             │
│              └──────────┬───────────────┘                             │
│                         ▼                                             │
│               Consensus Prediction                                    │
│          e.g. "Banh Mi" (95% confidence)                              │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Per-model breakdown (expandable dropdown)                    │    │
│  │                                                                │    │
│  │  EFF_B0 (arch: efficientnet_b0, acc: 92.35%)                 │    │
│  │    → Banh Mi at 95.3%                                         │    │
│  │  RESNET50_OLD (arch: resnet50, acc: 89.10%)                  │    │
│  │    → Banh Mi at 91.7%                                         │    │
│  │                                                                │    │
│  │  Agreement: 2/2 models = 100%                                 │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Timed ●                                                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      2. NUTRITION LOOKUP                             │
│                                                                      │
│  class_name ──► USDA_SEARCH_TERMS dict ──► USDA search term         │
│       "banh-mi"       103-class mapping        "Vietnamese           │
│                                                        baguette       │
│                                                        sandwich"      │
│                           │                                           │
│                           ▼                                           │
│              USDA FoodData Central API                                │
│         GET /fdc/v1/foods/search?query=...&api_key=...               │
│                           │                                           │
│                           ▼                                           │
│              Nutrition per 100g (calories, protein, fat,              │
│              carbs, fiber, sugars, sodium, calcium, iron,             │
│              vitamin C, vitamin A)                                    │
│                                                                      │
│  Fallback chain if no match:                                          │
│    1. Mapped Vietnamese term                                          │
│    2. English term without "Vietnamese"                               │
│    3. Raw class name                                                  │
│                                                                      │
│  Timed ●                                                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  3. INGREDIENT DETECTION                              │
│                                                                      │
│  class_name ──► INGREDIENT_PROMPTS["banh-mi"]                        │
│                   ["baguette bread.", "pork pate.",                   │
│                    "pickled daikon.", "pickled carrot.",              │
│                    "cilantro.", "jalapeno pepper.", "ham."]           │
│                           │                                           │
│                           ▼                                           │
│  ┌──────────────────────────────────────────┐                        │
│  │        Grounding DINO (tiny, ~660MB)      │                        │
│  │                                            │                        │
│  │  Input: image + text prompts               │                        │
│  │  Output: bounding boxes per ingredient      │                        │
│  │  Sensitivity: sidebar slider (0.1–0.9)      │                        │
│  └──────────────────────┬───────────────────┘                        │
│                         │                                             │
│                         ▼                                             │
│  ┌──────────────────────────────────────────┐                        │
│  │         SAM 2.1 Hiera Small (~184MB)       │                        │
│  │                                            │                        │
│  │  Input: image + bounding boxes              │                        │
│  │  Output: pixel-level segmentation masks     │                        │
│  │  per detected ingredient                    │                        │
│  └──────────────────────┬───────────────────┘                        │
│                         │                                             │
│                         ▼                                             │
│  ┌──────────────────────────────────────────┐                        │
│  │           Visualization Overlay             │                        │
│  │                                            │                        │
│  │  • Semi-transparent colored masks           │                        │
│  │    (12-color palette, alpha=0.35)           │                        │
│  │  • Bounding box outlines                    │                        │
│  │  • Labels with confidence %                 │                        │
│  │  • Color legend per ingredient              │                        │
│  └──────────────────────┬───────────────────┘                        │
│                         │                                             │
│                         ▼                                             │
│  ┌──────────────────────────────────────────┐                        │
│  │     Per-ingredient USDA Nutrition          │                        │
│  │                                            │                        │
│  │  Each detected ingredient is looked up     │                        │
│  │  individually via USDA API (expandable)    │                        │
│  │  e.g. "pork pate" → calories, protein...   │                        │
│  └──────────────────────────────────────────┘                        │
│                                                                      │
│  Timed ●                                                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  4. PORTION ESTIMATION                                │
│                                                                      │
│  ┌──────────────────────────────────────────┐                        │
│  │      Depth Anything V2 Small (~100MB)     │                        │
│  │                                            │                        │
│  │  Input: image                               │                        │
│  │  Output: relative depth map (0–1)           │                        │
│  │  Each pixel = relative distance from camera │                        │
│  └──────────────────────┬───────────────────┘                        │
│                         │                                             │
│                         ▼                                             │
│  ┌──────────────────────────────────────────┐                        │
│  │           Scale Estimation                 │                        │
│  │                                            │                        │
│  │  Sidebar option A: "Standard plate"         │                        │
│  │    → plate diameter = 25cm across image    │                        │
│  │    → scale = 25cm / image_width            │                        │
│  │                                            │                        │
│  │  Sidebar option B: Custom reference height   │                        │
│  │    → scale = reference_cm / image_height    │                        │
│  └──────────────────────┬───────────────────┘                        │
│                         │                                             │
│                         ▼                                             │
│  ┌──────────────────────────────────────────┐                        │
│  │          Volume Calculation                 │                        │
│  │                                            │                        │
│  │  For each pixel:                            │                        │
│  │    food_height = (1 - depth) × scale        │                        │
│  │    pixel_area  = scale²                      │                        │
│  │    volume     = Σ (food_height × area)      │                        │
│  │                                            │                        │
│  │  Result in cm³ (= mL)                      │                        │
│  └──────────────────────┬───────────────────┘                        │
│                         │                                             │
│                         ▼                                             │
│  ┌──────────────────────────────────────────┐                        │
│  │          Weight Estimation                 │                        │
│  │                                            │                        │
│  │  weight = volume × FOOD_DENSITIES[class]   │                        │
│  │                                            │                        │
│  │  e.g. banh_mi density = 0.45 kg/L          │                        │
│  │  Sanity clamp: if >5× typical portion,      │                        │
│  │  cap at 2× typical                          │                        │
│  └──────────────────────┬───────────────────┘                        │
│                         │                                             │
│                         ▼                                             │
│  ┌──────────────────────────────────────────┐                        │
│  │       Adjusted Nutrition                   │                        │
│  │                                            │                        │
│  │  multiplier = estimated_weight / 100        │                        │
│  │  for each nutrient:                          │                        │
│  │    adjusted = USDA_per_100g × multiplier    │                        │
│  └──────────────────────────────────────────┘                        │
│                                                                      │
│  Output:                                                             │
│    • Depth map heatmap (brighter = closer)                           │
│    • Plotly 3D surface plot                                           │
│    • Estimated weight, volume, density                               │
│    • Adjusted nutrition for estimated portion                        │
│                                                                      │
│  Timed ●                                                             │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PERFORMANCE SUMMARY                              │
│                     (expandable dropdown)                             │
│                                                                      │
│  Classification:  2.31s                                              │
│  Nutrition:        1.24s                                              │
│  Ingredients:     18.76s                                              │
│  Portion:         12.45s                                              │
│  ─────────────────────────                                            │
│  Total:           34.76s                                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

```
Image
  │
  ├─► Classification ──► class_name (e.g. "banh-mi")
  │                         │
  │                         ├── Per-model breakdown dropdown
  │                         │   (model name, architecture, accuracy,
  │                         │    top prediction, confidence %)
  │                         │
  │                         ├──► INGREDIENT_PROMPTS["banh-mi"]
  │                         │       │
  │                         │       ▼
  │                         │     Grounding DINO ──► bounding boxes
  │                         │       │
  │                         │       ▼
  │                         │     SAM 2 ──► segmentation masks
  │                         │       │
  │                         │       ├──► Visual overlay (colored masks + legend)
  │                         │       └──► Per-ingredient USDA lookup
  │                         │
  │                         ├──► USDA_SEARCH_TERMS["banh-mi"]
  │                         │       │
  │                         │       ▼
  │                         │     USDA API ──► Nutrition per 100g
  │                         │
  │                         └──► Depth Anything V2 ──► depth map
  │                                 │
  │                                 ▼
  │                               Scale (sidebar) ──► volume ──► weight
  │                                                          │
  │                                                          ▼
  │                                               Adjusted nutrition
  │                                               (per-100g × multiplier)
```

## UI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Sidebar                                                            │
│  ┌─────────────────────────────────┐                                │
│  │  Model Selection                 │                                │
│  │  ☐ Select All Models             │                                │
│  │  ☐ EFF_B0  ☐ RESNET50_OLD       │                                │
│  │                                  │                                │
│  │  ─────────────────────────       │                                │
│  │  USDA API Key                    │                                │
│  │  [password input]                │                                │
│  │                                  │                                │
│  │  ─────────────────────────       │                                │
│  │  Detection Settings              │                                │
│  │  Ingredient sensitivity [──●──] │                                │
│  │  Scale reference                 │                                │
│  │  ○ Standard plate (25cm)         │                                │
│  │  ○ Custom reference height       │                                │
│  └─────────────────────────────────┘                                │
│                                                                      │
│  Main Area                                                          │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Input Image                                                  │    │
│  │  [Upload] or [URL]                                            │    │
│  ├──────────────────────────────────────────────────────────────┤    │
│  │  CLASSIFICATION                                                │    │
│  │  [image]  │  Model Agreement: 100%                             │    │
│  │           │  ✓ Banh Mi (95.3% confidence)                     │    │
│  │           │  2/2 models agree                                  │    │
│  │           │  Top 3: ...                                        │    │
│  │           │  ▸ Per-model breakdown                              │    │
│  ├──────────────────────────────────────────────────────────────┤    │
│  │  NUTRITION                                                     │    │
│  │  USDA Match: ... │ Calories: ...  Protein: ...                  │    │
│  ├──────────────────────────────────────────────────────────────┤    │
│  │  INGREDIENT DETECTION                                         │    │
│  │  ▸ Expected ingredients for this dish                        │    │
│  │  [overlay image] │ Detected:                                  │    │
│  │                   │ ■ baguette bread (92%)                     │    │
│  │                   │ ■ pork pate (87%)                           │    │
│  │                   │ ■ pickled daikon (79%)                    │    │
│  │  ▸ Per-ingredient nutrition                                    │    │
│  ├──────────────────────────────────────────────────────────────┤    │
│  │  PORTION ESTIMATION                                            │    │
│  │  [depth map]     │  Estimated Weight: 280g                    │    │
│  │  [3D surface]    │  Estimated Volume: 311mL                   │    │
│  │                   │  Typical Portion: 250g                     │    │
│  │                   │  Adjusted nutrition for ~280g:            │    │
│  ├──────────────────────────────────────────────────────────────┤    │
│  │  ▸ Performance                                                │    │
│  │    Classification: 2.31s                                      │    │
│  │    Nutrition: 1.24s                                           │    │
│  │    Ingredients: 18.76s                                        │    │
│  │    Portion: 12.45s                                            │    │
│  │    Total: 34.76s                                               │    │
│  └──────────────────────────────────────────────────────────────┘    │
```

## File Structure

```
vn-food/
├── app.py                              # Entry point — 3 tabs
├── config.py                           # API keys, model IDs, constants
│
├── classification/
│   ├── model_loader.py                  # Load .pth / .onnx models
│   ├── predict.py                       # PyTorch & ONNX inference
│   ├── tab_classification.py            # Main tab: all 4 pipelines (auto-run)
│   └── tab_comparison.py                # Model comparison tab
│
├── nutrition/
│   ├── food_mapping.py                  # 103-class: USDA terms,
│   │                                    #   ingredient prompts, portions
│   ├── usda_client.py                   # USDA FoodData Central API
│   └── nutrition_display.py             # Streamlit nutrition UI helpers
│
├── segmentation/
│   ├── grounding_dino_loader.py         # @st.cache_resource loader
│   ├── sam2_loader.py                   # @st.cache_resource loader
│   ├── ingredient_detector.py           # Grounding DINO + SAM 2 pipeline
│   └── visualize.py                     # Mask overlay, legend, palette
│
├── depth/
│   ├── depth_loader.py                  # @st.cache_resource loader
│   ├── food_densities.py                # 103-class density table (kg/L)
│   └── portion_estimator.py             # Depth → volume → weight
│
├── info/
│   └── tab_info.py                      # About, tech stack, pipeline
│
├── utils/
│   └── image_utils.py                   # PIL ↔ numpy helpers
│
├── models/
│   ├── eff_b0.pth                       # EfficientNet-B0 checkpoint
│   ├── resnet50_old.pth                 # ResNet-50 checkpoint
│   └── class_names.json                 # 103 class names (extracted)
│
├── assets/
│   └── pipeline.png                     # Pipeline diagram
│
├── .streamlit/
│   └── secrets.toml                     # USDA_API_KEY (gitignored)
├── .gitignore
├── requirements.txt
└── PIPELINE.md                          # This file
```

## Models & Sizes

| Model | Purpose | Download Size | Loaded Via |
|---|---|---|---|
| EfficientNet-B0 / ResNet-50 | Food classification | ~47MB each | `torch.load` |
| Grounding DINO (tiny) | Ingredient detection | ~660MB | HuggingFace `transformers` |
| SAM 2.1 (hiera small) | Ingredient segmentation | ~184MB | HuggingFace `transformers` |
| Depth Anything V2 (small) | Portion depth estimation | ~100MB | HuggingFace `transformers` |

**Total additional download on first run: ~944MB**
Subsequent runs load from `~/.cache/huggingface/hub/`.

## Dictionaries (103 classes)

All dictionaries in `nutrition/food_mapping.py` and `depth/food_densities.py` cover every class:

- **`USDA_SEARCH_TERMS`** — Vietnamese class name → English search query for USDA API
- **`INGREDIENT_PROMPTS`** — Vietnamese class name → list of Grounding DINO text prompts (3–7 per dish, each ending with `.`)
- **`TYPICAL_PORTION_GRAMS`** — Vietnamese class name → typical serving size in grams
- **`FOOD_DENSITIES`** — Vietnamese class name → density in kg/L (default: 0.90)

## Performance Timing

All four pipeline stages run automatically on image upload and are timed with `time.time()`. Results shown in a collapsible **Performance** dropdown at the bottom:

| Stage | Typical Time (CPU) | Typical Time (GPU) |
|---|---|---|
| Classification | 2–5s | <1s |
| Nutrition (API) | 1–3s | 1–3s |
| Ingredient Detection | 15–30s | 5–10s |
| Portion Estimation | 10–20s | 3–5s |
| **Total** | **30–60s** | **10–20s** |

First run includes model download time (~944MB). Model loading from cache adds ~5–10s per model on first use per session.

## Limitations

- **USDA coverage**: Many Vietnamese dishes have no exact USDA match. The fallback chain tries English translations, then raw names, then individual ingredients.
- **Ingredient detection**: Grounding DINO works best with clear, well-lit photos. Overlapping or thin layers (pate inside banh mi) may be missed. Adjustable via sensitivity slider in sidebar.
- **Portion estimation**: Monocular depth produces *relative* depth, not metric. The "standard plate" assumption (25cm) is a rough heuristic. Results are approximate — not a substitute for weighing food.
- **Memory**: Running all models (classification + Grounding DINO + SAM 2 + Depth Anything) simultaneously requires ~6GB RAM minimum. May be slow on CPU-only machines.