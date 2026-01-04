# UI Tools for Japanese Comic Translation

This directory contains Gradio-based UI tools for comic translation annotation and evaluation.

## 🛠️ Available Tools

### 1. Translation Groundtruth Annotator
**File:** `translation_gt_annotater.py`
**Port:** `7861`

A lightweight, text-only Gradio UI for annotating Japanese-to-Traditional Chinese translations.

#### Features
✅ **Simplified Interface** - No image rendering, text-only for faster performance
✅ **Context Display** - See surrounding text boxes with visual indicators
✅ **Auto-load Progress** - Resume interrupted annotation sessions
✅ **Keyboard Support** - Press Enter to save and advance
✅ **TranslatedBoundingBox Schema** - Outputs to standard schema format

#### Quick Start
```bash
python ui/translation_gt_annotater.py
```
The UI will open at `http://localhost:7861`

---

### 2. Translation Result Visualization
**File:** `translation_result_visualization.py`
**Port:** `3862`

Compare prediction results with ground truth translations side by side.

#### Features
✅ **Multi-level Comparison** - Overall text similarity and box-by-box analysis
✅ **Visual Bounding Boxes** - Color-coded boxes on images (green/yellow/red)
✅ **Similarity Scoring** - Automatic similarity calculation for each text box
✅ **Easy Navigation** - Browse through results with prev/next/jump controls
✅ **Detailed Analysis** - See original Japanese, prediction, and ground truth side-by-side

#### Quick Start
```bash
python ui/translation_result_visualization.py
```
The UI will open at `http://localhost:3862`

#### Default Paths
- Prediction file: `results/2/translated.json`
- Ground truth directory: `data/benchmark/translation_groundtruth`

#### Color Legend
- 🟢 **Green** = Match (similarity >80%)
- 🟡 **Yellow** = Partial match (50-80%)
- 🔴 **Red** = Mismatch (<50%)

---

### 3. Dataset Annotator
**File:** `dataset_annotator.py`

General dataset annotation tool for OCR groundtruth creation.

## 📋 Common Workflows

### Workflow 1: Create Translation Ground Truth
**Tool:** Translation Groundtruth Annotator

1. **Load Files** - Point to `data/benchmark/ocr_groundtruth` directory
2. **Review** - Read the Japanese text for current box
3. **Translate** - Enter Traditional Chinese translation
4. **Save & Next** - Press Enter or click button to advance
5. **Save File** - Save all translations for current image
6. **Navigate** - Move to next file and repeat

**Interface Symbols:**
- `➤` = Current box being translated
- `✅` = Already translated
- `⬜` = Not yet translated

### Workflow 2: Evaluate Translation Results
**Tool:** Translation Result Visualization

1. **Load Files** - Specify prediction file and ground truth directory
2. **Review Overall** - Check overall similarity statistics
3. **Analyze Boxes** - Review box-by-box comparison details
4. **Visual Check** - View color-coded bounding boxes on images
5. **Navigate** - Browse through different images to analyze quality

## 📊 Output Schema

Translation output uses `TranslatedBoundingBox` schema:

```json
{
  "source": "images/example.jpg",
  "text": "原始日文",
  "translated_text": "翻譯的繁體中文",
  "image_path": "images/example.jpg",
  "bounding_boxes": [
    {
      "box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "text": "原始文字",
      "score": 0.98,
      "translated_text": "翻譯文字"
    }
  ]
}
```

## 💡 Tips

### For Annotation (translation_gt_annotater.py)
- **Resume work**: Existing translations auto-load when you reload a file
- **Skip uncertain boxes**: Use "Skip" button to come back later
- **Review/edit**: Use "Previous" button to go back
- **Keyboard shortcut**: Press Enter in translation field = Save & Next

### For Visualization (translation_result_visualization.py)
- **Jump to specific image**: Use the jump index input to go directly to a file
- **Compare multiple models**: Load different prediction files to compare
- **Focus on errors**: Red boxes indicate mismatches that need attention
- **Check context**: Box-by-box details show original Japanese for context

## 🔧 Port Configuration

| Tool | Default Port | Change in Code |
|------|-------------|----------------|
| Translation GT Annotator | 7861 | Line 419: `server_port=7861` |
| Translation Visualization | 3862 | Line 397: `server_port=3862` |
| Dataset Annotator | (varies) | Check file |

## 📂 Directory Structure

```
ui/
├── README.md                              # This file
├── translation_gt_annotater.py            # Create ground truth annotations
├── translation_result_visualization.py     # Compare predictions vs ground truth
└── dataset_annotator.py                   # OCR dataset annotation
```
