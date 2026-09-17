# Citrus Leaf Disease Detection

Base CNN (from scratch) + 3 transfer learning models (MobileNetV2, EfficientNetB0,
ResNet50V2) + evaluation metrics + Grad-CAM, for the Kaggle dataset
[Citrus-Diseases](https://www.kaggle.com/datasets/superlord/citrus-diseases).

Improvements over a plain single-ResNet50 pipeline: correct per-model preprocessing
(instead of one blanket `rescale=1/255` for every backbone), a fine-tuning stage after
initial frozen training, class-weighting for imbalanced real-world data, early
stopping / LR scheduling, and full metric + ROC-AUC evaluation for every model.

## 1. Install dependencies

```bash
pip install tensorflow==2.15.0 numpy opencv-python-headless matplotlib pandas scikit-learn split-folders kaggle
```

## 2. Get the dataset (pick one)

**Option A — Manual (simplest)**
1. Open https://www.kaggle.com/datasets/superlord/citrus-diseases and click Download.
2. Unzip it.
3. Put the extracted contents inside a folder named `citrus_raw` in the same
   directory as `citrus_disease_detection.py`. It's fine if there's an extra
   nested folder inside — the script auto-detects wherever the class folders are.

**Option B — Automatic via Kaggle API**
1. Go to https://www.kaggle.com/settings → API → "Create New Token". This
   downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`.
3. Just run the script (step 3 below) — if `citrus_raw/` doesn't exist, it will
   download and unzip the dataset automatically.

## 3. Run

```bash
python citrus_disease_detection.py
```

First run splits the data into `citrus_split/train`, `/val`, `/test` (70/20/10),
then trains all 4 models in sequence: `base_cnn`, `mobilenetv2`, `efficientnetb0`,
`resnet50v2`.

## 4. Outputs (all saved to `outputs/`)

- `<model>_final.h5` — trained model
- `<model>_training_curves.png` — accuracy & loss vs. epoch
- `<model>_confusion_matrix.png`
- `<model>_roc_curve.png` — per-class ROC with AUC
- `<model>_classification_report.txt` — per-class precision/recall/F1
- `<model>_gradcam_0/1/2.png` — Grad-CAM heatmap overlays on sample test images
- `model_comparison_metrics.csv` — Accuracy, Precision, Recall/Sensitivity,
  Specificity, F1, ROC-AUC for all 4 models side by side

## Notes

- A GPU (Colab, Kaggle Notebooks, or local CUDA) is strongly recommended;
  on CPU it will be slow. To do a quick smoke test, lower `EPOCHS_FROZEN`,
  `EPOCHS_FINETUNE`, and `EPOCHS_BASE_CNN` at the top of the script.
- All key settings (image size, batch size, epochs, learning rates, how many
  backbone layers to unfreeze for fine-tuning, whether to run Grad-CAM) are in
  the `CONFIG` block at the top of the script — no need to dig through the code.
- If the dataset's class-folder names differ from what you expect, that's fine —
  `flow_from_directory` reads class names straight from the folder names, and
  they're printed at the start of the run (`Classes: {...}`).
