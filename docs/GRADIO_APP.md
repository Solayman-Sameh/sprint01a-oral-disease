---
title: Oral Disease Classifier
emoji: 🦷
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "6.30.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: An educational oral-image classifier in English and Arabic
tags:
  - image-classification
  - pytorch
  - oral-health
---

# Oral Disease Classifier

A bilingual English/Arabic Gradio app for Sprint 02, Project 01. Upload an oral image, select **Classify**, and inspect the selected ResNet50 model's six ranked confidence scores. Arabic mode switches the interface and results to a right-to-left layout. Six training-set examples help demonstrate the workflow when available locally. They are omitted from the public Git project until their redistribution rights are established; the examples section is hidden if the images are absent.

This is an educational prototype, not a diagnostic tool. Softmax scores and the labels “Very Likely” through “Very Unlikely” are uncalibrated model confidence, not medical probabilities. The model has no healthy class and cannot reliably reject unrelated images.

## Run locally

Use Python 3.11. From the project directory, create and activate a virtual environment, then install the dependencies:

**Windows (Command Prompt):**

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

In PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead. If `py` is unavailable, use a Python 3.11 interpreter to create the environment.

If Windows reports a path-length error during setup, clone the project to a shorter path such as `C:\projects\oral-app` and retry there.

**macOS/Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The selected `best.pt` checkpoint is included alongside `class_to_idx.json` in `models/fine-tuned/`. In an exported Space package, these files are placed in `models/` instead.

Run the project app with the environment active:

```bash
python -m oral_app.app
```

Open <http://127.0.0.1:7860> and leave the terminal open. Press `Ctrl+C` to stop the app. If the port is in use, stop the existing app or set `GRADIO_SERVER_PORT` to another port before launching, then open that port in the browser. The app uses CPU inference; no GPU or ImageNet download is required.

The requirements pin Gradio 6.30.0, PyTorch 2.5.1, torchvision 0.20.1, Pillow 12.3.0 and NumPy 2.4.6. Linux explicitly selects CPU PyTorch wheels; Windows can retain the tested CUDA builds already installed. Training-only packages are unnecessary.

## How the app works

1. **Startup:** `oral_app/inference.py` verifies the checkpoint SHA-256, class mapping, data manifest and preprocessing recipe. It constructs ResNet50 with `weights=None`, replaces its head with Dropout(0.3) → Linear(2048, 6), loads the weights strictly and sets evaluation mode. The model is cached once per process.
2. **Upload:** Gradio decodes the image. The model input applies EXIF orientation, RGB conversion, centered black square padding, bilinear resizing to 224 × 224, tensor conversion and ImageNet normalization. This mirrors the original training evaluation transform exactly.
3. **Classify:** inference runs with gradients disabled. Softmax converts the six logits to scores, which are sorted in descending order. The top card and all six bars use the thresholds in the UI plan: 0.70, 0.40, 0.15 and 0.05.
4. **Language:** session state stores the scores and selected language. Switching language translates the existing result without another model call or clearing the image. A CSS rule switches layout and bar direction. Manrope and IBM Plex Sans Arabic are bundled locally, with their font licenses.
5. **Clear/change image:** the old scores are cleared when the image changes. Clear resets the image, results and selected example. Related events share a queue to prevent a pending classification from leaving results attached to a newer upload.

`oral_app/app.py` owns the interface and callbacks; `oral_app/inference.py` owns the model and preprocessing; `oral_app/translations.py` contains the app text; `oral_app/assets/style.css` owns the warm cream/sage/terracotta design. `oral_app/assets/model_info.json` records the saved evaluation, and `oral_app/assets/examples/manifest.json` records each example's training-split provenance.

Uploads are limited to 10 MB. The longest side must be at most 4,472 pixels because centered square padding also allocates that square. Uploads are processed in Gradio's temporary cache, configured for cleanup every 30 minutes for files older than 30 minutes; they are not added to the dataset or used for training. Model weights and local datasets are blocked from Gradio's file-serving routes.

## Selected model and evaluation

The selected model is **ResNet50 trial 1, checkpoint epoch 9**, with 23,520,326 parameters. It was selected using validation macro-F1 **before** test evaluation.

| Measure | Result |
|---|---:|
| Validation macro-F1 | 0.9047 |
| Held-out test accuracy | 91.84% |
| Held-out test macro-F1 | 0.8717 |
| Training images | 2,415 |
| Validation images | 518 |
| Test images | 515 |

| Class | Precision | Recall | F1 | Test images |
|---|---:|---:|---:|---:|
| Calculus | 0.851 | 0.905 | 0.877 | 63 |
| Caries | 0.690 | 0.714 | 0.702 | 28 |
| Gingivitis | 0.954 | 0.917 | 0.935 | 181 |
| Hypodontia | 0.952 | 0.973 | 0.962 | 182 |
| Mouth Ulcer | 0.941 | 0.889 | 0.914 | 36 |
| Tooth Discoloration | 0.840 | 0.840 | 0.840 | 25 |

Source: the saved Kaggle training notebook, ResNet50 final test output, and `model_comparison.csv`. Rounded per-class values are stored in the app metadata. The full five-architecture/two-trial comparison remains in the training project; only the selected model is used by the app.

Checkpoint SHA-256: `ce9aa13f69fb69ed067580044044ecef0c815715fff16d82409e0e71d338e1d7`.

## Dataset and limitations

[Kaggle: salmansajid05/oral-diseases, version 3](https://www.kaggle.com/datasets/salmansajid05/oral-diseases). After exact-duplicate filtering and quarantining conflicting similarity families, 3,448 images in 1,812 image families were split approximately 70/15/15. Related families remain within one split. Geometry flags were retained for review; not every image was clinically reviewed.

Patient independence is unverified because patient identifiers are absent. Labels are source-folder labels, not independently validated clinical diagnoses. Caries has the lowest F1 and only 28 test samples. These internal results do not establish generalization to clinical practice. A high confidence score can still be wrong.

## Checks

In the complete training project, run:

```powershell
python -m unittest discover -s tests -p test_inference.py -v
```

These checks load the real checkpoint, compare preprocessing with the original training source, exercise all six examples and non-RGB inputs, verify EXIF rotation, confirm deterministic six-class scores, and check translated results and input preservation. They do not replace a visual browser check or a live Spaces test.

## Prepare a Hugging Face Space

In the complete training project:

```powershell
python scripts/export_space.py
```

This creates `local/space-build/` containing `oral_app/`, a small `app.py` Space entry point, runtime requirements, this document as its Space README, the MIT code license, and `models/best.pt` plus `class_to_idx.json`. Run the packaged entry point with `python app.py`. The exporter copies locally available example photos, so review their rights before publishing the package. The exporter prefers a hard link for the immutable weights to avoid allocating another 94 MB locally. Do not modify the exported checkpoint in place. An existing output directory is never overwritten.

Create a **Gradio** Space using Python 3.11 and CPU hardware, then upload **only that package**. The YAML at the top of this README pins the Gradio runtime. Use the Hub uploader (which handles large files), or Git with Git LFS for `*.pt`; the checkpoint's size is not a reason to treat it as an ordinary Git blob. Restart the Space after a build and repeat upload, example, classification, clear and both-language checks. Local timing does not guarantee the same timing on shared Space hardware.

The app binds to `0.0.0.0` when Spaces supplies `SPACE_ID`, and to loopback locally. `GRADIO_SERVER_NAME` can override this. Optional `ORAL_MODEL_PATH` selects a checkpoint path; optional `ORAL_CPU_THREADS` controls CPU threads (default 2, maximum 4). Model loading accepts the original project's `models/fine-tuned/` layout and the exported `models/` layout.

## Licenses and example images

The **code and documentation** use the MIT license. Manrope and IBM Plex Sans Arabic use the SIL Open Font License 1.1; their notices and source URLs are in `oral_app/assets/fonts/`.

The dataset's license is listed as **Unknown**. The six bundled local demo photos come from the training split, retain source hashes in their manifest and are not licensed by this project's MIT license. Review permission for public image redistribution before publishing those examples; an educational purpose alone does not establish permission. The trained weights also derive from third-party data and pretrained weights, and their applicable terms require review before public release. This document does not grant rights to third-party materials.
