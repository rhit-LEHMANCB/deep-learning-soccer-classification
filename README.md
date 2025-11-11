# deep-learning-soccer-classification

High-level ResNet50-based multi-label classifier for soccer images. This repo contains a Streamlit demo app (app.py), a pretrained model snapshot (best_resnet50_multilabel.pth), training/analysis notebooks, and a simple dataset layout. Use this README to install dependencies, place your data, and run the demo or notebooks.

## What's included
- `app.py` — Streamlit UI to run inference with a saved ResNet50 multi-label model.
- `best_resnet50_multilabel.pth` — example saved model checkpoint (used by the demo).
- `image_labels.csv` — CSV of image filenames and labels (used by training/analysis notebooks).
- `dataset/` — expected place for image files. See "Data" below.
- `soccer-classification.ipynb`, `metrics.ipynb` — notebooks used for training/analysis and metrics.
- `requirements.txt` — pinned Python packages used by the environment in this repo.

## Quick contract (what this repo does)
- Input: RGB images of soccer scenes (placed under `dataset/`), optional CSV labels in `image_labels.csv`.
- Output: per-image multi-label predictions (displayed by `app.py`) and notebooks to reproduce/inspect training results.
- Error modes: the Streamlit demo will warn/stop if the model file is missing or if label count mismatches model outputs.

## Requirements
- Python 3.11
- The exact Python packages are listed in `requirements.txt`. The demo requires at minimum: `streamlit`, `torch`, `torchvision`, `pillow`, and `numpy`.

Recommended: create an isolated virtual environment before installing packages.

## Install
1. (Optional) Create and activate a virtual environment:

	python -m venv .venv
	source .venv/bin/activate

2. Install packages from `requirements.txt`:

	pip install -r requirements.txt

## Data
- Assumption: image files are stored under `dataset/` (this repo contains the `dataset/` directory). Place your input images in `dataset/` (create this folder if it doesn't exist).
- Images are downloaded from the dataset located here: https://sites.google.com/view/image-and-video-analysis/iaufd 
- Labels: `image_labels.csv` at the repository root is expected to contain image-level labels used by the notebooks. Typical CSV layout is:

	Image Number,label_1,label_2,...

	1,0,1
    
	2,1,1

Where to put files:
- Images: `dataset/`
- Labels CSV: `image_labels.csv`
- Saved model: `best_resnet50_multilabel.pth`

## How to run

Run the Streamlit demo (quickest way to try inference in a browser):

	streamlit run app.py

Usage notes for the Streamlit app:
- Upload an image in the browser UI.
- Optionally paste class labels (one per line) into the "Optional labels" text area.
- Adjust the probability threshold and top-K predictions.

Run the notebooks:

	jupyter lab
	# then open soccer-classification.ipynb or metrics.ipynb

If you prefer a simple script-based inference loop, inspect `app.py` for the model-loading and preprocessing functions (`load_model`, `preprocess_image`) and adapt them into a small script.


## License & contact
- This repository is provided for educational/demo purposes. If you need help running the code or adapting it, open an issue or contact the repository owner.

---
Last updated: 2025-11-11

