# Helmet Detection in Construction Sites – Computer Vision Pipeline

This repository contains the full source code for a computer vision pipeline
that detects whether construction workers are wearing safety helmets using
classical preprocessing and a deep-learning detector (YOLOv8 / Faster R-CNN).

## 1. Project Overview

Construction sites are hazardous environments where head protection is critical.
Manual monitoring of helmet compliance is:
- time-consuming,
- error-prone, and
- difficult to scale.

This project builds an automatic pipeline capable of:
- enhancing CCTV frames using classical filters (Gaussian, bilateral, CLAHE, Sobel, Laplacian, Canny, morphology),
- preparing clean inputs for a deep-learning detector, and
- evaluating helmet vs. no-helmet predictions with standard object-detection metrics.

## 2. Repository Structure

```text
src/
 └─ helmet_full_pipeline.py   # main script: preprocessing, metrics, plots
figures/
 ├─ confusion_matrix_helmet.png
 └─ training_and_pipeline.png
images/
 └─ example frames used in the poster (raw, preproc, yolo, post)
data/
 ├─ original_images/          # local dataset (ignored by git)
 └─ preprocessed/             # auto-generated outputs (ignored by git)
