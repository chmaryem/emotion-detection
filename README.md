FER-API — FastAPI Emotion Recognition (FER2013, ConvNeXt-Tiny)

A lightweight FastAPI service that predicts facial emotions from an image path on disk using a PyTorch model (ConvNeXt-Tiny backbone + custom head) with simple test-time augmentation (TTA).

Classes: angry, disgust, fear, happy, sad, surprise, neutral
Trained on: FER2013
Model file used: best_fer_model.pth (result of training the model on FER2013)

Features

FastAPI REST endpoint.

timm ConvNeXt-Tiny backbone + GELU head.

DropPath & Dropout regularization.

Test-time augmentation (random horizontal flips).

CUDA optional (falls back to CPU).

Loads and serves predictions using the trained .pth checkpoint.
