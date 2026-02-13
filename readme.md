# SEM Defect Classification – Edge AI (SIWaferx)

This repository contains an end-to-end Edge AI pipeline for classifying semiconductor defects from Scanning Electron Microscope (SEM) images. The project demonstrates training a lightweight CNN model and deploying it to edge devices using ONNX, with compatibility for NXP eIQ workflow.

---

## 🔍 Problem Statement
Manual inspection of SEM images in semiconductor manufacturing is:
- Time-consuming  
- Error-prone  
- Not scalable for real-time production environments  

This project automates defect classification using a lightweight deep learning model optimized for edge inference.

---

## 🧪 Defect Classes
The dataset contains 10 classes:

- Bridge  
- clean  
- cmp  
- contamination  
- crack  
- ler  
- opens  
- other  
- particle  
- via  

---

## 📁 Dataset Structure

dataset/
├── train/
│ ├── Bridge/
│ ├── clean/
│ ├── cmp/
│ └── ...
├── val/
│ └── (same class folders)
└── test/
└── (same class folders)


Split used:
- Train: 70%
- Validation: 15%
- Test: 15%

⚠️ Note: Full dataset is not uploaded due to size constraints.  
Only a small `dataset_sample/` is provided for structure reference.

---

## 🧠 Model Architecture

- Type: Custom CNN (3 Convolution Layers)
- Input size: 128×128 (Grayscale)
- Framework: TensorFlow / Keras
- Output: 10-class Softmax

**Architecture Summary:**
- Conv2D (16 filters) + MaxPool  
- Conv2D (32 filters) + MaxPool  
- Conv2D (64 filters) + MaxPool  
- Dense (128)  
- Dense (10, Softmax)

---

## ⚙️ Training Approach

- Training from scratch (no pretrained weights)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch Size: 16
- Epochs: 20–25
- Data normalization: Rescale (1/255)
- Grayscale SEM images

---

## 🧪 Training & Evaluation

Train the model:
```bash
python train_model.py

##

Evaluate Model on Test set

python evaluate_model.py


Export trained model to ONNX:

python export_onnx.py


(Optional) Quantize ONNX for edge:

python quantize_onnx.py


##📊 Model Performance (Test Set)

| Metric         | Value           |
| -------------- | --------------- |
| Accuracy       | **96.25%**      |
| Precision      | 97.27%          |
| Recall         | 96.25%          |
| F1-Score       | 96.11%          |
| Model Size     | ~2.1 MB (Keras) |
| ONNX Size      | ~1.8 MB         |
| Quantized ONNX | ~0.6–0.8 MB     |


##📤Edge Deployment

Export Format: ONNX

Optimization: INT8 quantization (optional)

Target Platform: NXP Edge devices (eIQ Toolkit compatible)

Runtime: ONNX Runtime
