

# 🌫️ A-Realtime-Image-Dehazing-using-CNN-and-DCP

<h3 align="center">
Real-Time Single Image Dehazing using CNN and Dark Channel Prior (DCP)
</h3>

<p align="center">
  Deep Learning • Computer Vision • Image Restoration • PyTorch
</p>

---

# 📌 Overview

Atmospheric haze significantly degrades image quality by reducing visibility, contrast, and color fidelity. This project presents a hybrid real-time image dehazing framework that combines:

* Convolutional Neural Networks (CNNs)
* Dark Channel Prior (DCP)
* PyTorch-based training pipeline

The framework restores visually clear images from hazy scenes while maintaining computational efficiency suitable for near real-time inference.

---

# ✨ Features

* Real-time image dehazing
* CNN-based image restoration
* Dark Channel Prior enhancement
* PyTorch training pipeline
* Evaluation metrics support
* Flask-based web application
* Modular architecture
* Custom image inference support

---

# 🏗️ Project Architecture

The proposed framework follows a hybrid image dehazing pipeline:

1. Input Hazy Image
2. Dark Channel Prior Processing
3. CNN-Based Feature Extraction
4. Image Reconstruction
5. Final Dehazed Output

---

# 📂 Project Structure

```bash
A-Realtime-Image-Dehazing-using-CNN-and-DCP/
│
├── assets/
├── Code/
├── synthetic_haze_demo.png
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

## RESIDE Dataset

https://sites.google.com/view/reside-dehaze-datasets/reside-v0

## O-HAZE Dataset

https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/

## Dense-Haze Dataset

https://data.vision.ee.ethz.ch/cvl/ntire18/dense-haze/

---

# ⚙️ Installation

```bash
git clone https://github.com/lalithtejakarnati/A-Realtime-Image-Dehazing-using-CNN-and-DCP.git

cd A-Realtime-Image-Dehazing-using-CNN-and-DCP

pip install -r requirements.txt
```

---

# 🚀 Usage

## Train

```bash
python Code/train.py
```

## Evaluate

```bash
python Code/evaluate.py
```

## Inference

```bash
python Code/inference.py
```

## Run Flask App

```bash
python Code/app.py
```

---

# 📈 Evaluation Metrics

* PSNR
* SSIM
* Visual Quality Assessment

---

# 🛠️ Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Flask
* CNN
* DCP

---

# 🔮 Future Improvements

* Transformer-based dehazing
* Real-time video dehazing
* TensorRT optimization
* Mobile deployment
* GAN-based enhancement

---

# 👨‍💻 Author

Karnati Lalith Teja

GitHub:
https://github.com/lalithtejakarnati
