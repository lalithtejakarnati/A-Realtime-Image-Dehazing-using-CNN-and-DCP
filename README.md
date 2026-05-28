# A-Realtime-Image-Dehazing-using-CNN-and-DCP

<p align="center">
  <img src="synthetic_haze_demo.png" width="850"/>
</p>

<h3 align="center">
Real-Time Single Image Dehazing using CNN and Dark Channel Prior (DCP)
</h3>

<p align="center">
  Deep Learning • Computer Vision • Image Restoration • PyTorch
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OpenCV-ComputerVision-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Flask-WebApp-black?style=for-the-badge"/>
</p>

---

# 📌 Overview

Atmospheric haze significantly degrades image quality by reducing visibility, contrast, and color fidelity. This project presents a hybrid real-time image dehazing framework that combines:

* **Convolutional Neural Networks (CNNs)** for feature extraction and restoration
* **Dark Channel Prior (DCP)** for atmospheric light estimation
* **PyTorch-based training pipeline** for efficient deep learning experimentation

The proposed framework restores visually clear images from hazy scenes while maintaining computational efficiency suitable for near real-time inference.

---

# ✨ Features

✅ Real-time image dehazing
✅ CNN-based image restoration
✅ Dark Channel Prior enhancement
✅ PyTorch training pipeline
✅ Evaluation metrics support
✅ Flask-based web application
✅ Modular and scalable architecture
✅ Custom image inference support

---

# 🖼️ Demo Results

## Synthetic Haze Example

<p align="center">
  <img src="synthetic_haze_demo.png" width="800"/>
</p>

---

## Example Workflow

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1200/1*UaQBNft8Cem2m4N0VwMrow.png" width="850"/>
</p>

---

# 🏗️ Project Architecture

```text id="a1"
Input Hazy Image
        │
        ▼
Dark Channel Prior Processing
        │
        ▼
CNN-based Feature Extraction
        │
        ▼
Image Reconstruction
        │
        ▼
Dehazed Output Image
```

---

# 📂 Project Structure

```bash id="a2"
A-Realtime-Image-Dehazing-using-CNN-and-DCP/
│
├── Code/
│   ├── app.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── interface.html
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── test.py
│   └── train.py
│
├── synthetic_haze_demo.png
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🧠 Methodology

The framework integrates traditional image priors with deep learning techniques.

---

## 1️⃣ Dark Channel Prior (DCP)

Dark Channel Prior estimates:

* atmospheric light
* transmission maps
* haze density

This improves haze removal performance before deep restoration.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*0w3l1vY0wM6hM9Vw4M4K5Q.png" width="700"/>
</p>

---

## 2️⃣ CNN-Based Restoration

The CNN learns haze-aware features and reconstructs visually enhanced images with:

* improved contrast
* sharper textures
* enhanced visibility
* restored color fidelity

<p align="center">
  <img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-07_at_11.49.00_PM_r4L4Y8W.png" width="800"/>
</p>

---

## 3️⃣ Post-Processing

Final enhancement improves:

* edge sharpness
* visual quality
* detail recovery
* artifact reduction

---

# 📊 Dataset

The dataset is excluded from this repository because of GitHub storage limitations.

## Recommended Datasets

---

## 🔹 RESIDE Dataset

Large-scale benchmark dataset for single image dehazing.

### Official Website

https://sites.google.com/view/reside-dehaze-datasets/reside-v0

### Direct Download

https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAfRYnhTR7My/view

<p align="center">
  <img src="https://sites.google.com/view/reside-dehaze-datasets/_/rsrc/1520591081060/home/fig1.png" width="800"/>
</p>

---

## 🔹 O-HAZE Dataset

Real outdoor hazy image benchmark dataset.

### Dataset Link

https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/

---

## 🔹 Dense-Haze Dataset

Dense haze benchmark dataset for evaluating dehazing performance.

### Dataset Link

https://data.vision.ee.ethz.ch/cvl/ntire18/dense-haze/

---

# ⚙️ Installation

## Clone Repository

```bash id="a3"
git clone https://github.com/lalithtejakarnati/A-Realtime-Image-Dehazing-using-CNN-and-DCP.git

cd A-Realtime-Image-Dehazing-using-CNN-and-DCP
```

---

## Install Dependencies

```bash id="a4"
pip install -r requirements.txt
```

---

# 🚀 Usage

## Train the Model

```bash id="a5"
python Code/train.py
```

---

## Evaluate Model Performance

```bash id="a6"
python Code/evaluate.py
```

---

## Run Inference

```bash id="a7"
python Code/inference.py
```

---

## Launch Web Application

```bash id="a8"
python Code/app.py
```

---

# 📈 Evaluation Metrics

The model performance can be evaluated using:

| Metric            | Description                   |
| ----------------- | ----------------------------- |
| PSNR              | Peak Signal-to-Noise Ratio    |
| SSIM              | Structural Similarity Index   |
| Visual Assessment | Subjective quality evaluation |

---

# 🛠️ Technologies Used

| Technology | Purpose               |
| ---------- | --------------------- |
| Python     | Core Programming      |
| PyTorch    | Deep Learning         |
| OpenCV     | Image Processing      |
| NumPy      | Numerical Computation |
| Flask      | Web Application       |
| CNN        | Feature Learning      |
| DCP        | Haze Estimation       |

---

# 🔮 Future Improvements

* Transformer-based dehazing
* Real-time video dehazing
* Mobile deployment
* TensorRT optimization
* GAN-based enhancement
* Attention-based restoration

---

# 🤝 Contribution

Contributions are welcome.

To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a pull request

---

# 👨‍💻 Author

## Karnati Lalith Teja

### GitHub

https://github.com/lalithtejakarnati

---

# 📜 License

This project is intended for:

* academic research
* educational purposes
* computer vision experimentation

---

# ⭐ Acknowledgements

Special thanks to:

* RESIDE Dataset creators
* PyTorch community
* OpenCV contributors
* Computer Vision research community
