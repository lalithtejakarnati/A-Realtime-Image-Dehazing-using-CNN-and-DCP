# 🌫️ A-Realtime-Image-Dehazing-using-CNN-and-DCP

<p align="center">
  <img src="assets/banner.png" width="1000"/>
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
  <img src="https://img.shields.io/badge/Status-Research_Project-success?style=for-the-badge"/>
</p>

---

# 📌 Overview

Atmospheric haze significantly degrades image quality by reducing visibility, contrast, and color fidelity. This project presents a **hybrid real-time image dehazing framework** that combines:

* **Convolutional Neural Networks (CNNs)** for feature extraction and restoration
* **Dark Channel Prior (DCP)** for atmospheric light estimation
* **PyTorch-based training pipeline** for deep learning experimentation

The framework restores visually clear images from hazy scenes while maintaining computational efficiency suitable for near real-time inference.

<p align="center">
  <img src="assets/overview.png" width="850"/>
</p>

---

# ✨ Features

| Feature               | Description               |
| --------------------- | ------------------------- |
| ⚡ Real-Time Dehazing  | Fast restoration pipeline |
| 🧠 CNN Restoration    | Deep feature learning     |
| 🌫️ DCP Enhancement   | Atmospheric estimation    |
| 📈 Evaluation Metrics | PSNR / SSIM support       |
| 🌐 Flask Web App      | Interactive inference     |
| 🧩 Modular Design     | Easy experimentation      |

<p align="center">
  <img src="assets/features.png" width="900"/>
</p>

---

# 🖼️ Demo Results

## Synthetic Haze Example

<p align="center">
  <img src="synthetic_haze_demo.png" width="850"/>
</p>

---

## Before vs After Dehazing

<p align="center">
  <img src="assets/results.png" width="950"/>
</p>

---

# 🏗️ Project Architecture

<p align="center">
  <img src="assets/architecture.png" width="1100"/>
</p>

The proposed framework follows a hybrid image dehazing pipeline integrating classical image priors with deep learning.

---

# 🔄 Pipeline Stages

---

## 1️⃣ Input Hazy Image

The hazy image is first provided to the dehazing pipeline.

<p align="center">
  <img src="assets/input_hazy.png" width="650"/>
</p>

---

## 2️⃣ Dark Channel Prior Processing

Dark Channel Prior estimates:

* atmospheric light
* transmission maps
* haze density

This stage improves visibility estimation before CNN processing.

<p align="center">
  <img src="assets/dcp_processing.png" width="850"/>
</p>

---

## 3️⃣ CNN-Based Feature Extraction

The CNN extracts haze-aware image representations and learns restoration mappings.

<p align="center">
  <img src="assets/cnn_feature_extraction.png" width="850"/>
</p>

---

## 4️⃣ Image Reconstruction

The network reconstructs:

* enhanced contrast
* sharper textures
* restored scene details

<p align="center">
  <img src="assets/reconstruction.png" width="850"/>
</p>

---

## 5️⃣ Final Dehazed Output

The final output image contains:

* improved visibility
* reduced haze
* restored color fidelity

<p align="center">
  <img src="assets/final_output.png" width="750"/>
</p>

---

# 🧠 Methodology

The framework combines traditional priors with modern deep learning methods.

<p align="center">
  <img src="assets/workflow.png" width="950"/>
</p>

---

## 🔹 Dark Channel Prior (DCP)

Dark Channel Prior is used to estimate atmospheric scattering and transmission maps.

<p align="center">
  <img src="assets/dcp.png" width="750"/>
</p>

---

## 🔹 CNN-Based Restoration

Deep convolutional layers learn haze-aware features for high-quality image reconstruction.

<p align="center">
  <img src="assets/cnn_model.png" width="850"/>
</p>

---

# 📂 Project Structure

```bash
A-Realtime-Image-Dehazing-using-CNN-and-DCP/
│
├── assets/
│   ├── architecture.png
│   ├── overview.png
│   ├── features.png
│   ├── workflow.png
│   ├── dcp.png
│   ├── cnn_model.png
│   ├── results.png
│   ├── input_hazy.png
│   ├── dcp_processing.png
│   ├── cnn_feature_extraction.png
│   ├── reconstruction.png
│   └── final_output.png
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

# 📊 Dataset

The dataset is excluded from this repository because of GitHub storage limitations.

---

## 🔹 RESIDE Dataset

Large-scale benchmark dataset for image dehazing.

### Official Website

https://sites.google.com/view/reside-dehaze-datasets/reside-v0

### Direct Download

https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAfRYnhTR7My/view

<p align="center">
  <img src="assets/reside_dataset.png" width="900"/>
</p>

---

## 🔹 O-HAZE Dataset

Real outdoor hazy image benchmark dataset.

### Dataset Link

https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/

<p align="center">
  <img src="assets/o_haze.png" width="850"/>
</p>

---

## 🔹 Dense-Haze Dataset

Dense haze benchmark dataset for evaluating dehazing performance.

### Dataset Link

https://data.vision.ee.ethz.ch/cvl/ntire18/dense-haze/

<p align="center">
  <img src="assets/dense_haze.png" width="850"/>
</p>

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/lalithtejakarnati/A-Realtime-Image-Dehazing-using-CNN-and-DCP.git

cd A-Realtime-Image-Dehazing-using-CNN-and-DCP
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

<p align="center">
  <img src="assets/installation.png" width="850"/>
</p>

---

# 🚀 Usage

---

## Train the Model

```bash
python Code/train.py
```

<p align="center">
  <img src="assets/training.png" width="850"/>
</p>

---

## Evaluate Model Performance

```bash
python Code/evaluate.py
```

<p align="center">
  <img src="assets/evaluation.png" width="850"/>
</p>

---

## Run Inference

```bash
python Code/inference.py
```

<p align="center">
  <img src="assets/inference.png" width="850"/>
</p>

---

## Launch Web Application

```bash
python Code/app.py
```

<p align="center">
  <img src="assets/webapp.png" width="900"/>
</p>

---

# 📈 Evaluation Metrics

| Metric            | Description                   |
| ----------------- | ----------------------------- |
| PSNR              | Peak Signal-to-Noise Ratio    |
| SSIM              | Structural Similarity Index   |
| Visual Assessment | Subjective quality evaluation |

<p align="center">
  <img src="assets/metrics.png" width="850"/>
</p>

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

<p align="center">
  <img src="assets/tech_stack.png" width="850"/>
</p>

---

# 🔮 Future Improvements

* Transformer-based dehazing
* Real-time video dehazing
* TensorRT optimization
* Mobile deployment
* GAN-based enhancement
* Attention-based architectures

<p align="center">
  <img src="assets/future_work.png" width="850"/>
</p>

---

# 🤝 Contribution

Contributions are welcome.

1. Fork repository
2. Create feature branch
3. Commit changes
4. Open pull request

<p align="center">
  <img src="assets/contribution.png" width="800"/>
</p>

---

# 👨‍💻 Author

## Karnati Lalith Teja

### GitHub

https://github.com/lalithtejakarnati

<p align="center">
  <img src="assets/author.png" width="300"/>
</p>

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
