# A-Realtime-Image-Dehazing-using-CNN-and-DCP

<p align="center">
  <img src="synthetic_haze_demo.png" width="750"/>
</p>

<h3 align="center">
Real-Time Single Image Dehazing using CNN and Dark Channel Prior (DCP)
</h3>

<p align="center">
  Deep Learning • Computer Vision • Image Restoration • PyTorch
</p>

---

## 📌 Overview

Atmospheric haze significantly degrades image quality by reducing visibility, contrast, and color fidelity. This project presents a hybrid real-time image dehazing framework that combines:

* **Convolutional Neural Networks (CNNs)** for feature extraction and restoration
* **Dark Channel Prior (DCP)** for haze estimation and enhancement
* **PyTorch-based training pipeline** for model optimization

The system is designed to restore clear scene details from hazy images while maintaining computational efficiency for near real-time inference.

---

# ✨ Features

* Real-time image dehazing
* CNN-based image restoration
* Dark Channel Prior enhancement
* PyTorch training pipeline
* Evaluation metrics support
* Flask-based web interface
* Modular and extensible architecture
* Inference support for custom images

---

# 🏗️ Project Architecture

```text id="n1"
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
<img width="1654" height="951" alt="image" src="https://github.com/user-attachments/assets/a8837453-fd26-4d1d-b4d7-7b8553e9bee7" />

---

# 📂 Project Structure

```bash id="n2"
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

The proposed framework integrates traditional image priors with deep learning techniques:

## 1. Dark Channel Prior (DCP)

Dark Channel Prior estimates atmospheric light and transmission maps to improve haze removal performance.

## 2. CNN-Based Restoration

The CNN learns haze-relevant features and reconstructs visually enhanced images with improved:

* contrast
* texture details
* visibility
* color consistency

## 3. Post-Processing

Final enhancement and refinement improve visual quality and reduce artifacts.

---

# 📊 Dataset

The dataset is excluded from this repository because of GitHub storage limitations.

## Recommended Datasets

### 🔹 RESIDE Dataset

Large-scale benchmark dataset for single image dehazing.

* Official Website:
  https://sites.google.com/view/reside-dehaze-datasets/reside-v0

* Direct Download:
  https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAfRYnhTR7My/view

---

### 🔹 O-HAZE Dataset

Real outdoor hazy image benchmark dataset.

* Dataset Link:
  https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/

---

### 🔹 Dense-Haze Dataset

Dense haze benchmark dataset for evaluating dehazing performance.

* Dataset Link:
  https://data.vision.ee.ethz.ch/cvl/ntire18/dense-haze/

---

# ⚙️ Installation

## Clone Repository

```bash id="n3"
git clone https://github.com/lalithtejakarnati/A-Realtime-Image-Dehazing-using-CNN-and-DCP.git

cd A-Realtime-Image-Dehazing-using-CNN-and-DCP
```

---

## Install Dependencies

```bash id="n4"
pip install -r requirements.txt
```

---

# 🚀 Usage

## Train the Model

```bash id="n5"
python Code/train.py
```

---

## Evaluate Model Performance

```bash id="n6"
python Code/evaluate.py
```

---

## Run Inference

```bash id="n7"
python Code/inference.py
```

---

## Launch Web Application

```bash id="n8"
python Code/app.py
```

---

# 📈 Evaluation Metrics

The model performance can be evaluated using:

* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity Index)
* Visual Quality Assessment

---

# 🖼️ Results

The proposed model improves:

* Scene visibility
* Contrast restoration
* Edge sharpness
* Color fidelity
* Texture recovery

| Input Hazy Image | Dehazed Output       |
| ---------------- | -------------------- |
| Hazy Scene       | Clear Restored Scene |

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

* Transformer-based dehazing architectures
* Real-time video dehazing
* Lightweight mobile deployment
* TensorRT optimization
* Attention-based restoration networks
* GAN-based enhancement models

---

# 🤝 Contribution

Contributions are welcome.

If you would like to improve the project:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

# 👨‍💻 Author

## Karnati Lalith Teja

* GitHub:
  https://github.com/lalithtejakarnati

---

# 📜 License

This project is intended for:

* academic research
* educational purposes
* experimentation in computer vision and deep learning

---

# ⭐ Acknowledgements

Special thanks to:

* RESIDE Dataset creators
* PyTorch community
* OpenCV contributors
* Computer Vision research community
