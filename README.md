# A-Realtime-Image-Dehazing-using-CNN-and-DCP

<img width="1654" height="951" alt="Screenshot 2026-05-28 at 4 39 47 PM" src="https://github.com/user-attachments/assets/06e6d0ef-0084-4451-865b-79ae846bd6a4" />


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
<img width="1654" height="951" alt="image" src="https://github.com/user-attachments/assets/fbda28c7-76ad-4213-ba8f-183296934697" />

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
<img width="119" height="83" alt="Screenshot 2026-05-28 at 4 41 08 PM" src="https://github.com/user-attachments/assets/b0f72984-dc00-4178-8848-1c0330cb87eb" />

Large-scale benchmark dataset for single image dehazing.

* Official Website:
  https://sites.google.com/view/reside-dehaze-datasets/reside-v0

* Direct Download:
  https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAfRYnhTR7My/view

---

### 🔹 O-HAZE Dataset
<img width="115" height="79" alt="Screenshot 2026-05-28 at 4 41 14 PM" src="https://github.com/user-attachments/assets/18c0bfb5-6aaf-496f-9a2f-386d978e25b0" />

Real outdoor hazy image benchmark dataset.

* Dataset Link:
  https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/

---

### 🔹 Dense-Haze Dataset
<img width="120" height="80" alt="Screenshot 2026-05-28 at 4 41 19 PM" src="https://github.com/user-attachments/assets/2bd1b6d1-a577-4735-a337-8d25dbe82df4" />

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

<img width="253" height="95" alt="Screenshot 2026-05-28 at 4 40 53 PM" src="https://github.com/user-attachments/assets/56dd45ce-f127-4771-9946-01e7099173fe" />

---

# 🛠️ Technologies Used
<img width="174" height="67" alt="Screenshot 2026-05-28 at 4 41 50 PM" src="https://github.com/user-attachments/assets/fc9cbeb8-872b-4bf1-a0b7-430918d8db68" />

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
<img width="248" height="72" alt="Screenshot 2026-05-28 at 4 41 57 PM" src="https://github.com/user-attachments/assets/f0fcdd48-5abd-451b-848f-44790e5fe60c" />

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
