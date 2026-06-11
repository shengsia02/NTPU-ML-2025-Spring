# NTPU Machine Learning 2025 Spring

This repository contains the source code and projects for the `Shallow Machine Learning` course at National Taipei University (NTPU), Department of Statistics.

The projects range from foundational unsupervised learning methods such as PCA and SVD to supervised classification models and deep learning-based image deblurring. Through Python implementations, visual analysis, model comparison, and experimental evaluation, these projects investigate how different machine learning methods can be applied to structured data, face image data, and image restoration tasks.

## 📋 Table of Contents

* [Project 1: Principal Component Analysis (PCA)](./Project_1)
* [Project 2: Singular Value Decomposition (SVD)](./Project_2)
* [Project 3: Face Image Classification](./Project_3)
* [Project 4: DeblurCNN for Image Deblurring](./Project_4/src)

## 📂 File Directory

| Project | Code | Report |
| :--- | :---: | :---: |
| **Project 1** | [`ipynb`](./Project_1/PCA.ipynb) | [`html`](https://htmlpreview.github.io/?https://github.com/shengsia02/NTPU-ML-2025-Spring/blob/main/Project_1/PCA.html) |
| **Project 2** | [`ipynb`](./Project_2/SVD.ipynb) | [`html`](https://htmlpreview.github.io/?https://github.com/shengsia02/NTPU-ML-2025-Spring/blob/main/Project_2/SVD.html) |
| **Project 3 - Part 1** | [`ipynb`](./Project_3/Classification_Part1.ipynb) | [`html`](https://htmlpreview.github.io/?https://github.com/shengsia02/NTPU-ML-2025-Spring/blob/main/Project_3/Classification_Part1.html) |
| **Project 3 - Part 2** | [`ipynb`](./Project_3/Classification_Part2.ipynb) | [`html`](https://htmlpreview.github.io/?https://github.com/shengsia02/NTPU-ML-2025-Spring/blob/main/Project_3/Classification_Part2.html) |
| **Project 4** | [`ipynb`](./Project_4/src/DeblurCNN.ipynb) | [`html`](https://htmlpreview.github.io/?https://github.com/shengsia02/NTPU-ML-2025-Spring/blob/main/Project_4/src/DeblurCNN.html) |

---

### Project 1: Principal Component Analysis (PCA)
This project studies **Principal Component Analysis (PCA)** on three structured datasets: the Wine dataset, the 2025 NUMBEO city quality-of-life dataset, and the Breast Cancer dataset.
* Used correlation heatmaps and boxplots to diagnose variable relationships and scaling problems, then compared PCA before and after standardization to show how feature scaling changes eigenvalues, explained variance ratios, scree plots, Pareto plots, and component interpretation.
* Visualized the first two or three principal components with 2D and 3D scatter plots to evaluate whether PCA-separated patterns match the original labels or grouped categories.
* Compared PCA loading structures with original correlation matrices to explain how highly correlated variables contribute to the same principal components.

### Project 2: Singular Value Decomposition (SVD)
This project investigates **Singular Value Decomposition (SVD)** for image compression, patch-based reconstruction, and eigenface-style image encryption and decryption.
* Compared PCA, NumPy SVD, and scikit-learn `TruncatedSVD` on handwritten digit image compression to verify their mathematical relationship and practical differences.
* Implemented Rank-$`q`$ approximation and showed that larger $q$ values preserve more image detail, while smaller $q$ values increase compression at the cost of blur.
* Tested patch-based SVD on images such as Lenna and Afghan Girl, then used Yale Faces singular-vector features to compare how well face-derived bases reconstruct face and non-face images.

### Project 3: Face Image Classification Using Logistic Regression, SVM, and Neural Network
This project compares **Multinomial Logistic Regression (MLR)**, **Support Vector Machine (SVM)**, and **Multilayer Perceptron (MLP)** models for face image classification under original and PCA-reduced feature representations.
* Part 1 uses the **AT&T face dataset** with 40 subjects and 400 images, where PCA keeps 90% cumulative explained variance and reduces the feature space to 59 principal components.
* On AT&T, MLR remains highly stable with and without PCA, SVM improves slightly after PCA, while MLP performs better on original features than on PCA features.
* Part 2 uses the **Yale Face dataset** with 38 subjects and 2410 images, showing that MLR is the strongest and most stable baseline, SVM is more sensitive to kernel and hyperparameter choices, and MLP benefits substantially from PCA in both accuracy and convergence speed.

### Project 4: DeblurCNN for Image Deblurring and Super-Resolution
This project applies **Convolutional Neural Networks (CNNs)** to image deblurring and super-resolution-style restoration under synthetic blur degradation.
* Prepared training data from the **T91 image dataset** by cropping 91 images into $35 \times 35$ patches, applying Gaussian blur, and adding 1/3 downscaling-upscaling degradation to simulate double blur.
* Trained and compared **DeblurCNN**, **DeblurCNN_RES**, **DeblurSuperResCNN**, **DnCNN**, and **EDSR** with patch-based inputs, tuned training settings, loss curves, and **PSNR (Peak Signal-to-Noise Ratio)**.
* Compared sharp, blurred, and restored outputs visually and quantitatively, then applied the selected model to an extra image outside the training and testing sets to evaluate generalization.

---

## ⚙️ Environment Setup

**1. Python Version** \
The development environment for this project is **Python 3.11.3**. It is recommended to refer to the `.python-version` file for configuration.

**2. Install PyTorch with CUDA 12.6** \
Before installing the remaining dependencies, please install PyTorch manually according to your operating system, Python version, and CUDA environment from the official PyTorch website.

For the CUDA 12.6 version used in this project, the pip installation command is:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

If your local GPU, driver, or CUDA environment is different, please visit the official PyTorch installation page and select the corresponding configuration before installation.

**3. Install Dependencies**  
After PyTorch is installed successfully, install the remaining packages using:

```bash
pip install -r requirements.txt
```
