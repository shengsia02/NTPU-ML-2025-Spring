# NTPU Machine Learning 2025 Spring

This repository contains the source code and projects for the `Shallow Machine Learning` course at National Taipei University (NTPU), Department of Statistics.

The projects range from foundational unsupervised learning methods such as PCA and SVD to supervised classification models and deep learning-based image deblurring. Through Python implementations, visual analysis, model comparison, and experimental evaluation, these projects investigate how different machine learning methods can be applied to structured data, face image data, and image restoration tasks.

## 📋 Table of Contents

* [Project 1: Principal Component Analysis (PCA)](./Project_1)
* [Project 2: Singular Value Decomposition (SVD)](./Project_2)
* [Project 3: Face Image Classification](./Project_3)
* [Project 4: DeblurCNN for Image Deblurring](./Project_4)

## 📂 File Directory

| Project | Code | Report |
| :--- | :---: | :---: |
| **Project 1** | [`.ipynb`](./Project_1/PCA.ipynb) | [`html`](https://htmlpreview.github.io/?https://github.com/shengsia02/NTPU-ML-2025-Spring/blob/main/Project_1/PCA.html) |
| **Project 2** | [`.ipynb`](./Project_2/SVD.ipynb) | [`html`](https://htmlpreview.github.io/?https://github.com/shengsia02/NTPU-ML-2025-Spring/blob/main/Project_2/SVD.html) |
| **Project 3 - Part 1** | [`.ipynb`](./Project_3/Classification_Part1.ipynb) | [`html`](https://htmlpreview.github.io/?https://github.com/shengsia02/NTPU-ML-2025-Spring/blob/main/Project_3/Classification_Part1.html) |
| **Project 3 - Part 2** | [`.ipynb`](./Project_3/Classification_Part2.ipynb) | [`html`](https://htmlpreview.github.io/?https://github.com/shengsia02/NTPU-ML-2025-Spring/blob/main/Project_3/Classification_Part2.html) |
| **Project 4** | [`.ipynb`](./Project_4/DeblurCNN.ipynb) | [`html`](https://htmlpreview.github.io/?https://github.com/shengsia02/NTPU-ML-2025-Spring/blob/main/Project_4/DeblurCNN.html) |

---

### Project 1: Principal Component Analysis (PCA)
This project explores the application of **Principal Component Analysis (PCA)** for dimensionality reduction and data visualization.
* Applied PCA to multiple datasets to investigate how high-dimensional variables can be transformed into lower-dimensional principal components.
* Compared PCA results before and after standardization to analyze how feature scaling affects eigenvalues, explained variance ratios, and the interpretation of principal components.
* Used visualization methods such as correlation heatmaps, boxplots, scree plots, Pareto plots, and 2D/3D scatter plots to examine whether PCA results align with the original group labels.
* Discussed the relationship between the original variable correlations and the principal component loading structure to better understand how PCA extracts important information from data.

### Project 2: Singular Value Decomposition (SVD)
This project investigates **Singular Value Decomposition (SVD)** and its applications in image matrix analysis.
* Compared PCA and SVD from the perspective of image compression to verify their mathematical relationship and practical differences.
* Implemented Rank-$q$ approximation to compress image matrices and observed how different values of $q$ affect image clarity and reconstruction quality.
* Explored whether dividing an image into smaller blocks before applying SVD can improve visual quality when using a small rank approximation.
* Applied face-based singular vector features to image encoding and decoding tasks, then compared reconstruction performance between face images and non-face images.

### Project 3: Face Image Classification Using Logistic Regression, SVM, and Neural Network
This project compares three supervised classification models for face image recognition: **Multinomial Logistic Regression**, **Support Vector Machine (SVM)**, and **Neural Network**.
* Used both original image features and PCA-reduced features to train and evaluate classification models.
* Compared classification accuracy, training efficiency, convergence behavior, and generalization ability under different feature representations.
* Part 1 focuses on the **AT&T face dataset**, which contains face images from 40 subjects.
* Part 2 extends the experiment to the **Yale Face dataset**, which contains face images from 38 subjects.
* Analyzed whether PCA can reduce feature dimensionality while preserving sufficient information for accurate face recognition.

### Project 4: DeblurCNN for Image Deblurring and Super-Resolution
This project applies **Convolutional Neural Networks (CNNs)** to image deblurring and super-resolution tasks.
* Prepared training data by cropping images into small patches, applying Gaussian blur, and using downscaling-upscaling degradation to simulate realistic blur conditions.
* Trained and compared several CNN-based models, including **DeblurCNN**, **DeblurCNN_RES**, **DeblurSuperResCNN**, **DnCNN**, and **EDSR**.
* Used loss curves and **PSNR (Peak Signal-to-Noise Ratio)** to evaluate model convergence and image restoration performance.
* Compared sharp images, blurred images, and restored images to visually and quantitatively assess each model's deblurring ability.
* Applied the selected trained model to an additional image outside the training and testing sets to evaluate practical generalization performance.

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
