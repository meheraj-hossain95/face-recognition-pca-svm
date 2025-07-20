# 🔍 Face Recognition with PCA-SVM
A beginner-friendly implementation of face recognition using Principal Component Analysis and Support Vector Machine.

## 📌 Overview
This project demonstrates classical face recognition techniques:
- **PCA**: Dimensionality reduction from 4096 to 108 features
- **SVM (RBF kernel)**: Classification with grid search optimization
- **Olivetti Dataset**: 400 face images, 40 people

## ✅ Results
- **Accuracy**: 96%
- **Dimensionality Reduction**: 97.4% (4096 → 108 features)
- **Variance Retained**: 95%

## ⚙️ Quick Start
```bash
# Install dependencies
pip install numpy matplotlib scikit-learn jupyter

# Run the notebook
jupyter notebook FaceRecognizer.ipynb
```

## 🔍 What's Included
- Face recognition pipeline with PCA preprocessing
- Hyperparameter tuning with GridSearchCV
- Visualization of eigenfaces and results
- Performance evaluation and confusion matrix

## Key Features
- Clean object-oriented design (`FaceRecognizer` class)
- Complete pipeline from data loading to evaluation
- Educational comments and step-by-step process
- Reproducible results with fixed random seed

## 📂 Dataset
**Olivetti Faces** (built-in with scikit-learn)
- 400 grayscale images (64×64 pixels)
- 40 subjects, 10 images per person

## 🛠️ Tools & Libraries
- Python 3.7+
- scikit-learn
- NumPy
- Matplotlib

## Learning Resources
Based on standard eigenfaces methodology from online tutorials and scikit-learn documentation.

## 📄 License
This project is released under the [MIT License](LICENSE).

---
*A learning project - feedback welcome!*
