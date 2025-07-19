# Olivetti Faces Recognition using PCA-SVM

A clean and simple implementation of face recognition using Principal Component Analysis (PCA) for dimensionality reduction and Support Vector Machine (SVM) with RBF kernel for classification.

## 📌 Overview

This project demonstrates a basic yet effective approach to face recognition using classical machine learning techniques:

* **PCA**: Reduces high-dimensional image data while preserving key facial features
* **SVM (RBF kernel)**: Performs classification with good generalization
* **Grid Search CV**: Tunes hyperparameters (C, gamma) for optimal model performance
* **Visualization**: Includes eigenfaces, explained variance, confusion matrix, and predictions

## 📂 Dataset

* **Name**: Olivetti Faces Dataset (from `scikit-learn`)
* **Size**: 400 grayscale images (64×64 pixels)
* **Subjects**: 40 individuals (10 images per person)

## ✅ Final Results

| Metric                   | Value                 |
| ------------------------ | --------------------- |
| Input Dimensions         | 4096 features (64×64) |
| PCA Components Used      | 108                   |
| Reduced Features         | 108                   |
| Variance Retained        | 95.03%                |
| Dimensionality Reduction | 97.4%                 |
| Test Accuracy            | 96.00%                |
| Model Used               | SVM (RBF kernel)      |

```python
{
  'accuracy': 0.96,
  'pca_components': 108,
  'original_features': 4096,
  'variance_retained': np.float32(0.9502931),
  'dimensionality_reduction': 97.36328125
}
```

## ⚙️ How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy matplotlib scikit-learn jupyter
```

### 2. Launch Jupyter Notebook

```bash
jupyter notebook FaceRecognizer.ipynb
```

## 🔍 Methodology

1. **Load Data**: Use `fetch_olivetti_faces()` from `sklearn.datasets`
2. **Preprocessing**: Standardize image features using `StandardScaler`
3. **PCA**: Reduce 4096 features to retain 95% variance (108 components used)
4. **SVM Classification**: Use `SVC(kernel='rbf')` with Grid Search CV for hyperparameter tuning
5. **Evaluation**: Generate classification report, confusion matrix, and supporting visualizations

## 📁 Project Structure

```
Algorithm/
├── FaceRecognizer.ipynb     # Main notebook with implementation
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

```


## 🧪 Tools & Libraries

* **Python 3.7+**
* **scikit-learn** – Machine learning models & utilities
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization

## 🧪 License

This project is released under the [MIT License](LICENSE).
