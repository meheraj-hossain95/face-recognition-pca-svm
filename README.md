# Olivetti Faces Recognition using PCA-SVM

A comprehensive face recognition implementation using Principal Component Analysis (PCA) for dimensionality reduction and Support Vector Machine (SVM) with RBF kernel for classification on the Olivetti faces dataset.

## Project Overview

This project implements a robust face recognition system using classical machine learning techniques:

- **PCA**: Reduces 4096 pixel features to optimal components (retaining 95% variance)
- **SVM with RBF kernel**: High-performance classification with hyperparameter tuning
- **Grid Search**: Automated hyperparameter optimization for best model performance
- **Comprehensive Evaluation**: Detailed metrics, confusion matrix, and visualizations

## Results

- **Dataset**: 400 images, 40 people (10 images each), 64x64 pixels
- **Test Accuracy**: ~90-95% (varies with random state)
- **Feature Reduction**: 4096 → ~87 features (98% reduction)
- **Variance Retained**: 95% of original information

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy matplotlib scikit-learn jupyter
```

### Run the Notebook
```bash
jupyter notebook FaceRecognizer.ipynb
```

## Implementation

### Methodology:
1. **Data Loading**: Fetch Olivetti faces dataset from scikit-learn
2. **Data Preprocessing**: Standardize pixel values using StandardScaler  
3. **Dimensionality Reduction**: Apply PCA to reduce from 4096 to ~87 features
4. **Model Training**: Train SVM with RBF kernel using Grid Search CV
5. **Model Evaluation**: Comprehensive testing with classification metrics
6. **Visualization**: Display results, eigenfaces, and performance metrics

### Key Features:
- **Object-Oriented Design**: Clean, modular `FaceRecognizer` class
- **Hyperparameter Tuning**: Automated grid search for C and gamma parameters
- **Rich Visualizations**: Sample faces, eigenfaces, confusion matrix, variance plots
- **Performance Metrics**: Detailed classification report and accuracy analysis

## Technologies Used

- **Python 3.7+**
- **scikit-learn**: Machine learning algorithms and utilities
- **NumPy**: Numerical computing and array operations
- **Matplotlib**: Data visualization and plotting

## Project Structure

```
Algorithm/
├── FaceRecognizer.ipynb    # Main notebook with complete implementation
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── .gitignore             # Git ignore file
```

## What You'll See

The notebook demonstrates:
- **Sample Faces**: Visualization of dataset samples
- **Eigenfaces**: Top 9 principal components as "ghost faces"
- **Variance Analysis**: How much information each component captures
- **Confusion Matrix**: Detailed classification performance
- **Performance Summary**: Comprehensive results and metrics

## Technical Details

- **Dataset Size**: 400 samples × 4096 features (64×64 pixel images)
- **Classes**: 40 different people (10 images each)
- **Train/Test Split**: 75%/25% with stratified sampling
- **PCA Components**: Dynamic selection to retain 95% variance
- **SVM Parameters**: Grid search over C=[1,10,100], gamma=['scale',0.001,0.01]

## Educational Value

This project is excellent for learning:
- Dimensionality reduction with PCA
- Classification with Support Vector Machines
- Cross-validation and hyperparameter tuning
- Model evaluation and visualization techniques
- Object-oriented programming in machine learning

## License

This project is open source and available under the MIT License.

---

*A comprehensive implementation demonstrating PCA-SVM pipeline for face recognition tasks.*


