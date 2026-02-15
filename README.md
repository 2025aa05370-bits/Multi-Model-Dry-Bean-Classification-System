# Multi Model Dry Bean Classification Application

## Problem Statement
This project aims to develop a machine learning application to automatically classify different varieties of dry beans based on their geometric and morphological features extracted using a computer vision system.  

The goal is to implement and compare multiple classification models, evaluate their performance using standard metrics, and deploy the trained models into an interactive Streamlit web application for real-time predictions.

This project demonstrates a complete end-to-end Machine Learning workflow including data preprocessing, model training, evaluation, and deployment.

---

## Dataset Description
**Dataset**: `Dry_Bean_Dataset.csv`  
**Source**: Kaggle / UCI Machine Learning Repository  

The dataset contains images of dry bean seeds captured using a high-resolution camera. A computer vision pipeline was used for segmentation and feature extraction. Each bean is represented using shape and dimensional measurements.

The task is to classify each bean into one of seven different registered varieties.

### Target Classes:
- Seker
- Barbunya
- Bombay
- Cali
- Dermosan
- Horoz
- Sira

### Key Features:
- `Area`: Pixel area of bean region
- `Perimeter`: Bean boundary length
- `MajorAxisLength`: Longest axis length
- `MinorAxisLength`: Shortest axis length
- `AspectRatio`: Ratio of major/minor axis
- `Eccentricity`: Ellipse eccentricity
- `ConvexArea`: Area of convex hull
- `EquivDiameter`: Diameter of equal area circle
- `Extent`: Bounding box ratio
- `Solidity`: Convexity ratio
- `Roundness`: (4πA / P²)
- `Compactness`: Roundness measure
- `ShapeFactor1`
- `ShapeFactor2`
- `ShapeFactor3`
- `ShapeFactor4`

**Number of Features**: 16  
**Number of Instances**: 13,611  
**Problem Type**: Multi-class classification  
**Missing Values**: None  

---

## Models Used and Evaluation Metrics

Six different classification models were implemented and evaluated on the same dataset.  

### Models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Evaluation Metrics:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:----------------|:---------|:---------|:-----------|:---------|:---------|:---------|
| Logistic Regression | 0.9210 | 0.9948 | 0.9351 | 0.9319 | 0.9333 | 0.9046 |
| Decision Tree | 0.8928 | 0.9454 | 0.9092 | 0.9097 | 0.9094 | 0.8704 |
| kNN | 0.9170 | 0.9833 | 0.9323 | 0.9274 | 0.9296 | 0.8996 |
| Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Random Forest (Ensemble) | 0.9210 | 0.9920 | 0.9360 | 0.9309 | 0.9333 | 0.9045 |
| XGBoost (Ensemble) | 0.9240 | 0.9950 | 0.9385 | 0.9335 | 0.9359 | 0.9080 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|-----------|------------------------------------|
| Logistic Regression | Performs strongly due to good linear separability of features |
| Decision Tree | Simple and interpretable but prone to overfitting |
| kNN | Good accuracy but slower prediction for larger datasets |
| Naive Bayes | Fast but assumes feature independence, reducing performance |
| Random Forest (Ensemble) | More stable and accurate due to ensemble averaging |
| XGBoost (Ensemble) | Best overall performance with highest accuracy, AUC and MCC |

Overall, **XGBoost and Random Forest achieved the best results**, showing that ensemble learning methods work effectively for this multi-class classification problem.

---

## Streamlit Web Application Features
The deployed web application includes:
- CSV dataset upload option
- Model selection dropdown
- Evaluation metrics display
- Classification report
- Confusion matrix visualization
- Model comparison table

---

## Project Structure
```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│     ├── *.pkl (saved models)
│     ├── scaler.pkl
│     ├── label_encoder.pkl
│     └── metrics.csv
│-- models/
│     └── train_models.py
```

---

## How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/2025aa05370-bits/Multi-Model-Dry-Bean-Classification-System
cd Multi-Model-Dry-Bean-Classification-System
```

2. Create a virtual environment (optional):
```bash
python -m venv myenv
myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train models:
```bash
python model/train_models.py
```

5. Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser.
