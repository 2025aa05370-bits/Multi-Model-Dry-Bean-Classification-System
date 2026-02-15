# Multi Model Dry Bean Classification Application

## Problem Statement
This project aims to develop a machine learning application to automatically classify different varieties of dry beans using geometric and morphological features extracted through a computer vision system. The goal is to implement and compare multiple machine learning classification models, evaluate their performance using standard metrics, and deploy the trained models into an interactive Streamlit web application for real-time predictions.

---

## Dataset Description
**Dataset**: `Dry_Bean_Dataset.csv`  
**Source**: Kaggle – Dry Bean Dataset by Murat Koklu  

This dataset contains measurements obtained from images of dry bean seeds captured using a high-resolution camera. A computer vision pipeline performs segmentation and feature extraction to compute shape-based and dimensional attributes for each bean.

Each row represents one bean sample and must be classified into one of seven bean varieties.

### Target Classes:
- SEKER
- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SIRA

---

## Dataset Features (Exact Column Names)

The dataset contains the following **16 numerical features + 1 target label**:

- `Area` – Pixel area of the bean region  
- `Perimeter` – Boundary length of the bean  
- `MajorAxisLength` – Longest axis of the bean  
- `MinorAxisLength` – Shortest axis of the bean  
- `AspectRation` – Ratio of major/minor axis length  
- `Eccentricity` – Ellipse eccentricity of the bean  
- `ConvexArea` – Area of the convex hull  
- `EquivDiameter` – Diameter of circle with equal area  
- `Extent` – Bounding box ratio  
- `Solidity` – Convexity ratio  
- `roundness` – Roundness measure (4πA / P²)  
- `Compactness` – Compactness of the bean  
- `ShapeFactor1`  
- `ShapeFactor2`  
- `ShapeFactor3`  
- `ShapeFactor4`  
- `Class` – Target label (bean type)

---

## Dataset Statistics
- **Number of Features**: 16  
- **Target Column**: Class  
- **Number of Instances**: 13,611  
- **Problem Type**: Multi-class classification  
- **Missing Values**: None  

---

## Models Used and Evaluation Metrics

Six different classification models were trained and compared.

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

- **Logistic Regression** performs well due to good feature separability  
- **Decision Tree** is interpretable but prone to overfitting  
- **kNN** provides strong accuracy but slower predictions  
- **Naive Bayes** is fast but assumes feature independence  
- **Random Forest (Ensemble)** improves stability using ensemble averaging  
- **XGBoost (Ensemble)** achieves the best overall performance across most metrics  

Overall, **XGBoost and Random Forest achieved the highest accuracy and robustness** for this dataset.

---

## Streamlit Web Application Features

The deployed web application provides:

- Upload test CSV file
- Model selection dropdown
- Classification report
- Confusion matrix visualization
- Performance metrics table
- Comparison of all trained models

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
│-- model/train_models.py
```

---

## How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/2025aa05370-bits/Multi-Model-Dry-Bean-Classification-System
cd Multi-Model-Dry-Bean-Classification-System
```

2. Create virtual environment (optional)
```bash
python -m venv myenv
myenv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Train models
```bash
python model/train_models.py
```

5. Run Streamlit app
```bash
streamlit run app.py
```

The application will open in your default browser.
