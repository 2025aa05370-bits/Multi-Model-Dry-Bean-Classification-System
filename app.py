import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Dry Bean Classification ML Dashboard")

st.title("Dry Bean Classification ML Dashboard")

st.markdown("""
This application performs Multi-Class Classification on the Dry Bean Dataset.

Each bean sample is classified into one of seven varieties:

SEKER, BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SIRA
""")

@st.cache_resource
def load_resources():
    models = {
        "Logistic Regression": joblib.load("model/Logistic Regression.pkl"),
        "Decision Tree": joblib.load("model/Decision Tree.pkl"),
        "KNN": joblib.load("model/KNN.pkl"),
        "Naive Bayes": joblib.load("model/Naive Bayes.pkl"),
        "Random Forest": joblib.load("model/Random Forest.pkl"),
        "XGBoost": joblib.load("model/XGBoost.pkl"),
    }
    scaler = joblib.load("model/scaler.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    metrics = pd.read_csv("model/metrics.csv")
    return models, scaler, label_encoder, metrics


models, scaler, le, metrics = load_resources()

st.sidebar.header("Controls")

model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
uploaded = st.sidebar.file_uploader("Upload Test CSV", type="csv")
run_btn = st.sidebar.button("Run Evaluation / Prediction")

if uploaded and run_btn:
    df = pd.read_csv(uploaded)

    X = df.drop(columns=["Class"], errors="ignore")

    X_scaled = scaler.transform(X)
    model = models[model_name]
    preds = model.predict(X_scaled)

    pred_labels = le.inverse_transform(preds)

    result_df = df.copy()
    result_df["Predicted_Class"] = pred_labels

    st.subheader("Predicted Bean Type Results")
    st.dataframe(result_df)

    st.download_button(
        "Download Predictions CSV",
        result_df.to_csv(index=False),
        "predictions.csv",
        "text/csv"
    ) 

    if "Class" in df.columns:
        y = le.transform(df["Class"])

        st.subheader("Classification Report")
        st.text(classification_report(y, preds))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, preds)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=ax
        )
        st.pyplot(fig)

st.subheader("Model Comparison Metrics")
st.dataframe(metrics)