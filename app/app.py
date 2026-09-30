import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib


# ============================================================
# NUTRIRATE AI - WORKING APPLICATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "nutrirate_aml_random_forest.joblib"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NutriRate AI",
    page_icon="🥗",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("NutriRate AI")

st.subheader(
    "AI-Based Nutritional Quality Prediction"
)

st.write(
    "Enter the nutritional values from a food product label "
    "to predict its Nutri-Score grade using a trained "
    "machine learning model."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Nutrition Information")

col1, col2, col3 = st.columns(3)

with col1:

    energy = st.number_input(
        "Energy (kcal)",
        min_value=0.0,
        value=300.0,
        step=1.0
    )

    fat = st.number_input(
        "Fat (g)",
        min_value=0.0,
        value=10.0,
        step=0.1
    )

    saturated_fat = st.number_input(
        "Saturated Fat (g)",
        min_value=0.0,
        value=3.0,
        step=0.1
    )

with col2:

    carbohydrates = st.number_input(
        "Carbohydrates (g)",
        min_value=0.0,
        value=40.0,
        step=0.1
    )

    sugars = st.number_input(
        "Sugars (g)",
        min_value=0.0,
        value=10.0,
        step=0.1
    )

    protein = st.number_input(
        "Protein (g)",
        min_value=0.0,
        value=10.0,
        step=0.1
    )

with col3:

    fiber = st.number_input(
        "Fiber (g)",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

    salt = st.number_input(
        "Salt (g)",
        min_value=0.0,
        value=1.0,
        step=0.01
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

EPS = 1e-6


def safe_ratio(a, b):

    if abs(b) <= EPS:
        return np.nan

    return a / b


def create_features():

    return pd.DataFrame([{

        "energy": energy,

        "fat": fat,

        "saturated_fat": saturated_fat,

        "carbohydrates": carbohydrates,

        "sugars": sugars,

        "protein": protein,

        "fiber": fiber,

        "salt": salt,

        "saturated_fat_ratio":
            safe_ratio(saturated_fat, fat),

        "sugar_carb_ratio":
            safe_ratio(sugars, carbohydrates),

        "fiber_carb_ratio":
            safe_ratio(fiber, carbohydrates),

        "protein_carb_ratio":
            safe_ratio(protein, carbohydrates),

        "protein_energy_ratio":
            safe_ratio(protein, energy),

        "fat_energy_ratio":
            safe_ratio(fat, energy),

        "sugar_energy_ratio":
            safe_ratio(sugars, energy),

        "fiber_energy_ratio":
            safe_ratio(fiber, energy),

        "salt_energy_ratio":
            safe_ratio(salt, energy)

    }])


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "Predict Nutri-Score",
    type="primary",
    use_container_width=True
):

    features = create_features()

    prediction = model.predict(features)[0]

    # Probability if available
    probabilities = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            features
        )[0]

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.success(
        f"Predicted Nutri-Score Grade: {prediction}"
    )

    st.metric(
        "Predicted Grade",
        prediction
    )

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    if probabilities is not None:

        st.subheader("Prediction Confidence")

        classes = model.classes_

        probability_df = pd.DataFrame({

            "Grade": classes,

            "Probability": probabilities

        })

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        )

        probability_df["Probability"] = (
            probability_df["Probability"].round(2)
        )

        st.bar_chart(
            probability_df.set_index("Grade")
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# INFORMATION
# ============================================================

st.divider()

st.header("About the Model")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:

    st.metric(
        "Model",
        "Random Forest"
    )

with info_col2:

    st.metric(
        "Test Accuracy",
        "88.85%"
    )

with info_col3:

    st.metric(
        "Macro F1",
        "0.8751"
    )

st.caption(
    "Model trained on the NutriRate AI nutritional dataset "
    "using nutritional attributes and engineered ratio features."
)