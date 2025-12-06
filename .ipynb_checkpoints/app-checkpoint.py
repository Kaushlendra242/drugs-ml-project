import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Drugs, Side Effects & Medical Condition – ML App",
    layout="centered"
)

# -------------------------
# Load models and encoders
# -------------------------
@st.cache_resource
def load_models_and_encoders():
    reg_model = joblib.load("drug_rating_regressor.pkl")   # RandomForestRegressor
    cls_model = joblib.load("rx_otc_classifier.pkl")       # RandomForestClassifier
    encoders = joblib.load("encoders.pkl")                 # dict of LabelEncoders
    return reg_model, cls_model, encoders

reg_model, cls_model, encoders = load_models_and_encoders()

le_generic = encoders["generic_name"]
le_medcond = encoders["medical_condition"]
le_csa = encoders["csa"]
le_preg = encoders["pregnancy_category"]
le_rxotc = encoders["rx_otc"]
le_side = encoders["side_effects"]

st.title("Drugs, Side Effects & Medical Condition – ML Predictions")

st.markdown(
    """
This app uses trained machine learning models on the **Drugs, Side Effects and Medical Condition** dataset.

You can:

- **Predict Drug Rating** (Regression – tuned Random Forest)
- **Predict Rx / OTC Type** (Classification – tuned Random Forest)

Feature selection is based on the encoded columns used during training.
"""
)

# -------------------------
# Task selection
# -------------------------
task = st.sidebar.selectbox(
    "Select Task",
    ["Predict Drug Rating (Regression)", "Predict Rx / OTC Type (Classification)"]
)

st.sidebar.markdown("### Input Features")

# -------------------------
# Dropdowns based on encoders
# -------------------------
# Generic name
generic_label = st.sidebar.selectbox(
    "Generic Name",
    options=list(le_generic.classes_)
)

# Medical condition
medical_label = st.sidebar.selectbox(
    "Medical Condition",
    options=list(le_medcond.classes_)
)

# Side effects (might be long list, but encoder-based)
side_effect_label = st.sidebar.selectbox(
    "Side Effects (encoded main pattern)",
    options=list(le_side.classes_)
)

# CSA schedule
csa_label = st.sidebar.selectbox(
    "CSA Schedule",
    options=list(le_csa.classes_)
)

# Pregnancy category
preg_label = st.sidebar.selectbox(
    "Pregnancy Category",
    options=list(le_preg.classes_)
)

# Alcohol interaction
alcohol_label = st.sidebar.selectbox(
    "Alcohol Interaction",
    options=["No interaction (0)", "Interacts with alcohol (1)"]
)
alcohol_val = 1.0 if "1" in alcohol_label else 0.0

# Number of reviews
no_of_reviews = st.sidebar.number_input(
    "Number of Reviews",
    min_value=0.0,
    step=1.0,
    value=0.0
)

# -------------------------
# Task-specific inputs
# -------------------------
if task == "Predict Drug Rating (Regression)":
    st.subheader("Task 1: Predict Drug Rating (Regression)")

    # For regression we need rx_otc_enc as feature
    rx_otc_label_input = st.sidebar.selectbox(
        "Rx / OTC Type (for rating model input)",
        options=list(le_rxotc.classes_)
    )

    # Encode categorical labels using encoders
    generic_enc = int(le_generic.transform([generic_label])[0])
    medcond_enc = int(le_medcond.transform([medical_label])[0])
    side_effect_enc = int(le_side.transform([side_effect_label])[0])
    csa_enc = int(le_csa.transform([csa_label])[0])
    preg_enc = int(le_preg.transform([preg_label])[0])
    rx_otc_enc_input = int(le_rxotc.transform([rx_otc_label_input])[0])

    if st.button("Predict Rating"):
        # Features for regression:
        # ['generic_name_enc','medical_condition_enc','no_of_reviews',
        #  'side_effects_enc','csa_enc','pregnancy_category_enc','rx_otc_enc','alcohol']
        data_dict = {
            "generic_name_enc": generic_enc,
            "medical_condition_enc": medcond_enc,
            "no_of_reviews": no_of_reviews,
            "side_effects_enc": side_effect_enc,
            "csa_enc": csa_enc,
            "pregnancy_category_enc": preg_enc,
            "rx_otc_enc": rx_otc_enc_input,
            "alcohol": alcohol_val,
        }

        X_new = pd.DataFrame([data_dict])

        pred_rating = reg_model.predict(X_new)[0]

        st.markdown("### Predicted Rating")
        st.write(f"**{pred_rating:.2f}** (on a 0–10 scale)")

else:
    st.subheader("Task 2: Predict Rx / OTC Type (Classification)")

    # For classification, rating is a feature instead of rx_otc_enc
    rating_val = st.sidebar.number_input(
        "Existing Rating (if known, else 0)",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        value=0.0
    )

    # Encode categorical labels
    generic_enc = int(le_generic.transform([generic_label])[0])
    medcond_enc = int(le_medcond.transform([medical_label])[0])
    side_effect_enc = int(le_side.transform([side_effect_label])[0])
    csa_enc = int(le_csa.transform([csa_label])[0])
    preg_enc = int(le_preg.transform([preg_label])[0])

    if st.button("Predict Rx / OTC Type"):
        # Features for classification:
        # ['generic_name_enc','medical_condition_enc','no_of_reviews',
        #  'side_effects_enc','rating','csa_enc','pregnancy_category_enc','alcohol']
        data_dict = {
            "generic_name_enc": generic_enc,
            "medical_condition_enc": medcond_enc,
            "no_of_reviews": no_of_reviews,
            "side_effects_enc": side_effect_enc,
            "rating": rating_val,
            "csa_enc": csa_enc,
            "pregnancy_category_enc": preg_enc,
            "alcohol": alcohol_val,
        }

        X_new = pd.DataFrame([data_dict])

        pred_class_enc = cls_model.predict(X_new)[0]
        pred_class_label = le_rxotc.inverse_transform([int(pred_class_enc)])[0]

        st.markdown("### Predicted Rx / OTC Type")
        st.write(f"Encoded class: **{int(pred_class_enc)}**")
        st.write(f"Original label: **{pred_class_label}**")
