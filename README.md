# 🔬 **Drugs, Side Effects & Medical Condition – Machine Learning Project**

### **🚀 Live Demo:** [https://drugs-ml-project-kaushlendra-ds.streamlit.app/]

## **End-to-End Data Science Pipeline**
**EDA → Feature Engineering → Regression → Classification → Deployment**

This repository contains a production-ready machine learning project built on a real-world pharmaceutical dataset covering:

- **Drug names & generic names**
- **Medical conditions**
- **Side effects**
- **Pregnancy category**
- **CSA schedule**
- **Rx/OTC regulatory classification**
- **Alcohol interaction**
- **User reviews & drug ratings**

The project solves two predictive ML tasks:

### ✅ **1. Predict Drug Rating (Regression)**  
### ✅ **2. Predict Rx / OTC Type (Classification)**  

---

# 📌 **Project Includes**
- ✔ **Complete Jupyter Notebook**
- ✔ **Thorough EDA**
- ✔ **Feature Engineering**
- ✔ **Regression & Classification Models**
- ✔ **Hyperparameter Tuning**
- ✔ **Final Evaluation Metrics**
- ✔ **Streamlit App**
- ✔ **Saved Label Encoders**
- ✔ **PDF / DOCX Reports**

---

# 📁 **Project Folder Structure**

drugs-ml-project/
├── drugs_side_effects_medical_condition_drugs_ml_predict.ipynb # Full workflow: EDA → ML → Tuning
├── app.py # Streamlit app
├── drug_rating_regressor.pkl # Tuned RandomForestRegressor
├── rx_otc_classifier.pkl # Tuned RandomForestClassifier
├── encoders.pkl # LabelEncoders for dropdowns
├── requirements.txt # Dependencies
│
├── data/
│ └── drugs_side_effects_drugs_com.csv
│
├── reports/
│ └── Drugs_Project_Summary_Basic.pdf
│
└── README.md


---

# 🧼 **Data Cleaning & Preprocessing**

### ✔ **Key Steps**
- Extracted and cleaned:
  - **generic_name**, **medical_condition**, **side_effects**, **drug_classes**
  - **rx_otc**, **pregnancy_category**, **csa**, **alcohol**
- Converted **activity → 0/1**
- Converted **alcohol: "X" → 1, NaN → 0**
- Missing text fields → **"Unknown"**
- Missing numeric fields → **0**
- Label Encoded:
  - **generic_name**, **medical_condition**, **side_effects**
  - **pregnancy_category**, **csa**, **rx_otc**

### ✔ **Final Dataset**
- **Rows:** 2,931  
- **Columns:** 16  
- **No missing values remain**

---

# 📊 **Exploratory Data Analysis (EDA)**

### 🔹 **Frequent Medical Conditions**
Pain, Colds & Flu, Acne, Hypertension, Infection, etc.

### 🔹 **Common Side Effects**
Hives, difficulty breathing, swelling, rash, dizziness, nausea.

### 🔹 **Most Common Drug Classes**
Upper respiratory combinations, topical steroids, acne agents, antibiotics.

Visualizations available in the notebook.

---

# 🤖 **Machine Learning – Regression (Predict Drug Rating)**

## **Baseline Model Comparison**

| Model                       | RMSE  | R²    |
|-----------------------------|-------|-------|
| LinearRegression            | 3.585 | 0.109 |
| **RandomForestRegressor**   | **1.425** | **0.859** |
| GradientBoostingRegressor   | 1.612 | 0.820 |
| CatBoostRegressor           | 1.568 | 0.830 |

### 🏆 **Best Baseline Model → RandomForestRegressor**

---

## **Hyperparameter Tuning (Random Forest)**

**Best Parameters:**

max_depth = 20
min_samples_leaf = 2
min_samples_split = 2
n_estimators = 200


**Best CV RMSE:** `1.6322`

---

## 🎯 **Final Regression Performance (Test Set)**

| Metric      | Score   |
|-------------|---------|
| **RMSE**    | **1.4618** |
| **MAE**     | **0.7877** |
| **R² Score** | **0.8520** |

### ✔ **Model Insights**
- Explains **85.2%** of rating variance  
- Performs strongly on a **0–10** scale  
- Robust for non-linear medical data  

---

# 🤖 **Machine Learning – Classification (Predict Rx/OTC Type)**

## **Baseline Model Comparison**

| Model                   | Accuracy | F1-Weighted |
|-------------------------|----------|-------------|
| Logistic Regression     | 0.686    | 0.616       |
| **RandomForestClassifier** | **0.901** | **0.899** |
| Gradient Boosting       | 0.882    | 0.878       |
| CatBoostClassifier      | 0.879    | 0.876       |

### 🏆 **Best Baseline Model → RandomForestClassifier**

---

## **Hyperparameter Tuning (Random Forest)**

**Best Parameters:**
max_depth = 20
min_samples_leaf = 1
min_samples_split = 2
n_estimators = 300


**Best CV Accuracy:** `0.8827`

---

## 🎯 **Final Classification Performance**

| Metric          | Score |
|------------------|--------|
| **Accuracy**     | **0.901** |
| **F1-weighted**  | **0.899** |

### **Classification Report**
Class 0 → F1 = 0.76
Class 1 → F1 = 0.95
Class 2 → F1 = 0.81
Overall Accuracy = 0.90


---

# 🌐 **Streamlit Deployment**

The Streamlit app (`app.py`) provides:

### ✔ **Prediction Modes**
- Drug Rating Prediction  
- Rx/OTC Prediction  

### ✔ Features
- Dropdowns powered by **LabelEncoders**
- Automatic encoding before prediction
- Clean UI

---

# ▶️ **Run the Streamlit App**

Install dependencies:

pip install -r requirements.txt


Run app:
streamlit run app.py


---

# 🚀 **Deploy on Streamlit Cloud**

1. Visit https://streamlit.io/cloud  
2. Connect GitHub  
3. Click **New App**  
4. Choose:
   - Repo: `drugs-ml-project`
   - Branch: `main`
   - File: `app.py`
5. Click **Deploy**

---

# 📦 **Installation (Local Machine)**

git clone https://github.com/<Kaushlendra242>/drugs-ml-project.git

cd drugs-ml-project

pip install -r requirements.txt

jupyter notebook drugs_side_effects_medical_condition_drugs_ml_predict.ipynb


---

# 🧾 **Reports Included**

Located in `/reports`:

- **Basic Project Summary (PDF/DOCX)**


Includes:
- Full results  
- Tables & scores  
- Diagrams  
- Conclusions  

---

# 🧠 **Key Insights**

- **Random Forest** is the best-performing algorithm for both tasks  
- Regression model predicts ratings with **R² = 0.852**  
- Classification model predicts Rx/OTC type with **~90% accuracy**  
- Encoded features significantly boost performance  
- Dataset offers deep insights into healthcare and drug analytics  

---

# 👤 **Author**

**Kaushlendra Pratap Singh**  
Data Analyst | Machine Learning | Data Science Practitioner  
GitHub: https://github.com/Kaushlendra242
