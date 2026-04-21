# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project builds a machine learning pipeline to predict the presence of heart disease using clinical and lifestyle data from the UCI Heart Disease dataset.

The goal is to compare different classification models and evaluate their ability to detect heart disease cases accurately.

---

## 🧠 Problem Type

* **Type:** Supervised Learning
* **Task:** Binary Classification
* **Target Variable:** `num`

  * `0` → No heart disease
  * `1` → Heart disease present

---

## 📊 Dataset

* Source: UCI Heart Disease Dataset
* Contains patient health indicators such as:

  * Age
  * Cholesterol
  * Blood pressure
  * Chest pain type
  * Exercise-induced angina
  * Maximum heart rate

---

## ⚙️ Workflow

### 1. Data Preprocessing

* Removed irrelevant columns (e.g., ID)
* Converted target variable into binary format
* Handled missing values using imputation
* Encoded categorical variables using One-Hot Encoding

---

### 2. Model Training

Two models were trained and compared:

* Logistic Regression (baseline)
* Random Forest (non-linear model)

---

### 3. Model Evaluation

Evaluation metrics used:

* Accuracy
* Confusion Matrix
* Precision, Recall, F1-score

---

## 📈 Results

| Model               | Accuracy   |
| ------------------- | ---------- |
| Logistic Regression | 84.24%     |
| Random Forest       | **85.33%** |

---

### 🔍 Key Insight

* Logistic Regression achieved **higher recall (0.90)** for detecting heart disease
* Random Forest achieved **higher overall accuracy and precision**

---

## 🧠 Feature Importance (Random Forest)

Top predictive features included:

* Chest pain type (asymptomatic)
* Cholesterol level
* Maximum heart rate
* Age
* ST depression (oldpeak)
* Exercise-induced angina

These features align with known medical risk factors for heart disease.

---

## ⚠️ Why Accuracy Alone is Not Enough

Accuracy does not capture the cost of misclassification.
In healthcare, failing to detect a disease (false negative) is more critical than a false positive.
Therefore, recall and precision are essential metrics.

---

## 🚀 Future Improvements

* Hyperparameter tuning
* Cross-validation
* ROC-AUC analysis
* Testing advanced models (e.g., Gradient Boosting)

---

## 🧩 Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib

---

## 📌 Conclusion

Both models successfully detected heart disease cases.
Random Forest performed best overall, while Logistic Regression demonstrated strong recall, making it suitable for sensitive detection tasks.

---

## 💼 Author

**Nqaba Jabulani Sikeyi**
MSc Computer Science
"""+
"""Machine Learning & Data Enthusiast
