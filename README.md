# Heart Disease Classification

**Comparative machine learning | Cross-validation · Hyperparameter tuning · Threshold optimization**

A reproducible binary classification study using **920 patient records** from the UCI Heart Disease dataset. The project compares **Logistic Regression** and **Random Forest** to predict recorded heart disease presence from 13 clinical features. Beyond overall accuracy, the analysis examines how hyperparameter selection and classification thresholds affect **recall, false negatives, and false positives**.

> **Scope:** A machine learning benchmark on observational data—not a clinically validated screening or diagnostic system.

## Project highlights

- **920 records**, **13 predictors**, and an **80/20 stratified train–test split** (736 training; 184 testing).
- **Stratified 5-fold cross-validation** for baseline comparison and training-only hyperparameter search.
- **`RandomizedSearchCV`** for both models, with ROC-AUC as the model-selection metric.
- **Out-of-fold threshold selection** using the F2-score, which gives more weight to recall than precision.
- **Held-out evaluation** reporting accuracy, precision, recall, F1, F2, ROC-AUC, average precision, and confusion matrices.
- **Permutation feature importance** to investigate which inputs influence the fitted model's predictions.

## Results

### 1. Hyperparameter tuning

Mean ROC-AUC from stratified 5-fold cross-validation on the **training partition**:

| Model | Baseline ROC-AUC | Tuned ROC-AUC | Change |
|---|---:|---:|---:|
| Logistic Regression | 0.8850 | 0.8864 | +0.0013 |
| **Random Forest** | 0.8838 | **0.8871** | **+0.0033** |

**Random Forest** was selected because it achieved the highest mean tuned cross-validation ROC-AUC. The gains were modest, and the difference between tuned models was too small to establish a meaningful performance advantage on its own.

### 2. Final model: effect of the classification threshold

Held-out performance of the **tuned Random Forest**, evaluated on **184 records** (102 positive, 82 negative):

| Metric | Default threshold (0.50) | F2-selected threshold (0.24) |
|---|---:|---:|
| Accuracy | **85.3%** | 79.9% |
| Precision | **85.0%** | 73.7% |
| Recall / sensitivity | 89.2% | **99.0%** |
| F1-score | **87.1%** | 84.5% |
| F2-score | 88.3% | **92.7%** |
| ROC-AUC | 0.924 | 0.924 |
| False negatives | 11 | **1** |
| False positives | **16** | 36 |

The lower decision threshold increased detection from **91/102** to **101/102** positive cases and reduced false negatives from **11 to 1**. This came at the cost of **20 additional false positives** and lower overall accuracy and precision.

**Key finding:** Hyperparameter tuning yielded small improvements in ROC-AUC; threshold selection produced a much larger change in sensitivity. ROC-AUC stayed constant because changing the threshold does not alter the model's probability ranking.

### 3. Feature interpretation

Permutation importance on the held-out partition identified the strongest predictive contributions from:

| Feature | Mean decrease in ROC-AUC when shuffled |
|---|---:|
| Chest pain type (`cp`) | 0.0650 |
| Serum cholesterol (`chol`) | 0.0344 |
| Exercise-induced angina (`exang`) | 0.0271 |
| ST depression (`oldpeak`) | 0.0190 |
| Sex (`sex`) | 0.0133 |

These are **model-specific predictive associations**, not estimates of causal effects. Correlated predictors and missing values can affect these rankings.

## Methodology

1. **Data preparation:** Read the dataset, check missing values, and binarize the original `num` outcome (`0` = no recorded heart disease; `1–4` = presence).
2. **Feature selection:** Exclude patient IDs and dataset-provenance fields; retain 13 clinical predictors.
3. **Data partitioning:** Create an 80/20 stratified training–test split with `random_state=42`.
4. **Preprocessing:** Apply median imputation and scaling to numeric fields; most-frequent imputation and one-hot encoding to categorical fields. Transformations are fitted inside model pipelines.
5. **Baseline benchmarking:** Compare Logistic Regression and Random Forest using stratified 5-fold cross-validation.
6. **Hyperparameter search:** Tune both models with 20-iteration randomized searches, optimizing validation ROC-AUC using training folds only.
7. **Threshold analysis:** Generate out-of-fold probabilities for the selected tuned configuration and choose the threshold maximizing F2 on training predictions.
8. **Evaluation:** Fit the chosen model on the full training partition and report both threshold operating points on the held-out test partition.
9. **Interpretation:** Examine confusion matrices, ROC behavior, and permutation feature importance.

## Dataset

The analysis uses a **920-row combined UCI Heart Disease dataset**, with the original multi-level target `num` converted into a binary outcome.

- **No recorded heart disease:** 411 records (44.7%)
- **Recorded heart disease:** 509 records (55.3%)
- **Predictors:** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalch`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

Several fields contain missing observations, especially `ca`, `thal`, and `slope`; preprocessing handles those values inside the cross-validation pipelines.

Dataset reference: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease). The exact combined CSV used by this notebook is expected at `data/heart_uci.csv`.

## Repository structure

```text
heart-disease-classification/
├── README.md
├── heart_disease_project.ipynb
├── data/
│   └── heart_uci.csv
└── outputs/                         # Generated after execution
    ├── baseline_cv_results.csv
    ├── hyperparameter_search_results.csv
    ├── training_threshold_results.csv
    ├── final_test_results.csv
    ├── permutation_importance.csv
    ├── experiment_metrics.json
    ├── selected_model.joblib
    └── github_result_line.md
```

## Reproducibility

**Requirements:** Python 3 and the following libraries:

```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib jupyter ipython
```

Place the dataset at `data/heart_uci.csv`, open `Heart_Disease_Classification_Tuned_GitHub.ipynb` in Jupyter, and execute the notebook from top to bottom. Results, fitted model, and CSV/JSON summaries are written to `outputs/`.

The notebook uses `random_state=42`. Small variations may occur across software versions or computational environments.

## Limitations

- This is a **single benchmark dataset**, not a representative prospective patient cohort or an externally validated clinical study.
- Several predictors have substantial missing values, and imputation cannot recover missing clinical measurements.
- Hyperparameter search and threshold selection reuse the same training partition; training-validation estimates may be optimistic.
- The hold-out partition was examined in earlier exploratory iterations of the project, so its results should be viewed as **reproducible benchmark estimates**, not a wholly independent prospective assessment.
- Lowering the decision threshold improves recall but increases false positives; the appropriate trade-off depends on the application and requires domain validation.
- Permutation importance describes prediction sensitivity, not causality.

## Conclusion

The tuned Random Forest achieved a **0.8871 cross-validation ROC-AUC** and a **0.924 held-out ROC-AUC**. At the default threshold it reached **85.3% accuracy**; the training-selected threshold of **0.24** increased recall to **99.0%**, reducing false negatives from **11 to 1** while increasing false positives from **16 to 36**.

The study illustrates why reliable ML evaluation requires more than a single accuracy score: **model discrimination, operating threshold, and the costs of different error types** should be considered separately.

## Tech stack

**Python · Pandas · NumPy · Scikit-learn · Matplotlib · Joblib · Jupyter Notebook**
