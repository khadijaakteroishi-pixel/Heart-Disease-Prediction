# Heart Disease Prediction using Machine Learning

A clinical machine learning project for heart disease prediction using structured patient data, multiple classification algorithms, ensemble learning, probability calibration, uncertainty-aware prediction, explainable AI, clinical feature selection, model locking, and external validation.

## Project Overview

This project develops an end-to-end machine learning workflow for predicting the presence of heart disease from clinical features.

The workflow goes beyond training a single classifier. It includes data validation and clinical cleaning, exploratory data analysis, stratified dataset splitting, preprocessing, nested cross-validation, individual model tuning, ensemble learning, probability calibration, uncertainty analysis, explainability, minimum-feature selection, model serialization, and external dataset validation.

The primary dataset contains **918 patient records**. After cleaning, all 918 records were retained.

## Dataset Summary

- Total records: **918**
- Target: `HeartDisease`
- Binary classes:
  - No Heart Disease
  - Heart Disease
- Training set: **642 samples**
- Validation set: **138 samples**
- Test set: **138 samples**
- Training class ratio: **287 negative / 355 positive**
- Resampling applied: **No**

The minority-to-majority ratio in the training set was approximately **0.8085**, so additional resampling was not considered necessary.

## Clinical Features

The project works with features such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- ST Slope

Missing-value indicators were also created for clinically invalid zero values in blood pressure and cholesterol.

## Data Cleaning

Several validation and cleaning steps were applied before modeling:

- Schema verification
- Duplicate checking
- Clinical value standardization
- Conversion of invalid `RestingBP = 0` to missing
- Conversion of invalid `Cholesterol = 0` to missing
- Missing-value indicators
- Exploratory distribution and outlier analysis

The dataset contained:

- **1** zero RestingBP value
- **172** zero Cholesterol values

These values were treated as missing rather than as true clinical measurements.

## Preprocessing

The preprocessing workflow includes:

- Numerical imputation
- Standard scaling
- Categorical encoding
- Binary feature handling
- Training-only preprocessing
- Unknown-category handling for external data

The main preprocessing pipeline is built using Scikit-learn pipelines and column transformers.

## Cross-Validation

A nested cross-validation strategy was configured using:

- **5-fold outer cross-validation**
- **5-fold inner cross-validation**

This was used to separate model assessment from hyperparameter tuning during the model-development stage.

## Machine Learning Models

Seven individual machine learning models were trained and tuned:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine
5. K-Nearest Neighbors
6. Gradient Boosting
7. XGBoost

### Individual Model Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.8859 | 0.9005 | 0.9308 |
| Decision Tree | 0.8152 | 0.8283 | 0.8947 |
| Random Forest | 0.8750 | 0.8878 | 0.9296 |
| Support Vector Machine | 0.8750 | 0.8920 | 0.9305 |
| K-Nearest Neighbors | 0.8859 | 0.8976 | **0.9514** |
| Gradient Boosting | 0.8913 | 0.9029 | 0.9372 |
| XGBoost | **0.8967** | **0.9091** | 0.9375 |

## Ensemble Learning

Two ensemble strategies were evaluated:

### Soft Voting Classifier

- Accuracy: **0.8967**
- ROC-AUC: **0.9385**
- F1 Score: **0.9082**

### Stacking Classifier

- Accuracy: **0.8967**
- ROC-AUC: **0.9389**
- F1 Score: **0.9082**

The stacking ensemble combines:

- Random Forest
- Gradient Boosting
- XGBoost
- Logistic Regression

with Logistic Regression as the final meta-classifier.

## Probability Calibration

The ensemble was evaluated with both isotonic and sigmoid probability calibration.

| Method | Brier Score | Log Loss | ROC-AUC |
|---|---:|---:|---:|
| Uncalibrated Ensemble | 0.0853 | 0.3004 | 0.9389 |
| Isotonic Calibration | **0.0749** | **0.2572** | **0.9504** |
| Sigmoid Calibration | 0.0845 | 0.2965 | 0.9389 |

Isotonic calibration produced the strongest probability-quality results in this comparison.

## Uncertainty-Aware Prediction

An entropy-based selective prediction mechanism was explored so that uncertain predictions could be referred for specialist review instead of being automatically classified.

At an entropy threshold of **0.85**:

- Auto-diagnosed cases: **95.1%**
- Referred to specialist: **4.9%**
- Accuracy on automatically diagnosed cases: **92.0%**

This experiment demonstrates how predictive uncertainty could be incorporated into a clinical decision-support workflow.

## Explainable AI

The project includes:

- Permutation Feature Importance
- SHAP-based global explanation
- SHAP-based individual patient explanation

The most influential features identified by permutation importance were:

1. ST Slope
2. Chest Pain Type
3. Cholesterol
4. Oldpeak
5. Sex
6. Fasting Blood Sugar
7. Resting Blood Pressure
8. Maximum Heart Rate
9. Exercise Angina
10. Age
11. Resting ECG

## Minimum Clinical Feature Selection

The project investigates whether strong performance can be retained using fewer clinical variables.

The final selected six features were:

- `ST_Slope`
- `ChestPainType`
- `Cholesterol`
- `Oldpeak`
- `Sex`
- `FastingBS`

The six-feature retrained model achieved:

- Accuracy: **0.8696**
- Precision: **0.8750**
- Recall: **0.8922**
- F1 Score: **0.8835**
- ROC-AUC: **0.9253**
- Brier Score: **0.0990**

This represents a trade-off between predictive performance and a smaller clinical feature set.

## Model Locking

The final retrained calibrated model was serialized using Joblib.

Model integrity was verified after reloading:

- Original ROC-AUC: **0.9253**
- Reloaded ROC-AUC: **0.9253**

This confirmed that the saved model reproduced the same predictive state.

## External Validation

The locked model was also evaluated on a separate external heart-disease dataset to assess generalizability.

| Metric | Internal Test | External Validation |
|---|---:|---:|
| Accuracy | 0.8696 | 0.7104 |
| Precision | 0.8750 | 0.6759 |
| Recall | 0.8922 | 0.7153 |
| F1 Score | 0.8835 | 0.6950 |
| ROC-AUC | 0.9253 | 0.7680 |
| Brier Score | 0.0990 | 0.2075 |

Bootstrap ROC-AUC confidence intervals were:

- Internal: **0.8825 – 0.9611**
- External: **0.7116 – 0.8238**

The performance drop on the external dataset highlights the importance of evaluating clinical machine learning systems across different populations and data sources.

## Repository Structure

```text
Heart-Disease-Prediction/
│
├── README.md
│
├── code/
│   └── Heart_Diseases_Project_Final.ipynb
│
├── images/
│   └── Important result figures
│
└── paper/
    └── Project paper / report
