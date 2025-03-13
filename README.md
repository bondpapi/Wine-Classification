# Wine Quality Classification - Linear Regression Model

## Overview
This project aims to classify red wine samples into **"Good"** or **"Not Good"** based on their physicochemical properties using **Logistic Regression**. The dataset is analyzed, preprocessed, and used to build a classification model that predicts wine quality effectively.

---

## Dataset Description
- The dataset contains various **chemical properties** of red wine, such as **acidity, sugar, chlorides, sulfur dioxide levels, density, pH, sulphates, and alcohol**.
- The target variable (**good_wine**) is binary:
  - **1** → High-quality wine
  - **0** → Lower-quality wine

### Key Dataset Information
- **Sample Size:** 1,599 wine samples
- **Features:** 11 numerical attributes
- **Target Variable:** Binary classification (Good vs. Not Good)

---

## Project Workflow

### 1️. Data Understanding & Hypothesis Formulation
- Understand what each variable measures and how it impacts wine quality.
- Formulated a hypothesis to test whether certain features (e.g., **alcohol, sulphates**) significantly influence perceived wine quality.

### 2. Exploratory Data Analysis (EDA) & Feature Selection
- Inspected the **distribution of features**, detected **outliers**, and checked for **multicollinearity** using correlation matrices and **VIF (Variance Inflation Factor)**.
- Selected the **most relevant features** that have a significant impact on wine quality based on **logistic regression coefficients and p-values**.

### 3. Model Development & Training
- **Preprocessing Steps:**
  - Standardized features using **StandardScaler** to ensure uniform scale.
  - Addressed class imbalance using **class_weight="balanced"** instead of SMOTE.
- **Logistic Regression Model Training:**
  - Used **Statsmodels** for interpretability of coefficients and significance tests.
  - Evaluated feature importance and confidence intervals.

### 4. Model Evaluation on Test Set
- **Performance Metrics:**
  - Accuracy: **89%**
  - AUC-ROC: **0.91** (strong discrimination ability)
  - Precision, Recall, and F1-score analyzed using a **classification report**
- **Threshold Tuning:**
  - Chose **0.26** as the optimal threshold, balancing **true positive rate** without increasing false positives significantly.
- **Confusion Matrix Analysis:**
  - Evaluated model misclassifications and improved recall for detecting good wine.

### 5. Model Fit & Generalization Check
- **Log Loss Comparison:**
  - Train Log Loss: **0.2777**
  - Test Log Loss: **0.2438** (Good generalization, no overfitting)
- **Mean Squared Error (MSE):** **0.0759**
- **Pseudo R² (McFadden’s R²):** **0.31**, meaning 31% of variance is explained by the model.

---

## Key Findings & Business Implications
- **Alcohol & Sulphates** have the strongest positive impact on wine quality—winemakers should optimize these for better quality perception.
- **Volatile Acidity & Total Sulfur Dioxide** negatively impact quality—monitoring these can prevent lower ratings.
- **Threshold Tuning** at **0.26** significantly improves classification performance.
- **Logistic Regression is interpretable**, but ensemble models (e.g., **Random Forest, Gradient Boosting**) might provide better accuracy.

---

## Future Improvements
a)  **Try alternative models** like **Random Forest, XGBoost, or Neural Networks**.

b) **Feature Engineering** (e.g., interaction terms between acidity & pH).

c) **Collect more data** to enhance the model’s predictive power.

d) **Hyperparameter tuning** for optimal regularization and feature selection.

---

## How to Use This Notebook
- Clone the repository or download the dataset.

-  Open `wine_quality.ipynb` in **Jupyter Notebook**.

- Run the cells sequentially to train and evaluate the model.

- Adjust the **classification threshold** to see how it affects precision & recall.

**Recommended Tools:** Python 3, Jupyter Notebook, Pandas, Scikit-Learn, Statsmodels, Seaborn, Matplotlib.

- Install library package below 

``` python
pip install -r requirements.txt
```
---

## Contributors
📌 **Author:** *Michael Bond*

📌 **Date:** March 2025

🚀 **Cheers to Better Wine! 🍷**


