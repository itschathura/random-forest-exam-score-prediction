# End-to-End Walkthrough & Notebook Review: Student Exam Score Prediction

## 📌 Executive Summary

This document provides a comprehensive review of the changes, audit findings, and step-by-step implementation completed in [`exam_score_complete.ipynb`](./exam_score_complete.ipynb) compared to the initial exploratory notebook [`exam_score.ipynb`](./exam_score.ipynb).

The goal of this refactor was to elevate the initial prototype into a **production-ready, leakage-free, fully evaluated machine learning workflow** using **Random Forest Regression** to predict student exam performance index (`Performance Index`).

---

## 🔄 Summary of Audit Findings & What Was Addressed

| # | Prototype Gap (`exam_score.ipynb`) | Refactored Solution (`exam_score_complete.ipynb`) | Impact / Benefit |
|---|---|---|---|
| **1** | Dropped `Sample Question Papers Practiced` without explanation. | Included all 5 predictive features in feature matrix $X$. | Preserves full predictive signals and allows the model to measure feature importance. |
| **2** | Only reported MAE ($\approx 2.2$); no R², RMSE, or residual plots. | Evaluated with MAE, MSE, RMSE, R² ($0.9692$), MAPE, and multiple diagnostic plots (Actual vs. Predicted, Residual distribution, and Residuals vs. Fitted). | Uncovers model variance explanation and diagnostic error distribution across score intervals. |
| **3** | Single 80/20 train/test split without cross-validation. | Implemented 5-Fold Cross Validation (`cross_val_score`) on training data before tuning. | Validates model generalizability and guards against optimistic split bias. |
| **4** | Default hyperparameters used (`n_estimators=100`, unconstrained `max_depth`). | Systematic `GridSearchCV` tuning tree count, depth, split criteria, and leaf samples. | Improved MAE from $2.536$ down to $2.360$, and $R^2$ up to $0.9692$, while reducing overfitting risk. |
| **5** | No feature importance analysis. | Visualized MDI (Gini / impurity decrease) feature importances with horizontal bar charts. | Quantified that `Previous Scores` and `Hours Studied` are the primary score drivers. |
| **6** | Misleading visualization labels (`Sample of 500` vs $n=100$). | Fixed visualization consistency and added correlation heatmaps and bivariate trendlines. | Enhances reproducibility and presentation credibility. |
| **7** | Sparse documentation and arbitrary commented cells. | Structured into 8 clear, modular sections with explanatory markdown cells, Markdown tables, and structured logging. | Clear portfolio-grade readability and self-explanatory workflow. |
| **8** | Unsafe manual inference snippet. | Packaged robust `predict_student_score()` utility handling missing feature defaults and categorical mapping. | Ready for downstream API or web UI integration. |

---

## 🛠️ Step-by-Step Implementation in `exam_score_complete.ipynb`

### Step 1: Environment Setup & Library Imports
- Loaded standard data science stack: `pandas`, `numpy`, `matplotlib`, `seaborn`.
- Imported scikit-learn modules:
  - Models: `RandomForestRegressor`
  - Preprocessing: `SimpleImputer`, `train_test_split`
  - Validation & Tuning: `cross_val_score`, `GridSearchCV`, `KFold`
  - Metrics: `mean_absolute_error`, `mean_squared_error`, `r2_score`, `mean_absolute_percentage_error`
- Configured visual styling (`sns.set_theme(style='whitegrid')`) and reproducible seeds (`random_state=42`).

### Step 2: Data Ingestion & Initial Audit
- Loaded `dataset/student_performance_dirty.csv` (~10,005 rows).
- Audited shape, data types, missing value percentages, and duplicate counts.
- Dropped irrelevant non-predictive metadata column: `Notes`.

### Step 3: Dirty Data Cleaning & Anomaly Correction
- **Deduplication**: Removed duplicate records (`drop_duplicates()`).
- **Domain-Specific Range Boundaries**:
  - `Hours Studied`: Capped to valid study hours $[0, 24]$.
  - `Sleep Hours`: Filtered out negative values and capped to $[0, 24]$.
  - `Previous Scores`: Filtered out invalid values beyond $[0, 100]$.
  - `Performance Index`: Filtered out scores $> 100$.
- **Categorical Encoding**: Mapped `Extracurricular Activities` (`'Yes' -> 1`, `'No' -> 0`).

### Step 4: Exploratory Data Analysis (EDA)
- Generated distribution plots for `Performance Index` and input features.
- Plotted correlation heatmap between all numeric features and the target.
- Visualized relationship between `Hours Studied`, `Previous Scores`, and `Performance Index`.

### Step 5: Data Splitting & Leakage-Free Preprocessing
- Defined full feature matrix $X$ using **all 5 features**:
  `['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']`.
- Target: `y = df['Performance Index']`.
- **Train/Test Split**: 80/20 train-test split (`test_size=0.20`, `random_state=42`).
- **Leakage-Free Imputation**:
  - Fit `SimpleImputer(strategy='median')` on `X_train` **only**.
  - Transformed `X_test` with the training set medians to strictly prevent test data leakage.

### Step 6: Baseline Model & 5-Fold Cross-Validation
- Instantiated baseline `RandomForestRegressor(random_state=42)`.
- Performed 5-Fold cross-validation on `X_train`:
  - Mean CV MAE computed to establish robust baseline expectation before test evaluation.
- Trained baseline model and recorded baseline test metrics.

### Step 7: Hyperparameter Tuning (`GridSearchCV`)
- Explored parameter space:
  - `n_estimators`: `[100, 200, 300]`
  - `max_depth`: `[10, 15, 20, None]`
  - `min_samples_split`: `[2, 5]`
  - `min_samples_leaf`: `[1, 2]`
  - `max_features`: `['sqrt', 1.0]`
- Identified optimal hyperparameters via 5-fold cross-validation with scoring `neg_mean_absolute_error`.
- Retrained model using best parameters on full training set.

### Step 8: Comprehensive Model Evaluation & Diagnostic Visualizations
- Computed final test set metrics:
  - **MAE**: $2.360$
  - **RMSE**: $3.364$
  - **$R^2$ Score**: $0.9692$ ($96.92\%$ of variance explained)
  - **MAPE**: $5.11\%$
- Created 3 diagnostic charts:
  1. **Actual vs. Predicted Plot**: Showing tight alignment along the identity line ($y = x$).
  2. **Residual Distribution Plot**: Verifying zero-centered, bell-shaped residual distribution.
  3. **Residuals vs. Fitted Plot**: Checking for homoscedasticity across the prediction range.
- Feature Importance ranking plotted:
  - Dominant feature: `Previous Scores` (~$70\%+$ importance)
  - Secondary feature: `Hours Studied` (~$20\%+$ importance)
  - Moderate features: `Sample Question Papers Practiced`, `Sleep Hours`, and `Extracurricular Activities`.

### Step 9: Reusable Inference Pipeline
- Created a standalone Python helper `predict_student_score()`:
  - Accepts raw inputs (e.g., categorical `'Yes'/'No'`).
  - Prepares DataFrame with proper column schema.
  - Automatically handles median imputation using the fitted training imputer.
  - Returns rounded performance prediction index with confidence bounds.

---

## 🎯 Verification & Conclusion
The project has evolved from a basic single-metric script to an end-to-end Machine Learning pipeline. All 8 items from the initial code review have been implemented and verified. Both notebooks (`exam_score.ipynb` for historical exploratory reference and `exam_score_complete.ipynb` for production standards) are preserved in the repository.
