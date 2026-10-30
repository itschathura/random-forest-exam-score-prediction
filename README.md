# 🎓 Student Exam Score Prediction (Random Forest Regression)

An end-to-end Machine Learning project that uses **Random Forest Regression** to predict student exam performance index based on study habits, prior academic history, and behavioral factors.

---

## 🚀 Project Overview

This repository contains both the initial exploratory prototype and an industry-grade, production-ready machine learning pipeline:

- **[`exam_score_complete.ipynb`](./exam_score_complete.ipynb)**: Full production-grade notebook with data cleaning, exploratory data analysis, leakage-free imputation, 5-fold cross-validation, hyperparameter tuning (`GridSearchCV`), comprehensive diagnostic evaluation, and a deployment-ready inference function.
- **[`exam_score.ipynb`](./exam_score.ipynb)**: Initial exploratory prototype.

---

## 📊 Features & Target Variable

The model utilizes **all 5 predictive features** from the dataset:

| Feature | Description | Range / Type |
| :--- | :--- | :--- |
| **`Hours Studied`** | Daily study hours | Float (0 – 24) |
| **`Previous Scores`** | Marks earned in preceding examinations | Float / Int (0 – 100) |
| **`Extracurricular Activities`** | Participation in school/college activities | Binary (1 = Yes, 0 = No) |
| **`Sleep Hours`** | Average hours of sleep per night | Float (0 – 24) |
| **`Sample Question Papers Practiced`** | Number of past/mock papers completed | Int (0+) |
| **`Performance Index`** *(Target)* | Overall student exam performance index | Float (10 – 100) |

---

## 📈 Model Performance & Benchmarks

Evaluating on 1,982 held-out test records ($20\%$ test partition):

| Model | MAE | RMSE | $R^2$ Score | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Random Forest** | 2.536 | 3.609 | 0.9646 | 5.50% |
| **Tuned Random Forest (`GridSearchCV`)** | **2.360** | **3.364** | **0.9692** | **5.11%** |

### 🔍 Key Insights:
- **High Explanatory Power ($R^2 = 0.9692$)**: Explains over $96.9\%$ of the variance in final student grades.
- **Low Error ($\text{MAE} = 2.36$)**: Average prediction deviation is just ~2.3 points on a 100-point scale.
- **Feature Importance**: `Previous Scores` is the strongest predictor, followed by `Hours Studied` and `Sample Question Papers Practiced`.

---

## 🛠️ Project Structure

```text
├── dataset/
│   └── student_performance_dirty.csv   # Raw dirty dataset with duplicates, anomalies & nulls
├── exam_score_complete.ipynb           # Production-ready, fully executed ML notebook
├── exam_score.ipynb                    # Initial prototype notebook
├── .gitignore                          # Standard Python gitignore rules
├── LICENSE                             # MIT License
└── README.md                           # Documentation
```

---

## 💻 Quick Start & Setup

### 1. Clone the repository
```bash
git clone https://github.com/itschathura/random-forest-exam-score-prediction.git
cd random-forest-exam-score-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the complete notebook
Launch Jupyter Lab or VS Code and open `exam_score_complete.ipynb`:
```bash
jupyter notebook exam_score_complete.ipynb
```

---

## 🔮 Interactive Prediction Example

```python
from exam_score_complete import predict_student_score

predicted_score = predict_student_score(
    hours_studied=8.0,
    previous_scores=85,
    extracurricular="Yes",
    sleep_hours=7.0,
    sample_papers=5,
    model=best_model,
    imputer=imputer
)

print(f"Predicted Performance Index: {predicted_score}")
# Predicted Performance Index: 81.13
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
