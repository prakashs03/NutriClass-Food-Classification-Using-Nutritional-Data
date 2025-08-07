
# NutriClass - Food Classification Using Nutritional Data

NutriClass is a mini-project focused on **multi-class classification** using machine learning models to classify food items based on nutritional values. The goal is to predict the `Meal_Type` (breakfast, lunch, dinner, snack) using a synthetic, imbalanced dataset.

---

## 🧠 Project Overview

- **Dataset**: `synthetic_food_dataset_imbalanced.csv`
- **Target**: `Meal_Type` (4 classes → multi-class classification)
- **Goal**: Build, evaluate, and compare ML models for food classification.

---

## 📌 Step-by-Step Process

### ✅ Step 1: Data Understanding & Exploration
- Loaded the dataset using Pandas
- Checked dataset shape (31,700 rows × 16 columns)
- Identified:
  - Missing values in 375 rows
  - 313 duplicate rows
  - Target variable: `Meal_Type` (multi-class)
- Class Distribution:
  - breakfast: 7970
  - lunch: 7856
  - dinner: 7873
  - snack: 8001 (balanced)

### ✅ Step 2: Data Preprocessing
- **Removed missing rows** (375 rows with NaNs)
- **Removed duplicates** (65 remaining duplicates dropped)
- **Outlier Capping**: Used IQR method on numerical columns
- **Normalization**: Applied MinMaxScaler to scale features between 0 and 1

### ✅ Step 3: Feature Engineering
- Dropped non-informative columns: `Food_Name`
- **Label Encoding**: Converted `Meal_Type` into numeric labels (0–3)
- **One-Hot Encoding**: Encoded `Preparation_Method`
- Final features stored in `X`, target in `y`
- Skipped PCA (not required due to low dimensionality)

### ✅ Step 4: Model Training & Evaluation
Trained and evaluated the following models:

| Model               | Accuracy | F1 Score |
|--------------------|----------|----------|
| Logistic Regression| 24.6%    | 24.1%    |
| Decision Tree      | 25.4%    | 25.4%    |
| Random Forest      | 25.4%    | 25.4%    |
| KNN (Best)         | 25.8%    | 25.3%    |
| SVM (Worst)        | 23.8%    | 23.8%    |
| XGBoost            | 25.5%    | 25.5%    |
| Gradient Boosting  | 24.6%    | 24.5%    |

- Used `train_test_split` with `random_state=42`
- Evaluated using:
  - Accuracy
  - Precision (macro)
  - Recall (macro)
  - F1 Score (macro)
  - Confusion Matrix

### ✅ Step 5: Visualization
Generated the following plots:
- 📊 Bar chart: Accuracy vs F1 Score (All models)
- 📉 Confusion Matrix Heatmap: KNN
- 📉 Confusion Matrix Heatmap: XGBoost

### ✅ Step 6: Conclusion & Insights
- KNN achieved the highest accuracy (~25.8%)
- Random Forest and XGBoost were close
- SVM underperformed due to overlapping features and default kernel
- Overall, feature discrimination was weak → further feature engineering or additional features recommended

---

## 📂 Project Structure

```
NutriClass-Food-Classifier/
├── data/
│   └── synthetic_food_dataset_imbalanced.csv
├── notebooks/
│   └── NutriClass.ipynb
├── visuals/
│   ├── 1.png  -> Accuracy vs F1 Score
│   ├── 2.png  -> Confusion Matrix - KNN
│   ├── 3.png  -> Confusion Matrix - XGBoost
├── report/
│   └── NutriClass_Report.pdf (or .docx)
├── README.md
└── requirements.txt
```

---

## 🧪 Evaluation Metrics Summary

- ✅ Multi-class classification handled correctly (4 target classes)
- ✅ Accuracy, Precision, Recall, F1 Score computed using `average='macro'`
- ✅ Confusion matrices visualized for model interpretation

---

## ⚙️ Setup Instructions

```bash
pip install -r requirements.txt
```

---

## 🔖 Tags / Domains

- Data Science
- Machine Learning
- Multi-class Classification
- Model Evaluation
- Feature Engineering
- Data Preprocessing
- Git/GitHub
- Visualization
- Scikit-learn / XGBoost
- Python (pandas, matplotlib, seaborn)

---

## 📃 License

This project is a part of academic coursework for educational purposes only.

