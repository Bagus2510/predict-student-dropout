# 🎓 Predict Students Dropout and Academic Success

<div align="center">

![Project Banner](images/image.jpg)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)

**Machine Learning Project for Predicting Student Dropout and Academic Outcomes**

[Dataset](#-dataset) • [Key Findings](#-inspiration--key-findings) • [Notebooks](#-notebooks) • [Results](#-model-performance)

</div>

---

## 📖 About The Project

This project aims to predict **student dropout, enrollment, and graduation outcomes** based on academic, demographic, and socioeconomic features. Using advanced machine learning techniques including **Random Forest**, **Decision Tree**, and **XGBoost**, the models are trained on multiple dataset variants to identify the most robust configuration for predicting student academic success.

### 🎯 Key Objectives

- 🔍 Analyze student academic and demographic data
- 🧹 Perform comprehensive data preprocessing including outlier handling and feature selection
- 📊 Conduct exploratory data analysis (EDA) to uncover patterns and insights
- 🤖 Build and compare multiple classification models across 4 dataset variants
- 🎯 Fine-tune models using Coarse-to-Fine Search (RandomizedSearchCV → GridSearchCV)
- 📈 Evaluate and compare all tuned models with full metrics, confusion matrix, ROC curve, and feature importance

---

## 💡 Inspiration & Key Findings

### ❓ Can we predict whether a student will dropout, stay enrolled, or graduate?

**✅ YES!** Our machine learning models successfully classify student outcomes with strong performance:

- **XGBoost (Tuned)** achieved the highest **ROC-AUC of 0.8905** with the strongest **CV Mean of 0.7799**
- **Random Forest (Tuned)** achieved the highest **Accuracy of 77.63%** and nearly matched XGBoost on ROC-AUC
- **Decision Tree (Tuned)** remains competitive as a simpler baseline with **ROC-AUC of 0.8191**

All models were validated through:
- ✅ 5-Fold Cross Validation with GridSearchCV
- ✅ Coarse-to-Fine hyperparameter tuning strategy
- ✅ Evaluation on 4 dataset variants (All/No Outliers × All/Selected Features)
- ✅ Final fair comparison on the **same evaluation split** (same train/test data for all models)
- ✅ Consistent performance across confusion matrix, ROC curve, class-level recall, and feature importance analysis

### ❓ Which features significantly affect student dropout prediction?

Based on **Feature Importance Analysis** across all three models, the top influencing factors are:

1. 🥇 **Curricular Units 2nd Sem (Approved)** — Most critical predictor across all models
2. 🥈 **Curricular Units 2nd Sem (Grade)** — Second most important feature
3. 🥉 **Curricular Units 1st Sem (Approved)** — Strong predictor of early academic momentum
4. **Tuition Fees Up to Date** — Financial status strongly correlates with dropout risk
5. **Age at Enrollment** — Non-traditional age students show distinct dropout patterns

> **💡 Key Insight:** Academic performance in the first two semesters is the dominant predictor of dropout. Students who fail to pass sufficient curricular units early are at the highest risk and require immediate intervention.

---

## 📊 Dataset

The dataset is sourced from the UCI Machine Learning Repository:

🔗 **[Predict Students Dropout and Academic Success Dataset](https://www.kaggle.com/datasets/naveenkumar20bps1137/predict-students-dropout-and-academic-success)**

### Dataset Variants Used:
| Variant | Description | Shape |
|---|---|---|
| All Outliers + All Features | Full dataset, no outlier removal | 4424 × 35 |
| All Outliers + Selected Features | Full dataset, top 5 features only | 4424 × 6 |
| No Outliers + All Features | Outliers removed, all features | 4335 × 35 |
| No Outliers + Selected Features | Outliers removed, top 5 features | 4335 × 6 |

### Target Classes:
- **Dropout** — Student left the institution
- **Enrolled** — Student is currently enrolled
- **Graduate** — Student successfully graduated

---

## 🗂️ Project Structure

```
Predict Students Dropout Academic Success/
│
├── data/
│   ├── dataset.csv                                        # Raw dataset
│   ├── data_cleaned_with_all_outliers_and_all_features.csv
│   ├── data_cleaned_with_no_outliers_and_all_features.csv
│   ├── data_cleaned_selected_with_all_outliers.csv
│   └── data_cleaned_selected_with_no_outliers.csv
│
├── models/
│   ├── random_forest_classifier_model/
│   │   ├── best_rf_classifier_model_TIMESTAMP.pkl
│   │   └── best_rf_classifier_model_TIMESTAMP_metadata.json
│   ├── decision_tree_classifier_model/
│   │   ├── best_dt_classifier_model_TIMESTAMP.pkl
│   │   └── best_dt_classifier_model_TIMESTAMP_metadata.json
│   └── xgboost_classifier_model/
│       ├── best_xgb_classifier_model_TIMESTAMP.pkl
│       └── best_xgb_classifier_model_TIMESTAMP_metadata.json
│
├── notebooks/
│   ├── 01_eda.ipynb                          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb                # Data cleaning & preprocessing
│   ├── 03_01_random_forest_classifier_model.ipynb
│   ├── 03_02_decision_tree_classifier_model.ipynb
│   ├── 03_03_xgboost_classifier_model.ipynb
│   └── 04_model_comparison.ipynb             # Final model comparison
│
├── images/                                   # Saved plots and visualizations
├── README.md
└── requirements.txt
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/Bagus2510/predict-student-dropout.git
cd student-dropout-prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📓 Notebooks

### 1️⃣ Exploratory Data Analysis (`01_eda.ipynb`)
- Distribution analysis of all features
- Target class balance visualization
- Correlation heatmap and bivariate analysis
- Outlier detection and statistical summary

### 2️⃣ Data Preprocessing (`02_preprocessing.ipynb`)
- Missing value handling
- Outlier removal using IQR method
- Feature selection based on correlation and importance
- Generation of 4 dataset variants for experimentation

### 3️⃣ Random Forest Classifier (`03_01_random_forest_classifier_model.ipynb`)
- Baseline Random Forest on all 4 dataset variants
- Coarse-to-Fine tuning: RandomizedSearchCV → GridSearchCV
- Full evaluation: classification report, confusion matrix, ROC curve, feature importance
- Best model saved with metadata

### 4️⃣ Decision Tree Classifier (`03_02_decision_tree_classifier_model.ipynb`)
- Baseline Decision Tree on all 4 dataset variants
- Coarse-to-Fine tuning: RandomizedSearchCV → GridSearchCV
- Full evaluation pipeline identical to Random Forest notebook
- Best model saved with metadata

### 5️⃣ XGBoost Classifier (`03_03_xgboost_classifier_model.ipynb`)
- Baseline XGBoost on all 4 dataset variants
- Coarse-to-Fine tuning with hardware-aware optimization (`tree_method='hist'`)
- Full evaluation pipeline with ROC-AUC multi-class support
- Best model saved with metadata

### 6️⃣ Model Comparison (`04_model_comparison.ipynb`)
- Load all 3 best saved models from disk
- Re-train all 3 models on the same train split (fair comparison protocol)
- Side-by-side metrics comparison (Accuracy, F1, ROC-AUC, CV Mean, CV Std)
- Combined ROC curve and confusion matrix visualization
- Feature importance comparison (Top 15 per model)
- Cost-benefit simulation and fairness proxy checks (age group and tuition status)
- Final model recommendation based on ROC-AUC + CV stability

---

## 📈 Model Performance

### Overall Comparison — Tuned Models

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | ROC-AUC | CV Mean (5-Fold) | CV Std |
|---|---|---|---|---|
| Random Forest | **0.7763** | 0.7063 | 0.7654 | 0.8876 | 0.7751 | **0.0088** |
| Decision Tree | 0.7277 | 0.6514 | 0.7182 | 0.8191 | 0.7222 | 0.0146 |
| **XGBoost** | 0.7751 | **0.7095** | **0.7675** | **0.8905** | **0.7799** | 0.0102 |

### 🏆 Best Model: XGBoost — All Outliers + All Features

#### Best Hyperparameters:
```python
{
    'n_estimators'    : 400,
    'max_depth'       : 2,
    'learning_rate'   : 0.15,
    'subsample'       : 0.9,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'gamma'           : 0,
    'reg_alpha'       : 0.1,
    'reg_lambda'      : 1.5
}
```

#### Model Generalization:
- ✅ **ROC-AUC: 0.8905** — Best class discrimination on fair evaluation split
- ✅ **CV Mean: 0.7799 (Std: 0.0102)** — Stable cross-validation performance
- ✅ **Balanced Performance** — Strong weighted F1 with robust Graduate and Dropout recall

### 🧪 Validation Protocol (Updated)

- Final comparison uses **the same dataset and same train-test split** for Random Forest, Decision Tree, and XGBoost.
- Each model is rebuilt from saved best hyperparameters, then trained on that shared train set.
- This avoids invalid comparison caused by testing different models on different datasets.

---

## 🔍 Feature Importance & Business Insights

### Top Features (Consistent Across All Models)

| Rank | Feature | Business Implication |
|---|---|---|
| 1 | Curricular Units 2nd Sem (Approved) | Early academic failure = highest dropout risk |
| 2 | Curricular Units 2nd Sem (Grade) | Grade quality predicts persistence |
| 3 | Curricular Units 1st Sem (Approved) | First semester sets academic trajectory |
| 4 | Tuition Fees Up to Date | Financial distress triggers dropout |
| 5 | Age at Enrollment | Non-traditional students need flexible support |

### 💡 Actionable Insights

1. **Implement Early Warning System** — Flag students with low unit approval rates after Semester 1 and trigger counseling before Semester 2 begins.
2. **Proactive Financial Support** — Identify students with overdue tuition and offer financial aid or installment plans before dropout occurs.
3. **Flexible Programs for Mature Students** — Design part-time pathways for older enrollees who face competing work and family commitments.
4. **Sustain Monitoring Into Semester 2** — The transition into Semester 2 is a critical dropout risk window that is often overlooked.

### 📉 Cost-Benefit Simulation (Best Model)

Using the best model (XGBoost) on the fair evaluation split with simple assumptions:

- Intervention cost per at-risk student: **150**
- Estimated loss if one student drops out: **1200**
- True Positive dropout alerts: **213**
- False Positive alerts: **50**
- Missed dropout cases (FN): **71**
- Estimated net impact: **+130,950**

This indicates a practical positive return for targeted intervention, while also highlighting that improving recall on the **Enrolled** class remains an important next step.

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** — Data manipulation
- **NumPy** — Numerical computing
- **Matplotlib & Seaborn** — Data visualization
- **Scikit-learn** — Machine learning models and metrics
- **XGBoost** — Gradient boosting classifier
- **Joblib** — Model serialization
- **Jupyter Notebook** — Interactive development

---

## 💡 Usage

### Run Notebooks Sequentially:

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_01_random_forest_classifier_model.ipynb
jupyter notebook notebooks/03_02_decision_tree_classifier_model.ipynb
jupyter notebook notebooks/03_03_xgboost_classifier_model.ipynb
jupyter notebook notebooks/04_model_comparison.ipynb
```

### Load and Use Saved Model:
```python
import joblib
import os

# Load best model
MODEL_DIR = "models/xgboost_classifier_model"
model_file = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')][0]
model = joblib.load(os.path.join(MODEL_DIR, model_file))

# Predict
prediction = model.predict(X_test)
print(f"Predicted class: {prediction[0]}")  # Dropout / Enrolled / Graduate
```

---

## 🔮 Future Improvements

- [ ] Add deep learning models (Neural Networks / TabNet)
- [ ] Implement SHAP for model interpretability
- [ ] Create web application for real-time dropout prediction
- [ ] Add more socioeconomic features (scholarship history, part-time work)
- [ ] Implement ensemble stacking across RF, DT, and XGBoost
- [ ] Add model monitoring and data drift detection
- [ ] Deploy as REST API for integration with student information systems

---

## 🙏 Acknowledgments

- Dataset sourced from the [UCI Machine Learning Repository](https://www.kaggle.com/datasets/naveenkumar20bps1137/predict-students-dropout-and-academic-success)
- Built as part of a Data Science portfolio project

---

## 👤 Author

**Bagus Rahmadani**

- GitHub: [@Bagus2510](https://github.com/Bagus2510)
- LinkedIn: [bagusrahmadani](https://www.linkedin.com/in/bagusrahmadani/)
- Portfolio Website: [bagusrahmadani.vercel.app](https://bagusrahmadani.vercel.app/)
- Email: bagusrajin465@gmail.com

---

## ⭐ Show Your Support

Give a ⭐️ if this project helped you learn or inspired your own work!

---

<div align="center">

**Made with ❤️ for Student Academic Success**

</div>