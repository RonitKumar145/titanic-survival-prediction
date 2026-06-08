# 🚢 Titanic Survival Prediction

A Machine Learning web application that predicts whether a passenger would have survived the Titanic disaster based on passenger demographics, ticket information, and travel class.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![Status](https://img.shields.io/badge/Status-Deployed-success)

---

## 🌐 Live Demo

Add your Streamlit URL here:

```text
https://ronit-titanic-survival-prediction.streamlit.app/
```

---

## 📌 Project Overview

The Titanic disaster is one of the most famous shipwrecks in history.

This project uses Machine Learning to predict passenger survival based on factors such as:

* Passenger Class
* Gender
* Age
* Fare
* Family Size
* Port of Embarkation

The application allows users to enter passenger details and instantly receive:

* Survival Prediction
* Survival Probability
* Model Confidence Score

---

## 🎯 Objectives

* Perform Exploratory Data Analysis (EDA)
* Handle Missing Values
* Engineer New Features
* Train Classification Models
* Evaluate Model Performance
* Build an Interactive Web Application
* Deploy the Model Online

---

## 📊 Dataset Information

### Dataset

Titanic Passenger Dataset

### Records

```text
891 passengers
12 original features
```

### Target Variable

```text
Survived

0 = Did Not Survive
1 = Survived
```

---

## 📂 Project Structure

```text
titanic-survival-prediction/
│
├── data/
│   └── titanic.csv
│
├── models/
│   └── titanic_survival_model.pkl
│
├── notebooks/
│   └── titanic-test.ipynb
│
├── screenshots/
│   ├── home.png
│   └── prediction.png
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔍 Exploratory Data Analysis

### Missing Values

| Feature  | Missing Values |
| -------- | -------------- |
| Age      | 177            |
| Cabin    | 687            |
| Embarked | 2              |

### Key Findings

#### Survival by Gender

| Gender | Survival Rate |
| ------ | ------------- |
| Female | 74.2%         |
| Male   | 18.9%         |

Women were significantly more likely to survive.

---

#### Survival by Passenger Class

| Class     | Survival Rate |
| --------- | ------------- |
| 1st Class | 63.0%         |
| 2nd Class | 47.3%         |
| 3rd Class | 24.2%         |

Passengers in higher classes had better survival chances.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

* Missing Value Treatment
* Feature Engineering
* One-Hot Encoding
* Train-Test Split
* Pipeline Creation

### Missing Value Handling

```python
Age -> Median Imputation
Embarked -> Mode Imputation
Cabin -> Dropped
```

---

## 🧠 Feature Engineering

Created a new feature:

```python
Family_Size = SibSp + Parch + 1
```

This feature represents the total number of family members traveling together.

---

## 🤖 Machine Learning Models

### Logistic Regression

Accuracy:

```text
80.45%
```

---

### Random Forest Classifier

Accuracy:

```text
82.12%
```

---

## 🏆 Best Model

Random Forest Classifier achieved the highest accuracy and was selected for deployment.

---

## 📈 Model Performance

| Metric       | Value                    |
| ------------ | ------------------------ |
| Accuracy     | 82.12%                   |
| Dataset Size | 891 Rows                 |
| Model Type   | Random Forest Classifier |

---

## 🚀 Streamlit Web Application

### Features

✅ Interactive User Interface

✅ Survival Probability Prediction

✅ Passenger Profile Summary

✅ Model Confidence Score

✅ Prediction History

✅ Real-Time Results

---

## 📸 Application Screenshots

### 🏠 Homepage

```markdown
![Homepage](screenshots/home.png)
```

### 🎯 Prediction Result

```markdown
![Prediction](screenshots/prediction.png)
```

Replace image names if different.

---

## 🛠 Tech Stack

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit
* Matplotlib
* Seaborn

### Tools

* VS Code
* Jupyter Notebook
* Git
* GitHub

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/RonitKumar145/titanic-survival-prediction.git
```

### Move into Project Folder

```bash
cd titanic-survival-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 💡 Future Improvements

* Hyperparameter Tuning
* XGBoost Classifier
* Feature Importance Dashboard
* Model Explainability using SHAP
* Advanced Visualizations
* Cloud Deployment Monitoring

---

## 👨‍💻 Author

### Ronit Kumar

Artificial Intelligence & Machine Learning Student

GitHub:
https://github.com/RonitKumar145

LinkedIn:
https://www.linkedin.com/in/ronit-kumar-64271b258

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.
