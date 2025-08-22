# Predicting Student Academic Performance

## Overview
This project explores how demographic, social, and academic factors influence student success.  
Using regression models (Linear, Ridge, Lasso, and ElasticNet), we predict **final grades (G3)** of secondary school students based on historical data from the UCI Machine Learning Repository.

---

## 1. Problem Context
Educational performance is shaped by various factors:  
- **Student background** – age, gender, family support, lifestyle.  
- **Learning environment** – study time, school choice, internet access.  
- **Past academic records** – previous grades (G1, G2).  

Understanding these factors helps design early intervention strategies, improve teaching methods, and support students at risk.

---

## 2. Dataset Overview
- **Target Variable:** `G3` (Final grade, 0–20)
- **Features:** Demographics, family background, lifestyle, past grades, school factors.
- **Source:** [UCI Machine Learning Repository – Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

---

## 3. Workflow
1. Data Collection  
2. Data Inspection & Cleaning  
3. Exploratory Data Analysis (EDA)  
4. Feature Engineering  
5. Model Training & Evaluation  
6. Save Best Model (`.pkl`)  
7. Flask App Deployment

## 4. Tech Stack
- **Python:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, Flask
