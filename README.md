# PRODIGY_ML_01
# 🏠 House Price Prediction using Linear Regression

## 📌 Project Overview
This project builds a Linear Regression model to predict house prices using selected features from the Kaggle House Prices dataset.

The goal is to demonstrate a complete machine learning pipeline including:
- Data preprocessing
- Feature selection
- Model training
- Performance evaluation
- Interpretation of results

---

## 📊 Features Used
- GrLivArea (Above ground living area in square feet)
- BedroomAbvGr (Number of bedrooms)
- FullBath (Number of full bathrooms)

---

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

---

## 📈 Model Performance
- R² Score: 0.6341
- RMSE: 52975.72

The model explains approximately 63% of the variance in house prices using only three core features.

---

## 🔍 Key Insights
- Living area has the strongest positive impact on price.
- Bathrooms significantly increase house value.
- Bedrooms show a negative coefficient due to correlation with overall living area.

---

## 📁 Project Structure
task1_linear_regression/
│
├── task1.py
├── train.csv
└── README.md

---

## 🚀 How to Run
1. Install required libraries:
   pip install pandas numpy scikit-learn

2. Run the script:
   python task1.py

---

## 🎯 Conclusion
This project demonstrates the implementation of a supervised learning regression model and provides insights into the factors influencing house prices.
