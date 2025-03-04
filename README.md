# 🏡 House Price Prediction using OLS Regression  

## 📌 Overview  
This project uses **Ordinary Least Squares (OLS) Regression** to predict **house prices** based on various factors such as median income, house age, number of rooms, and population density. The goal is to build a reliable predictive model using **statistical analysis and machine learning techniques**.  

---

## 📊 Dataset  
We use the **California Housing Dataset**, which contains:  

### 📌 Features (Independent Variables)  
- `MedInc` → Median income in the area  
- `HouseAge` → Age of houses  
- `AveRooms` → Average number of rooms per house  
- `AveBedrms` → Average number of bedrooms per house  
- `Population` → Total population in the area  
- `AveOccup` → Average number of occupants per household  
- `Latitude` & `Longitude` → Geographic location  

### 🎯 Target (Dependent Variable)  
- `PRICE` → Median house price in the area  

---

## 🛠️ Technologies Used  
- **Python** 🐍  
- **Libraries:**  
  - `pandas` → Data manipulation  
  - `numpy` → Numerical operations  
  - `matplotlib & seaborn` → Data visualization  
  - `statsmodels` → Statistical analysis (**OLS Regression**)  
  - `scikit-learn` → Data preprocessing & model evaluation  

---

## 🔍 Methodology  

### ✅ 1. Data Preprocessing  
- Checked for missing values  
- Standardized numerical features  
-

### ✅ 2. Exploratory Data Analysis (EDA)  
- **Correlation Heatmap** → Identified key features influencing house prices  
 

### ✅ 3. Model Training - OLS Regression  
- Used **Ordinary Least Squares (OLS) Regression** from `statsmodels.api`  
- Fitted the model to the training data  
- Analyzed **p-values & coefficients** to interpret feature importance  

```python
import statsmodels.api as sm

# Add constant term for OLS regression
X = sm.add_constant(X)

# Fit OLS model
ols_model = sm.OLS(y, X).fit()

# Summary of the model
print(ols_model.summary())
📈 Results & Insights
Median Income (MedInc) was the most significant predictor of house prices 📈
