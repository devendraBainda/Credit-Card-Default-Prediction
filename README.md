# Credit Card Default Prediction

A machine learning project to predict credit card customer default behavior using behavioral and financial data. 

## 🎯 Project Overview

Bank A aims to improve its credit risk management by developing a forward-looking **Behavior Score** - a classification model that predicts whether a credit card customer will default in the next billing cycle. This predictive model helps proactively manage credit exposure and minimize financial losses.

## 📊 Dataset Description

The dataset contains anonymized behavioral data of over **30,000 credit card customers** with the following features:

### Customer Demographics
- **Customer Id**: Unique customer identifier
- **sex**: Gender (1 = Male, 0 = Female)
- **marriage**: Marital status (1 = Married, 2 = Single, 3 = Others)
- **education**: Education level (1 = Graduate School, 2 = University, 3 = High School, 4 = Others)
- **age**: Customer age in years

### Financial Variables
- **LIMIT_BAL**: Credit limit assigned to the customer
- **PAY_0 to PAY_6**: Payment status for past 6 months
  - `-2`: No credit consumption
  - `-1`: Bill paid in full on time
  - `0`: Partial/minimum payment made
  - `≥1`: Payment delayed by specified months
- **BILL_AMT1 to BILL_AMT6**: Monthly bill amounts for past 6 months
- **PAY_AMT1 to PAY_AMT6**: Monthly payment amounts for past 6 months

### Engineered Features
- **AVG_BILL_AMT**: Average bill amount over 6 months
- **PAY_TO_BILL_RATIO**: Ratio of total payment to total bill over 6 months

### Target Variable
- **next_month_default**: Binary indicator (1 = Default, 0 = No Default)

## 🔄 Project Workflow

### 1. Data Loading and Preprocessing
- Loaded training (~25,000 records) and validation (~5,000 records) datasets
- Removed Customer Id as it's only an identifier
- Verified and analyzed class imbalance

### 2. Exploratory Data Analysis (EDA)
- Generated descriptive statistics for all features
- Created correlation heatmap to identify feature relationships
- Analyzed feature distributions and patterns
- Investigated behavioral trends like payment delays and repayment consistency

### 3. Feature Engineering
Created financially meaningful features:
- **avg_bill_amt**: Average bill amount across 6 months
- **pay_to_bill_ratio**: Payment efficiency ratio
- Additional risk indicators like credit utilization patterns

### 4. Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to balance target classes
- Evaluated impact on model performance, particularly recall improvement

### 5. Model Building
Trained and compared multiple classification models:
- **Logistic Regression**: Baseline linear model
- **Random Forest Classifier**: Ensemble method with feature importance
- **XGBoost Classifier**: Gradient boosting for high performance
- **LightGBM Classifier**: Efficient gradient boosting variant

### 6. Model Evaluation
- **Primary Metrics**: Accuracy, Precision, Recall, F1-score
- **Business Focus**: F2-score for validation dataset (emphasizes recall)
- **Threshold Optimization**: Aligned with bank's risk appetite
- **Cross-validation**: Ensured model robustness

### 7. Prediction Generation
- Selected best performing model based on comprehensive evaluation
- Generated predictions on validation dataset
- Created submission file following project specifications

## 🛠️ Tools & Technologies

### Core Libraries
- **Python 3.x**: Primary programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Data visualization

### Machine Learning
- **scikit-learn**: Model building and evaluation
- **imbalanced-learn**: SMOTE implementation
- **xgboost**: Extreme Gradient Boosting
- **lightgbm**: Light Gradient Boosting Machine

### Optional Enhancements
- **SHAP**: Model interpretability and feature importance
- **GridSearchCV**: Hyperparameter optimization

## 📈 Key Results & Insights

### Model Performance
- Successfully handled class imbalance improving recall significantly
- Multiple model comparison identified robust solutions
- Optimized classification threshold for business requirements

### Financial Insights
- Payment history (PAY_0 to PAY_6) strongly correlates with default risk
- Credit utilization patterns are key predictors
- Engineered features enhanced model interpretability and performance

### Business Impact
- Forward-looking predictions enable proactive risk management
- Balanced precision-recall trade-off aligned with credit risk priorities
- Model supports early warning systems and credit exposure adjustments

## 🎯 Key Learnings

1. **Class Imbalance**: Major challenge requiring specialized techniques like SMOTE
2. **Feature Engineering**: Financially meaningful features significantly improve performance
3. **Model Selection**: Multiple model comparison ensures robust solutions
4. **Business Metrics**: Threshold tuning crucial for credit risk management
5. **Interpretability**: Understanding model decisions essential for business adoption

## 🚀 Next Steps for Improvement

### Technical Enhancements
- Implement comprehensive hyperparameter tuning using GridSearchCV
- Develop additional financial risk features (credit utilization ratio, delinquency streaks)
- Apply advanced threshold optimization techniques
- Integrate SHAP for enhanced model interpretability

### Business Applications
- Deploy model in production environment
- Implement real-time scoring system
- Develop monitoring and model drift detection
- Create business dashboards for risk management

## 📁 Repository Structure

```
Credit-Card-Default-Prediction/
├── main.ipynb                    # Complete project notebook with all analysis
├── predictions.csv               # Final model predictions on validation set
├── train_dataset_final1.csv      # Training dataset (~25,000 records)
├── validate_dataset_final.csv    # Validation dataset (~5,000 records)
└── README.md                     # Project documentation
```

### File Descriptions
- **main.ipynb**: Jupyter notebook containing the complete machine learning pipeline from data loading to final predictions
- **predictions.csv**: Final submission file with Customer_Id and next_month_default predictions
- **train_dataset_final1.csv**: Training data with features and target variable for model development
- **validate_dataset_final.csv**: Validation data without target variable for final predictions

## 👨‍💻 Author

**Devendra Bainda**

---

*This project demonstrates the application of machine learning techniques to real-world financial risk management, combining technical expertise with business understanding to create actionable predictive models.*
