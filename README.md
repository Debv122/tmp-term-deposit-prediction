# Term Deposit Prediction Project

This project predicts whether a bank client will subscribe to a term deposit based on marketing campaign data using machine learning techniques.

## 📊 Project Overview

- **Dataset:** 41,188 client records with 20 features
- **Target:** Binary classification (subscribe: yes/no)
- **Success Rate:** 11.3% subscription rate (imbalanced dataset)
- **Data Quality:** Complete dataset with 0% missing values

## 🚀 Quick Start

### 1. Setup the environment:
```bash
python setup_and_run.py
```

### 2. Run the complete analysis:
```bash
python term_deposit_prediction.py
```

### 3. Run the simple version (for quick testing):
```bash
python simple_prediction.py
```

## 📋 What This Project Does

### 1. **Exploratory Data Analysis (EDA)**
- Comprehensive data quality assessment
- Statistical analysis of 41,188 client records
- Outlier detection and feature distribution analysis
- Correlation analysis between features

### 2. **Data Preprocessing**
- Label encoding for 10 categorical variables
- Feature scaling with StandardScaler
- Stratified train-test split (80/20)
- Class balancing for imbalanced dataset

### 3. **Model Development**
- **Algorithms:** Random Forest and Logistic Regression
- **Cross-validation:** 5-fold stratified CV
- **Model selection:** Based on F1-score (optimal for imbalanced data)
- **Performance metrics:** Accuracy, Precision, Recall, F1-score, AUC-ROC

### 4. **Model Evaluation**
- Comprehensive performance metrics
- Confusion matrix and ROC curve visualizations
- Feature importance analysis
- Model comparison and selection

### 5. **Business Insights & Recommendations**
- Client segmentation analysis
- Campaign optimization strategies
- Actionable recommendations for marketing team
- ROI improvement suggestions

## 📈 Key Findings

### Client Demographics
- **Job Distribution:** Admin (25.3%), Blue-collar (22.5%), Technician (16.4%)
- **Education:** University degree holders (29.5%) most common
- **Contact Preference:** Cellular (63.5%) vs Telephone (36.5%)
- **Campaign Timing:** May campaigns most frequent (33.4%)

### Success Factors
- **Call Duration:** Strong predictor of subscription success
- **Economic Indicators:** Significant impact on campaign outcomes
- **Contact Frequency:** Optimal range of 1-3 contacts per campaign
- **Previous Success:** Strong predictor for future subscriptions



## 📖 Detailed Analysis Report

For comprehensive analysis results, methodology, and business recommendations, please refer to:

**[📊 Complete Analysis Report](reports/Reports.md)**

The report includes:
- Detailed methodology and technical specifications
- Complete model performance metrics
- Feature importance rankings
- Business insights and actionable recommendations
- Implementation guidelines for production deployment

## 🎯 Model Performance

*Note: Specific performance metrics are available in the detailed report after running the analysis*

- **Best Model:** Selected based on F1-score optimization
- **Cross-validation:** 5-fold stratified validation
- **Metrics:** Comprehensive evaluation including precision, recall, and AUC-ROC
- **Production Ready:** Includes prediction function for new clients

## 💼 Business Value

### Marketing Optimization
- **Improved Targeting:** Data-driven client segmentation
- **Cost Efficiency:** Reduced wasted marketing contacts
- **ROI Enhancement:** Higher conversion rates through better targeting
- **Customer Experience:** More relevant and timely interactions

### Operational Benefits
- **Lead Scoring:** Prioritize high-potential prospects
- **Resource Allocation:** Optimize budget distribution across channels
- **Campaign Planning:** Strategic timing and approach optimization
- **Performance Monitoring:** Data-driven campaign effectiveness tracking

## 🛠️ Technical Requirements

- **Python 3.x**
- **Key Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Data:** Bank marketing dataset (included in data/data/ directory)
- **Output:** Trained models, visualizations, and prediction functions

## 📊 Dataset Information

The dataset contains direct marketing campaign data from a Portuguese banking institution:
- **Source:** UCI Machine Learning Repository
- **Records:** 41,188 client interactions
- **Features:** Demographics, campaign details, economic indicators
- **Target:** Term deposit subscription (yes/no)

## 🚀 Getting Started

1. **Clone the repository**
2. **Install dependencies:** `python setup_and_run.py`
3. **Run analysis:** `python term_deposit_prediction.py`
4. **Review results:** Check generated visualizations and [detailed report](reports/Reports.md)
5. **Deploy model:** Use the trained model for new predictions

## 📞 Model Deployment

The trained model includes a production-ready prediction function that can:
- Score new client prospects in real-time
- Provide subscription probability estimates
- Handle categorical encoding and feature scaling automatically
- Return confidence levels for business decision-making

---

**For detailed technical analysis, methodology, and business recommendations, see the [Complete Analysis Report](reports/Reports.md)**

*Project developed for predictive analytics and marketing optimization*
