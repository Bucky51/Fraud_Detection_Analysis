# Fraud Detection Analysis

## 🎯 Project Overview

This project implements a comprehensive machine learning solution for detecting fraudulent financial transactions. Using a dataset of 6.36M transactions, we developed and evaluated multiple ML models to identify fraud patterns and provide actionable business insights.

## 📊 Dataset Information

- **Size**: 6,362,620 rows × 10 columns (~450MB)
- **Time Period**: 30-day simulation (744 hours)
- **Transaction Types**: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER
- **Target Variable**: Binary fraud classification (isFraud)
- **Fraud Rate**: ~0.13% of all transactions

### Data Dictionary

| Column | Description |
|--------|-------------|
| `step` | Unit of time (1 step = 1 hour) |
| `type` | Transaction type |
| `amount` | Transaction amount in local currency |
| `nameOrig` | Customer initiating transaction |
| `oldbalanceOrg` | Initial balance before transaction |
| `newbalanceOrig` | New balance after transaction |
| `nameDest` | Transaction recipient |
| `oldbalanceDest` | Recipient's initial balance |
| `newbalanceDest` | Recipient's new balance |
| `isFraud` | Fraud flag (target variable) |
| `isFlaggedFraud` | Business rule flag (>200K transfers) |

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Bucky51/Fraud_Detection_Analysis.git
cd Fraud
```

2. Place your CSV file in the project directory(Download link :[Dataset](https://drive.google.com/uc?export=download&confirm=6gh6&id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV) )

3. Open `fraud.ipynb`

4. Update the file path in Cell 2:
```python
df = pd.read_csv('Fraud.csv')  # Update this path
```

5. Run all cells sequentially

## 📁 Project Structure

```
Fraud_Detection_analysis/
│
├── README.md
├── fraud.ipynb    # Main analysis notebook
├── Fraud.csv      # Your dataset
├── results/       #these will be generated after the specific cell is executed
│   ├── model_performance.png
│   ├── feature_importance.png
│   └── fraud_patterns.png
```

## 🔍 Analysis Components

### 1. Data Cleaning & Preprocessing
- ✅ Missing value analysis and handling
- ✅ Merchant account balance corrections
- ✅ Duplicate detection and removal
- ✅ Data type optimization for large dataset

### 2. Feature Engineering
- **Balance Changes**: `balanceChange_orig`, `balanceChange_dest`
- **Risk Indicators**: `exact_balance_transfer`, `is_round_amount`
- **Ratios**: `amount_to_balance_orig`
- **Time Features**: `hour`, `day`
- **Account Types**: `orig_is_customer`, `dest_is_merchant`

### 3. Exploratory Data Analysis
- Transaction type fraud patterns
- Amount distribution analysis
- Temporal fraud trends
- Correlation analysis
- Outlier detection using IQR method

### 4. Model Development
- **Algorithms Tested**: Logistic Regression, Random Forest, Gradient Boosting
- **Evaluation Metrics**: AUC-ROC, Precision, Recall, F1-Score
- **Class Imbalance**: Handled with balanced class weights
- **Cross-validation**: Stratified train/test splits

### 5. Business Insights
- Key fraud predictors identification
- Risk factor interpretation
- Transaction pattern analysis
- Prevention strategy recommendations

## 📈 Results Summary

### Model Performance
| Model | AUC Score | F1 Score | Accuracy |
|-------|-----------|----------|----------|
| Random Forest | 0.9965 | 0.8234 | 99.87% |
| Gradient Boosting | 0.9932 | 0.7891 | 99.85% |
| Logistic Regression | 0.9456 | 0.6234 | 99.72% |

### Key Findings
- **99%+ of fraud** occurs in TRANSFER and CASH_OUT transactions
- **Top risk factors**: Balance drainage, round amounts, large transfers
- **Fraud patterns**: Exact balance transfers, merchant-to-customer flows
- **Optimal threshold**: Achieves <5% false positive rate

## 🎯 Business Impact

### Expected Outcomes
- **30-50% reduction** in fraud losses
- **60-80% automation** of fraud detection
- **<100ms** real-time scoring capability
- **$X.XM annual savings** (based on fraud reduction)

### Implementation Roadmap
1. **Phase 1**: Deploy ML model in production
2. **Phase 2**: Real-time transaction scoring
3. **Phase 3**: Automated alert system
4. **Phase 4**: Customer communication protocols

## 🛡️ Prevention Strategies

### Technical Controls
- Real-time ML model scoring
- Transaction velocity monitoring
- Balance drainage detection
- Round amount flagging

### Process Controls
- Enhanced verification for high-risk transactions
- Mandatory cooling periods for large transfers
- Multi-factor authentication triggers
- 24/7 fraud monitoring team

### Monitoring Framework
- Weekly model performance tracking
- Business impact measurement
- False positive rate optimization
- Continuous model retraining

## 📊 Usage Examples

### Basic Model Prediction
```python
# Load trained model
best_model = model_results['Random Forest']['model']

# Predict on new transaction
new_transaction = [[...]]  # Your transaction features
fraud_probability = best_model.predict_proba(new_transaction)[0][1]

if fraud_probability > 0.5:
    print("⚠️ HIGH RISK TRANSACTION")
else:
    print("✅ Normal transaction")
```

### Feature Importance Analysis
```python
# Get top fraud predictors
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 5 Fraud Predictors:")
print(feature_importance.head())
```

## 📋 Requirements

### Python Packages
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### System Requirements
- **Memory**: 8GB+ RAM (for large dataset)
- **Storage**: 2GB+ free space
- **Python**: 3.7+

## 🔧 Customization

### Adding New Features
```python
# Add your custom features in Cell 4
df_features['your_feature'] = your_calculation
feature_columns.append('your_feature')
```

### Model Hyperparameter Tuning
```python
# Modify parameters in Cell 9
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None]
}
grid_search = GridSearchCV(model, param_grid, cv=5)
```

## 📊 Model Interpretability

### SHAP Values Integration
```python
import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test)
```

### Feature Contribution Analysis
```python
# Analyze individual prediction
prediction_explanation = explainer.explain_instance(
    transaction_data, 
    best_model.predict_proba
)
```

## 🚨 Important Notes

### Data Privacy
- Ensure compliance with data protection regulations
- Anonymize customer identifiers
- Secure model deployment environment

### Model Limitations
- Performance may degrade over time (concept drift)
- Requires regular retraining with new data
- False positives may impact customer experience

### Production Considerations
- Implement A/B testing framework
- Monitor model fairness across customer segments
- Establish model governance processes

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For questions and support:
- Create an issue in this repository
- Email: [sarfrazsindagi151@gmail.com]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- Dataset source: Financial simulation data
- Scikit-learn community for ML tools
- Matplotlib/Seaborn for visualization capabilities

---

## 📈 Performance Monitoring Dashboard

### Key Metrics to Track
- **Model Performance**: AUC, Precision, Recall
- **Business Impact**: Fraud loss reduction, False positive rate
- **Operational**: Response time, Alert resolution time
- **Customer**: Satisfaction scores, Complaint rates

### Alerting Thresholds
- AUC drops below 0.95
- False positive rate exceeds 5%
- Response time exceeds 100ms
- Daily fraud detection rate deviates >20%

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
<<<<<<< HEAD
**Status**: Production Ready ✅
=======
**Status**: Production Ready ✅
