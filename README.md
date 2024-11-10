# NETWORK ANOMALY DETECTION PROJECT
> An anomaly detection system for identifying unusual network behavior using machine learning algorithms in Python. This project offers an end-to-end approach to network anomaly detection, starting with data exploration and visualization in **Tableau**, followed by exploratory data analysis (EDA), hypothesis testing, machine learning modeling with experimental tracking, and concluding with deployment.

## PROBLEM STATEMENT
> In the realm of cybersecurity, network anomaly detection is a critical task that involves identifying unusual patterns or behaviors that deviate from the norm within network traffic. These anomalies could signify a range of security threats, from compromised devices and malware infections to large-scale cyber-attacks like **DDoS (Distributed Denial of Service)**.

> As a data scientist working in the cybersecurity department, your task is to visualize and analyze the provided **network data (NSL-KDD dataset)**. Apply supervised learning algorithms to find the best model for classifying the data using the attack column. Utilize unsupervised algorithms to improve the performance and deploy the machine learning models using **Streamlit**.

## TARGET METRIC
Classification metrics such as **accuracy**, **precision**, **recall**, and **F1-score** were used. However, **recall** and **accuracy** were prioritized for network anomaly detection due to their importance in identifying anomalies effectively.

## CONTENTS AVAILABLE IN `NETWORK_ANOMALY_DETECTION_PROJECT`
1. **Definitions & Problem Statement**: Detailed feature definitions, dataset origin, and the problem statement.
2. **Tableau Visualization**:
   - **Network Connection Metrics Dashboard** (Basic and Content-Related Features)
   - **Advanced Traffic Patterns Dashboard** (Time and Host-Related Features)
3. **Exploratory Data Analysis Using Python**: Data cleaning, feature engineering, and univariate, bivariate, and multivariate data visualizations.
4. **Hypothesis Testing**: Conducted between important features and binary/multi-class targets, verifying assumptions and applying non-parametric tests when necessary.
5. **Feature Engineering Using Unsupervised Algorithms**: Added features using anomaly detection algorithms and **KMeans clustering** to enhance classification performance.
6. **Machine Learning Modeling for Binary Classification**: Applied and tuned 10 different classification models using **MLflow** for experimental tracking.
7. **Evaluation of Results for Binary Classification**: Plotted and compared metrics of all 10 models.
8. **Multi-Class Classification Utilizing Optimal Models**: Applied optimal models after preprocessing and compared their results.
9. **Deployment of the Optimal Model Using Streamlit**: Deployed locally and on **Streamlit Cloud** for real-time predictions.
10. **Conclusion, Actionable Insights, Recommendations, Future Scope & References**: Summarized findings and provided relevant links.

## FINAL SCORES ACHIEVED

### BINARY CLASS CLASSIFICATION
- **Model**: Tuned Adaboost on Balanced Dataset
- **Parameters**: `{'algorithm': 'SAMME', 'learning_rate': 0.8065, 'n_estimators': 114}`
- **Training Time**: 98.13 seconds
- **Testing Time**: 2.87 seconds
- **Tuning Time**: 227.52 seconds

**Train Metrics**:
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-score: 1.0000
- F2-score: 1.0000
- ROC-AUC: 1.0000
- PR-AUC: 1.0000

**Test Metrics**:
- Accuracy: 0.9992
- Precision: 0.9996
- Recall: 0.9988
- F1-score: 0.9992
- F2-score: 0.9990
- ROC-AUC: 1.0000
- PR-AUC: 1.0000

### MULTI-CLASS CLASSIFICATION
- **Model**: Tuned_Light_GBM_on_Balanced_Dataset
- **Parameters**: `{'bagging_fraction': np.float64(0.7809345096873808), 'bagging_freq': 9, 'boosting_type': 'gbdt', 'colsample_bytree': np.float64(1.3021969807540397), 'feature_fraction': np.float64(0.5745506436797708), 'learning_rate': np.float64(0.3060660809801552), 'max_depth': 10, 'min_child_samples': 10, 'min_child_weight': np.float64(0.020871568153417244), 'min_data_in_leaf': 82, 'min_split_gain': np.float64(0.08154614284548342), 'n_estimators': 180, 'num_leaves': 52, 'objective': 'binary', 'random_state': 42, 'reg_alpha': np.float64(0.03702232586704518), 'reg_lambda': np.float64(1.0376985928164089), 'scale_pos_weight': np.float64(0.7317381190502594), 'subsample': np.float64(1.3631034258755936)}`
- **Training Time**: 5.65 seconds
- **Testing Time**: 1.84 seconds
- **Tuning Time**: 167.39 seconds

**Train Metrics**:
- Accuracy: 1.0000
- Precision (macro): 1.0000
- Recall (macro): 1.0000
- F1-score (macro): 1.0000
- F2-score (macro): 1.0000

**Test Metrics**:
- Accuracy: 0.9990
- Precision (macro): 0.9447
- Recall (macro): 0.9761
- F1-score (macro): 0.9590
- F2-score (macro): 0.9689

## IMPORTANT LINKS
1. [Network Anomaly Detection Project - GitHub](https://github.com/Sainadh-Bahadursha/Network_Anomaly_Detection_Project)
2. [LinkedIn Profile (Sainadh Bahadursha)](https://www.linkedin.com/in/sainadh-bahadursha-67b121171/)
3. [Tableau Dashboards](https://public.tableau.com/views/FINAL_NETWORK_ANOMALY_DETECTION_TABLEAU_WORKBOOK/AdvancedTrafficPatternsDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
4. [Data Science Portfolio](https://www.datascienceportfol.io/sainadhbahadursha)
5. [Project Video Presentation](https://drive.google.com/file/d/12A06FdNMlJWUX8BSoHqq2FWSkPfG1Ok_/view?usp=sharing)
6. [Project Part 1 - Medium Blog](https://medium.com/@sainadhbahadursha/end-to-end-network-anomaly-detection-project-from-data-exploration-to-deployment-part-1-76a3e156527e)
7. [Project Part 2 - Medium Blog](https://medium.com/@sainadhbahadursha/end-to-end-network-anomaly-detection-project-from-data-exploration-to-deployment-part-2-f72892d5e734)

This formatted markdown ensures a clear, professional presentation for your GitHub README.
