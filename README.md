# Network_Anomaly_Detection_Project
> An anomaly detection system for identifying unusual network behavior using machine learning algorithms in Python. . This project offers an end-to-end approach to network anomaly detection, starting with data exploration and visualization in Tableau, followed by exploratory data analysis (EDA), hypothesis testing, machine learning modeling with experimental tracking, and concluding with deployment.

# Problem Statement
> In the realm of cybersecurity, network anomaly detection is a critical task that involves identifying unusual patterns or behaviors that deviate from the norm within network traffic. These anomalies could signify a range of security threats, from compromised devices and malware infections to large-scale cyber-attacks like DDoS (Distributed Denial of Service).

> So assume you are working as a data scientist with cyber security department. You are provided with the Networking data (NSL-KDD dataset). Your task is to visualise and analyse the given data. Apply Supervised Learning algorithms to find the best model which can classify the data using attack column. Unsupervised algorithms can be used to improve the performance. Finally deploy the machine learning models via Streamlit.

# Target Metric
Used all the classification metrics like accuracy, precision, recall, f1 score. But for network anomaly detection, Recall and accuracy plays important role than others. 

# Contents available in `Network_Anomaly_Detection_Project` 
1. Definitions & Problem Statement - All the feature definitions, dataset origin, problem statement were discussed. 
2. Tableau Visualization - Two dashboards are created - a) Network Connection Metrics Dashboard based on Basic and Content Related features b) Advanced Traffic Patterns Dashboard based on Time and Host Related Features
3. Exploratory Data Analysis Using Python - Has performed all the EDA steps. Cleaned the data. Has done feature engineering us. Univariate, Bivariate, Multi Variate Data Visualisation using python 
4. Hypothesis Testing - Hypothesis Testing has performed between important features vs binary target as well as multi-class target. Verified the assumptions. If failed applied alternative non parametric testings too
5. Feature Engineering using Unsupervised algorithms - Used Anomaly Detection algorithms to add additional features. By using an assumption from the journal, Performed Kmeans_advanced clustering to get the features which can improve the classification performance.
6. Machine Learning Modeling for Binary Classification - Tuned and applied 10 different classification models and plotted all the metrics. Used Mlflow experimental tracking approach.
7. Evaluation of Results for Binary Classification - Compared all the 10 models by plotting all the logged metrics of Mlflow.
8. Multi-Class Classification Utilizing Optimal Models - Applied preprocessing, Applied only optimal models and compared the results.
9. Deployment of the Optimal Model Using Streamlit - Deployed using streamlit locally as well as in streamlit cloud.
10. Conclusion, Actionable Insights, Recommendations, Future Scope & References - Summarized and concluded. Provided all the links.

# Final Scores Achieved - 
**Binary Class Classification** - 
Model: Tuned_Adaboost_on_Balanced_Dataset
params: {'algorithm': 'SAMME', 'learning_rate': np.float64(0.8065429868602328), 'n_estimators': 114}
Training Time: 98.1315 seconds
Testing Time: 2.8727 seconds
Tuning Time: 227.5235 seconds

Train Metrics:
Accuracy_train: 1.0000
Precision_train: 1.0000
Recall_train: 1.0000
F1_score_train: 1.0000
F2_score_train: 1.0000
Roc_auc_train: 1.0000
Pr_auc_train: 1.0000

Test Metrics:
Accuracy_test: 0.9992
Precision_test: 0.9996
Recall_test: 0.9988
F1_score_test: 0.9992
F2_score_test: 0.9990
Roc_auc_test: 1.0000
Pr_auc_test: 1.0000

**Multi- Class Classification**
Training Time: 5.6453 seconds
Testing Time: 1.8362 seconds
Tuning Time: 167.3905 seconds

Train Metrics:
Accuracy_train: 1.0000
Precision_train_macro: 1.0000
Recall_train_macro: 1.0000
F1_score_train_macro: 1.0000
F2_score_train_macro: 1.0000

Test Metrics:
Accuracy_test: 0.9990
Precision_test_macro: 0.9447
Recall_test_macro: 0.9761
F1_score_test_macro: 0.9590
F2_score_test_macro: 0.9689

# IMPORTANT LINKS
1. [NETWORK ANOMALY DETECTION PROJECT-GITHUB LINK](https://github.com/Sainadh-Bahadursha/Network_Anomaly_Detection_Project)
2. [MY LINKED IN PROFILE LINK (Sainadh Bahadursha)](https://www.linkedin.com/in/sainadh-bahadursha-67b121171/)
3. [NETWORK ANOMALY DETECTION PROJECT- TABLEAU DASHBOARDS LINK](https://public.tableau.com/views/FINAL_NETWORK_ANOMALY_DETECTION_TABLEAU_WORKBOOK/AdvancedTrafficPatternsDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
4. [MY DATASCIENCE PORTFOL.IO LINK](https://www.datascienceportfol.io/sainadhbahadursha)
5. [NETWORK ANOMALY DETECTION PROJECT VIDEO PRESENTATION LINK](https://drive.google.com/file/d/12A06FdNMlJWUX8BSoHqq2FWSkPfG1Ok_/view?usp=sharing)
6. [NETWORK ANOMALY DETECTION PROJECT PART 1 MEDIUM BLOG](https://medium.com/@sainadhbahadursha/end-to-end-network-anomaly-detection-project-from-data-exploration-to-deployment-part-1-76a3e156527e)
7. [NETWORK ANOMALY DETECTION PROJECT PART 2 MEDIUM BLOG](https://medium.com/@sainadhbahadursha/end-to-end-network-anomaly-detection-project-from-data-exploration-to-deployment-part-2-f72892d5e734)


