# import numpy as np
# from sklearn.neighbors import LocalOutlierFactor
# import pickle,gzip,warnings
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.ensemble import IsolationForest
# from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
# from sklearn.covariance import EllipticEnvelope
# from sklearn.svm import OneClassSVM
# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import DBSCAN,KMeans
# from sklearn.ensemble import AdaBoostClassifier

# # Load globally
# with open('ohe_encoder.pkl', 'rb') as f:
#     ohe_encoder = pickle.load(f)

# with open('service_encoding.pkl', 'rb') as f:
#     service_encoding = pickle.load(f)

# with open('nadp_X_train_binary_scaler.pkl', 'rb') as f:
#     nadp_X_train_binary_scaler = pickle.load(f)

# with gzip.open('nadp_X_train_binary_final.pkl', 'rb') as f:
#     nadp_X_train_binary_final = pickle.load(f)

# with open("train_binary_iforest.pkl","rb") as f:
#     train_binary_iforest = pickle.load(f)

# with open("train_binary_robust_cov.pkl","rb") as f:
#     train_binary_robust_cov = pickle.load(f)

# with open("train_binary_one_class_svm.pkl","rb") as f:
#     train_binary_one_class_svm = pickle.load(f)

# with open("train_binary_knn.pkl","rb") as f:
#     train_binary_knn = pickle.load(f)

# with open("train_binary_gmm.pkl","rb") as f:
#     train_binary_gmm = pickle.load(f)

# with open("k_means_scaler.pkl","rb") as f:
#     k_means_scaler = pickle.load(f)

# with open("kmeans_best.pkl","rb") as f:
#     kmeans_best = pickle.load(f)

# with open("kmeans_adv_scaler.pkl","rb") as f:
#     kmeans_adv_scaler = pickle.load(f)

# with open("best_binary_classification_model.pkl","rb") as f:
#     best_binary_classification_model = pickle.load(f)

def label_clusters(labels, centroids, points, alpha, beta, gamma):
    """
    Labels clusters as normal or anomaly based on size, density, and extreme density.
    """
    unique_labels = np.unique(labels[labels != -1])
    n_clusters = len(unique_labels)
    cluster_sizes = np.array([np.sum(labels == i) for i in unique_labels])
    N = len(points)
    anomaly_labels = np.full(labels.shape, 'normal')

    # Calculate within-cluster sum of squares
    within_cluster_sums = []
    for i in unique_labels:
        cluster_points = points[labels == i]
        centroid = centroids[i]
        sum_of_squares = np.sum(np.linalg.norm(cluster_points - centroid, axis=1)**2)
        within_cluster_sums.append(sum_of_squares / len(cluster_points) if len(cluster_points) > 0 else np.inf)

    median_within_sum = np.median(within_cluster_sums)

    # Label clusters based on the given conditions
    for i, label in enumerate(unique_labels):
        size = cluster_sizes[i]
        average_within_sum = within_cluster_sums[i]

        if size < alpha * (N / n_clusters):
            anomaly_labels[labels == label] = 'anomal'
        elif average_within_sum > beta * median_within_sum:
            anomaly_labels[labels == label] = 'anomal'
        elif average_within_sum < gamma * median_within_sum:
            anomaly_labels[labels == label] = 'anomal'

    return anomaly_labels

# The existing function definition
def binary_classification_prediction(df):

    warnings.filterwarnings('ignore')

    # feature engineering
    # Time and host related features
    df['serrors_count'] = df['serrorrate']*df['count']
    df['rerrors_count'] = df['rerrorrate']*df['count']
    df['samesrv_count'] = df['samesrvrate']*df['count']
    df['diffsrv_count'] = df['diffsrvrate']*df['count']
    df['serrors_srvcount'] = df['srvserrorrate']*df['srvcount']
    df['rerrors_srvcount'] = df['srvrerrorrate']*df['srvcount']
    df['srvdiffhost_srvcount'] = df['srvdiffhostrate']*df['srvcount']
    df['dsthost_serrors_count'] = df['dsthostserrorrate']*df['dsthostcount']
    df['dsthost_rerrors_count'] = df['dsthostrerrorrate']*df['dsthostcount']
    df['dsthost_samesrv_count'] = df['dsthostsamesrvrate']*df['dsthostcount']
    df['dsthost_diffsrv_count'] = df['dsthostdiffsrvrate']*df['dsthostcount']
    df['dsthost_serrors_srvcount'] = df['dsthostsrvserrorrate']*df['dsthostsrvcount']
    df['dsthost_rerrors_srvcount'] = df['dsthostsrvrerrorrate']*df['dsthostsrvcount']
    df['dsthost_samesrcport_srvcount'] = df['dsthostsamesrcportrate']*df['dsthostsrvcount']
    df['dsthost_srvdiffhost_srvcount'] = df['dsthostsrvdiffhostrate']*df['dsthostsrvcount']

    # drop numoutboundcmd
    df = df.drop(["numoutboundcmds"], axis=1)

    # Data speed features
    df['srcbytes/sec'] = df.apply(lambda row: row['srcbytes'] / row['duration'] if row['duration'] != 0 else row['srcbytes'] / (row['duration'] + 0.001), axis=1)
    df['dstbytes/sec'] = df.apply(lambda row: row['dstbytes'] / row['duration'] if row['duration'] != 0 else row['dstbytes'] / (row['duration'] + 0.001), axis=1)

    # modify suattempted to binary
    df["suattempted"] = df["suattempted"].apply(lambda x: 0 if x == 0 else 1)

    # encoding categorical features using ohe_encoder and service_encoder
    encoded_test_data = ohe_encoder.transform(df[['protocoltype', 'flag']])
    encoded_df = pd.DataFrame(encoded_test_data, columns=ohe_encoder.get_feature_names_out(['protocoltype', 'flag']))
    df = pd.concat([df.drop(columns=['protocoltype', 'flag']), encoded_df], axis=1)
    df['service'] = df['service'].map(service_encoding)

    # For any new service types in the test dataset that weren't in the training set, assign max + 1 = 70
    df["service"] = df['service'].fillna(70)

    # Scale the test features using the same scaler
    df_scaled = nadp_X_train_binary_scaler.transform(df)

    # Convert the scaled test features back to a DataFrame
    df = pd.DataFrame(df_scaled, columns=df.columns)

    # Removing VIF features
    VIF_reduced_columns = ['duration', 'srcbytes', 'dstbytes', 'wrongfragment', 'urgent', 'hot',
        'numfailedlogins', 'numcompromised', 'numfilecreations', 'numshells',
        'numaccessfiles', 'serrorrate', 'rerrorrate', 'diffsrvrate',
        'srvdiffhostrate', 'dsthostcount', 'dsthostsrvcount',
        'dsthostdiffsrvrate', 'dsthostsamesrcportrate',
        'dsthostsrvdiffhostrate', 'serrors_count', 'rerrors_count',
        'samesrv_count', 'diffsrv_count', 'serrors_srvcount',
        'rerrors_srvcount', 'srvdiffhost_srvcount', 'dsthost_rerrors_count',
        'dsthost_samesrv_count', 'dsthost_serrors_srvcount',
        'dsthost_rerrors_srvcount', 'dsthost_samesrcport_srvcount',
        'dsthost_srvdiffhost_srvcount', 'srcbytes/sec', 'dstbytes/sec']
    cat_features = ['flag_REJ','flag_RSTO', 'flag_RSTOS0', 'flag_RSTR', 'flag_S0',
                    'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH', 'isguestlogin', 
                    'ishostlogin','land', 'loggedin', 'protocoltype_tcp', 'protocoltype_udp',
                    'rootshell','service', 'suattempted']
    final_selected_features = VIF_reduced_columns + cat_features
    df = df[final_selected_features]

    # Now, add the query point as an extended dataset for LOF calculation
    query_point_df = df.copy(deep = True)
    query_point = df.values  # The query point from the DataFrame `df`
    extended_data = np.vstack([nadp_X_train_binary_final, query_point])  # Extend the training data with the query point

    # Fit the LOF model on the extended data (training data + query point)
    train_binary_lof = LocalOutlierFactor(n_neighbors=20, contamination="auto", n_jobs=-1)
    extended_data_lof_labels = train_binary_lof.fit_predict(extended_data)  # Fit on extended data

    # Extract the Negative Outlier Factor (NOF) for the last point (query point)
    query_point_lof_nof = train_binary_lof.negative_outlier_factor_[-1]  # Get the NOF for the last point

    # Add the NOF as a new feature for the query point
    df['binary_lof_nof'] = query_point_lof_nof  # Only add the NOF for the query point

    # Add the decision_function of train_binary_iforest
    df["binary_iforest_df"]= train_binary_iforest.decision_function(query_point_df)

    # Add the decision_function of train_binary_robust_cov
    df["binary_robust_cov_df"]= train_binary_robust_cov.decision_function(query_point_df)

    # Add the decision_function of train_binary_one_class_svm
    df["binary_one_class_svm_df"]= train_binary_one_class_svm.decision_function(query_point_df)

    # Fit the dbscan model on the extended data (training data + query point)
    train_binary_dbscan = DBSCAN(eps = 0.5, min_samples=5,n_jobs=-1)
    df["binary_dbscan_labels"] = train_binary_dbscan.fit_predict(extended_data)[-1]  # Fit on extended data

    # Add the kth neighbor distance using train_binary_knn
    test_distances, test_indices = train_binary_knn.kneighbors(query_point_df)
    df["binary_knn_kth_distance"]= test_distances[:,-1]

    # Add the decision_function of train_binary_gmm
    df["binary_gmm_score"]= train_binary_gmm.score_samples(query_point_df)

    # Apply k_means_scaler transform
    data_train = k_means_scaler.transform(df)
    train_labels = label_clusters(kmeans_best.predict(df), kmeans_best.cluster_centers_, data_train, 0.01, 2.0, 0.25)
    df["binary_kmeans_adv"] = np.where(train_labels == "anomal",1,0)

    # Apply Final scaling before classification algorithms
    df_array = kmeans_adv_scaler.transform(df)
    df = pd.DataFrame(df_array,columns=df.columns)

    # Classification prediction
    prediction = best_binary_classification_model.predict(df)

    if prediction[0] == 0:
        return "NORMAL"
    elif prediction[0] == 1:
        return "ATTACK"
    return None