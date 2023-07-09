'''
Module Name: Cluster Helper
Description: Functions to help in performing k means clustering on features from input excel file for use in site risk profiling.
Date Created: 2023-04-26
Date Last Modified: 2023-04-26
Version: 0.1
Author: Meg Sharma, PPAD
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

class clusterHelper:
    
    def __init__(self):
        pass
 
    def select_features(t_df: pd.DataFrame, features_list: list, t_label: pd.Series):
        """
        selects features using an input feature list and label pandas Series, scales the features and returns scaled features and label Series
        """
        feat = t_df[features_list]
        X = t_df[t_label]
        feat = feat.fillna(0)
        features = list(feat.itertuples(index=False))
        scaler = StandardScaler()
        scaler.fit(features)
        scaled_features = scaler.fit_transform(features)
        return scaled_features, X
    
    def visualize_results(t_X, t_labels_scale):
        """
        helper function to plot clustering results
        arguments:
        t_X: label name
        t_labels_scale: clustering labels
        """
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(10,10))
        # scat = sns.scatterplot(t_X, t_labels_scale,
        #     hue = t_labels_scale,
        #     style = t_labels_scale,
        #     palette="Set2")
        plt.scatter(t_X, t_labels_scale)
        plt.title("K Means Clusters")
        plt.xlabel("Site")
        plt.ylabel("Grouping")
        plt.show()