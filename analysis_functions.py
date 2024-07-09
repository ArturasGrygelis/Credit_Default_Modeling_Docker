from collections import Counter
import optuna
import lightgbm as lgb
import xgboost as xgb
from functools import reduce
from typing import List, Tuple, Union,Dict
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import t, wilcoxon
from scipy.stats import ttest_1samp,chi2_contingency
from scipy.stats import mannwhitneyu
import scipy.stats as stats
from xml.etree.ElementTree import fromstring, ElementTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, roc_auc_score


def lasso_classifier(X_train,y_train,X_test,y_test,X):
    
    lasso_classifier = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
    lasso_classifier.fit(X_train, y_train)


    y_pred = lasso_classifier.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    precision = precision_score(y_test, y_pred)
    print("Precision:", precision)

    lasso_coefficients = lasso_classifier.coef_[0]


    lasso_abs_coefficients = np.abs(lasso_coefficients)

    top_20_lasso_indices = np.argsort(lasso_abs_coefficients)[-20:]


    top_20_lasso_feature_names = X.columns[top_20_lasso_indices]


    top_20_lasso_coefficients = lasso_coefficients[top_20_lasso_indices]

    # Create a bar plot to visualize the top 20 most important features for Lasso
    plt.figure(figsize=(12, 8))
    plt.barh(top_20_lasso_feature_names, top_20_lasso_coefficients)
    plt.xlabel("Coefficient Value (Lasso)")
    plt.title("Most Important Features - Lasso")
    plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
    plt.show()    

def ridge_classifier(X_train,y_train,X_test,y_test,X):
    ridge_classifier = LogisticRegression(penalty="l2", solver="liblinear", random_state=42)
    ridge_classifier.fit(X_train, y_train)

   
    y_pred = ridge_classifier.predict(X_test)

   
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    precision = precision_score(y_test, y_pred)
    print("Precision:", precision)

   
    ridge_coefficients = ridge_classifier.coef_[0]

    
    ridge_abs_coefficients = np.abs(ridge_coefficients)

    
    top_20_ridge_indices = np.argsort(ridge_abs_coefficients)[-20:]

   
    top_20_ridge_feature_names = X.columns[top_20_ridge_indices]

    
    top_20_ridge_coefficients = ridge_coefficients[top_20_ridge_indices]

    
    plt.figure(figsize=(12, 8))
    plt.barh(top_20_ridge_feature_names, top_20_ridge_coefficients)
    plt.xlabel("Coefficient Value (Ridge)")
    plt.title("Top 20 Most Important Features - Ridge")
    plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
    plt.show()  

def is_binary(series):
    unique_values = series.unique()
    return len(unique_values) == 2 and set(unique_values) == {0, 1}


def read_stratified_sample(file_path, num_records=200000, chunk_size=10000, selected_columns=all, stratify_column='grade'):
    # If selected_columns is not provided, read all columns
    if selected_columns is None:
        selected_columns = []

    # Initialize an empty list to store the sampled data chunks
    stratified_samples = []

    # Iterate through chunks of the CSV file
    for chunk in pd.read_csv(file_path, usecols=selected_columns + [stratify_column], chunksize=chunk_size):
        # Stratify the chunk by the specified column and sample
        chunk_sample = chunk.groupby(stratify_column, group_keys=False).apply(lambda x: x.sample(min(len(x), num_records)))

        # Append the sampled chunk to the list
        stratified_samples.append(chunk_sample)

        # Break the loop if the desired number of records is reached
        if sum(len(chunk) for chunk in stratified_samples) >= num_records:
            break

    # Concatenate the list of sampled chunks into a single DataFrame
    stratified_sample = pd.concat(stratified_samples, ignore_index=True)

    return stratified_sample



def read_sample(file_path, num_records=200000, chunk_size=10000, selected_columns=None, random_state=None):
    # If selected_columns is not provided, read all columns
    if selected_columns is None:
        selected_columns = []

    # Initialize an empty list to store the sampled data chunks
    sampled_chunks = []

    # Iterate through chunks of the CSV file
    for chunk in pd.read_csv(file_path, usecols=selected_columns, chunksize=chunk_size):
        # Perform a simple random sample
        chunk_sample = chunk.sample(n=min(len(chunk), num_records), random_state=random_state)

        # Append the sampled chunk to the list
        sampled_chunks.append(chunk_sample)

        # Break the loop if the desired number of records is reached
        if sum(len(chunk) for chunk in sampled_chunks) >= num_records:
            break

    # Concatenate the list of sampled chunks into a single DataFrame
    sampled_data = pd.concat(sampled_chunks, ignore_index=True)

    return sampled_data


def chi_squared_test(features: List[str], target: str, df: pd.DataFrame) -> Dict[str, float]:
    """
    Perform chi-squared test for each specified feature against the target variable.

    Parameters:
    - features (List[str]): List of feature column names to test.
    - target (str): Name of the target variable column.
    - df (pd.DataFrame): DataFrame containing the features and target variable.

    Returns:
    Dict[str, float]: A dictionary where keys are significant features and values are their p-values.
    """
    significant_features = {}

    for feature in features:
        # Create a contingency table (cross-tabulation) for the feature and target
        contingency_table = pd.crosstab(df[feature], df[target])

        # Perform the chi-squared test
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        # Check if the p-value is less than the significance level (e.g., 0.05)
        if p_value < 0.05:
            significant_features[feature] = p_value
            print(
                f"Feature '{feature}' is significant with p-value = {p_value:.4f}"
            )
        else:
            print(
                f"Feature '{feature}' is not significant with p-value = {p_value:.4f}"
            )
    return significant_features


def mann_whitney_test(features: List[str], target: str, df: pd.DataFrame) -> Dict[str, float]:
    """
    Perform Mann-Whitney U test for each specified feature against the target variable.

    Parameters:
    - features (List[str]): List of feature column names to test.
    - target (str): Name of the target variable column.
    - df (pd.DataFrame): DataFrame containing the features and target variable.

    Returns:
    Dict[str, float]: A dictionary where keys are significant features and values are their p-values.
    """
    significant_features = {}

    for feature in features:
        # Perform Mann-Whitney U test
        stat, p_value = mannwhitneyu(df[df[target] == 0][feature], df[df[target] == 1][feature])

        # Check if the p-value is less than the significance level (e.g., 0.05)
        if p_value < 0.05:
            significant_features[feature] = p_value
            print(
                f"Feature '{feature}' is significant with p-value = {p_value:.4f}"
            )
        else:
            print(
                f"Feature '{feature}' is not significant with p-value = {p_value:.4f}"
            )

    return significant_features


def keep_top_categories(data, column_name, top_n):
    """
    Keep the top N most frequent categories in the specified column and label the rest as 'Other'.

    Parameters:
    - data: DataFrame
    - column_name: str
      The name of the column for which you want to keep the top categories.
    - top_n: int
      The number of top categories to keep.

    Returns:
    - DataFrame
      A new DataFrame with only the top N categories in the specified column.
    """
    # Step 1: Find the top N most frequent values in the specified column
    top_values = data[column_name].value_counts().nlargest(top_n).index

    # Step 2: Replace values not in the top N with 'Other'
    data[column_name] = data[column_name].apply(lambda x: x if x in top_values else 'Other')

    return data


