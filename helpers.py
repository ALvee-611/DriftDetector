import numpy as np
import pandas as pd
import sqlite3
from scipy.stats import truncnorm
from collections import defaultdict
from scipy.stats import ks_2samp, chi2_contingency
import scipy.stats as stats
import streamlit as st
import matplotlib.pyplot as plt


def generate_data(n_samples):
    # set random seed for reproducibility
    #np.random.seed(123)
    
    # define means and standard deviations for continuous variables
    age_mean = 35
    age_std = 10
    hourly_rate_mean = 65
    hourly_rate_std = 15
    monthly_income_mean = 5000
    monthly_income_std = 2000
   
    # generate continuous variables
    age = truncnorm((18 - age_mean) / age_std, (65 - age_mean) / age_std, loc=age_mean, scale=age_std).rvs(size=n_samples).astype(int)
    hourly_rate = np.random.normal(hourly_rate_mean, hourly_rate_std, n_samples)
    monthly_income = np.random.normal(monthly_income_mean, monthly_income_std, n_samples)
    
    # define parameter values for discrete variables
    distance_param = 0.3
    job_inv_param = 2
    job_level_param = 2
    job_sat_param = 3   
    work_life_bal_param = 3
    
    # generate discrete variables
    distance_from_home = np.random.negative_binomial(distance_param, 1-distance_param, n_samples)
    job_involvement = np.random.poisson(job_inv_param, n_samples)
    job_level = np.random.poisson(job_level_param, n_samples)
    job_satisfaction = np.random.poisson(job_sat_param, n_samples)
    work_life_balance = np.random.poisson(work_life_bal_param, n_samples)
    
    # define category names and probabilities for categorical variables
    business_travel_cat = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
    business_travel_prob = [0.7, 0.2, 0.1]
    department_cat = ['Sales', 'Research & Development', 'Human Resources']
    department_prob = [0.4, 0.5, 0.1]
    education_field_cat = ['High School', 'Undergraduate', 'Masters', 'Phd']
    education_field_prob = [0.2, 0.4, 0.2, 0.2]
    marital_status_cat = ['Married', 'Single', 'Divorced']
    marital_status_prob = [0.4, 0.4, 0.2]
    gender_cat = ['Male', 'Female']
    gender_prob = [0.6, 0.4]

    # generate categorical variables
    business_travel = np.random.choice(business_travel_cat, n_samples, p=business_travel_prob)
    department = np.random.choice(department_cat, n_samples, p=department_prob)
    education_level = np.random.choice(education_field_cat, n_samples, p=education_field_prob)
    marital_status = np.random.choice(marital_status_cat, n_samples, p=marital_status_prob)
    gender = np.random.choice(gender_cat, n_samples, p=gender_prob)


    # combine all variables into a pandas dataframe
    data = pd.DataFrame({'Age': age,
                        'BusinessTravel': business_travel,
                        'Department': department,
                        'DistanceFromHome': distance_from_home,
                        'Education': education_level,
                        'Gender': gender,
                        'HourlyRate': hourly_rate,
                        'JobInvolvement': job_involvement,
                        'JobLevel': job_level,
                        'JobSatisfaction': job_satisfaction,
                        'MaritalStatus': marital_status,
                        'MonthlyIncome': monthly_income,
                        'OverTime': np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75]),
                        'TotalWorkingYears': np.random.randint(0, 40, n_samples),
                        'WorkLifeBalance': work_life_balance,
                        'YearsAtCompany': np.random.randint(0, 20, n_samples),
                        'YearsInCurrentRole': np.random.randint(0, 15, n_samples),
                        'YearsSinceLastPromotion': np.random.randint(0, 15, n_samples),
                        'YearsWithCurrManager': np.random.randint(0, 15, n_samples)})

    return data


def create_label(row):
    if row['Age'] < 35 and ((row['BusinessTravel'] == 'Travel_Rarely') or  (row['BusinessTravel'] =='Non-Travel')) and row['MaritalStatus'] == 'single':
        return 0  
    elif row['Age'] > 45 and row['MonthlyIncome'] > 4000 and row['MaritalStatus'] == 'married':
        return 0
    elif (row['JobSatisfaction']>3 and row['YearsAtCompany'] > 8) or (row['OverTime'] == 'No' and row['YearsSinceLastPromotion'] < 4):
        return 0
    else:
        return 1  


def save_to_db(data):
    # Connect to an SQLite database
    conn = sqlite3.connect('example.db')

    # Save the DataFrame to a table in the database
    data.to_sql('Tab', conn, index=False)

    # Close the database connection
    conn.close()


#new_data = generate_data(5000)
# apply the function to create the classification label for each row in the dataset
#new_data['Attrition'] = new_data.apply(create_label, axis=1)

#save_to_db(new_data)


def detect_data_drift(reference_data, new_data, threshold=0.05):
    
    drift_features = []
    drift_features_cat = []
    
    # check feature types and calculate distance for each feature
    for feature in reference_data.columns:
        old_batch = reference_data[feature]
        new_batch = new_data[feature]

        # check if feature is categorical or numeric
        if np.issubdtype(old_batch.dtype, np.number):
            ks_statistic, p_val = ks_2samp(old_batch, new_batch)
            if p_val < threshold:
                #drift_dict['ks_2samp'] = p_val
                drift_features.append(feature)
                #feature_names.append(feature)
        else:
            #feature_types[feature] = 'categorical'
            old_distribution = pd.crosstab(old_batch, new_batch)
            chi2_stat, p_val, dof, expected = chi2_contingency(old_distribution)
            if p_val < threshold:
                drift_features_cat.append(feature)
                #feature_names_cat.append(feature)
    return drift_features_cat, drift_features


def psi_numeric(observed, expected):
    buckets = 10


    # Calculate the bucket boundaries
    boundaries = np.quantile(observed.index.values, np.linspace(0, 1, buckets + 1))
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    # Group the observed and expected data based on the bucket boundaries
    observed_groups = observed.groupby(pd.cut(observed.index.values, boundaries)).count()
    expected_groups = expected.groupby(pd.cut(expected.index.values, boundaries)).count()

    # Calculate the observed and expected proportions for each group
    observed_proportions = observed_groups / observed.sum()
    expected_proportions = expected_groups / expected.sum()

    # Add missing buckets to expected proportions
    missing_buckets = observed_proportions.index.difference(expected_proportions.index)
    for bucket in missing_buckets:
        expected_proportions[bucket] = 0

    # Sort the data by the bucket boundaries
    observed_proportions.sort_index(inplace=True)
    expected_proportions.sort_index(inplace=True)

    # Calculate the PSI value
    psi_value = np.sum((observed_proportions - expected_proportions) * np.log(observed_proportions / expected_proportions)) * 100

    return psi_value

def psi_cat(ref_feature, new_feature):
    # Calculate distribution of actual and expected values
    actual_counts = ref_feature.value_counts(normalize=True, sort=False)
    expected_counts = new_feature.value_counts(normalize=True, sort=False)

    # Calculate the proportion of each distribution
    actual_prop = actual_counts / actual_counts.sum()
    expected_prop = expected_counts / expected_counts.sum()

    # Calculate the PSI value
    psi_value = ((expected_prop - actual_prop) * np.log(expected_prop / actual_prop)).sum()

    return psi_value


def calculate_psi(old_batch, current_batch, threshold=0.1):
    drift_features = []

    for feature in old_batch.columns:
        df_1 = old_batch[feature]
        df_2 = current_batch[feature]
        if np.issubdtype(df_1.dtype, np.number):
            stat = psi_numeric(df_1, df_2)

            if stat > threshold:
                drift_features.append(feature)
        else:
            stat = psi_cat(df_1, df_2)

            if stat > threshold:
                drift_features.append(feature)
    return drift_features


def compare_preds(old_batch, current_batch, alpha = 0.05):
    """
    compare_preds(old_batch, current_batch) takes two datsets and then retruns true if the target variable in the two sets are 
    statistically different
    DataFrame DataFrame -> Bool
    """
    # create two sets of predictions
    predictions_1 = old_batch['Attrition']
    predictions_2 = current_batch['Attrition']

    # create a contingency table
    contingency_table = pd.crosstab(predictions_1, predictions_2)

    # perform the chi-squared test
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

    if p_val < alpha:
        return "Significant difference between the two sets of predictions"
    else:
        return "No significant difference between the two sets of prediction"
    

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from collections import Counter

def compute_numerical_distance(ref_data, prod_data):
    """
    Compute the Euclidean distance between the summary statistics of two numerical columns
    """
    ref_stats = np.array([ref_data.mean(), ref_data.std(), ref_data.var()])
    prod_stats = np.array([prod_data.mean(), prod_data.std(), prod_data.var()])
    return cdist(ref_stats.reshape(1, -1), prod_stats.reshape(1, -1), metric='euclidean')[0, 0]

def compute_categorical_distance(ref_data, prod_data):
    """
    Compute the Jensen-Shannon distance between the empirical distributions of two categorical columns
    """
    ref_counts = dict(Counter(ref_data))
    prod_counts = dict(Counter(prod_data))
    all_values = list(set(ref_counts.keys()) | set(prod_counts.keys()))
    ref_probs = np.array([ref_counts.get(val, 0) / len(ref_data) for val in all_values])
    prod_probs = np.array([prod_counts.get(val, 0) / len(prod_data) for val in all_values])
    avg_probs = 0.5 * (ref_probs + prod_probs)
    js_divergence = 0.5 * (np.sum(ref_probs * np.log(ref_probs / avg_probs)) + np.sum(prod_probs * np.log(prod_probs / avg_probs)))
    #print(js_divergence)
    return np.sqrt(js_divergence)

def compute_overall_distance(ref_df, prod_df, numerical_weights=None):
    """
    Compute the overall distance metric between the reference and production dataframes
    """
    numerical_cols = ref_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = ref_df.select_dtypes(include=['object']).columns.tolist()

    if numerical_weights is None:
        numerical_weights = [1] * len(numerical_cols)
    num_distance = 0
    cat_distance = 0
    for i, col in enumerate(numerical_cols):
        num_distance += numerical_weights[i] * compute_numerical_distance(ref_df[col], prod_df[col])
        # print(num_distance)
    for col in categorical_cols:
        cat_distance += compute_categorical_distance(ref_df[col], prod_df[col])

       # print(cat_distance)
    return num_distance + cat_distance

def test_for_drift(ref_data, prod_data, numerical_weights=None, threshold=0.1):
    """
    Test for covariate drift between the reference and production dataframes
    """
    distance = compute_overall_distance(ref_data, prod_data, numerical_weights)
    return distance >= threshold


## Model KPIs

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def calculate_model_KPI(y_true, y_pred):
    # assuming y_true and y_pred are the true and predicted labels, respectively
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, auc_roc, cm

def plot_roc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return fpr, tpr, thresholds
    
# # plot the ROC curve
# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()

@st.cache_resource
def drift_detector(old_data, new_data):
    drift_dict = defaultdict()

    # KS_test and Chi_sq test for numeric and categorical features respectively
    cat_features, num_features = detect_data_drift(old_data, new_data, threshold=0.05)
    drift_dict['KS_test'] = num_features
    drift_dict['Chi_Sq'] = cat_features

    psi_features = calculate_psi(old_data, new_data)
    drift_dict['PSI'] = psi_features

    numerical_cols = []
    categorical_cols = []
    for feature in old_data.columns:
        if np.issubdtype(old_data[feature].dtype, np.number):
            numerical_cols.append(feature)
        else:
            categorical_cols.append(feature)
    
    val = drift_dict.values()
    
    # True only if all the tests are empty
    any_drift = all(not v for v in val)

    return drift_dict, any_drift

@st.cache_resource
def draw_fig(old_df, new_df, feature_to_plot):
    old_dist = old_df[feature_to_plot]
    new_dist = new_df[feature_to_plot]

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))

    axs.hist(old_dist, alpha=0.5)
    axs.set_title("Old Distrubtion for" + feature_to_plot)

    axs.hist(new_dist,alpha=0.5)
    axs.set_title("New Distrubtion for" + feature_to_plot)

    st.pyplot(fig)
