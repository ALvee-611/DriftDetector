import numpy as np
import pandas as pd

from scipy.stats import truncnorm
from collections import defaultdict
from scipy.stats import ks_2samp, chi2_contingency
import scipy.stats as stats
import streamlit as st
import matplotlib.pyplot as plt
import os
import joblib



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

@st.cache_resource
def load_model(model_name):
    path = os.path.join( model_name)
    model = joblib.load(path)
    return model

#new_data = generate_data(5000)
# apply the function to create the classification label for each row in the dataset
#new_data['Attrition'] = new_data.apply(create_label, axis=1)

#save_to_db(new_data)

@st.cache_resource
def get_num_cat_columns(data):
    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    return numerical_cols, categorical_cols


def detect_data_drift(reference_data, new_data, threshold=0.05):
    
    drift_features = []
    #drift_features_cat = []
    
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
        # else:
        #     freq1 = np.bincount(old_batch)
        #     freq2 = np.bincount(new_batch)

        #     contingency_table = np.array([freq1, freq2])

        #     stat, pval, dof, expected = chi2_contingency(contingency_table)
        #     # #feature_types[feature] = 'categorical'
        #     # old_batch_codes, _ = pd.factorize(old_batch)
        #     # new_batch_codes, _ = pd.factorize(new_batch)
        #     # old_distribution = pd.crosstab(old_batch_codes, new_batch_codes)
        #     # chi2_stat, p_val, dof, expected = chi2_contingency(old_distribution)
        #     # if p_val < threshold:
        #     #     drift_features_cat.append(feature)
        #     #     #feature_names_cat.append(feature)

        #     # cont_table = pd.crosstab(old_batch, new_batch)
        #     # chi2_statistic, p_val, dof, expected = chi2_contingency(cont_table)
        #     if p_val < threshold:
        #         drift_features_cat.append(feature)

    #return drift_features_cat, drift_features
    return drift_features


def psi_numeric(observed, expected):
    # Calculate the difference between the mean of the observed and expected data
    diff = abs(observed.mean() - expected.mean())
    
    # Set a threshold for detecting drift
    threshold = 0.1 * abs(expected.mean())
    
    # If the difference exceeds the threshold, return the PSI value
    if diff > threshold:
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
    else:
        return 0.0



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

def calculate_psi(old_batch, current_batch):
    drift_features = []

    for feature in old_batch.columns:
        df_1 = old_batch[feature]
        df_2 = current_batch[feature]

        if np.issubdtype(df_1.dtype, np.number):
            psi_threshold = 0.1 * abs(df_1.mean())
            # psi_threshold = 0.1 # Change this value to set the desired PSI threshold
        else:
            psi_threshold = 0.05

        if np.issubdtype(df_1.dtype, np.number):
            stat = psi_numeric(df_1, df_2)

            # if stat > optimal_threshold:
            if stat > psi_threshold:
                drift_features.append(feature)
        else:
            stat = psi_cat(df_1, df_2)

            if stat > psi_threshold:
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
from joblib import load

def calculate_model_KPI(model, new_df):
    loc = os.getcwd()
    p = os.path.join(loc,'preprocess_pipeline', 'preprocess_pipeline.joblib')
    Preprocess = load(p)
    y_true = new_df['Attrition']
    new_df['OverTime'] = new_df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
    X_df = new_df.drop('Attrition', axis=1)
    new_data = Preprocess.transform(X_df)
    X = pd.DataFrame(new_data)
    y_pred = model.predict(X)

    # assuming y_true and y_pred are the true and predicted labels, respectively
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return accuracy, precision, recall, f1, auc_roc, cm, fpr, tpr, thresholds

#############################################################################################################################

@st.cache_resource
def drift_detector(old_data, new_data):
    drift_dict = defaultdict()

    # KS_test and Chi_sq test for numeric and categorical features respectively
    #cat_features, 
    num_features = detect_data_drift(old_data, new_data, threshold=0.05)
    drift_dict['KS_test'] = num_features
    #drift_dict['Chi_Sq'] = cat_features

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

def draw_figures(feature, old_dist, new_dist, numerical_col):
    fig, ax = plt.subplots()
    if feature in numerical_col:
        iqr = np.percentile(old_dist, 75) - np.percentile(old_dist, 25)
        if iqr == 0:
            bin_width = 1
        else:
            bin_width = 2 * iqr / (len(old_dist) ** (1/3))

        num_bins = int(np.ceil((old_dist.max() - old_dist.min()) / bin_width))

        ax.hist(old_dist, alpha=0.6, label='Old Batch', bins=num_bins, edgecolor='black')
        ax.hist(new_dist,alpha=0.6, label='New Batch', bins=num_bins, edgecolor='black')
        ax.set_title("Distribution for " + feature)
        ax.legend()
    else:
        old_dist_counts = old_dist.value_counts()
        new_dist_counts = new_dist.value_counts()
        categories = set(list(old_dist_counts.index) + list(new_dist_counts.index))

        old_counts = [old_dist_counts.get(cat, 0) for cat in categories]
        new_counts = [new_dist_counts.get(cat, 0) for cat in categories]

        x = np.arange(len(categories))
        width = 0.35

        rects1 = ax.bar(x - width/2, old_counts, width, alpha=0.6, label='Old Batch')
        rects2 = ax.bar(x + width/2, new_counts, width, alpha=0.6, label='New Batch')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title("Distribution for " + feature)
        ax.legend()

    st.pyplot(fig)



### Introduce categorical drift
@st.cache_resource
def cat_drift(data, feature_name, drift_type='GRADUAL'):

    num_steps = 15
    # Get current category names and probabilities
    cat_prob_now = data[feature_name].value_counts(normalize=True).sort_index().values
    cat_names = data[feature_name].value_counts(normalize=True).sort_index().index

    alpha = [1] * len(cat_prob_now)
    new_cat_prob = np.random.dirichlet(alpha, size=1).flatten() * sum(cat_prob_now)

    new_prob = new_cat_prob.tolist()

    new_data = pd.DataFrame()

    # Gradual drift
    if drift_type == 'GRADUAL':
        step_sizes = [(new_p - curr_p) / num_steps for new_p, curr_p in zip(new_prob, cat_prob_now)]
        
        # new probabilities for each step added gradually
        for i in range(num_steps+1):

            # Calculate new probabilities for this step
            step_prob = [curr_prob + (step_size * i) for curr_prob, step_size in zip(cat_prob_now, step_sizes)]
            step_prob = np.clip(step_prob, 0, 1) 
            
            # Create a new DataFrame with the drifted data for this step
            step_data = data.copy()
            step_data[feature_name] = np.random.choice(cat_names, size=len(data), p=step_prob)
            
            # AddING new data to the main DataFrame
            new_data = pd.concat([new_data, step_data], axis=0)
    
    # Sudden drift
    elif drift_type == 'SUDDEN':
        # Create a new DataFrame with the drifted data
        new_data = data.copy()
        new_data[feature_name] = np.random.choice(cat_names, size=len(data), p=new_prob)
    
    return np.random.choice(cat_names, size=len(data), p=new_prob)

## DRIFT FOR NUMERIC DATA
@st.cache_resource
def num_drift(data, feature_name, drift_type='GRADUAL'):
    data_to_drift = data[feature_name]

    if drift_type == 'GRADUAL':
        delta = np.linspace(0, data_to_drift.mean() * 0.5, len(data_to_drift))
        new_drift_data = data_to_drift + delta
    else:
        sudden = data_to_drift.mean() * 0.75
        new_drift_data = data_to_drift + sudden

    #data[feature_name] = new_drift_data
    return new_drift_data