import numpy as np
import pandas as pd
import sqlite3
from scipy.stats import truncnorm

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


new_data = generate_data(5000)
# apply the function to create the classification label for each row in the dataset
new_data['Attrition'] = new_data.apply(create_label, axis=1)

#save_to_db(new_data)

new_data.to_csv('Batch_1.csv', index=False)

new_data_2 = generate_data(5000)
# apply the function to create the classification label for each row in the dataset
new_data_2['Attrition'] = new_data_2.apply(create_label, axis=1)

new_data_2.to_csv('Batch_2.csv', index = False)
