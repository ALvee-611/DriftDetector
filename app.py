import numpy as np
import pandas as pd
import streamlit as st
from helpers import *
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Monitoring",
                   layout="wide")

# Import FontAwesome CSS
FA_CSS = f"""<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">"""
st.markdown(FA_CSS, unsafe_allow_html=True)

# read csv
dataset_1 = "Batch_1.csv"
dataset_2 = "Batch_2.csv"

# read csv from a URL
@st.cache_data
def get_data(dataset_url) -> pd.DataFrame:
    return pd.read_csv(dataset_url)

df_1 = get_data(dataset_1)
df_2 = get_data(dataset_2)

# dashboard title
st.title("Drift Detection Dashboard")

# create two sets of predictions
predictions_1 = df_1['Attrition']
predictions_2 = df_2['Attrition']

# create a contingency table
contingency_table = pd.crosstab(predictions_1, predictions_2)


# perform the chi-squared test
#chi2, p, dof, expected = stats.chi2_contingency(contingency_table)


tab1, tab2, tab3 = st.tabs(["Drift Monitoring Dashboard", "Model Performance", "About"])

with tab1:
   st.header('Drift Detection Dashboard')
   st.caption('This is the live overview of the Model Performance where data is being added in batch and the true Attrition label is not known.')
   st.write("---")
   
   col1, col2, col3 = st.columns([1,1,1])
   col1.metric("Total Predictions", "300")
   col2.metric("Model Used", "XGBoost")
   col3.metric("Total Features with Potential Data Drift", "0", "0")

   st.title("Model Prediction on Current Batch VS Old Batch")
   st.text('This is the live overview of the Model Performance where data is being added in batch and the true Attrition label is not known.')
   col_pred, col_stat = st.columns([1,2])
   
   with col_pred:
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'Yes', 'No'
        sizes = [629, 1135]

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        ax1.set_title('Breakdown of Prediction Made By Model on Current Batch')

        st.pyplot(fig1, use_container_width = True)
   with col_stat:
      if False:
            st.markdown('<p style="color: green; font-size:3rem;"><i class="fas fa-check-circle"></i> No significant difference</p>', unsafe_allow_html=True)
      else:
            st.markdown("""<p style="color: red; font-size:3rem; text-align:center;"><i class="fas fa-check-circle"></i> Failure</p>
                        <p style="text-align: center;">This gives us  </p>""", unsafe_allow_html=True)
   
   st.write("---")

   st.title("Distance Based")

   col_desc,col_space, col_stat_check = st.columns([2,1,2])

   with col_desc:
      st.subheader("Distance-Based Approach to detecting Covariare Drift")
      st.text("Here we are checking if there is any feature drift using distanced based approach:")
   with col_stat_check:
      if True:
            st.markdown('<p style="color: green; font-size:3rem;"><i class="fas fa-check-circle"></i> No significant difference</p>', unsafe_allow_html=True)
      else:
            st.markdown("""<p style="color: red; font-size:3rem; text-align:center;"><i class="fas fa-check-circle"></i> Failure</p>
                        <p style="text-align: center;">This gives us  </p>""", unsafe_allow_html=True)
      with st.expander("See explanation"):
         st.write("""
                  The chart the code used to get this:
               """)
         st.text("Here I explain the approach")
   
   st.write("---")

   st.title("Statistical Approach to Data Drift detection")

   col_desc,col_space, col_stat_check = st.columns([2,1,2])

   with col_desc:
      st.subheader("Distance-Based Approach to detecting Covariare Drift")
      st.text("Here we are checking if there is any feature drift using distanced based approach:")
   with col_stat_check:
      if True:
            st.markdown('<p style="color: green; font-size:3rem;"><i class="fas fa-check-circle"></i> No significant difference</p>', unsafe_allow_html=True)
      else:
            st.markdown("""<p style="color: red; font-size:3rem; text-align:center;"><i class="fas fa-check-circle"></i> Failure</p>
                        <p style="text-align: center;">This gives us  </p>""", unsafe_allow_html=True)
      with st.expander("See explanation"):
         st.write("""
                  The chart the code used to get this:
               """)
         st.text("Here I explain the approach")
   
   st.write("---")

   # with col2:
   #  st.subheader("Prediction on Current Batch VS Last Batch")
   #  st.table(contingency_table)

   # data= {'Attrition_last': df_1['Attrition'].value_counts(), 'Attrition_current': df_2["Attrition"].value_counts()}
   # df = pd.DataFrame(data)

   # df
   # st.bar_chart(df)
   
with tab2:
   col1, col2, col3, col4 = st.columns([1,1,1,1])
   col1.metric("Total Predictions", "300")
   col2.metric("Current Batch VS Last Batch", "statistically same")
   col3.metric("Current Batch VS Last Batch", "statistically same")
   col4.metric("Total Features with Potential Data Drift", "0", "0")

with tab3:
   st.header("Data Drift")
   st.markdown("""One of the major tasks for any data sceintist is to constantly monitor the performance of the ML models that have been deployed into production. The deployed model may start to perform poorly over time. This can happen for various reasons: changes in the data distribution, changes in the data generating process or changes in the model itself. Model performance degradation is a real problem which can lead to incorrect predictions and so need to be continuously monitored and retrained incase a loss of performance is detected.
   
   One of the main reasons for a loss in performance is due to `Data Drift`. Data changes over time and these changes cause the model which was trained on old data to be inconsistent and unable to produce similar results using the new data. Once such change could be changes in the input variables used to make the predictions. The distribution of the inpit variables could change or the relationship between the input variables and target variables could change. This is called `Covariate drift`.
   
   We will start by implementing a framework to detect such changes in the distribution of the input variables.""")

   st.markdown("""PSI is primarily used for detecting drift in categorical features, and it compares the distribution of the feature between a reference dataset and a new dataset to determine if there is a significant change. PSI calculates a score that reflects the amount of change between the two datasets and can be used as a threshold for detecting significant drift.
   
   In general, PSI is more suitable for detecting drift in categorical features, while the KS test is more suitable for detecting drift in numeric features. However, it's important to note that both methods have limitations and may not detect certain types of drift, and it's often necessary to use multiple methods in combination to comprehensively detect data drift.""")

   st.markdown("""the contingency tables of categorical features can be treated as multidimensional frequency distributions, and the Chi-square test can be used to compare the distance between these distributions.""")




st.write(drift_detector(df_1, df_2))