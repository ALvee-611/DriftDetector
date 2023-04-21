import numpy as np
import pandas as pd
import streamlit as st
from helpers import *
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Model Monitoring",
                   layout="wide")

Xg_model = load_model('xgb_model.joblib')


# Import FontAwesome CSS
FA_CSS = f"""<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">"""
st.markdown(FA_CSS, unsafe_allow_html=True)

# read csv
dataset_1 = "Batch_1.csv"
dataset_2 = "Batch_2.csv"
main_data = 'batch_data.csv'

# read csv from a URL
@st.cache_data
def get_data(dataset_url) -> pd.DataFrame:
    return pd.read_csv(dataset_url)

#### THIS WILL BE OUR DEFAULT BASELINE DATA
main_df = get_data(main_data)

all_columns = main_df.columns

#st.dataframe(main_df)

#total_pred = df_1["Attrition"].shape[0] + df_2["Attrition"].shape[0]


# dashboard title
st.title("Drift Detection Dashboard")

# create two sets of predictions
# predictions_1 = df_1['Attrition']
# predictions_2 = df_2['Attrition']

# create a contingency table
# contingency_table = pd.crosstab(predictions_1, predictions_2)

# Metrics
accuracy, precision, recall, f1, auc_roc, cm, fpr, tpr, thresholds = calculate_model_KPI(Xg_model, main_df)


#comparing_predictions = compare_preds(df_1, df_2)

# if comparing_predictions:
#    res = "statistically same"
# else:
#    res = "statistically different"

tab1, tab2, tab3 = st.tabs(["Drift Monitoring Dashboard", "Model Performance", "About"])

st.button("Generate Random")
# Using "with" notation
with st.sidebar:
    st.header("Choose all of the features you want to add drift to:")
    options = st.multiselect(
    'Choose from the droopdown menu',
    all_columns,
    )

st.write('You selected:', options)

with tab1:
   st.header('Drift Detection Dashboard')
   st.caption('This is the live overview of the Model Performance where data is being added in batch and the true Attrition label is not known.')
   st.write("---")
   
   col1, col2, col3 = st.columns([1,3,2])
   #col1.metric("Total Predictions", total_pred)
   # col2.metric("Current Batch Predictions VS Last Batch Predictions", res)
   col3.metric("# Features with Potential Data Drift", "0")

   st.subheader("Model Prediction on Current Batch VS Old Batch")
   st.text('This is the live overview of the Model Performance where data is being added in batch and the true Attrition label is not known.')
   

   col_pred, col_stat = st.columns([1,1])
   
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
      if True:# comparing_predictions:
            st.markdown('<p style="color: green; font-size:1.5rem; text-align: center;"><i class="fas fa-check-circle"></i> No significant difference between the two sets of predictions!</p>', unsafe_allow_html=True)
      else:
            st.markdown("""<p style="color: red; font-size:1.5rem; text-align:center;"><i class="fas fa-check-circle"></i> Failure</p>
                        <p style="text-align: center;">Possible Concept Drift!!</p>""", unsafe_allow_html=True)
   
   st.write("---")

   st.title("Distance Based")

   col_desc,col_space, col_stat_check = st.columns([2,1,2])

   with col_desc:
      st.subheader("Distance-Based Approach to detecting Covariare Drift")
      st.text("Here we are checking if there is any feature drift using distanced based approach:")
   with col_stat_check:
      if True: #test_for_drift(df_1, df_2):
            st.markdown('<p style="color: green; font-size:2rem;"><i class="fas fa-check-circle"></i> No significant difference</p>', unsafe_allow_html=True)
      else:
            st.markdown("""<p style="color: red; font-size:2rem; text-align:center;"><i class="fas fa-check-circle"></i> Failure</p>
                        """, unsafe_allow_html=True)
      with st.expander("See explanation"):
         st.write("""
                  The chart the code used to get this:
               """)
         st.text("Data Drift")
   
   st.write("---")

   st.title("Statistical Approach to Data Drift detection")

   col_desc,col_space, col_stat_check = st.columns([2,1,2])

   with col_desc:
      st.subheader("Statistical Tests to detect Covariare Drift")
      st.text("Here we are checking if there is any feature drift using distanced based approach:")
   with col_stat_check:
      #x, any_true = drift_detector(df_1,df_2)
      any_true = True
      if any_true:
            st.markdown('<p style="color: green; font-size:2rem;"><i class="fas fa-check-circle"></i> No significant difference</p>', unsafe_allow_html=True)
      else:
            st.markdown("""<p style="color: red; font-size:2rem; text-align:center;"><i class="fas fa-check-circle"></i> Failure</p>
                        <p style="text-align: center;">This gives us  </p>""", unsafe_allow_html=True)
   with st.expander("See explanation"):
      table_col = st.columns([1,2,1])
      # with table_col[1]:
      #    for k in x.keys():
      #       for i in x[k]:
      #          st.subheader("from '" + k + "' test:")
      #          draw_fig(df_1, df_2, i)
   
   st.write("---")

with tab2:
   
   st.title("Monitoring Model Performance:")
   st.caption("""Here we are tracking all the useful metrics to make sure our model is performing well. We are using XGBoost since it had the best performance out of all the models trained on the training data.
           """)
   
   col1, col2, col3, col4 = st.columns([1,1,1,1])
   col1.metric("Model Accuracy", str((round(accuracy,2) * 100)) + "%")
   col2.metric("Model Precision", str((round(precision,3)* 100)) + "%")
   col3.metric("Model Recall", str((round(recall,2)* 100)) + "%")
   col4.metric("F1 Score", str((round(f1, 2)* 100)) + "%")

   if accuracy > 0.75:
      st.markdown('<p style="color: green; font-size:2rem; text-align: center;"><i class="fas fa-check-circle"></i>Model Performing fine with no need for retraining</p>', unsafe_allow_html=True)
   else:
      st.markdown("""<p style="color: red; font-size:2rem; text-align:center;"><i class="fas fa-check-circle"></i> Warning!</p>
                        <p style="text-align: center;">Model Needs retraining since accuracy fell below 0.8 threshold!</p>""", unsafe_allow_html=True)
   # assuming you have a list of accuracy values for each time point
   accuracy_dummy = np.random.rand(10)
   f1_dummy = np.random.rand(10)

   # assuming you have a list of time points
   time_points = [1, 2, 3, 4, 5, 6, 7,8,9,10]
   
   # Create a figure with two subplots
   fig, ac = plt.subplots(1, 1)

   # Plot accuracy on first subplot
   ac.plot(time_points, accuracy_dummy, label='Accuracy', color='firebrick')
   ac.plot(time_points, f1_dummy, label='F1 Score', color = 'steelblue')
   ac.legend()

   ac.set_xlabel('Date')
   ac.set_ylabel('Model Score')
   ac.set_title('Model Performance')
   ac.legend()

   # Show the plot in Streamlit
   st.pyplot(fig,use_container_width=True)

   #with roc_col:
      # Plot ROC curve
   fig, (ax1, ax2) = plt.subplots(ncols=2)

   ax1.plot(fpr, tpr, label=f'AUC = {auc_roc:.3f}')
   ax1.plot([0, 1], [0, 1], 'k--')
   ax1.set_xlabel('False Positive Rate')
   ax1.set_ylabel('True Positive Rate')
   ax1.set_title('ROC Curve')
   ax1.legend()
   ax1.set_aspect('equal')
   plt.gca().set_position([0, 0, 1, 1])
   ax1.patch.set_alpha(0.0)
   
   # Plot confusion matrix
   sns.heatmap(cm, annot = True,fmt='d', ax=ax2)
   ax2.set_xlabel('Predicted')
   ax2.set_ylabel('True')
   ax2.set_title('Confusion Matrix')
   
   #.pyplot()
      
   st.pyplot(fig, use_container_width=True)

with tab3:
   st.header("Data Drift")
   st.markdown("""One of the major tasks for any data sceintist is to constantly monitor the performance of the ML models that have been deployed into production. The deployed model may start to perform poorly over time. This can happen for various reasons: changes in the data distribution, changes in the data generating process or changes in the model itself. Model performance degradation is a real problem which can lead to incorrect predictions and so need to be continuously monitored and retrained incase a loss of performance is detected.
   
   One of the main reasons for a loss in performance is due to `Data Drift`. Data changes over time and these changes cause the model which was trained on old data to be inconsistent and unable to produce similar results using the new data. Once such change could be changes in the input variables used to make the predictions. The distribution of the inpit variables could change or the relationship between the input variables and target variables could change. This is called `Covariate drift`.
   
   We will start by implementing a framework to detect such changes in the distribution of the input variables.""")

   st.markdown("""PSI is primarily used for detecting drift in categorical features, and it compares the distribution of the feature between a reference dataset and a new dataset to determine if there is a significant change. PSI calculates a score that reflects the amount of change between the two datasets and can be used as a threshold for detecting significant drift.
   
   In general, PSI is more suitable for detecting drift in categorical features, while the KS test is more suitable for detecting drift in numeric features. However, it's important to note that both methods have limitations and may not detect certain types of drift, and it's often necessary to use multiple methods in combination to comprehensively detect data drift.""")

   st.markdown("""the contingency tables of categorical features can be treated as multidimensional frequency distributions, and the Chi-square test can be used to compare the distance between these distributions.""")
