import numpy as np
import pandas as pd
import streamlit as st
from helpers import *
import matplotlib.pyplot as plt
import seaborn as sns
import random

st.set_page_config(page_title="Model Monitoring",
                   layout="wide")

Xg_model = load_model('xgb_model.joblib')


# Import FontAwesome CSS
FA_CSS = f"""<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">"""
st.markdown(FA_CSS, unsafe_allow_html=True)

# read csv
main_data = 'batch_data.csv'

# read csv from a URL
@st.cache_data
def get_data(dataset_url) -> pd.DataFrame:
    return pd.read_csv(dataset_url)

## Define session states
if 'show_options' not in st.session_state:
   st.session_state['show_options'] = False
if 'features_chosen' not in st.session_state:
   st.session_state['features_chosen'] = []




#### THIS WILL BE OUR DEFAULT BASELINE DATA
main_df = get_data(main_data)

all_columns = main_df.columns.tolist()
all_columns.remove('Attrition')
numerical_col, cat_columns = get_num_cat_columns(main_df)

# Get all model metrics
accuracy, precision, recall, f1, auc_roc, cm, fpr, tpr, thresholds = calculate_model_KPI(Xg_model, main_df)

#################################################### DASHBOARD STARTS #########################################################
# dashboard title
st.title("Drift Detection Dashboard")

st.header('Drift Detection Dashboard')
st.text('This is the live overview of the Model Performance where data is being added in batch and the true Attrition label is not known.')
st.markdown("""One of the major tasks for any data sceintist is to constantly monitor the performance of the ML models that have been deployed into production. The deployed model may start to perform poorly over time. This can happen for various reasons: changes in the data distribution, changes in the data generating process or changes in the model itself. Model performance degradation is a real problem which can lead to incorrect predictions and so need to be continuously monitored and retrained incase a loss of performance is detected.

One of the main reasons for a loss in performance is due to `Data Drift`. Data changes over time and these changes cause the model which was trained on old data to be inconsistent and unable to produce similar results using the new data. Once such change could be changes in the input variables used to make the predictions. The distribution of the inpit variables could change or the relationship between the input variables and target variables could change. This is called `Covariate drift`.
   
We will start by implementing a framework to detect such changes in the distribution of the input variables.""")
st.write("---")

placeholder = st.empty()
placeholder_multi = st.empty()


st.header("Choose features you want to add variations to:")
st.text("Once you have chosen the features, this drift detection framework will detect the changes using different methods:")
st.markdown(" ")

st.subheader("Choose 3 features randomly")
if st.button("Random features"):
   random_col_names = random.sample(all_columns, k=3)
   st.write(random_col_names)
   st.session_state['features_chosen'] = random_col_names
   st.session_state['show_options'] = True

st.subheader("You have Chosen: " + str(st.session_state['features_chosen']))

# st.header("OR")

# st.subheader("Choose the features you want to add drift to (max 5):")
# options = st.multiselect('Choose from the droopdown menu',all_columns, max_selections = 5 )

# st.subheader("You have Chosen: " + str(options))
# st.session_state['show_options'] = True

# if options:
#    for i in options:
#       if i in numerical_col:
#          print (i)
#       else:
#          print('Num')

if st.session_state['show_options']:
   new_data = pd.DataFrame()
   new_data = main_df.copy()
   for i in st.session_state['features_chosen']:
      #st.write(main_df[i])
      # main_data[i]
      drift_typ = random.choice(['GRADUAL', 'SUDDEN'])
      if i in  numerical_col:
         #st.write(str(i))
         #st.write(main_df[i])
         new_data[i] = num_drift(main_df, i, drift_type=drift_typ)
      else:
         new_data[i] = cat_drift(main_df, i, drift_type=drift_typ)
   
   # create two sets of predictions
   predictions_1 = main_df['Attrition']
   predictions_2 = new_data['Attrition']
   new_df = new_data.drop(columns=['Attrition'], axis=1)

   # Metrics
   accuracy, precision, recall, f1, auc_roc, cm, fpr, tpr, thresholds = calculate_model_KPI(Xg_model, new_data)

   comparing_labels = compare_preds(main_df, new_data)

   # if comparing_labels:
   #    res = "statistically same"
   # else:
   #    res = "statistically different"

   #col1, col2, col3 = st.columns([1,3,2])
   #col2.metric("Current Batch True Labels VS Last Batch True Labels", res)

   col1, col2, col3, col4 = st.columns([1,1,1,1])
   col1.metric("Model Accuracy", str((round(accuracy,2) * 100)) + "%")
   col2.metric("Model Precision", str((round(precision,3)* 100)) + "%")
   col3.metric("Model Recall", str((round(recall,2)* 100)) + "%")
   col4.metric("F1 Score", str((round(f1, 2)* 100)) + "%")

   if accuracy > 0.75 and f1 > 0.75:
      st.markdown('<p style="color: green; font-size:2rem; text-align: center;"><i class="fas fa-check-circle"></i>Model Performing fine with no need for retraining</p>', unsafe_allow_html=True)
   else:
      st.markdown("""<p style="color: red; font-size:2rem; text-align:center;"><i class="fas fa-check-circle"></i> Warning!</p>
                        <p style="text-align: center;">Model Needs retraining since accuracy fell below 0.8 threshold!</p>""", unsafe_allow_html=True)

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
      if comparing_labels:
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
      if test_for_drift(main_df, new_data):
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
      x, any_true = drift_detector(main_df,new_data)
      if any_true:
            st.markdown('<p style="color: green; font-size:2rem;"><i class="fas fa-check-circle"></i> No significant difference</p>', unsafe_allow_html=True)
      else:
            st.markdown("""<p style="color: red; font-size:2rem; text-align:center;"><i class="fas fa-check-circle"></i> Failure</p>
                        <p style="text-align: center;">This gives us  </p>""", unsafe_allow_html=True)
   with st.expander("See explanation"):
      table_col = st.columns([1,2,1])
      with table_col[1]:
         for k in x.keys():
            for i in x[k]:
               st.subheader("from '" + k + "' test:")
               #draw_fig(main_df, new_data, i)
               st.header(i)
   
   st.write("---")


   st.subheader('Model Performance:')

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