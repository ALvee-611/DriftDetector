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
   loc = os.getcwd()
   #print(os.listdir(os.getcwdb()))
   path = os.path.join(loc,'data', main_data)
   # path = os.path.join('..\\data',main_data) ### Chnage to top one
   return pd.read_csv(path)

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

#numerical_col.remove('Attrition')

# Get all model metrics
accuracy, precision, recall, f1, auc_roc, cm, fpr, tpr, thresholds = calculate_model_KPI(Xg_model, main_df)

#################################################### DASHBOARD STARTS #########################################################
# dashboard title
st.title("Drift Detection Dashboard")

st.subheader('This is the live overview of the Model Performance where data is being added in batch and the true Attrition label is not known.')
st.markdown("""This allows the users visualize and monitor changes in the data distributions and ML model performance changes. It randomly introduces sudden or gradual drift in 3 features which then get detected by the underlying data detection framework working in the background.""")
st.write("---")


# st.header("Choose features you want to add variations to:")
# st.text("Once you have chosen the features, this drift detection framework will detect the changes using different methods:")
# st.markdown(" ")

st.subheader("Choose 3 features randomly")
st.text("Once you have chosen the features, this drift detection framework will detect the changes using different methods:")
if st.button("Random features"):
   random_col_names = random.sample(all_columns, k=3)
   # st.write(random_col_names)
   st.session_state['features_chosen'] = random_col_names
   st.session_state['show_options'] = True
feature_statement = ""
for i in st.session_state['features_chosen']:
   feature_statement = feature_statement + ' ' + str(i) 
st.subheader("You have Chosen: " + feature_statement)

st.write("---")

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
   new_data['Attrition'] = new_data.apply(create_label, axis=1)
   # create two sets of predictions
   predictions_1 = main_df['Attrition']
   predictions_2 = new_data['Attrition']
   new_df = new_data.drop(columns=['Attrition'], axis=1)

   # Metrics
   accuracy, precision, recall, f1, auc_roc, cm, fpr, tpr, thresholds = calculate_model_KPI(Xg_model, new_data)

   comparing_labels = compare_preds(main_df, new_data)

   
   st.subheader('Model KPI:')
   col1, col2, col3, col4 = st.columns([1,1,1,1])
   col1.metric("Model Accuracy", str((round(accuracy,2) * 100)) + "%")
   col2.metric("Model Precision", str((round(precision,3)* 100)) + "%")
   col3.metric("Model Recall", str((round(recall,2)* 100)) + "%")
   col4.metric("F1 Score", str((round(f1, 2)* 100)) + "%")

   st.markdown("  ")

   if accuracy >= 0.55 and f1 >= 0.50:
      st.markdown('<p style="color: green; font-size:2rem; text-align: center;"><i class="fas fa-check-circle"></i> Model Performing fine with no need for retraining</p>', unsafe_allow_html=True)
   elif accuracy < 0.55 and f1 >= 0.50:
      st.markdown('<p style="color: orange; font-size:2rem; text-align: center;"><i class="fas fa-check-circle"></i> Model accuracy has dropped below the set threshold but f1 still above threshold. Model retraining might be required!</p>', unsafe_allow_html=True)
   elif accuracy >= 0.55 and f1 < 0.50:
      st.markdown('<p style="color: orange; font-size:2rem; text-align: center;"><i class="fas fa-check-circle"></i> Model f1 score has dropped below the set threshold but accuracy still above threshold. Model retraining might be required!</p>', unsafe_allow_html=True)
   else:
      st.markdown("""<p style="color: red; font-size:2rem; text-align:center;"><i class="fas fa-check-circle"></i> Warning!</p>
                        <p style="text-align: center;">Model Needs retraining since accuracy and f1 score both fell below set threshold!</p>""", unsafe_allow_html=True)
      st.subheader('Model Performance:')

   with st.expander("Model Performance Visualizations"):
      tab1, tab2 = st.tabs(["ROC", "Confusion Matrix"])
      with tab1:
         st.subheader("ROC Curve to evaluate the trade-off between the true positive rate (TPR) and the false positive rate (FPR) for different threshold values.")
         #fig, (ax1, ax2) = plt.subplots(ncols=2)
         fig,ax1 = plt.subplots(nrows = 1, ncols = 1)
         ax1.plot(fpr, tpr, label=f'AUC = {auc_roc:.3f}')
         ax1.plot([0, 1], [0, 1], 'k--')
         ax1.set_xlabel('False Positive Rate')
         ax1.set_ylabel('True Positive Rate')
         ax1.set_title('ROC Curve')
         ax1.legend()
         ax1.set_aspect('equal')
         plt.gca().set_position([0, 0, 1, 1])
         ax1.patch.set_alpha(0.0)
         st.pyplot(fig, use_container_width=True)

      with tab2:
         # Confusion matrix
         st.subheader("Confusion Matrix to evaluate the performance of a classification model by comparing actual and predicted values for a given dataset.")
         fig,ax2 = plt.subplots(nrows = 1, ncols = 1)
         sns.heatmap(cm, annot = True,fmt='d', ax=ax2)
         ax2.set_xlabel('Predicted')
         ax2.set_ylabel('True')
         ax2.set_title('Confusion Matrix')
      
         st.pyplot(fig, use_container_width=True)
   
   st.write("---")

   col_pred, col_stat = st.columns([1,1])
   
   with col_pred:
      st.subheader("Is there a difference in the distribution of the the target label in the old batch VS new batch (with drift introduced): ")
      st.markdown("Here we are using the Chi-Sq test to check if there is any difference in the distributions of the target label")
        
   with col_stat:
      if comparing_labels:
            st.markdown('<p style="color: green; font-size:1.5rem; text-align: center;"><i class="fas fa-check-circle"></i> No significant difference between the two sets of predictions!</p>', unsafe_allow_html=True)
      else:
            st.markdown("""<p style="color: red; font-size:1.5rem; text-align:center;"><i class="fas fa-check-circle"></i> Failure</p>
                        <p style="text-align: center;">Possible Label Drift!!</p>""", unsafe_allow_html=True)
   
   st.write("---")

   st.title("Distance Based")

   col_desc,col_space, col_stat_check = st.columns([2,1,2])

   with col_desc:
      st.subheader("Distance-Based Approach to detecting Covariare Drift")
      st.markdown("""The Distance-Based Approach to detecting Covariate Drift is a technique used to detect if there are changes or differences between two datasets that are supposed to be similar.
It works by calculating the distance between the two datasets based on the values of the features they share. If the distance between the datasets is larger than a certain threshold, then it is likely that the two datasets are different, and there may be covariate drift.""")
   with col_stat_check:
      if test_for_drift(main_df, new_data):
            st.markdown('<p style="color: green; font-size:2rem;"><i class="fas fa-check-circle"></i> No significant difference</p>', unsafe_allow_html=True)
      else:
            st.markdown("""<p style="color: red; font-size:2rem; text-align:center;"><i class="fas fa-check-circle"></i> Failure</p>
                        """, unsafe_allow_html=True)
   
   st.write("---")

   st.title("Statistical Approach to Data Drift detection")

   col_desc,col_space, col_stat_check = st.columns([2,1,2])

   with col_desc:
      st.subheader("Statistical Tests to detect Covariare Drift")
      st.markdown("Here we compare the distribution of features between the two batches of data using the Kolmogorov-Smirnov test and Population Stability Index (PSI)")
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
         my_list = []
         for k in x.keys():
            for i in x[k]:
               if i not in my_list:
                  my_list.append(i)
         fig, axes = plt.subplots(nrows=len(my_list), ncols=1)
         if 'Attrition' in my_list:
             my_list.remove('Attrition')
         for feature in my_list:
            old_dist = main_df[feature]
            new_dist = new_df[feature]
            draw_figures(feature, old_dist, new_dist, numerical_col)
            
   st.write("---")