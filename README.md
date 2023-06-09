﻿# DriftDetector

One of the major tasks for any data sceintist is to constantly monitor the performance of the ML models that have been deployed into production. The deployed model may start to perform poorly over time. This can happen for various reasons: changes in the data distribution, changes in the data generating process or changes in the model itself. Model performance degradation is a real problem which can lead to incorrect predictions and so need to be continuously monitored and retrained incase a loss of performance is detected.

In this project a framework to detect model and data drift is built that will track the performance of the deployed model and notify if it detects any drift in data or model. Thie live app can be viewed here: [CLICK ME](https://alvee-611-driftdetector-framework.streamlit.app/)

This project is organized in a modular structure with the code written in Python and the repository is structured as follows:

```
DRIFT DETECTOR
│   README.md
│   requirements.txt
│
└───data
│   │   model_metric.csv
│   │   batch_data.csv
|   
└───models
│   │   logistic_model.joblib
│   │   svm_model.joblib
│   │   xgb_model.joblib
│   
└───notebook
│   │   build_framework.ipynb
│   │   data_generating.ipynb
│   
└───preprocess_pipeline
│   │   preprocess_pipeline.joblib
│   
└───scripts
│   │   app.py
│   │   helpers.py

```

## Project Structure

- `data`: This folder contains the baseline data used by the project and also the model KPIs for all the models trained.
- `models`: This folder contains all the models trained on the data.
- `notebook`: This folder contains project documentation and information on how the functions were implemented.
- `preprocess_pipeline`: This folder contains the data preprocessing pipeline.
- `scripts`: This folder contains all the python scripts for the project.

## File Descriptions

- `README.md`: This file contains a brief introduction and overview of the project.
- `requirements.txt`: This file contains the required packages and dependencies for running the project.
- `app.py`: This file is the entry point for the streamlit project and contains the main functions to run the code.
- `helpers.py`: This file contains helper functions used by the main script.
- `preprocess_pipeline.joblib`: This is the data preprocessing pipeline trained for this project.
- `batch_data.csv`: This file contains the baseline raw data used by the project.
- `model_metric.pdf`: This contains the model KPIs for all the models trained on the batch_data training data.
- `logistic_model.joblib`: This is the logistic model trained on the batch_data dataset.
- `svm_model.joblib`: This is the svm model trained on the batch_data dataset.
- `xgb_model.joblib`: This is the XGBoost model trained on the batch_data dataset and is the model picked for this project.
- `test_module2.py`: This file contains the unit tests for module2.

## Conclusion

This contains a basic framework for data and model drift detection. Detecting data and model drift is a complex task and varies based on context so eve though this framework is a good starting point, more complex frameworks are required to deal with more complex scenarios.
