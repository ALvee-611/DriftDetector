# DriftDetector

One of the major tasks for any data sceintist is to constantly monitor the performance of the ML models that have been deployed into production. The deployed model may start to perform poorly over time. This can happen for various reasons: changes in the data distribution, changes in the data generating process or changes in the model itself. Model performance degradation is a real problem which can lead to incorrect predictions and so need to be continuously monitored and retrained incase a loss of performance is detected.

In this project a framework to detect model and data drift is built that will track the performance of the deployed model and notify if it detects any drift in data or model. Thie live app can be viewed here: CLICK ME


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

- `data`: This folder contains the source code of the project, which includes Python scripts, modules, and packages.
- `models`: This folder contains data files used by the project.
- `notebook`: This folder contains project documentation such as reports, user manuals, and design documents.
- `preprocess_pipeline`: This folder contains the unit tests for the project.
- `scripts`: This folder contains the unit tests for the project.

## File Descriptions

- `README.md`: This file contains a brief introduction and overview of the project.
- `requirements.txt`: This file contains the required packages and dependencies for running the project.
- `main.py`: This file is the entry point of the project and contains the main function to run the code.
- `module1.py`: This file contains the code related to the module1 of the project.
- `module2.py`: This file contains the code related to the module2 of the project.
- `data.csv`: This file contains the raw data used by the project.
- `report.pdf`: This file contains the project report and documentation.
- `test_module1.py`: This file contains the unit tests for module1.
- `test_module2.py`: This file contains the unit tests for module2.

## Conclusion

This is a basic project structure that can be used as a template for organizing projects of different sizes and complexity. It provides a clear separation of concerns, easy maintenance, and scalability.