# Home Credit Default Risk

This project aims to develop a machine learning model to predict the repayment behavior of loan applicants. The dataset used for this project is the "Home Credit Default Risk" dataset obtained from Kaggle. The dataset contains application, demographic, and historical credit behavior data for various loan applicants.

## Dataset

The "Home Credit Default Risk" dataset can be found on Kaggle at the following link:
[https://www.kaggle.com/yourusername/home-credit-default-risk](https://www.kaggle.com/yourusername/home-credit-default-risk)

## Project Overview

The goal of this project is to develop a machine learning model that can accurately predict whether a loan applicant is likely to default or prepay. The project follows a systematic methodology and involves the following steps:

1. **Data Gathering**: The dataset is explored to understand its structure and contents. Basic exploratory data analysis (EDA) is performed to identify any anomalies, missing data, or irregularities.

2. **Data Preprocessing**: The dataset is cleaned and preprocessed to handle missing values, outliers, and other data quality issues. Statistical analysis is conducted on numerical and categorical features, and visual exploration is performed to gain insights and identify patterns.

3. **Model Development**: Various machine learning models are trained and evaluated on the preprocessed dataset. Models such as Logistic Regression, Naive Bayes, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Support Vector Machines (SVM), AdaBoost, XGBoost, and LightGBM are considered. Hyperparameter tuning is performed using techniques like GridSearchCV to optimize model performance.

4. **Evaluation Metrics**: The models are evaluated based on metrics such as Accuracy, Precision, Recall, F1 Score, Area Under the ROC Curve (AUC-ROC), and Confusion Matrix. These metrics provide a comprehensive evaluation of the model's performance in predicting loan repayment behavior.

5. **Machine Learning Pipelines**: Machine learning pipelines are implemented to streamline the data preprocessing and modeling process. Pipelines include steps for handling missing values, scaling features, and training multiple models. Stratified K-Fold Cross Validation is used to obtain reliable model performance estimates.

## Requirements

The following software and libraries are required to run the code:

- Python (version 3.8)
- Docker Container
- Jupyter Notebook or JupyterLab
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- XGBoost
- LightGBM

Please refer to the documentation and source code for more detailed instructions on running and using the application.

