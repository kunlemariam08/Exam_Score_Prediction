# Student Exam Score Prediction Using Machine Learning

## Project Overview

This project focuses on predicting students' examination scores using machine learning regression algorithms. It explores how academic behaviours, study habits, demographic characteristics, learning environments, and other educational factors relate to students' academic performance.

The project applies an end-to-end machine learning workflow, including data preparation, exploratory data analysis, feature engineering, model development, hyperparameter tuning, and model evaluation.

Several regression algorithms were trained and compared to identify the model that provides the most accurate predictions of students' examination scores.

The primary objective is to develop a predictive model that can help educational institutions better understand factors associated with academic performance and support data-driven educational decision-making.

## Problem Statement

Academic performance is influenced by several factors, including students' study habits, attendance, sleep patterns, learning environments, and access to educational resources.

Educational institutions often need reliable ways to understand these factors and estimate students' academic performance.

This project addresses the problem by developing machine learning models capable of predicting examination scores based on available student-related information.

By analysing these variables, the project aims to demonstrate how predictive analytics can support academic performance monitoring and educational planning.

## Project Objectives

The major objectives of this project are to:

1. Explore and understand the relationship between student characteristics and examination performance.
2. Perform data cleaning and preprocessing to prepare the dataset for machine learning.
3. Identify relevant features that may influence students' examination scores.
4. Develop multiple regression models to predict examination scores.
5. Apply hyperparameter tuning to improve model performance.
6. Compare different machine learning algorithms using appropriate regression evaluation metrics.
7. Identify the best-performing model based on validation performance.
8. Evaluate the selected model on unseen test data.

## Dataset Description

The dataset contains **20,000 student records and 13 columns**, including demographic information, academic behaviours, learning conditions, and examination scores.

### Dataset Features

| Feature          | Description                                | Data Type   |
| ---------------- | ------------------------------------------ | ----------- |
| student_id       | Unique identifier assigned to each student | Integer     |
| age              | Age of the student                         | Integer     |
| gender           | Gender of the student                      | Categorical |
| course           | Course or programme of study               | Categorical |
| study_hours      | Number of hours spent studying             | Float       |
| class_attendance | Student's class attendance                 | Float       |
| internet_access  | Availability of internet access            | Categorical |
| sleep_hours      | Average number of sleeping hours           | Float       |
| sleep_quality    | Student's sleep quality                    | Categorical |
| study_method     | Study method adopted by the student        | Categorical |
| facility_rating  | Rating of available learning facilities    | Categorical |
| exam_difficulty  | Perceived examination difficulty           | Categorical |
| exam_score       | Student's examination score                | Float       |

**Target Variable:** `exam_score`

The target variable is numerical, making this a supervised machine learning regression problem.

## Technologies and Libraries Used

The project was implemented using Python and the following libraries:

- **Python:** Main programming language.
- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical computations.
- **Matplotlib:** Data visualization.
- **Seaborn:** Statistical data visualization.
- **Scikit-learn:** Data preprocessing, model development, hyperparameter tuning, and evaluation.
- **XGBoost:** Additional gradient boosting regression algorithm.
- **Jupyter Notebook:** Development and experimentation environment.

## Project Workflow

### 1. Data Loading and Inspection

The dataset was loaded into a Pandas DataFrame and examined to understand its structure.

The inspection involved:

- Checking the number of observations and features.
- Examining data types.
- Identifying missing values.
- Understanding categorical and numerical variables.
- Checking the distribution of the target variable.

### 2. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the dataset and identify patterns that could influence examination scores.

The analysis focused on:

- Distribution of examination scores.
- Relationship between study hours and examination scores.
- Relationship between class attendance and academic performance.
- Influence of sleep hours and sleep quality.
- Differences in examination scores across study methods.
- Relationship between examination difficulty and student performance.

Visualizations were used to identify trends, patterns, and potential relationships between the independent variables and the target variable.

### 3. Data Preprocessing and Feature Engineering

The dataset was prepared for machine learning by separating the target variable from the independent features.

The target variable was defined as:

`exam_score`

The remaining columns were used as model input features.

Categorical and numerical variables were transformed into numerical representations using Scikit-learn's `DictVectorizer`.

The transformation was performed using:

- `fit_transform()` on the training data.
- `transform()` on the validation data.
- `transform()` on the test data.

This ensured that the feature representation learned from the training data was consistently applied to the validation and test datasets.

### 4. Train, Validation and Test Split

The dataset was divided into three subsets:

| Dataset    | Percentage | Purpose                                        |
| ---------- | ---------- | ---------------------------------------------- |
| Training   | 60%        | Training machine learning models               |
| Validation | 20%        | Comparing models and selecting hyperparameters |
| Testing    | 20%        | Final evaluation on unseen data                |

A random state of 42 was used to ensure reproducibility.

### 5. Model Development

Several supervised machine learning regression algorithms were implemented and compared.

The models included:

**Linear Regression**

Used as the baseline model to establish the relationship between student-related features and examination scores.

**Ridge Regression**

Applied L2 regularization to reduce the impact of large model coefficients and help control overfitting.

**Lasso Regression**

Applied L1 regularization, which can reduce less important feature coefficients to zero.

**Decision Tree Regressor**

Used to capture nonlinear relationships and interactions between student characteristics and examination performance.

**Random Forest Regressor**

Combined multiple decision trees to improve prediction stability and capture complex patterns in the data.

**Gradient Boosting Regressor**

Built an ensemble of decision trees sequentially, with each new tree attempting to correct errors made by previous trees.

**XGBoost Regressor**

Considered as an additional boosting algorithm for modelling complex relationships and improving predictive performance.

### 6. Hyperparameter Tuning

Hyperparameter tuning was introduced to identify suitable model configurations and improve predictive performance.

Scikit-learn's `GridSearchCV` was used to test different hyperparameter combinations.

Five-fold cross-validation (`cv=5`) was applied to evaluate model configurations.

The scoring metric used was:

`neg_root_mean_squared_error`

This allowed the tuning process to identify parameter combinations that minimized the Root Mean Squared Error (RMSE).

Some of the hyperparameters explored included:

| Model             | Hyperparameters                                                              |
| ----------------- | ---------------------------------------------------------------------------- |
| Ridge             | Alpha                                                                        |
| Lasso             | Alpha                                                                        |
| Decision Tree     | Max depth, minimum samples split, minimum samples leaf                       |
| Random Forest     | Number of estimators, max depth, minimum samples split, minimum samples leaf |
| Gradient Boosting | Number of estimators, learning rate, max depth                               |

The best estimator from each GridSearchCV process was retained for subsequent evaluation.

### 7. Model Evaluation

The models were evaluated using four major regression metrics.

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual examination scores and predicted scores. Lower values indicate better performance.

**Mean Squared Error (MSE)**

Measures the average squared prediction error. Lower values indicate better performance.

**Root Mean Squared Error (RMSE)**

Measures prediction error in the same unit as the examination score. Lower values indicate better predictive accuracy.

**R-squared (R²)**

Measures the proportion of variation in examination scores explained by the model. Higher values indicate better explanatory performance.

## Model Performance and Results

The following table presents the validation results obtained from the model comparison.

| Model             | MAE     | MSE      | RMSE       | R²         |
| ----------------- | ------- | -------- | ---------- | ---------- |
| Linear Regression | 7.8224  | 94.5062  | **9.7214** | **0.7350** |
| Ridge             | 7.8225  | 94.5075  | 9.7215     | 0.7350     |
| Gradient Boosting | 7.9186  | 96.3948  | 9.8181     | 0.7297     |
| Random Forest     | 8.2494  | 104.5706 | 10.2260    | 0.7068     |
| Lasso             | 8.7424  | 117.6750 | 10.8478    | 0.6701     |
| Decision Tree     | 11.7408 | 216.2633 | 14.7059    | 0.3937     |

### Best-Performing Model

Based on the validation results, **Linear Regression achieved the best overall performance** among the evaluated models.

Its performance metrics were:

- MAE: 7.8224
- MSE: 94.5062
- RMSE: 9.7214
- R² Score: 0.7350

The model achieved an R² score of approximately 73.5%, indicating that it explains around 73.5% of the variation in examination scores within the validation dataset.

Ridge Regression produced almost identical results, suggesting that regularization did not substantially change the predictive performance in this experiment.

Gradient Boosting also performed competitively, while Decision Tree produced the weakest results among the reported models.

### Key Findings

The model comparison revealed the following:

1. Linear Regression provided the lowest validation RMSE and MAE.
2. Ridge Regression performed almost identically to Linear Regression.
3. Gradient Boosting demonstrated competitive predictive performance.
4. Random Forest performed reasonably well but did not outperform the linear models.
5. Decision Tree had the highest prediction error, suggesting that a single decision tree may not adequately capture the underlying patterns in this dataset.
6. The results suggest that the relationships between the selected student features and examination scores can be reasonably represented using a linear model.

## Conclusion

This project demonstrates the application of supervised machine learning techniques to predict students' examination scores using demographic, academic, and behavioural features.

Several regression algorithms were developed and evaluated, with Linear Regression achieving the best validation performance.

The results indicate that student-related variables can provide useful information for predicting examination outcomes.

Although the selected model achieved an R² score of approximately 0.735, further improvements could be explored through feature selection, additional feature engineering, alternative modelling techniques, and more extensive hyperparameter optimization.

The project provides a foundation for applying predictive analytics to educational data and supporting evidence-based academic decision-making.

## Future Improvements

Potential areas for further development include:

- Performing more detailed feature importance analysis.
- Investigating the effects of individual study behaviours on examination performance.
- Exploring additional machine learning algorithms.
- Applying advanced hyperparameter optimization techniques such as RandomizedSearchCV and Bayesian optimization.
- Comparing model performance across different student groups.
- Developing an interactive web application for examination score prediction.
- Deploying the selected model through a REST API.
- Integrating the prediction system into a Learning Management System (LMS).

## Project Structure

A suggested repository structure is:

```
student-exam-score-prediction/
│
├── data/
│   └── student_performance.csv
│
├── notebooks/
│   └── student_exam_score_prediction.ipynb
│
├── models/
│   └── best_model.pkl
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Usage

### 1. Clone the repository

```
git clone https://github.com/your-username/student-exam-score-prediction.git
```

### 2. Navigate into the project directory

```
cd student-exam-score-prediction
```

### 3. Install the dependencies

```
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```
jupyter notebook
```

Open the notebook in the `notebooks` directory and execute the cells sequentially.

## Requirements

The major dependencies include:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

## Disclaimer

This project is an educational and experimental machine learning project. Its predictions should not be used as the sole basis for academic decisions, student assessment, or institutional policies.

Model performance depends on the quality, representativeness, and reliability of the underlying dataset.

## Author

**Mariam Osunkunle**

Mathematics Graduate | Data Analytics | Machine Learning

Interested in applying mathematics, statistical analysis, and machine learning to real-world problems, particularly in education and predictive analytics.

---

**Project Focus:** Machine Learning | Regression | Educational Data Analytics | Predictive Modelling | Hyperparameter Optimization

**Tools:** Python | Pandas | NumPy | Scikit-learn | XGBoost | Matplotlib | Seaborn

## **Important improvements before publishing on GitHub**

There are four things I recommend checking before you publish this README.

1. **Remove&#x20;**`student_id`**&#x20;from your model features.** It is an identifier rather than a meaningful academic predictor, unless you have a specific reason to retain it.
2. **Confirm whether the reported results are from the tuned models or baseline models.** The README currently calls them validation results without claiming they are final test results.
3. **Add your actual EDA graphs.** Include examination-score distribution, study-hours versus exam-score, and attendance versus exam-score plots.
4. **Add your final test results** after evaluating the selected model on `X_test`. Do not present validation metrics as test performance.

## **Suggested GitHub repository details**

Repository name

**student-exam-score-prediction**

&#x20;Copy

Repository description

A machine learning regression project that predicts students' examination scores using academic, demographic, and behavioural features, with model comparison and hyperparameter tuning.

&#x20;Copy description

GitHub topics

machine-learning

python

regression

data-science

educational-data

scikit-learn

predictive-analytics

hyperparameter-tuning



## Final Model Evaluation

After comparing the validation performance of the candidate regression models, Linear Regression was selected as the final model because it achieved the lowest validation RMSE.

The selected model was subsequently evaluated on the independent test dataset to assess its ability to generalise to unseen student records.

### Final Test Results

| Evaluation Metric | Test Score |
|---|---:|
| Mean Absolute Error (MAE) | 8.3251 |
| Root Mean Squared Error (RMSE) | 10.3187 |
| R-squared (R²) | 0.7023 |

### Validation vs Test Performance

| Metric | Validation | Test |
|---|---:|---:|
| MAE | 7.8224 | 8.3251 |
| RMSE | 9.7214 | 10.3187 |
| R² Score | 0.7350 | 0.7023 |

### Interpretation of Results

The final Linear Regression model achieved an R² score of 0.7023 on the test dataset, indicating that approximately 70.23% of the variation in students' examination scores is explained by the features included in the model.

The test MAE of 8.3251 indicates that the model's predictions differ from actual examination scores by approximately 8.33 points on average in absolute terms.

The test RMSE of 10.3187 indicates the magnitude of prediction errors, with larger errors receiving greater weight.

The slight reduction in performance between the validation and test datasets suggests that the model generalises reasonably well to unseen observations, although further investigation and improvement may be beneficial.

### Final Conclusion

This project demonstrates how machine learning and predictive analytics can be applied to educational data to estimate students' examination performance.

Multiple regression algorithms were developed, compared, and evaluated using appropriate regression metrics. Hyperparameter tuning and cross-validation were also incorporated into the model development process.

Linear Regression emerged as the best-performing model based on validation RMSE and achieved an R² score of 0.7023 on the independent test dataset.

These findings demonstrate the potential of machine learning in identifying patterns associated with academic performance and supporting data-informed educational planning.

Further improvements could include feature engineering, feature selection, residual analysis, model optimization, and deployment through a REST API.
