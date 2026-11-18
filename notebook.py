#!/usr/bin/env python
# coding: utf-8

# # Exam Score: A Regression Problem

# ## Data Preparation

# ### Load and inspect. Check the shape, data types, missing values and duplicates.

# In[63]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


# In[64]:


df = pd.read_csv(r"C:\Users\pc\Downloads\Exam_Score_Prediction.csv")


# In[65]:


df


# In[66]:


df.info()


# In[67]:


df.head().T


# In[68]:


# Checking some statistical information on Numerical fields. Trying to pinpoint some outliers.
df.describe()


# In[69]:


df.dtypes


# In[70]:


# Checking for any missing information.
df.isnull().sum()


# In[71]:


# checking duplicates
df.duplicated().sum()


# ## EDA

# In[72]:


sns.histplot(df.exam_score, bins=50)


# In[73]:


def barplotsColumns(autos,nLower = 2,nUpper = 100):
    nColumns = 3
    autos = autos[[col for col in autos if autos.nunique()[col] > nLower and autos.nunique()[col] < nUpper]] 
    _, nCol = autos.shape
    nGraphRow = (nCol + nColumns - 1) / nColumns
    fig = plt.figure(num = None, figsize = (7 * 6, int(4 * nGraphRow)), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(nCol):
        plt.subplot(int(nGraphRow), nColumns, i + 1)
        columnDf = autos.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('Counts')
        plt.title('{}'.format(list(autos)[i]))

    fig.savefig('high_view.jpg')

    plt.show()


# In[74]:


# general view of feature distributions
barplotsColumns(df, nLower = 1, nUpper = 1000)


# In[75]:


plt.figure()
sns.boxplot(data=df, x='internet_access', y='age', palette='Set2')
plt.title('Age Distribution by Internet Access')
plt.show()


# In[76]:


plt.figure()
sns.lineplot(data=df, x='age', y='class_attendance', marker='o', color='pink')
plt.title('Average Class Attendance across Ages')
plt.show()


# In[77]:


plt.figure()
sns.countplot(data=df, x='study_method', hue='gender', palette='Pastel1')
plt.title('Study Method Preference by Gender')
plt.show()


# In[78]:


plt.figure()
course_diff = pd.crosstab(df['course'], df['exam_difficulty'], normalize='index')
sns.heatmap(course_diff, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Proportion of Exam Difficulty per Course')
plt.show()


# In[79]:


# 1. Score distribution
print(df["exam_score"].describe())
print("Skew:", df["exam_score"].skew())

df["exam_score"].hist(bins=30)
plt.xlabel("Exam score"); plt.ylabel("Number of students")
plt.title("Exam score distribution")
plt.show()

# 2. Correlation of each numeric feature with the score
num = df.select_dtypes("number").drop(columns="student_id")   # drop the ID
corr = num.corr()["exam_score"].drop("exam_score").sort_values(ascending=False)
print(corr)

corr.plot.barh(title="Correlation with exam score")
plt.show()

# 3. Scatter plots to check the shape of each relationship
for col in corr.index:
    plt.scatter(df[col], df["exam_score"], alpha=0.1, s=5)
    plt.xlabel(col); plt.ylabel("exam_score"); plt.show()


# In[18]:


num = df.select_dtypes("number").drop(columns="student_id")   # numeric only, no ID
correlation_matrix = num.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="PiYG", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()


# ## Model Evaluation

# In[19]:


df_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full, test_size=0.25, random_state=42)


# In[20]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[21]:


y_train = df_train['exam_score']
y_val = df_val['exam_score']
y_test = df_test['exam_score']


# In[22]:


del df_train['exam_score']
del df_val['exam_score']
del df_test['exam_score']


# In[23]:


df_train.head()


# ### Features importance analysis

# In[24]:


from sklearn.metrics import mutual_info_score


# In[26]:


import pandas as pd
from sklearn.metrics import mutual_info_score

# target as a Series aligned with df_train's rows
y = pd.Series(y_train, index=df_train.index)

# the target is continuous, so bin it before using mutual information
y_binned = pd.qcut(y, q=5, duplicates='drop')

categorical = df_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

def calculate_mi(series):
    return mutual_info_score(series, y_binned)

df_mi = df_train[categorical].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')
display(df_mi)


# In[31]:


numerical = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
target = pd.Series(y_train, index=df_train.index)   # y_train must match df_train's rows

a = df_train[numerical].corrwith(target).to_frame('correlation')
a['abs_corr'] = a['correlation'].abs()
a.sort_values('abs_corr', ascending=False)


# #this shows that study hours IS important!

# In[32]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text
import matplotlib.pyplot as plt
import xgboost as xgb


# In[33]:


from sklearn.feature_extraction import DictVectorizer

# 1. drop the ID: it's unique per row, so it's useless as a feature
for d in (df_train, df_val, df_test):
    d.drop(columns=['student_id'], inplace=True, errors='ignore')

# 2. rows -> dictionaries
train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

# 3. dictionaries -> numeric matrix
dv = DictVectorizer(sparse=False)          # dense avoids the int64 sparse error
X_train = dv.fit_transform(train_dicts)    # learn the columns from TRAIN only
X_val = dv.transform(val_dicts)            # reuse them for validation
X_test = dv.transform(test_dicts)          # and for test

print(X_train.shape, X_val.shape, X_test.shape)
print(dv.get_feature_names_out()[:10])


# In[80]:


from sklearn.feature_extraction import DictVectorizer

# Convert DataFrames into dictionaries
train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

# Convert dictionaries into numerical matrices
dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)

print(X_train.shape, X_val.shape, X_test.shape)


# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

# target as a Series aligned with df_train's rows
y = pd.Series(y_train, index=df_train.index)

numerical = df_train.select_dtypes(include='number').columns.tolist()
categorical = df_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
print('numerical  :', numerical)
print('categorical:', categorical)

# 1. numeric features: correlation with the target (strongest first)
corr = df_train[numerical].corrwith(y).to_frame('correlation')
corr['abs_corr'] = corr['correlation'].abs()
corr = corr.sort_values('abs_corr', ascending=False)
display(corr)

# 2. categorical features: mutual information with a binned target
y_binned = pd.qcut(y, q=5, duplicates='drop')

def calculate_mi(series):
    return mutual_info_score(series, y_binned)

df_mi = df_train[categorical].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')
display(df_mi)

# 3. quick visual comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
corr['abs_corr'].sort_values().plot.barh(ax=axes[0], title='|Correlation| (numeric)')
df_mi['MI'].sort_values().plot.barh(ax=axes[1], title='Mutual information (categorical)')
plt.tight_layout()
plt.show()


# In[81]:


import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def evaluate_model(model, model_name, X_train, y_train, X_val, y_val):

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_val)

    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)

    return {
        'Model': model_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    }


# In[84]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

results = []

for name, model in models.items():

    result = evaluate_model(
        model,
        name,
        X_train,
        y_train,
        X_val,
        y_val
    )

    results.append(result)

# Display comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by='RMSE')

print(results_df)


# In[83]:


from xgboost import XGBRegressor

models['XGBoost'] = XGBRegressor(
    random_state=42,
    n_estimators=100
)


# ###  Hyperparameter Tuning

# In[85]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },

    'Ridge': {
        'model': Ridge(),
        'params': {
            'alpha': [0.01, 0.1, 1, 10, 100]
        }
    },

    'Lasso': {
        'model': Lasso(max_iter=10000),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1, 10]
        }
    },

    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },

    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },

    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [2, 3, 5]
        }
    }
}


# In[86]:


best_model = RandomForestRegressor(random_state=42)

best_model.fit(X_train, y_train)

y_test_pred = best_model.predict(X_test)

print("Test MAE:", mean_absolute_error(y_test, y_test_pred))

print("Test RMSE:", np.sqrt(
    mean_squared_error(y_test, y_test_pred)
))

print("Test R2:", r2_score(y_test, y_test_pred))


# In[87]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_test_pred, alpha=0.5)

plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Actual vs Predicted Exam Scores")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.show()


# In[ ]:




