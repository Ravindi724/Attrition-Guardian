#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pandas')
import pandas as pd
import numpy as np


# In[ ]:


data =pd.read_csv("/content/KDU DataSet - Sheet1.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data


# In[ ]:


data.columns


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# In[ ]:


# Assuming your dataset is stored in a DataFrame named 'data'
employee_ids = [
    "F9FADDE1-995A-4FC6-A388-19C2149A71DB",
    "25083D48-1EB1-4541-8FA6-C3993EC700FF",
    "BB279AE7-7B24-454C-9611-9F63E3363522",
    "E6B4B54E-E80F-4249-8E2E-3052DD1114AA",
    "22028EEA-B141-4C3B-A46D-60071CA0AF34",
    "711E1F3D-E507-42E3-B794-818DD208935F"
]

# Remove rows based on employee IDs
data = data[~data['EmployeeId'].isin(employee_ids)]

# Displaying the updated dataset
print(data)


# In[ ]:


data.shape


# In[ ]:


data.describe()


# In[ ]:


data = data.rename(columns={'Deleted': 'Resign'})


# In[ ]:


data['Resign'].describe() #y variable


# In[ ]:


data['Age'].describe()


# In[ ]:


data = data.drop(columns=['GenderName'
,'Dateofbirth'
,'Level1Code' #0
,'Level4Code' #72
,'UnionMember' #0
,'UnionStatus' #0
,'Salary' #0
,'Overtime' #0
,'TravelMethodName'
,'AvgFinalScore' #132
,'PromotionTypeName' #62

,'maxeffectdate' #177
,'deleted_resign'  #0
,'GradeCode' #0
,'ResignationTypeName' #177
,'PromotionDate' #62
,'OldDesignationCode' #62
,'OldCategoryCode' #62
,'OldGradeCode' #62
,'latestpromotionday' #62
,'avgyearlyhrsholidyhrsworked' #137
,'avgyearlynoOfHolidaysworked' #137
,'avgyearlywkndhrsworked' #156
,'avgnofwkndsworked'  #156
,'AvgShrtlveHrs' #0
,'expectdwrkhrsR'  #123
,'expectdwrkhrsEX' #123
,'sumofworkhrsRe' #123
,'sumofworkhrsEx' #129
,'sumwrhrsR' #43
,'sumwrhrsE' #29
,'noofweekendswrkd6E' #29
,'noofweekendswrkd6R' #43
,'Noofweekendsworked6'  #71
,'sumwrholihre'  #28
,'sumwrholihrex' #54
,'holidywrkpercentage6' #78
,'noofholidaysworked6' #78
,'sumOverTimeRate6monthsE' #123
,'sumOverTimeRate6monthsR' #123
,'overtimewrknpercntage'  #218
,'exs' #0
,'sre' #0
,'sumofshrtlvehrs6'
,'ResignationTypeId'])


# In[ ]:


data.columns


# In[ ]:


data.describe(include=['object'])


# In[ ]:


from google.colab import data_table
data_table.enable_dataframe_formatter()
from google.colab import data_table
data_table.disable_dataframe_formatter()


# In[ ]:


# Define a function to convert numbers to 0 if they are not 0, 1, or 2
def convert_to_zero(num):
    if num not in [0, 1, 2]:
        return 0
    return num

# Apply the conversion function to the 'travelmodecode' column
data['TravelModeCode'] = data['TravelModeCode'].apply(convert_to_zero)


# In[ ]:


mode_level3code = data['Level3Code'].mode()[0]

# Replace null values with the mode
data['Level3Code'].fillna(mode_level3code, inplace=True)


# In[ ]:


# Replace negative values with 0 in the 'tenure' column
data.loc[data['tenure'] < 0, 'tenure'] = 0


# In[ ]:


# Calculate the mean of the 'Average_LeavesPerMonth' column
mean_leaves = data['Average_LeavesPerMonth'].mean()

# Replace null values with the mean
data['Average_LeavesPerMonth'].fillna(mean_leaves, inplace=True)


# In[ ]:


# Calculate the mode of the 'ReportingPersonCode' column
mode_reporting_person_code = data['ReportingPersonCode'].mode()[0]

# Replace 0 values with the mode
data.loc[data['ReportingPersonCode'] == 0, 'ReportingPersonCode'] = mode_reporting_person_code


# In[ ]:


# Calculate the mean of the 'AverageMonthlyWorkhrs' column
mean_work_hours = data['AverageMonthlyWorkhrs'].mean()

# Replace null values with the mean
data['AverageMonthlyWorkhrs'].fillna(mean_work_hours, inplace=True)


# In[ ]:


# Calculate the mean of the 'HolidayworkingPercentage' column
mean_holiday_percentage = data['HolidayworkingPercentage'].mean()

# Replace null values with the mean
data['HolidayworkingPercentage'].fillna(mean_holiday_percentage, inplace=True)


# In[ ]:


# Replace null values with 0 in the 'PromotionTypeId' column
data['PromotionTypeId'].fillna(0, inplace=True)


# In[ ]:


#convert into numaric
data['Category'] = data['Category'].replace(['PER', 'PROB', 'CON', 'CAS'], [0, 1, 2, 3])
data['Category']


# In[ ]:


data['AbsentiesmPercentage'].describe()


# In[ ]:


# Calculate the mean of the 'AbsentiesmPercentage' column
mean_absentiesm = data['AbsentiesmPercentage'].mean()

# Replace null values with the mean
data['AbsentiesmPercentage'].fillna(mean_absentiesm, inplace=True)


# In[ ]:


data['AbsentiesmPercentage'].describe()


# In[ ]:


data['weekendworkingPercentage'].describe()


# In[ ]:


data.isnull().sum()


# In[ ]:


threshold = 35
null_counts = data.isnull().sum()
columns_to_drop = null_counts[null_counts > threshold].index.tolist()
data = data.drop(columns=columns_to_drop)


# In[ ]:


data.isnull().sum()


# In[ ]:


count = len(data[data['Resign'] == 0])
print(count)


# In[ ]:


count = len(data[data['Resign'] == 1])
print(count)


# In[ ]:


haveduprows = data.duplicated().any()
haveduprows


# In[ ]:


def normalize_column_z_score(dataframe, column_name):
    column_data = dataframe[column_name]
    mean_val = column_data.mean()
    std_val = column_data.std()
    normalized_column = (column_data - mean_val) / std_val
    dataframe[column_name] = normalized_column


# In[ ]:


column_name = 'AbsentiesmPercentage'  # Replace with the column name to be normalized

normalize_column_z_score(data, column_name)


# In[ ]:


column_name = 'leave_PercentagALL'  # Replace with the column name to be normalized

normalize_column_z_score(data, column_name)


# In[ ]:


column_name = 'Average_LeavesPerMonth'  # Replace with the column name to be normalized

normalize_column_z_score(data, column_name)


# In[ ]:


column_name = 'HolidayworkingPercentage'  # Replace with the column name to be normalized

normalize_column_z_score(data, column_name)


# In[ ]:


numeric_df = data.select_dtypes(include=['number'])
numeric_df


# In[ ]:


from scipy.stats import spearmanr


# In[ ]:


# Use the .corr() function to calculate the Pearson correlation coefficient
correlation_coefficient = data['Resign'].corr(data['Age'])
correlation_coefficient


# In[ ]:


selected_columnsA = ['Resign','Level2Code','JobCategoryCode','tenure','DesignationCode','EmploymentTypeCode','ReportingPersonCode', 'Category','AbsentiesmPercentage','PromotionTypeId',
                    'ReportingDesignationCode','AverageMonthlyWorkhrs','HolidayworkingPercentage','GenderCode', 'Age','Level3Code', 'TravelModeCode','Average_LeavesPerMonth','allocated_leaves','usedleaves','LeavePercntagenew','leave_PercentagALL' ]


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
correlation_matrix = data[selected_columnsA].corr()
plt.figure(figsize=(15, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="viridis")
plt.title("Correlation Matrix")
plt.show()


# In[ ]:


selected_columnsX = ['Level2Code','JobCategoryCode','tenure','DesignationCode','EmploymentTypeCode','ReportingPersonCode','AbsentiesmPercentage','PromotionTypeId',
                    'ReportingDesignationCode','AverageMonthlyWorkhrs','HolidayworkingPercentage', 'Age','allocated_leaves','usedleaves']


# In[ ]:


data = data.drop(columns=['EmployeeId','GenderCode','Level3Code','DateOfAppointment','A_ResignedDate','A_DateOfRetirement','Category','Average_LeavesPerMonth','allocatedleaves','usedleaves.1','leave_PercentagALL','LeavePercntagenew'])


# In[ ]:


data.columns


# In[ ]:


import sklearn as sl
X = data.drop(['Resign'], axis=1)
Y = data['Resign']
Y
X


# In[ ]:


from sklearn.model_selection import train_test_split
# Split the data into a 70% training set and a 30% testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Print the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[ ]:


#Random Forest
#Logistic Regression
#KNN
#Gausian Naive Bayes
#Decision Tree
#Support Vector Machine


# In[ ]:


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# Create and train a Random Forest model
RMmodel = RandomForestClassifier()
RMmodel.fit(X_train, y_train)

# Make predictions on the test set
y_predRF = RMmodel.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predRF)
print(f'Accuracy of Random Forest: {accuracy:.2f}')

# Generate a confusion matrix
cm = confusion_matrix(y_test, y_predRF)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Random forest Confusion Matrix')
plt.show()


# In[ ]:


# Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_predLR = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predLR)
print(f'Accuracy of Logistic Reggression : {accuracy:.2f}')

# Generate a confusion matrix
cm = confusion_matrix(y_test, y_predLR)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Logistic Regression Confusion Matrix')
plt.show()


# In[ ]:


# Create and train a Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_predDT = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predDT)
print(f'Accuracy of Decision Tree: {accuracy:.2f}')

# Generate a confusion matrix
cm = confusion_matrix(y_test, y_predDT)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Decision Tree Confusion Matrix')
plt.show()


# In[ ]:


# Create and train a KNN model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_predKC = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predKC)
print(f'Accuracy of KNN : {accuracy:.2f}')

# Generate a confusion matrix
cm = confusion_matrix(y_test, y_predKC)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('KNN Confusion Matrix')
plt.show()


# In[ ]:


from os import GRND_NONBLOCK
# Create and train a Naive Bayes (Gaussian Naive Bayes) model
GNBmodel = GaussianNB()
GNBmodel.fit(X_train, y_train)

# Make predictions on the test set
y_predGN = GNBmodel.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predGN)
print(f'Accuracy of Gaussian Naive Bayes : {accuracy:.2f}')

# Generate a confusion matrix
cm = confusion_matrix(y_test, y_predGN)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Gaussian Naive Bayes Confusion Matrix')
plt.show()


# In[ ]:


from sklearn.svm import SVC
# Create and train a Support Vector Machine (SVM) model
model = SVC()
model.fit(X_train, y_train)

# Make predictions on the test set
y_predSVM = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predSVM)
print(f'Accuracy: {accuracy:.2f}')

# Generate a confusion matrix
cm = confusion_matrix(y_test, y_predSVM)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Set3")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:


#Evaluating the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Initialize and train the models
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Gaussian Naive Bayes': GaussianNB(),
    'KNN Classification': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector machine': SVC()
}

results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

# Create a DataFrame to display the results
results_df = pd.DataFrame(results)
print(results_df)


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame with the model performance metrics
data = {
    'Model': ['Random Forest', 'Logistic Regression', 'Gaussian Naive Bayes', 'KNN Classification','Decision Tree','SVM'],
    'Accuracy': [0.83, 0.66, 0.72, 0.67,0.62,0.58],
    'Precision': [0.83, 0.66, 0.73, 0.67,0.66,0.58],
    'Recall': [0.90, 0.86, 0.82, 0.86,0.72,0.9],
    'F1-Score':[0.86, 0.75, 0.77, 0.75,0.69,0.73]

}

df = pd.DataFrame(data)

# Set the model names as the index for the DataFrame
df.set_index('Model', inplace=True)

# Define custom colors for the bars
colors = ['#1f77b4', '#aec7e8', '#9467bd', '#c5b0d5']

# Create a bar graph with custom colors
df.plot(kind='bar', figsize=(10, 6), color=colors)
plt.title('Model Performance Metrics')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
#RM: 0.79
#gnb : 0.75


# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV

# Make predictions on both training and testing data
y_train_pred = RMmodel.predict(X_train)
y_test_pred = RMmodel.predict(X_test)

# Calculate accuracy on both datasets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Determine if the model is overfit, underfit, or well-fit
if train_accuracy > test_accuracy:
    print("Model is likely overfitting.")
elif train_accuracy < test_accuracy:
    print("Model is likely underfitting.")
else:
    print("Model is performing well on both training and testing data.")

# Calculate the overfit percentage
overfit_percentage = (train_accuracy - test_accuracy) / train_accuracy * 100

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing Accuracy: {test_accuracy:.2f}')
print(f'Overfit Percentage: {overfit_percentage:.2f}%')


# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=RMmodel, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with the best hyperparameters
best_rf_classifier = RandomForestClassifier(random_state=42, **best_params)
best_rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy of Random Forest:", accuracy)


# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV
# Make predictions on both training and testing (or validation) data
y_train_pred = best_rf_classifier.predict(X_train)
y_test_pred = best_rf_classifier.predict(X_test)

# Calculate accuracy on both datasets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Determine if the model is overfit, underfit, or well-fit
if train_accuracy > test_accuracy:
    print("Model is likely overfitting.")
elif train_accuracy < test_accuracy:
    print("Model is likely underfitting.")
else:
    print("Model is performing well on both training and testing data.")

# Calculate the overfit percentage
overfit_percentage = (train_accuracy - test_accuracy) / train_accuracy * 100

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing Accuracy: {test_accuracy:.2f}')
print(f'Overfit Percentage: {overfit_percentage:.2f}%')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create and train a KNN model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Make predictions on both training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy on both datasets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Determine if the model is overfit, underfit, or well-fit
if train_accuracy > test_accuracy:
    print("Model is likely overfitting.")
elif train_accuracy < test_accuracy:
    print("Model is likely underfitting.")
else:
    print("Model is performing well on both training and testing data.")

# Calculate the overfit percentage
overfit_percentage = (train_accuracy - test_accuracy) / train_accuracy * 100

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing Accuracy: {test_accuracy:.2f}')
print(f'Overfit Percentage: {overfit_percentage:.2f}%')


# In[ ]:


####GNB
# Make predictions on both training and testing data
y_train_pred = GNBmodel.predict(X_train)
y_test_pred = GNBmodel.predict(X_test)

# Calculate accuracy on both datasets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Determine if the model is overfit, underfit, or well-fit
if train_accuracy > test_accuracy:
    print("Model is likely overfitting.")
elif train_accuracy < test_accuracy:
    print("Model is likely underfitting.")
else:
    print("Model is performing well on both training and testing data.")

# Calculate the overfit percentage
overfit_percentage = (train_accuracy - test_accuracy) / train_accuracy * 100

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing Accuracy: {test_accuracy:.2f}')
print(f'Overfit Percentage: {overfit_percentage:.2f}%')


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
# Define hyperparameters to tune
param_grid = {
    'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6,1e-5,0,1e+1,1e+2]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(GNBmodel, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model with optimal hyperparameters
best_GNBmodel = grid_search.best_estimator_

# Make predictions on both training and testing data
y_train_pred = best_GNBmodel.predict(X_train)
y_test_pred = best_GNBmodel.predict(X_test)

# Calculate accuracy on both datasets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Determine if the model is overfit, underfit, or well-fit
if train_accuracy > test_accuracy:
    print("Model is likely overfitting.")
elif train_accuracy < test_accuracy:
    print("Model is likely underfitting.")
else:
    print("Model is performing well on both training and testing data.")

# Calculate the overfit percentage
overfit_percentage = (train_accuracy - test_accuracy) / train_accuracy * 100

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing Accuracy: {test_accuracy:.2f}')
print(f'Overfit Percentage: {overfit_percentage:.2f}%')


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np


# Define the parameter grid for var_smoothing
param_grid = {'var_smoothing': np.logspace(1e-9, -9, num=10)}

# Create a Gaussian Naive Bayes classifier
GNB_Model = GaussianNB()

# Perform grid search with 3-fold cross-validation
grid_search = GridSearchCV(GNB_Model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameter value
best_var_smoothing = grid_search.best_params_['var_smoothing']
print(f'Best var_smoothing: {best_var_smoothing:.4e}')

# Print the mean cross-validated accuracy with the best hyperparameter
best_mean_accuracy = grid_search.best_score_
print(f"Mean Cross-Validation Accuracy with Best Hyperparameter: {best_mean_accuracy:.4f}")

# Train the model on the entire training dataset with the best hyperparameter
best_GNB_Model = GaussianNB(var_smoothing=best_var_smoothing)
best_GNB_Model.fit(X_train, y_train)

# Make predictions on the test dataset
y_pred_best = best_GNB_Model.predict(X_test)

# Calculate accuracy on the test dataset
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Accuracy on the Test Dataset with Best Hyperparameter: {accuracy_best:.4f}')

# Determine if the model is overfit, underfit, or well-fit with the best hyperparameter
tolerance = 1e-4  # You can adjust the tolerance based on your needs
if best_mean_accuracy - accuracy_best > tolerance:
    print("Model is likely overfitting.")
    overfit_percentage = ((best_mean_accuracy - accuracy_best) / best_mean_accuracy) * 100
    print(f'Overfit Percentage: {overfit_percentage:.2f}%')
elif accuracy_best - best_mean_accuracy > tolerance:
    print("Model is likely underfitting.")
    underfit_percentage = ((accuracy_best - best_mean_accuracy) / best_mean_accuracy) * 100
    print(f'Underfit Percentage: {underfit_percentage:.2f}%')
else:
    print("Model is well-fit, performing consistently with cross-validation and the test dataset.")


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold


# Define the parameter grid for var_smoothing
param_grid = {'var_smoothing': np.logspace(1e-9, -9, num=100)}

# Create a Gaussian Naive Bayes classifier
GNB_Model = GaussianNB()

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(GNB_Model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameter value
best_var_smoothing = grid_search.best_params_['var_smoothing']
print(f'Best var_smoothing: {best_var_smoothing:.4e}')

# Print the mean cross-validated accuracy with the best hyperparameter
best_mean_accuracy = grid_search.best_score_
print(f"Mean Cross-Validation Accuracy with Best Hyperparameter: {best_mean_accuracy:.4f}")

# Train the model on the entire training dataset with the best hyperparameter
best_GNB_Model = GaussianNB(var_smoothing=best_var_smoothing)
best_GNB_Model.fit(X_train, y_train)

# Make predictions on the test dataset
y_pred_best = best_GNB_Model.predict(X_test)

# Calculate accuracy on the test dataset
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Accuracy on the Test Dataset with Best Hyperparameter: {accuracy_best:.4f}')

# Determine if the model is overfit, underfit, or well-fit with the best hyperparameter
tolerance = 1e-4  # You can adjust the tolerance based on your needs
if best_mean_accuracy - accuracy_best > tolerance:
    print("Model is likely overfitting.")
    overfit_percentage = ((best_mean_accuracy - accuracy_best) / best_mean_accuracy) * 100
    print(f'Overfit Percentage: {overfit_percentage:.2f}%')
elif accuracy_best - best_mean_accuracy > tolerance:
    print("Model is likely underfitting.")
    underfit_percentage = ((accuracy_best - best_mean_accuracy) / best_mean_accuracy) * 100
    print(f'Underfit Percentage: {underfit_percentage:.2f}%')
else:
    print("Model is well-fit, performing consistently with cross-validation and the test dataset.")


# In[ ]:


###decision tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create and train a Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on both training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy on both datasets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate fitting percentages
fitting_percentage_train = (1 - train_accuracy) * 100
fitting_percentage_test = (1 - test_accuracy) * 100

# Determine if the model is overfit, underfit, or well-fit
if train_accuracy > test_accuracy:
    print("Model is likely overfitting.")
elif train_accuracy < test_accuracy:
    print("Model is likely underfitting.")
else:
    print("Model is performing well on both training and testing data.")

# Calculate the overfit percentage
overfit_percentage = (train_accuracy - test_accuracy) / train_accuracy * 100

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing Accuracy: {test_accuracy:.2f}')
print(f'Overfit Percentage: {overfit_percentage:.2f}%')


# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define the hyperparameters to tune
param_grid = {
    'max_depth': [None, 10, 20, 30],  # You can adjust these values based on your dataset
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create and train a Decision Tree model with GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Create a Decision Tree model with the best hyperparameters
best_model = DecisionTreeClassifier(**best_params)
best_model.fit(X_train, y_train)

# Make predictions on both training and testing data
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Calculate accuracy on both datasets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[ ]:


# Determine if the model is overfit, underfit, or well-fit
if train_accuracy > test_accuracy:
    print("Model is likely overfitting.")
elif train_accuracy < test_accuracy:
    print("Model is likely underfitting.")
else:
    print("Model is performing well on both training and testing data.")

# Calculate the overfit percentage
overfit_percentage = (train_accuracy - test_accuracy) / train_accuracy * 100

print(f'Best Hyperparameters: {best_params}')
print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing Accuracy: {test_accuracy:.2f}')
print(f'Overfit Percentage: {overfit_percentage:.2f}%')


# In[ ]:


###############################deployement##############################################


# In[ ]:


#######conditional probability######################


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Prepare input features for prediction
input_data_employeeA = pd.DataFrame({
    'Age': [55],
    'Level2Code': [6],
    'JobCategoryCode': [134],
    'DesignationCode': [9011],
    'tenure': [0],
    'TravelModeCode': [0],
    'EmploymentTypeCode': [54],
    'PromotionTypeId': [0],
    'ReportingPersonCode': [17099],
    'ReportingDesignationCode': [9065],
    'allocated_leaves': [28],
    'usedleaves': [1],
    'AbsentiesmPercentage': [34.03],
    'AverageMonthlyWorkhrs': [159.45],
    'HolidayworkingPercentage': [45],
})

# Use the model to make predictions
predicted_attrition = best_GNB_Model.predict(input_data_employeeA)

# Interpret the predictions
if predicted_attrition == 1:
    print("The model predicts that the employee is likely to leave the company.")
else:
    print("The model predicts that the employee is likely to stay with the company.")


# Analyze the impact of each feature for the specific employee
class_0_means = best_GNB_Model.theta_[0]
class_1_means = best_GNB_Model.theta_[1]

# Ensure that feature_names is a list
feature_names = input_data_employeeA.columns.tolist()

# Calculate the absolute differences in means between the classes
mean_diff = abs(class_1_means - class_0_means)

# Identify the most affected feature
most_affected_feature = feature_names[mean_diff.argmax()]

print(f"The most affected feature for the specific employee is: {most_affected_feature}")

# Plot the impact of each feature
fig, ax = plt.subplots()
ax.barh(feature_names, mean_diff)
ax.set_title("Impact of Each Feature on Predictions")
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Prepare input features for prediction
input_data_employeeB = pd.DataFrame({
    'Age': [19],
    'Level2Code': [6],
    'JobCategoryCode': [139],
    'DesignationCode': [9032],
    'tenure': [1],
    'TravelModeCode': [4],
    'EmploymentTypeCode': [54],
    'PromotionTypeId': [1],
    'ReportingPersonCode': [17310],
    'ReportingDesignationCode': [9128],
    'allocated_leaves': [30],
    'usedleaves': [5],
    'AbsentiesmPercentage': [2.4],
    'AverageMonthlyWorkhrs': [189],
    'HolidayworkingPercentage': [0],
})

# Use the model to make predictions
predicted_attrition = best_GNB_Model.predict(input_data_employeeB)

# Interpret the predictions
if predicted_attrition == 1:
    print("The model predicts that the employee is likely to leave the company.")
else:
    print("The model predicts that the employee is likely to stay with the company.")

# Analyze the impact of each feature using class conditional probabilities
class_0_means = best_GNB_Model.theta_[0]
class_1_means = best_GNB_Model.theta_[1]

feature_names = input_data_employeeB.columns
feature_impact = class_1_means - class_0_means

# Plot the impact of each feature
fig, ax = plt.subplots()
ax.barh(feature_names, feature_impact)
ax.set_title("Impact of Each Feature on Predictions")
plt.show()


# In[ ]:


#################################bayes theorem############################################


# In[ ]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB


# Prepare input features for a specific employee
input_data_employee = pd.DataFrame({
    'Age': [45],
    'Level2Code': [11],
    'JobCategoryCode': [139],
    'DesignationCode': [9032],
    'tenure': [1],
    'TravelModeCode': [0],
    'EmploymentTypeCode': [54],
    'PromotionTypeId': [1],
    'ReportingPersonCode': [17310],
    'ReportingDesignationCode': [9128],
    'allocated_leaves': [2.5],
    'usedleaves': [9.5],
    'AbsentiesmPercentage': [2.4],
    'AverageMonthlyWorkhrs': [189],
    'HolidayworkingPercentage': [0.02],
})

# Get the class prior probabilities
class_priors = GNB_Model.class_prior_

# Get the probabilities of each feature given the class
feature_probabilities = GNB_Model.predict_proba(input_data_employee)

# Calculate the posterior probability for each class
posterior_probabilities = class_priors * feature_probabilities.prod(axis=1)

# Identify the most affecting attribute based on the highest posterior probability
most_affecting_attribute = X.columns[feature_probabilities.argmax(axis=1)]

# Display the most affecting attribute
print(f"\nThe most affecting attribute(s) for the chosen employee is/are: {most_affecting_attribute}")


# Use the model to make predictions
predicted_attrition = GNB_Model.predict(input_data_employee)

# Interpret the predictions
if predicted_attrition == 1:
    print("The model predicts that the employee is likely to leave the company.")
else:
    print("The model predicts that the employee is likely to stay with the company.")


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Prepare input features for prediction
input_data_employee = pd.DataFrame({
    'Age': [38],
    'Level2Code': [11],
    'JobCategoryCode': [139],
    'DesignationCode': [9032],
    'tenure': [1],
    'TravelModeCode': [0],
    'EmploymentTypeCode': [54],
    'PromotionTypeId': [1],
    'ReportingPersonCode': [17310],
    'ReportingDesignationCode': [9128],
    'allocated_leaves': [98],
    'usedleaves': [94],
    'AbsentiesmPercentage': [24],
    'AverageMonthlyWorkhrs': [189],
    'HolidayworkingPercentage': [0],
})

# Use the model to make predictions
predicted_attrition = best_GNB_Model.predict(input_data_employee)

# Interpret the predictions
if predicted_attrition == 1:
    print("The model predicts that the employee is likely to leave the company.")
else:
    print("The model predicts that the employee is likely to stay with the company.")

# Get the class prior probabilities
class_priors = best_GNB_Model.class_prior_

# Get the probabilities of each feature given the class
feature_probabilities = best_GNB_Model.predict_proba(input_data_employee)

# Calculate the posterior probability for each class
posterior_probabilities = class_priors * feature_probabilities.prod(axis=1)

# Identify the most affecting attribute based on the highest posterior probability
most_affecting_attribute = input_data_employee.columns[feature_probabilities.argmax()]

# Display the most affecting attribute
print(f"\nThe most affecting attribute for the chosen employee is: {most_affecting_attribute}")     #level02: level of the job position


# In[ ]:


###############################PCA####################################################


# In[ ]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming X_train and y_train are your training features and labels
best_GNB_Model.fit(X_train, y_train)

model_feature_names = [
    'Age','Level2Code','JobCategoryCode','DesignationCode','tenure','TravelModeCode','EmploymentTypeCode','PromotionTypeId','ReportingPersonCode',
    'ReportingDesignationCode','allocated_leaves','usedleaves','AbsentiesmPercentage','AverageMonthlyWorkhrs','HolidayworkingPercentage',
]
# Prepare input features for prediction
input_data_employeeM = pd.DataFrame({
    'Age': [38],
    'Level2Code': [11],
    'JobCategoryCode': [139],
    'DesignationCode': [9032],
    'tenure': [1],
    'TravelModeCode': [0],
    'EmploymentTypeCode': [54],
    'PromotionTypeId': [1],
    'ReportingPersonCode': [17310],
    'ReportingDesignationCode': [9128],
    'allocated_leaves': [98],
    'usedleaves': [94],
    'AbsentiesmPercentage': [24],
    'AverageMonthlyWorkhrs': [189],
    'HolidayworkingPercentage': [0],
})

# Handle NaN values (replace NaN with mean or use another imputation method)
input_data_employeeA.fillna(input_data_employeeM.mean(), inplace=True)

# Apply PCA with standard scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(input_data_employeeM)
pca = PCA(n_components=0.95)

try:
    principalComponents = pca.fit_transform(scaled_data)

    # Explain the importance of each attribute using PCA
    for i in range(len(pca.explained_variance_ratio_)):
        print('Attribute: ', model_feature_names[i], ', Importance: ', pca.explained_variance_ratio_[i])

    # Use the model to make predictions
    predicted_attrition = best_GNB_Model.predict(input_data_employeeA)

    # Interpret the predictions
    if predicted_attrition == 1:
        print("The model predicts that the employee is likely to leave the company.")
    else:
        print("The model predicts that the employee is likely to stay with the company.")

except IndexError as e:
    print(f"Error: {e}. Check if there is sufficient variability in the data for PCA.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# In[ ]:


#PCA introduces challenges in interpretability due to its transformation of original features into linear combinations,
#assuming linearity, potentially losing crucial information,
#and may not align with the context of predicting specific outcomes, such as employee attrition.


# In[ ]:


###########################################################################################

