# all full library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# all sklearn specific imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('ai4i2020.csv')
df.head()
df.describe()
df.info()

# Check for duplicate values and drop them
print("Checking for duplicate values: ", df.duplicated().sum() != 0)

# set the numeric columns data type to float
df['Process temperature [K]'] = df['Process temperature [K]'].astype(float)
df['Rotational speed [rpm]'] = df['Rotational speed [rpm]'].astype(float)
df['Torque [Nm]'] = df['Torque [Nm]'].astype(float)
df['Tool wear [min]'] = df['Tool wear [min]'].astype(float)
df['Machine failure'] = df['Machine failure'].astype(float)

# Remove first character and set to numeric dtype
df['Product ID'] = df['Product ID'].apply(lambda x: x[1:])
df['Product ID'] = pd.to_numeric(df['Product ID'])

# Convert the Type column M,L,H to 0,1,2
ds = df["Type"].value_counts().reset_index()[:28]
df['Type'] = df['Type'].apply(lambda x: 0 if x == 'L' else 1 if x == 'M' else 2)

# Show both plots in the same figure using subplots
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].pie(df['Type'].value_counts(), labels=df['Type'].value_counts().index, autopct='%1.1f%%')
ax[0].set_title('Pie chart of Type')
sns.histplot(data=df, x='Product ID', hue='Type', ax=ax[1])
ax[1].set_title('Histogram of Type')
plt.show()

# Drop the UDI and Product ID column as it is not needed for the model training
df_ = df.copy()
df = df_.drop(['UDI', 'Product ID'], axis=1)

# Check for missing values
df.isnull().sum()

# Print count of Unique values of each column
for col in df.columns:
    print(col, df[col].nunique())

# Plot the correlation matrix 
corr = df.corr()
plt.figure(figsize=(5, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.1f')
plt.show()

# Plot visualizations of the data and their correlations with the target variable
numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]','Type']
sns.pairplot(df.loc[:,numeric_cols],hue="Type",diag_kind='kde',kind='scatter')
plt.show()

##
## Feature Engineering
##

# Standardize the data 
X = df.drop(['Machine failure','TWF','HDF','PWF','OSF','RNF'], axis=1)
y = df['Machine failure']

print(X.shape, y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

 # Print the shapes of the training and testing sets and top 5 rows of the training set , X_train, y_train and X_test, y_test
print("----Training Data ----")
print(X_train.shape, y_train.shape)
print("----Test Data ----")
print(X_test.shape, y_test.shape)

for i in range(5):
    print("Training Set:", X_train[i], y_train.iloc[i])
    print("Testing Set:", X_test[i], y_test.iloc[i])

##
## Logistic Regression
##

# lr : Logistic Regression model  , y_pred_lr : Predicted values of y_test 

# Create linear regression object
lr = LogisticRegression()

# Train the model using the training sets
lr.fit(X_train, y_train)

# Evaluation metrics

y_pred_lr = lr.predict(X_test)
log_train = round(lr.score(X_train, y_train) * 100, 2)
log_accuracy = round(accuracy_score(y_pred_lr, y_test) * 100, 2)

# Show Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr)
plt.title('Confusion Matrix')
plt.show()

##
## Naive Bayes Classifier
##

# gnb : Naive Bayes Model  , y_pred_gnb : Predicted values of y_test

# Create Gaussian Naive Bayes object
gnb = GaussianNB()

# Train the model using the training sets
gnb.fit(X_train, y_train)

# Evaluation metrics

y_pred_gnb = gnb.predict(X_test)
gnb_train = round(gnb.score(X_train, y_train) * 100, 2)
gnb_accuracy = round(accuracy_score(y_pred_gnb, y_test) * 100, 2)

# Show Confustion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_gnb)
plt.title('Confusion Matrix')
plt.show()

# Logistic Classifier Results
print("Training Accuracy    :",log_train ,"%")
print("Model Accuracy Score :",log_accuracy ,"%")
print("\033[1m--------------------------------------------------------\033[0m")
print("Classification_Report: \n",classification_report(y_test,y_pred_lr,zero_division=0))
print("\033[1m--------------------------------------------------------\033[0m")

# Naive Bayes Results
print("Training Accuracy    :",gnb_train ,"%")
print("Model Accuracy Score :",gnb_accuracy ,"%")
print("\033[1m--------------------------------------------------------\033[0m")
print("Classification_Report: \n",classification_report(y_test,y_pred_gnb))
print("\033[1m--------------------------------------------------------\033[0m")

# Compare the results of the models
results = {'Logistic Regression': [log_accuracy, log_train]}
results['Naive Bayes'] = [gnb_accuracy, gnb_train]
results_df = pd.DataFrame(results, index=['Accuracy', 'Training Accuracy']).T
print(results_df)