# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


import joblib

# Load the dataset, remove unnecessary index column if present, and display the first 5 rows.
df = pd.read_csv('german_credit_data.csv') 
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
print(df.head())

print(df.info())

print(df.describe())
print("Missing values")
print(df.isnull().sum())

# Fill missing values with 'unknown'
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

print("Missing values")
print(df.isnull().sum())
categorical = ['Checking account', 'Saving accounts', 'Purpose']
numerical_features = ['Age', 'Credit amount', 'Duration']
df[numerical_features].hist(bins=20, figsize=(15, 10), edgecolor='black')
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout()
plt.show()

# Feature Engineering: Creating new features

# 1. Credit amount per month
df['Credit_amount_per_month'] = df['Credit amount'] / df['Duration']

# 2. Age category
def categorize_age(age):
    if age <= 30:
        return 'Young'
    elif age <= 50:
        return 'Adult'
    else:
        return 'Senior'

df['Age_category'] = df['Age'].apply(categorize_age)

# Display first few rows to confirm
print(df.head())

# Define the conditions for Good and Bad Credit Risk
def define_credit_risk(row):
    if row['Credit amount'] <= 5000 and row['Duration'] <= 24 and row['Checking account'] != 'little':
        return 'Good'
    else:
        return 'Bad'

# Apply the function to create the 'Result' column
df['Result'] = df.apply(define_credit_risk, axis=1)

# Verify the distribution
print(df['Result'].value_counts())

print(df.head())

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded.info()

X = df_encoded.drop('Result_Good', axis=1)  # Features (all columns except 'Result_Good')
y = df_encoded['Result_Good']  # Target variable (1: Good, 0: Bad)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)


# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Logistic Regression model with increased max_iter
logreg = LogisticRegression(
    random_state=14,    # Random state for reproducibility
    max_iter=100,      # Maximum number of iterations
    C=0.1,                # Regularization strength
    solver='liblinear', # Solver 
    penalty='l1'        # Regularization type
)  
logreg.fit(X_train_scaled, y_train)

# Make predictions on the scaled test set
y_pred_logreg = logreg.predict(X_test_scaled)

# Evaluate the model's performance
print("Logistic Regression Accuracy: ", accuracy_score(y_test, y_pred_logreg))

print("\nClassification Report:\n", classification_report(y_test, y_pred_logreg))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))



# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(
    random_state=14,
    n_estimators=100,  # Number of trees
    max_depth=None,    # Allow trees to grow until pure
    criterion='gini'   # Split quality: Gini impurity
)

rf_model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test set
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the model's performance
print("Random Forest Accuracy: ", accuracy_score(y_test, y_pred_rf))

print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Initialize and train the SVM model
svm_model = SVC(
    random_state=14,
    kernel='rbf',    # Radial basis function kernel
    C=1.0,           # Regularization parameter
    gamma='scale'    # Kernel coefficient
)
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test set
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluate the model's performance
print("SVM Accuracy: ", accuracy_score(y_test, y_pred_svm))

print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))



# Number of folds
k = 5

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=k, scoring='accuracy')

# Print the cross-validation scores and average
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {np.mean(cv_scores)}")

# Set up the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],   # Number of trees
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples to split an internal node
    'min_samples_leaf': [1, 2, 4]     # Minimum number of samples required to be a leaf node
}

# Initialize GridSearchCV
grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1, verbose=1)

# Fit GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
print("Best Hyperparameters: ", best_params)

# Use the best model from GridSearchCV to predict
best_rf = grid_search.best_estimator_

# Make predictions
y_pred = best_rf.predict(X_test_scaled)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Cross-validation scores with the best model
cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5)
print("Cross-validation scores: ", cv_scores)
print("Average cross-validation score: ", cv_scores.mean())


from sklearn.model_selection import KFold, cross_val_score

# After completing GridSearchCV and getting best_rf (best Random Forest model)
# Initialize KFold with 10 splits
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Apply K-fold Cross-validation
kfold_cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=kfold)

# Print K-fold results
print("K-fold Cross-validation scores: ", kfold_cv_scores)
print("Average K-fold cross-validation score: ", kfold_cv_scores.mean())

# Get the feature importances and feature names
importances = best_rf.feature_importances_
feature_names = X.columns

# Create a DataFrame to display feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort the features by importance
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display the top 10 most important features
print(feature_importance.head(10))

# Plot the top 10 most important features
top_features = feature_importance.head(10)
plt.figure(figsize=(10,6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Absolute Coefficient')
plt.title('Top 10 Most Important Features in Logistic Regression')
plt.gca().invert_yaxis()  # To have the most important feature at the top
plt.show()

#Test Case
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = best_rf.predict(X_test_scaled)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
