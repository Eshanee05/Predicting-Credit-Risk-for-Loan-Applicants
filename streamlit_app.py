# streamlit_app.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# App title
st.title("Credit Risk Prediction App 🚀")

# Load the dataset
st.header("1. Load and Preprocess Data")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if st.checkbox('Show dataset info'):
        st.write(df.info())

    if st.checkbox('Show dataset description'):
        st.write(df.describe())
    
    st.subheader("Missing values before filling:")
    st.write(df.isnull().sum())

    # Fill missing values
    df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
    df['Checking account'] = df['Checking account'].fillna('unknown')

    st.subheader("Missing values after filling:")
    st.write(df.isnull().sum())

    # Histograms of numerical features
    st.subheader("Numerical Features Distribution")
    numerical_features = ['Age', 'Credit amount', 'Duration']
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(numerical_features):
        sns.histplot(df[col], bins=20, ax=ax[i], kde=True, edgecolor='black')
    st.pyplot(fig)

    # Feature Engineering
    df['Credit_amount_per_month'] = df['Credit amount'] / df['Duration']

    def categorize_age(age):
        if age <= 30:
            return 'Young'
        elif age <= 50:
            return 'Adult'
        else:
            return 'Senior'

    df['Age_category'] = df['Age'].apply(categorize_age)

    # Create target variable
    def define_credit_risk(row):
        if row['Credit amount'] <= 5000 and row['Duration'] <= 24 and row['Checking account'] != 'little':
            return 'Good'
        else:
            return 'Bad'

    df['Result'] = df.apply(define_credit_risk, axis=1)

    st.subheader("Target Variable Distribution")
    st.bar_chart(df['Result'].value_counts())

    # One-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Features and Target
    X = df_encoded.drop('Result_Good', axis=1)
    y = df_encoded['Result_Good']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.success('Data Preprocessing Completed!')

    # Model training
    st.header("2. Train Models")

    model_choice = st.selectbox(
        "Choose a model",
        ["Logistic Regression", "Random Forest(Selected model)", "SVM"]
    )

    if model_choice == "Logistic Regression":
        model = LogisticRegression(
            random_state=14, max_iter=100, C=0.1, solver='liblinear', penalty='l1'
        )
    elif model_choice == "Random Forest(Selected model)":
        model = RandomForestClassifier(
            random_state=14, n_estimators=100, criterion='gini'
        )
    elif model_choice == "SVM":
        model = SVC(
            random_state=14, kernel='rbf', C=1.0, gamma='scale'
        )

    if st.button('Train Selected Model'):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        st.subheader("Model Evaluation")
        st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

        # Cross Validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        st.subheader("Cross-validation Scores (5-Fold)")
        st.write(cv_scores)
        st.write(f"Average CV Score: {np.mean(cv_scores):.4f}")

        # K-Fold CV
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        kfold_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold)
        st.subheader("K-Fold (10-Fold) Cross-validation")
        st.write(kfold_scores)
        st.write(f"Average K-Fold CV Score: {np.mean(kfold_scores):.4f}")

        # Feature importance if Random Forest
        if model_choice == "Random Forest(Selected model)":
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

        

            st.subheader("Top 10 Feature Importances (Random Forest)")
            st.dataframe(feature_importance.head(10))

            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='viridis')
            plt.title('Top 10 Important Features')
            st.pyplot(fig)

else:
    st.warning("Please upload a dataset to continue.")

