import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import gradio as gr

# Load your dataset
df = pd.read_csv('customer_data.csv')

# Define the categorical columns for one-hot encoding
categorical_columns = ['Location', 'Occupation', 'Laptop Brands', 'Frequency of Use', 'Purpose', 'Tech Knowledge Level']

# Load the pre-trained Decision Tree model
model1 = joblib.load('decision_tree_model.joblib')

# Function to make predictions
def predict_cart_abandonment(age, income, price, satisfaction_rating, location, occupation, laptop_brands, frequency_of_use, purpose, tech_knowledge_level):
    input_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'Price': [price],
        'Satisfaction Rating': [satisfaction_rating],
        'Location': [location],
        'Occupation': [occupation],
        'Laptop Brands': [laptop_brands],
        'Frequency of Use': [frequency_of_use],
        'Purpose': [purpose],
        'Tech Knowledge Level': [tech_knowledge_level]
    })

    input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns)

    # Use the trained models to make predictions
    dt_prediction = model1.predict(input_data_encoded)[0]

    return {
        "Decision Tree Prediction": "Yes" if dt_prediction == 1 else "No"
    }

# Gradio Interface
iface = gr.Interface(
    fn=predict_cart_abandonment,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Income"),
        gr.Number(label="Price"),
        gr.Number(label="Satisfaction Rating"),
        gr.Dropdown(list(df['Location'].unique()), label="Location"),
        gr.Dropdown(list(df['Occupation'].unique()), label="Occupation"),
        gr.Dropdown(list(df['Laptop Brands'].unique()), label="Laptop Brands"),
        gr.Dropdown(list(df['Frequency of Use'].unique()), label="Frequency of Use"),
        gr.Dropdown(list(df['Purpose'].unique()), label="Purpose"),
        gr.Dropdown(list(df['Tech Knowledge Level'].unique()), label="Tech Knowledge Level"),
    ],
    outputs=[gr.Label(label="Decision Tree Prediction")],
    live=True,
    theme="clean",
    title="Cart Abandonment Prediction",
    layout="vertical",
    enable_queue=True,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
