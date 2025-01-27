import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from etl_pipeline.exceptions import ModelTrainingError
from etl_pipeline.data_ingestion import load_data
from etl_pipeline.data_transformation import clean_data

def train_model(file_path):
    
    try:
        df = load_data(file_path)
        df_cleaned = clean_data(df)
        label_encoder = LabelEncoder()
        df_cleaned['size_encoded'] = label_encoder.fit_transform(df_cleaned['size'])
        X = df_cleaned[['weight', 'age', 'height']]
        y = df_cleaned['size_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        with open('model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        with open('scaler.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
        with open('label_encoder.pkl', 'wb') as le_file:
            pickle.dump(label_encoder, le_file)

        logging.info("Model, scaler, and label encoder saved successfully.")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise ModelTrainingError(f"Error during model training: {e}")


def load_model():
    """
    Load the trained logistic regression model.
    """
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise ModelTrainingError(f"Error loading model: {e}")


def load_scaler():
    """
    Load the trained scaler.
    """
    try:
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        logging.info("Scaler loaded successfully.")
        return scaler
    except Exception as e:
        logging.error(f"Error loading scaler: {e}")
        raise ModelTrainingError(f"Error loading scaler: {e}")


def load_label_encoder():
    """
    Load the trained label encoder.
    """
    try:
        with open('label_encoder.pkl', 'rb') as le_file:
            label_encoder = pickle.load(le_file)
        logging.info("Label encoder loaded successfully.")
        return label_encoder
    except Exception as e:
        logging.error(f"Error loading label encoder: {e}")
        raise ModelTrainingError(f"Error loading label encoder: {e}")


def predict_size(weight, age, height):
    """
    Predict the clothing size based on the input features.
    """
    try:
        model = load_model()
        scaler = load_scaler()
        label_encoder = load_label_encoder()

        input_data = np.array([[weight, age, height]])
        input_data_scaled = scaler.transform(input_data)

        predicted_size = model.predict(input_data_scaled)
        predicted_size_label = label_encoder.inverse_transform(predicted_size)
        logging.info(f"Prediction made successfully: {predicted_size_label[0]}")
        return predicted_size_label[0]

    except Exception as e:
        logging.error(f"Error during model prediction: {e}")
        raise ModelTrainingError(f"Error during model prediction: {e}")
