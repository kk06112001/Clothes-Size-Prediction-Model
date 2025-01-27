from sklearn.preprocessing import LabelEncoder
import pandas as pd
from etl_pipeline.exceptions import DataTransformationError
import logging
from scipy.stats import zscore

def clean_data(df):
    try:               
        df['age'] = df['age'].fillna(df['age'].median())
        df['height'] = df['height'].fillna(df['height'].median())
        logging.info("Missing values handled successfully")
        z_scores = zscore(df[['weight', 'age', 'height']])
        outliers = (z_scores > 3) | (z_scores < -3)
        df_cleaned = df[~outliers.any(axis=1)]
        logging.info("Outliers removed.")
        return df_cleaned
    except Exception as e:
        logging.error(f"Error in data cleaning: {e}")
        raise DataTransformationError(f"Error in data cleaning: {e}")
