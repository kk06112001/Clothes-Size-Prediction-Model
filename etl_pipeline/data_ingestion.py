import pandas as pd
from etl_pipeline.exceptions import DataIngestionError
import logging

def load_data(file_path):
    
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data successfully loaded from {file_path}.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise DataIngestionError(f"Error loading data from {file_path}: {e}")
