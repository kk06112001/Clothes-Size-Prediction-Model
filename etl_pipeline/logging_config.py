import logging
import os

def setup_logging():
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = os.path.join(log_dir, 'etl_pipeline.log')
    handler = logging.FileHandler(log_filename)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            handler,
            logging.StreamHandler()
        ]
    )
