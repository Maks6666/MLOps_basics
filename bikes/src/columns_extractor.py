import pandas as pd
import logging


class Extractor:
    def __init__(self):
        ... 

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data['hour'] = pd.to_datetime(data['datetime']).dt.hour
            data['month'] = pd.to_datetime(data['datetime']).dt.month

            return data
        except Exception as e: 
            logging.error(f'Error: {e}')
            raise