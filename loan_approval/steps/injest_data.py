from zenml import step
import pandas as pd
import logging

class IngestData:
    def __init__(self, data_path: str) -> pd.DataFrame:
        self.data_path = data_path
    def extract_data(self):
        try:
            return pd.read_csv(self.data_path)
        except Exception as e:
            logging(f"Error: {e}")
            raise e
        

@step
def injest_data(data_path: str) -> pd.DataFrame:
    injestor = IngestData(data_path)
    try:
        return injestor.extract_data()
    except Exception as e:
        logging.error(f'Error: {e}')
        raise e 
