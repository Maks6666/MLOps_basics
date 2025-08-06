import logging
import pandas as pd
from zenml import step


class IngestData:
    '''
    Ingesting data from data_path
    '''
    def __init__(self, data_path: str):
        '''
        Args:
            data_path: path to the data
        '''
        self.data_path = data_path
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    '''
    Ingesting data from data_path

    Args: 
        data_path: path to data
    Returns:
        pd.DataFrame: the ingested data
    '''

    try:
        ingest_data = IngestData(data_path)
        data = ingest_data.get_data()
        return data
    except Exception as e:
        logging.error(f'Error while ingesting data: {e}')
        raise e

