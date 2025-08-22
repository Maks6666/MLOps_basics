from zenml import step
import pandas as pd
import logging


class IngestData:
    def __init__(self, data: str) -> pd.DataFrame:
        self.data = data
    def read_data(self):
        try:
            return pd.read_csv(self.data)
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e




@step
def ingest_data(data: str) -> pd.DataFrame:
    ingester = IngestData(data)
    try:
        return ingester.read_data()
    except Exception as e:
        logging.error(f"Error: {e}")
        raise e
