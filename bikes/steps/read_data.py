from zenml import step 
import pandas as pd
import logging


@step
def read_data(link: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(link)
        return data
    except Exception as e: 
        logging.error(f'Error: {e}')
        raise 
