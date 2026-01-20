from zenml import step
import pandas as pd


@step()
def extract_data(data: str) -> pd.DataFrame:
    data = pd.read_csv(data)
    return data

