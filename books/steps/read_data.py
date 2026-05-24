import pandas as pd
from zenml import step 

@step
def read_data(link: str) -> pd.DataFrame:
    data = pd.read_csv(link)
    return data 