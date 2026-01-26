from zenml import step
import pandas as pd

@step
def injest_data(link: str) -> pd.DataFrame:
    data = pd.read_csv(link)
    return data