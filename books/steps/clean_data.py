from zenml import step
import pandas as pd
from src.cleaner import Cleaner

@step
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaner = Cleaner(data)

    data = cleaner.set_target('is_successful', 'units sold')

    columns_to_delete = ['index', 'Book Name', 'gross sales', 'publisher revenue', 'sales rank', 'units sold', 'language_code']
    data = cleaner.delete_columns(columns_to_delete)

    data = cleaner.set_freq('Author')

    data = cleaner.rename('genre', {'genre fiction': "fiction"})

    return data 

    