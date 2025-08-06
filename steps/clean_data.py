import logging
import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreporcessStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    try:
        process_strategy = DataPreporcessStrategy()
        divide_strategy = DataDivideStrategy()

        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        data_cleaning = DataCleaning(processed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        return x_train, x_test, y_train, y_test
        logging.info("Data cleaning completed")
    except Exception as e:
        logging.error(f"Error {e}")
        raise e




