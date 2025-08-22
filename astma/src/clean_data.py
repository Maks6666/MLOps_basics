from abc import ABC, abstractmethod
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split
import logging

class DataCleaner(ABC):
    @abstractmethod
    def clean_data(data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        ...



class CleanData(DataCleaner):
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        
        try:

            data = data.drop(["Patient_ID", "Asthma_Control_Level"], axis="columns")

            data["Allergies"].fillna(data["Allergies"].value_counts().idxmax(), inplace=True)
            data["Comorbidities"].fillna(data["Comorbidities"].value_counts().idxmax(), inplace=True)

            data.replace({"Gender": {"Female": 0, "Male": 1, "Other": 2}}, inplace=True)
            data.replace({"Allergies": {"Dust": 0, "Pollen": 1, "Pets": 2, "Multiple": 3}}, inplace=True)
            data.replace({"Smoking_Status": {"Never": 0, "Former": 1, "Current": 2}}, inplace=True)
            data.replace({"Air_Pollution_Level": {"Low": 0, "Moderate": 1, "High": 2}}, inplace=True)
            data.replace({"Physical_Activity_Level": {"Sedentary": 0, "Moderate": 1, "Active": 2}}, inplace=True)
            data.replace({"Occupation_Type": {"Indoor": 0, "Outdoor": 1}}, inplace=True)
            data.replace({"Comorbidities": {"Diabetes": 0, "Hypertension": 1, "Both": 2}}, inplace=True)

            return data

        except Exception as e:
            logging.error(f"Eroro: {e}")



class SplitData(DataCleaner):
    def clean_data(data: pd.DataFrame) -> pd.Series:
        try:
            x = data.drop(["Has_Asthma"], axis="columns")
            y = data["Has_Asthma"]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error: {e}")



class DataTool:
    def __init__(self, data, tool):
        self.data = data
        self.tool = tool
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.tool.clean_data(self.data)
        except Exception as e:
            logging.error(f"Error: {e}")
