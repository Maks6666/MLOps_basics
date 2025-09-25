from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Union
import logging
from sklearn.model_selection import train_test_split


class DataCleaner(ABC):
    @abstractmethod
    def preprocess_data(data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        ...



class CleanData(DataCleaner):
    def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
        try:
            
            data = data.drop(["car_ID", "CarName", "symboling"], axis="columns")

            data.replace({"fueltype": {"gas": 0, "diesel": 1}}, inplace=True)
            data.replace({"doornumber": {"two": 0, "four": 1}}, inplace=True)
            data.replace({"aspiration": {"std": 0, "turbo": 1}}, inplace=True)
            data.replace({"carbody": {"convertible": 0, "hatchback": 1, "sedan": 2, "wagon": 3, "hardtop": 4}}, inplace=True)
            data.replace({"drivewheel": {"rwd": 0, "fwd": 1, "4wd": 2}}, inplace=True)
            data.replace({"enginelocation": {"front": 0, "rear": 1}}, inplace=True)
            data.replace({"enginetype": {"dohc": 0, "ohcv": 1, "ohc": 2, "l": 3, "rotor": 4, "ohcf": 5, "dohcv": 6}}, inplace=True)
            data.replace({"cylindernumber": {"four": 0, "six": 1, "five": 2, "three": 3, "twelve": 4, "two": 5, "eight": 6}}, inplace=True)
            data.replace({"fuelsystem": {"mpfi": 0, "2bbl": 1, "mfi": 2, "1bbl": 3, "spfi": 4, "4bbl": 5, "idi": 6, "spdi": 7}}, inplace=True)

            return data
        
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        

class DivideData(DataCleaner):
    def preprocess_data(data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            x = data.drop("price", axis="columns")
            y = data["price"]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            return x_train, x_test, y_train, y_test
        except Exception as e: 
            logging.error(f"Error: {e}")
            raise e
        
    

class DataTool:
    def __init__(self, data: pd.DataFrame, tool: DataCleaner):
        self.data = data
        self.tool = tool
    def preprocess_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.tool.preprocess_data(self.data)
        except Exception as e: 
            logging.error(f"Error: {e}")
            raise e
    

         
    
