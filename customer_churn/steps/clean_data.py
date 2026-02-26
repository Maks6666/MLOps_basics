from zenml import step
import pandas as pd
import numpy as np

from src.cleaner import Cleaner

@step
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaner = Cleaner()

    columns_to_rename = ['MultipleLines']
    replacer = {'No phone service': 'No'}
    data = cleaner.clean(data, columns_to_rename, replacer)
 

    columns_to_rename = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies']
    replacer = {'No internet service': 'No'}
    data = cleaner.clean(data, columns_to_rename, replacer)


    columns_to_rename = ['TotalCharges']
    replacer = {' ': np.nan}

    data = cleaner.clean(data, columns_to_rename, replacer)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    
    data = data.dropna(subset=['TotalCharges'])
    data = data.drop('customerID', axis='columns')
    

    columns_to_rename = [
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Churn']
    
    replacer = {'Yes': 1, 'No': 0}

    data = cleaner.clean(data, columns_to_rename, replacer)

    columns_to_rename = ['gender']
    replacer = {'Female': 0, 'Male': 1}

    data = cleaner.clean(data, columns_to_rename, replacer)

    return data

