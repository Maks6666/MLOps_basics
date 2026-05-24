from zenml import pipeline

from steps.read_data import read_data
from steps.clean_data import clean_data
from steps.resample_data import resample_data
from steps.split_data import split_data
from steps.preprocess_data import preprocess_data

@pipeline
def run_pipeline(link, target):
    data = read_data(link)
    data = clean_data(data)
    x, y = resample_data(data, target)
    x_train, x_test, y_train, y_test = split_data(x, y)
    x_train, x_test = preprocess_data(x_train, x_test)


