from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.tune_model import tune_model
from steps.train_model import train_model
from steps.model_eval import model_eval

@pipeline(enable_cache=False)
def run_pipeline(data_path, n_trials, model_name):
    data = ingest_data(data_path)
    x_train, x_test, y_train, y_test = clean_data(data)
    best_params, name = tune_model(x_train, y_train, x_test, y_test, n_trials, model_name)
    trained_model = train_model(x_train, y_train, best_params, name)
    f1, a_s, rmse = model_eval(x_test, y_test, trained_model)