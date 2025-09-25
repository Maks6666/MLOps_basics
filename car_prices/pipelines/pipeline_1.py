from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import preprocess_data
from steps.tune_model import tune_model
from steps.train_model import train_model
from steps.eval_model import model_eval

@pipeline(enable_cache=False)
def run_pipeline(data_link, n_trials, model_name):
    data = ingest_data(data_link=data_link)
    x_train, x_test, y_train, y_test = preprocess_data(data)
    best_params = tune_model(x_train, x_test, y_train, y_test, n_trials, model_name)
    model = train_model(x_train, y_train, best_params, model_name)
    mse, mae, rmse = model_eval(model, x_test, y_test)
