from zenml import pipeline 

from steps.injest_data import injest_data
from steps.preprocess_data import preprocess_data
from steps.split_data import split_data
from steps.encode_data import encode_data
from steps.tune_model import tune_model
from steps.train_model import train_model
from steps.eval_model import eval
@pipeline(enable_cache=False)
def run_pipeline(link, target, model_name, n_trials):
    data = injest_data(link)
    clean_data = preprocess_data(data)
    x_train, x_test, y_train, y_test = split_data(clean_data, target)
    x_train, x_test = encode_data(x_train, x_test)
    best_params = tune_model(x_train, x_test, y_train, y_test, model_name=model_name, n_trials=n_trials)
    trained_model = train_model(x_train, y_train, model_name, best_params)
    acc, f1, rmse = eval(trained_model, x_test, y_test)