from zenml import pipeline 

from steps.read_data import read_data
from steps.preprocess_data import preprocess_data
from steps.split_data import split_data
from steps.scale_data import scale_data
from steps.tune_model import tune_model
from steps.train_model import train_model
from steps.eval_model import model_eval


@pipeline
def run_pipeline(link, model_name, n_trials):
    data = read_data(link=link)
    data = preprocess_data(data=data)
    x_train, x_test, y_train, y_test = split_data(data=data)
    x_train, x_test = scale_data(x_train, x_test)

    best_params = tune_model(x_train, x_test, y_train, y_test, model_name, n_trials)
    trained_model = train_model(x_train, y_train, best_params, model_name)
    mse, mae = model_eval(trained_model, x_test, y_test)
