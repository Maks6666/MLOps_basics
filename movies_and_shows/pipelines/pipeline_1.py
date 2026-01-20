from zenml import pipeline
from steps.extract import extract_data
from steps.preprocess import prerocess_data
from steps.split import split
from steps.tune import tune_model
from steps.train import train_model
from steps.eval_model import model_eval

@pipeline(enable_cache=False)
def run_pipeline(data_link: str, model_name: str, n_trials: int) -> None:
    data = extract_data(data_link)
    cleaned_data = prerocess_data(data)
    x_train, x_test, y_train, y_test = split(cleaned_data)
    best_params = tune_model(x_train, x_test, y_train, y_test, model_name, n_trials)
    trained_model = train_model(best_params, model_name, x_train, y_train)
    acc_score, f_score, rmse_score = model_eval(trained_model, x_test, y_test)

