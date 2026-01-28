from zenml import pipeline

from steps.extract_data import extract_data
from steps.preprocess_data import preprocess_data
from steps.split_data import split_data
from steps.scale_data import scale_data
from steps.encode_data import encode_data
from steps.tune_model import tune_model
from steps.train_model import train_model
from steps.eval_model import eval_model

@pipeline(enable_cache=False)
def run(link, target, model_name, n_trials):
    data = extract_data(link=link)
    data = preprocess_data(data=data)
    x_train, x_test, y_train, y_test = split_data(data, target)

    x_train, x_test = scale_data(x_train=x_train, x_test=x_test)
    x_train, x_test = encode_data(x_train=x_train, x_test=x_test)

    best_params = tune_model(x_train, x_test, y_train, y_test, model_name, n_trials)

    trained_model = train_model(x_train, y_train, model_name, best_params)
    mae, mse, r2 = eval_model(trained_model, x_test, y_test)


    