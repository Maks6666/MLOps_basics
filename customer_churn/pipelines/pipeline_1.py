from zenml import pipeline
from steps.read_data import read_data
from steps.clean_data import clean_data
from steps.resample_data import resample_data
from steps.split_data import split_data
from steps.preprocess_data import preprocess_data
from steps.tune_model import tune_model
from steps.train_model import train_model
from steps.eval_model import eval_model

@pipeline
def run_pipeline(link, target, model_name, n_trials):
    data = read_data(link=link)
    data = clean_data(data)
    x, y = resample_data(data, target)
    x_train, x_test, y_train, y_test = split_data(x, y)
    x_train, x_test = preprocess_data(x_train, x_test)
    best_params = tune_model(x_train, x_test, y_train, y_test, model_name, n_trials)
    trained_model = train_model(x_train, y_train, model_name, best_params)
    
    test_roc_auc, test_pr_auc, test_f1, test_pr, test_r = eval_model(trained_model, x_test, y_test)