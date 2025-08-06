from piplines.training_pipline import training_pipline
from zenml.client import Client

if __name__ == "__main__":
    # Эта строка возвращает URI (путь) к месту, где MLflow (или другой трекер)
    #  хранит все данные об экспериментах, запусках, метриках, моделях и т.д.
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipline(data_path="/Users/maxkucher/preprocessing/mlops/data/archive (2)/olist_customers_dataset.csv")
    print("Pipeline completed!")


# export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# установка mlflow через zenml
# zenml integration install mlflow

# создание трекера эскпериметнов 
# zenml experiment-tracker register mlflow_tracker --flavor=mlflow

# создание деплойера модели 
# zenml model-deployer register mlflow --flavor=mlflow

# создание стека !!!
# zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set

# -e - это название созданного трекера
# -d - это нахвание созданного деплойера модели

# стек - это набор компонентов инфраструктуры, которые работают вместе, чтобы запускать, отслеживать и разворачивать ML-проекты. 
# Это своего рода "рабочая среда" для ML-пайплайнов, где у каждого компонента своя функция.

# запуск веб-интерфейса ml-flow
# mlflow ui --backend-store-uri "file:/Users/maxkucher/Library/Application Support/zenml/local_stores/0cf0c7cc-d186-488b-90d0-24105f08400a/mlruns"