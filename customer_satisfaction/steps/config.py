from zenml.config import BaseParameters  # ✅ правильно



class ModelNameConfig(BaseParameters):
    model_name: str = "LinearRegression"