from src.preprocessor import Preprocessor
from src.preprocessor import Encoder
from zenml import step
import pandas as pd

@step
def prerocess_data(data: pd.DataFrame):
    prerocessor = Preprocessor()
    encoder = Encoder()
    
    upd_data = prerocessor.drop_values(data=data, columns=["show_id", "title", "director", "cast", "description"])
    upd_data = prerocessor.drop_nulls(data=upd_data, columns=["date_added", "rating", "duration"])
    upd_data = prerocessor.fill_empties(data=upd_data, columns=["director", "cast", "country"], fill_value="Unknown")
    upd_data = prerocessor.replace(data=upd_data, column="type", replacer={"Movie": 0, "TV Show": 1})
    upd_data = prerocessor.leave_one(data=upd_data, columns=["country"])

    upd_data = encoder.one_hot(data=upd_data, column="country")
    upd_data = encoder.one_hot(data=upd_data, column="rating")
    upd_data = encoder.datetime(data=upd_data, column="date_added")
    upd_data = encoder.cycle_encoder(data=upd_data, column="month")
    upd_data = encoder.ecnode_duration(data=upd_data, columns=["duration_minutes", "duration_seasons"], column="duration")
    upd_data = encoder.process_multi_labels(data=upd_data, column="listed_in", new_column="genres")

    upd_data = prerocessor.drop_nulls(data=upd_data, columns=["year", "month_sin", "month_cos"])

    return upd_data

