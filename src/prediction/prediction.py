from time import sleep

from src.prediction.reader import csv_to_dataframe

class Prediction:
    def __init__(self):
        self._df = csv_to_dataframe("/home/tphiepbk/workspace/air_quality_index_api/dataset/df_aod_raw.csv")

    def dummy_action(self, data):
        print("sleeping")
        sleep(5)
        return {"data": data}
