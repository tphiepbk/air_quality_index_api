from src.preprocessor.preprocessor import Preprocessor
from src.prediction.prediction import PredictionModel
from src.schema.schema import PredictionResponse

import os
import pandas as pd
import numpy as np

class RequestHandler():
    def __init__(self):
        self.__df_vientham = None
        self.__prep = None
        self.__pred_model = None

    def handle(self, vienthamdata):
        # Create dataframe
        self.__df_vientham = pd.DataFrame(vienthamdata.model_dump())

        # Preprocess data
        self.__prep = Preprocessor(n_past=7, n_future=1)
        scaled_data, label_scaler = self.__prep.execute(self.__df_vientham)
        scaled_data = np.expand_dims(scaled_data, axis=0)
        print(scaled_data.shape)

        # Predict
        pred_model_path = os.path.join("models", "prediction", "aod_LSTMPrediction_no_dim_reduction.keras")
        self.__pred_model = PredictionModel(pred_model_path)
        predicted_pm25 = self.__pred_model.predict(scaled_data)[0]

        # Proceed the result
        res = PredictionResponse(value_1d=predicted_pm25, value_2d=[], value_3d=[])
        return res

