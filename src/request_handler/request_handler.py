import pandas as pd

from src.preprocessor.preprocessor import Preprocessor
from src.prediction.prediction import PredictionModel
from src.reduction.reduction import ReductionModel
from src.schema.schema import VienThamResponse
from src.logger.logger import info

class RequestHandler():
    def __init__(self):
        func_name = "RequestHandler.__init__()"
        info("{}: is called", func_name)

    def handleVienThamRequest(self, vientham_request, reduction_model_name, prediction_model_name):
        # Logger
        func_name = "RequestHandler.handleVienThamRequest()"
        info("{}: is called", func_name)

        # Read the VienTham request
        n_future = vientham_request.n_future
        df_vientham = pd.DataFrame(vientham_request.data.model_dump())

        # Preprocess data
        X_scaled, y_scaled = Preprocessor(data_type="aod").execute(input_data=df_vientham)
        info("{}: X_scaled.shape = {}", func_name, X_scaled.shape)
        info("{}: X_scaled = \n{}", func_name, X_scaled)

        # Reduction model
        X_scaled_encoded = ReductionModel(data=X_scaled,
                                          data_type="aod",
                                          n_future=n_future,
                                          reduction_model_name=reduction_model_name).encode()
        info("{}: X_scaled_encoded.shape = {}", func_name, X_scaled_encoded.shape)
        info("{}: X_scaled_encoded = \n{}", func_name, X_scaled_encoded)

        # Prediction model
        predicted_pm25 = PredictionModel(feature_data=X_scaled_encoded,
                                         label_data=y_scaled,
                                         data_type="aod",
                                         n_future=n_future,
                                         reduction_model_name=reduction_model_name,
                                         prediction_model_name=prediction_model_name).predict()
        info("{}: predicted_pm25.shape = \n{}", func_name, predicted_pm25.shape)
        info("{}: predicted_pm25 = \n{}", func_name, predicted_pm25)
        predicted_pm25_reshaped = predicted_pm25.reshape(-1)
        info("{}: predicted_pm25_reshaped.shape = \n{}", func_name, predicted_pm25_reshaped.shape)
        info("{}: predicted_pm25_reshaped = \n{}", func_name, predicted_pm25_reshaped)

        # Proceed the result
        return VienThamResponse(data=predicted_pm25_reshaped)

    def handleCMAQRequest(self, cmaq_request):
        pass

