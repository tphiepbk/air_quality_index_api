import pandas as pd

from src.preprocessor.preprocessor import Preprocessor
from src.prediction.prediction import PredictionModel
from src.reduction.reduction import ReductionModel
from src.schema.schema import CMAQResponse, VienThamResponse
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
                                          n_past=7,
                                          n_future=n_future,
                                          reduction_model_name=reduction_model_name).encode()
        info("{}: X_scaled_encoded.shape = {}", func_name, X_scaled_encoded.shape)
        info("{}: X_scaled_encoded = \n{}", func_name, X_scaled_encoded)

        # Prediction model
        predicted_pm25 = PredictionModel(feature_data=X_scaled_encoded,
                                         label_data=y_scaled,
                                         data_type="aod",
                                         n_past=7,
                                         n_future=n_future,
                                         reduction_model_name=reduction_model_name,
                                         prediction_model_name=prediction_model_name).predict()
        info("{}: predicted_pm25.shape = \n{}", func_name, predicted_pm25.shape)
        info("{}: predicted_pm25 = \n{}", func_name, predicted_pm25)

        # Proceed the result
        return VienThamResponse(data=predicted_pm25)

    def handleCMAQRequest(self, cmaq_request, reduction_model_name, prediction_model_name):
        # Logger
        func_name = "RequestHandler.handleCMAQRequest()"
        info("{}: is called", func_name)

        # Read the VienTham request
        n_future = cmaq_request.n_future
        df_cmaq = pd.DataFrame(cmaq_request.data.model_dump())

        # Preprocess data
        X_scaled, y_scaled = Preprocessor(data_type="cmaq").execute(input_data=df_cmaq)
        info("{}: X_scaled.shape = {}", func_name, X_scaled.shape)
        info("{}: X_scaled = \n{}", func_name, X_scaled)

        # Reduction model
        X_scaled_encoded = ReductionModel(data=X_scaled,
                                          data_type="cmaq",
                                          n_past=168,
                                          n_future=n_future,
                                          reduction_model_name=reduction_model_name).encode()
        info("{}: X_scaled_encoded.shape = {}", func_name, X_scaled_encoded.shape)
        info("{}: X_scaled_encoded = \n{}", func_name, X_scaled_encoded)

        # Prediction model
        predicted_no = PredictionModel(feature_data=X_scaled_encoded,
                                         label_data=y_scaled,
                                         data_type="cmaq",
                                         n_past=168,
                                         n_future=n_future,
                                         reduction_model_name=reduction_model_name,
                                         prediction_model_name=prediction_model_name).predict()
        info("{}: predicted_pm25.shape = \n{}", func_name, predicted_no.shape)
        info("{}: predicted_pm25 = \n{}", func_name, predicted_no)

        # Proceed the result
        return CMAQResponse(data=predicted_no)

