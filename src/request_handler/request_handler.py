import pandas as pd
import os
import pickle

import lightgbm as lgb

from src.preprocessor.preprocessor import Preprocessor
from src.prediction.prediction import PredictionModel
from src.reduction.reduction import ReductionModel
from src.schema.schema import CMAQResponse, VienThamResponse, QuanTracResponse
from src.logger.logger import info
from src.lightgbm_wrapper.feature_engineer import add_time_features, add_lag_features, add_rolling_features
from src.lightgbm_wrapper.station_embedding import attach_station_embedding

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

    def handleQuanTracRequest(self, quantrac_request, target_col):
        # Logger
        func_name = "RequestHandler.handleQuanTracRequet()"
        info("{}: is called", func_name)

        # Read the QuanTrac request
        df_quantrac = pd.DataFrame(quantrac_request.data.model_dump())

        # Convert "date" column to datetime
        df_quantrac["date"] = pd.to_datetime(df_quantrac["date"])

        # Rename columns
        column_mapper = {}
        for col in df_quantrac.columns:
            if col in ["date", "station_id"]:
                pass
            elif col in ["temperature", "humid"]:
                column_mapper[col] = f"{col.capitalize()}_quantrac"
            else:
                column_mapper[col] = f"{col.upper()}_quantrac"
        df_quantrac = df_quantrac.rename(column_mapper, axis=1)

        # Print the dataset
        info("{}: df_quantrac: \n{}", func_name, df_quantrac)
        info("{}: df_quantrac.columns: {}", func_name, list(df_quantrac.columns))

        # Define information
        BASE_FEATURE_COLS = [
            "NO2_quantrac",
            "PM25_quantrac",
            "O3_quantrac",
            "CO_quantrac",
            "Temperature_quantrac",
            "Humid_quantrac",
        ]
        LAG_STEPS = [3, 6, 12, 24, 48, 72]
        ROLL_WINDOWS = [3, 6, 12, 24, 48, 72]
        HORIZONS = [1, 24, 48, 72]
        model_path = os.path.join("models", "lightgbm")

        # Processing
        df_time_feats = add_time_features(df_quantrac)
        df_lag_feats = add_lag_features(df_time_feats, group_col="station_id", target_cols=BASE_FEATURE_COLS, lag_steps=LAG_STEPS)
        df_rolling_feats = add_rolling_features(df_lag_feats, group_col="station_id", target_cols=[target_col], windows=ROLL_WINDOWS)

        info("{}: df_rolling_feats: \n{}", func_name, df_rolling_feats)
        info("{}: df_rolling_feats.columns: {}", func_name, list(df_rolling_feats.columns))

        # Add station embedding
        with open(os.path.join(model_path, f"{target_col}_station_embedding.pkl"), "rb") as no2_emb_file:
            station_to_embedding = pickle.load(no2_emb_file)
        df_embedded, _ = attach_station_embedding(df_rolling_feats, station_to_embedding, station_col="station_id")

        info("{}: df_embedded: \n{}", func_name, df_embedded)
        info("{}: df_embedded.columns: {}", func_name, list(df_embedded.columns))

        # Drop station and date before prediciting
        df_final = df_embedded.drop(columns=["station_id", "date", target_col])

        predicted_values = []
        for horizon_h in HORIZONS:
            model = lgb.Booster(model_file=os.path.join(model_path, f"{target_col}_lightgbm_{horizon_h}h"))
            predicted_value = model.predict(df_final, num_iteration=getattr(model, "best_iteration", None))[0]
            print(f"Horizon: {horizon_h}h - predicted: {predicted_value}")
            predicted_values.append(predicted_value)

        # Proceed the result
        return QuanTracResponse(data=predicted_values)

