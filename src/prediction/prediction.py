import glob

import pandas as pd

from keras.models import load_model

from src.logger.logger import info
from src.preprocessor.data_utils import reframePastFuture
from src.preprocessor.preprocessor import Preprocessor

# Now we only have one model to predict which is LSTM
# But we have different reduction model and n_future
class PredictionModel:
    def __init__(self, feature_data, label_data,
                 data_type="aod",
                 n_past=7,
                 n_future=1,
                 reduction_model_name="LSTMSeq2SeqReduction",
                 prediction_model_name="LSTMPrediction"):
        # Logger
        func_name = "PredictionModel.__init__()"
        info("{}: is called", func_name)

        self.__feature_data = feature_data
        self.__label_data = label_data
        self.__n_past = n_past
        self.__n_future = n_future
        self.__reduction_model_name = reduction_model_name
        self.__prediction_model_name = prediction_model_name
        self.__data_type = data_type

        # Get the model pattern to search
        model_pattern = "_".join([self.__data_type,
                                  self.__prediction_model_name,
                                  f"{self.__n_future}_future",
                                  "with",
                                  self.__reduction_model_name])

        # Search and load model
        model_path = glob.glob(f"models/prediction/{model_pattern}*.keras")[0]
        self.__model = load_model(model_path)
        info("{}: loaded model {}", func_name, model_path)

    def predict(self):
        # Logger
        func_name = "PredictionModel.predict()"
        info("{}: is called", func_name)

        # Combined feature and label
        combined_data = pd.concat((pd.DataFrame(self.__feature_data), self.__label_data), axis=1)
        info("{}: combined_data = \n{}", func_name, combined_data)

        # Reframe data
        reframed_combined_data, _ = reframePastFuture(combined_data, self.__n_past, self.__n_future)
        info("{}: reframed_combined_data = \n{}", func_name, reframed_combined_data)

        # Prediction
        print(self.__model.summary())
        prediced_values = self.__model.predict(reframed_combined_data)
        info("predicted_values.type {}", type(prediced_values))
        info("predicted_values.shape {}", prediced_values.shape)

        # Reshape and inverse transform
        # The predicted values have shape (1, n_future, 1), reshape it
        predicted_values_reshaped = prediced_values.reshape(-1, 1)
        inverted_predicted_values = Preprocessor(data_type=self.__data_type).inverse_transform(predicted_values_reshaped)

        # Results
        return inverted_predicted_values

