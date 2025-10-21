import glob

from keras.models import load_model
from src.preprocessor.data_utils import reframePastFuture, padPastFuture
from src.logger.logger import info

class ReductionModel:
    def __init__(self, data,
                 data_type="aod",
                 n_past=7,
                 n_future=1,
                 reduction_model_name="LSTMSeq2SeqReduction"):
        # Logger
        func_name = "ReductionModel.__init__()"
        info("{}: is called", func_name)

        self.__data = data
        self.__n_past = n_past
        self.__n_future = n_future
        self.__reduction_model_name = reduction_model_name
        self.__data_type = data_type

        # Get the model pattern to search
        model_pattern = "_".join([self.__data_type,
                                  self.__reduction_model_name,
                                  f"{self.__n_future}_future"])

        # Search and load model
        model_path = glob.glob(f"models/reduction/{model_pattern}*.keras")[0]
        self.__model = load_model(model_path)
        info("{}: loaded model {}", func_name, model_path)


    def encode(self):
        # Logger
        func_name = "ReductionModel.encode()"
        info("{}: is called", func_name)

        # Pad and reframe data
        padded_data = padPastFuture(self.__data, self.__n_past, self.__n_future)
        info("{}: padded_data.shape = {}", func_name, padded_data.shape)
        reframed_data, _ = reframePastFuture(padded_data, self.__n_past, self.__n_future)
        info("{}: reframed_data.shape = {}", func_name, reframed_data.shape)
        info("{}: reframed_data = \n{}", func_name, reframed_data)

        # Prediction
        print(self.__model.summary())
        return self.__model.predict(reframed_data)

