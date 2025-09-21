import os

from keras.models import load_model

class ReductionModel:
    def __init__(self) -> None:
        self.__model = load_model(os.path.join("models", "reduction", "aod_LSTMSeq2SeqReduction_13_features_encoder.keras"))

    def predict(self, vienthamdata):
        print(self.__model)
