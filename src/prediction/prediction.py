from keras.models import load_model

class PredictionModel:
    def __init__(self, model_path):
        self.__model = load_model(model_path)

    def predict(self, data):
        print(self.__model.summary())
        return self.__model.predict(data)

