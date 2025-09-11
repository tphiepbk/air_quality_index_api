from time import sleep

class Prediction:
    def __init__(self):
        pass

    def dummy_action(self, data):
        print("sleeping")
        sleep(5)
        return {"data": data}
