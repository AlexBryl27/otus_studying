import pickle
import contract
import numpy as np


class ClassificationAPI:

    def __init__(self, model_path):

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def process(self, request):

        arr = np.asarray([float(x) for x in request]).reshape(1, -1)
        prediction = self.model.predict(arr)

        return contract.Response(disease=int(prediction[0]))
