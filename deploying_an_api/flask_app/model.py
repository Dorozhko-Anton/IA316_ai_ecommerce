import numpy as np

class RecommenderModel():
    def __init__(self, history):
        kwargs = {
        }
        self.model = Random(**kwargs)
        self.model.train(history)

    def predict(self, input_data):
        prediction = self.model.predict(input_data)
        return prediction

class Random:
    def __init__(self):
        pass

    def train(self, history):
        pass

    def predict(self, input_data):
        self.last_pred = np.random.choice([1, 2, 3, 4, 5])
        return self.last_pred

    def store_reward(self, reward):
        pass
