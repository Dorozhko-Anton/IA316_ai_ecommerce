import numpy as np

class Model():
    def __init__(self):
        pass

    def train(self, history):
        raise NotImplemented

    def predict(self, input_data):
        raise NotImplemented

class Random(Model):
    def __init__(self):
        pass

    def train(self, history):
        pass
    def predict(self, input_data):
        self.last_state = input_data['state']
        prediction = np.random.choice(range(len(self.last_state)))
        return prediction

    def store_reward(self, reward):
        self.last_reward     = reward


class MostExpensive(Model):
    def __init__(self):
        pass

    def train(self, history):
        pass
    def predict(self, input_data):
        self.last_state = input_data['state']

        prices = np.array(self.last_state)[:, 2]

        prediction = np.argmax(prices)
        return prediction

    def store_reward(self, reward):
        self.last_reward     = reward


from sklearn.linear_model import LogisticRegression

class LogReg(Model):
    def __init__(self, price_reweight=False, retrain=None):
        self.price_reweight = price_reweight
        self.retrain = retrain
        self.logreg = None

        self.step = 0

    def train(self, history):
        self.X = []
        self.Y = []
        for a, r, items in zip(history['action_history'],
                               history['rewards_history'],
                               history['state_history']):

            x = items[a][2:]
            y = r > 0

            self.X.append(x)
            self.Y.append(y)

        self.logreg = LogisticRegression()
        self.logreg.fit(self.X, self.Y)

    def predict(self, input_data):
        self.step += 1
        self.last_state = np.array(input_data['state'])


        xs = self.last_state[:, 2:]

        scores = self.logreg.predict_proba(xs)[:, 1]

        if self.price_reweight:
            weights = np.array(self.last_state)[:, 2]
            scores = scores * weights

        prediction = np.argmax(scores)
        self.last_prediction = prediction

        return prediction

    def store_reward(self, reward):
        self.last_reward     = reward

        self.X.append(self.last_state[:, 2:][self.last_prediction])
        self.Y.append(self.last_reward > 0)
        if self.retrain is not None:
            if self.step % self.retrain == 0:
                self.logreg.fit(self.X, self.Y)


##
from sklearn.svm import SVC
class SVCModel(Model):
    def __init__(self, price_reweight=False, retrain=None):
        self.price_reweight = price_reweight
        self.retrain = retrain
        self.model = None

        self.step = 0

    def train(self, history):
        self.X = []
        self.Y = []
        for a, r, items in zip(history['action_history'],
                               history['rewards_history'],
                               history['state_history']):

            x = items[a][2:]
            y = r > 0

            self.X.append(x)
            self.Y.append(y)

        self.model = SVC(probability=True)
        self.model.fit(self.X, self.Y)

    def predict(self, input_data):
        self.step += 1
        self.last_state = np.array(input_data['state'])


        xs = self.last_state[:, 2:]

        scores = self.model.predict_proba(xs)[:, 1]

        if self.price_reweight:
            weights = np.array(self.last_state)[:, 2]
            scores = scores * weights

        prediction = np.argmax(scores)
        self.last_prediction = prediction

        return prediction

    def store_reward(self, reward):
        self.last_reward     = reward

        self.X.append(self.last_state[:, 2:][self.last_prediction])
        self.Y.append(self.last_reward > 0)
        if self.retrain is not None:
            if self.step % self.retrain == 0:
                self.model.fit(self.X, self.Y)
