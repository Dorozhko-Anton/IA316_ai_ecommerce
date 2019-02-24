import numpy as np

class RecommenderModel():
    def __init__(self, history):
        kwargs = {
        }
        self.model = SVDAgent(**kwargs)
        self.model.train(history)

    def predict(self, input_data):
        prediction = self.model.predict(input_data)
        return prediction



import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD


class SVDAgent:
    def __init__(self):
        pass

    def train(self, history):

        self.data = pd.DataFrame(np.array([history['item_history'],
                                           history['user_history'],
                                           history['rating_history']]).T,
                    columns=['item', 'user', 'rating'])

        reader = Reader(rating_scale=(1, 5))
        train_spr = Dataset.load_from_df(self.data[['user','item','rating']],reader).build_full_trainset()
        self.algo = SVD(n_factors=20)
        self.algo.fit(train_spr)


        self.user_history = []
        self.item_history = []
        self.predicted_scores = []
        self.true_scores = []


    def predict(self, input_data):
        self.user_history.append(input_data['user'])
        self.item_history.append(input_data['item'])

        self.last_pred = self.algo.predict(uid = self.user_history[-1], iid = self.item_history[-1]).est
        return self.last_pred


    def store_reward(self, true_score):
        self.predicted_scores.append(self.last_pred)
        self.true_scores.append(true_score)
