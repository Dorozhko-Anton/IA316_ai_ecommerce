import numpy as np

class RecommenderModel():
    def __init__(self, history):
        kwargs = {
            'embedding_size':15,
            'loss_name' : 'mse',
            'retrain':False,
            'retrain_after':20
        }
        self.model = DeepMFAgent(**kwargs)
        self.model.train(history)

    def predict(self, input_data):
        prediction = self.model.predict(input_data)
        if input_data['rating'] is not None:
            self.model.store_reward(input_data['rating'])
        return prediction

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot
from tensorflow.keras.models import Model

class DeepMFAgent:
    def __init__(self, embedding_size = 30, loss_name='mse', retrain=False, retrain_after=20):
        self.embedding_size = embedding_size
        self.loss_name = loss_name
        self.retrain = retrain
        self.retrain_after = retrain_after


    def train(self, history):
        self.n_user = history['nb_users']
        self.n_item = history['nb_items']
        self.step_counter = 0

        self.data = pd.DataFrame(np.array([history['item_history'],
                                           history['user_history'],
                                           history['rating_history']]).T,
                    columns=['item', 'user', 'rating'])

        self.memory = self.data.values

        self._create_model()
        self._train()

        self.user_history = []
        self.item_history = []
        self.predicted_scores = []
        self.true_scores = []

    def _create_model(self):
        user_id_input = Input(shape=[1],name='user')
        item_id_input = Input(shape=[1], name='item')


        user_embedding = Embedding(output_dim=self.embedding_size, input_dim=self.n_user + 1,
                                   input_length=1, name='user_embedding')(user_id_input)

        item_embedding = Embedding(output_dim=self.embedding_size, input_dim=self.n_item + 1,
                                   input_length=1, name='item_embedding')(item_id_input)

        # reshape from shape: (batch_size, input_length, embedding_size)
        # to shape: (batch_size, input_length * embedding_size) which is
        # equal to shape: (batch_size, embedding_size)
        user_vecs = Flatten()(user_embedding)
        item_vecs = Flatten()(item_embedding)

        y = Dot(axes=1)([user_vecs, item_vecs])

        model = Model(inputs=[user_id_input, item_id_input], outputs=y)
        model.compile(optimizer='adam', loss=self.loss_name)
        self.model = model

    def _train(self):
        history = self.model.fit([self.memory[:, 1], self.memory[:, 0]], self.memory[:, 2],
                    batch_size=100, epochs=80, validation_split=0.1,
                    shuffle=True)

    def _predict(self):
#         return np.clip(self.model.predict([[self.user_history[-1]],
#                                            [self.item_history[-1]]]), a_min=1, a_max=5)
          return float(self.model.predict([[self.user_history[-1]],
                                            [self.item_history[-1]]]))

    def predict(self, input_data):
        self.user_history.append(input_data['user'])
        self.item_history.append(input_data['item'])

        self.step_counter += 1

        self.last_pred = self._predict()

        return self.last_pred

    def store_reward(self, true_score):

        self.predicted_scores.append(self.last_pred)
        self.true_scores.append(true_score)

        if self.retrain:
            if self.step_counter % self.retrain_after == 0:

                new_memory = np.array(list(zip(self.item_history,
                                               self.user_history,
                                                               self.true_scores)))

                self.model.fit([new_memory[:, 1], new_memory[:, 0]], new_memory[:, 2],
                        batch_size=self.retrain_after//4, epochs=1, validation_split=0.1,
                        shuffle=True)
