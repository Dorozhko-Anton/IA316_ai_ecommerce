

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense
from tensorflow.keras.models import Model

class DeepMFCovariatesAgent:
    
    def __init__(self, key, history, embedding_size = 30, loss_name='mse', retrain=False, retrain_after=20):
        self.key = key
        self.n_user = history['nb_users']
        self.n_item = history['nb_items']
        self.embedding_size = embedding_size
        self.loss_name = loss_name
        self.retrain = retrain
        self.retrain_after = retrain_after
        self.step_counter = 0
        self.variables_size = 5


        self.memory = np.vstack([history['item_history'], history['user_history'], history['rating_history']]).T
        self.variables = np.array(history['variables_history'])

        self._create_model()
        self._train()

        self.user_history = []
        self.item_history = []
        self.next_variables = []
        self.predicted_scores = []
        self.true_scores = []

        self.user_history.append(history['next_user'])
        self.item_history.append(history['next_user'])
        self.next_variables.append(history['next_variables'])

    def _create_model(self):
        user_id_input = Input(shape=[1],name='user')
        item_id_input = Input(shape=[1], name='item')

        variables = Input(shape=[self.variables_size], name='variables')


        user_embedding = Embedding(output_dim=self.embedding_size, input_dim=self.n_user + 1,
                                   input_length=1, name='user_embedding')(user_id_input)

        item_embedding = Embedding(output_dim=self.embedding_size, input_dim=self.n_item + 1,
                                   input_length=1, name='item_embedding')(item_id_input)

        user_vecs = Flatten()(user_embedding)
        item_vecs = Flatten()(item_embedding)


        features = Concatenate(axis=-1)([user_vecs, item_vecs, variables])

        net = Dense(units=50)(features)
        net = Dense(units=50)(net)

        y = Dense(units=1)(net)

        model = Model(inputs=[user_id_input, item_id_input, variables], outputs=y)
        model.compile(optimizer='adam', loss=self.loss_name)
        self.model = model

    def _train(self):
        history = self.model.fit([self.memory[:, 1], self.memory[:, 0], self.variables], self.memory[:, 2],
                    batch_size=32, epochs=60, validation_split=0.1,
                    shuffle=True)

    def _predict(self):
        return float(self.model.predict([
                                         [self.user_history[-1]],
                                         [self.item_history[-1]],
                                         [self.next_variables[-1]]
                                        ]))

    def step(self, input_data):
        self.step_counter += 1
        self.true_scores.append(input_data['rating'])

        pred = self._predict()

        self.predicted_scores.append(pred)
        self.user_history.append(input_data['next_user'])
        self.item_history.append(input_data['next_item'])
        self.next_variables.append(input_data['next_variables'])

#         if self.retrain:
#             if self.step_counter > 0 and self.step_counter % self.retrain_after == 0:

#                 new_memory = np.array(list(zip(self.item_history[:-1],
#                                                                self.user_history[:-1],
#                                                                self.true_scores)))

#                 self.model.fit([new_memory[:, 1], new_memory[:, 0], np.array(self.next_variables[:-1])], new_memory[:, 2],
#                         batch_size=self.retrain_after//4, epochs=10, validation_split=0.1,
#                         shuffle=True)

        return pred
