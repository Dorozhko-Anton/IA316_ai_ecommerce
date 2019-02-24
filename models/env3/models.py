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
        self.last_r     = reward


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
        self.last_r     = reward


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
        self.last_r     = reward

        self.X.append(self.last_state[:, 2:][self.last_prediction])
        self.Y.append(self.last_reward > 0)
        if retrain is not None:
            if self.step % self.retrain == 0:
                self.logreg.fit(self.X, self.Y)


##
from sklearn.svm import SVC
class SVCModel(Model):
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

        self.logreg = SVC(probability=True)
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
        self.last_r     = reward

        self.X.append(self.last_state[:, 2:][self.last_prediction])
        self.Y.append(self.last_reward > 0)
        if retrain is not None:
            if self.step % self.retrain == 0:
                self.logreg.fit(self.X, self.Y)

import tensorflow as tf
from keras.models import Model as KerasModel, Sequential
from keras.layers import Embedding, Flatten, Input, Dense, Dropout
from keras.layers import Concatenate, Lambda
from keras.regularizers import l2
from collections import defaultdict

def identity_loss(y_true, y_pred):
    """Ignore y_true and return the mean of y_pred

    This is a hack to work-around the design of the Keras API that is
    not really suited to train networks with a triplet loss by default.
    """
    return tf.reduce_mean(y_pred + 0 * y_true)

def margin_comparator_loss(inputs):
    """Comparator loss for a pair of precomputed similarities"""
    positive_pair_sim, negative_pair_sim, positive_reward, negative_reward = inputs
    return tf.maximum(negative_pair_sim - positive_pair_sim + positive_reward - negative_reward, 0)

def make_interaction_mlp(input_dim, n_hidden=1, hidden_size=64,
                         dropout=0, l2_reg=None):
    """Build the shared multi layer perceptron"""
    mlp = Sequential()
    if n_hidden == 0:
        # Plug the output unit directly: this is a simple
        # linear regression model. Not dropout required.
        mlp.add(Dense(1, input_dim=input_dim,
                      activation='relu', kernel_regularizer=l2_reg))
    else:
        mlp.add(Dense(hidden_size, input_dim=input_dim,
                      activation='relu', kernel_regularizer=l2_reg))
        mlp.add(Dropout(dropout))
        for i in range(n_hidden - 1):
            mlp.add(Dense(hidden_size, activation='relu',
                          W_regularizer=l2_reg))
            mlp.add(Dropout(dropout))
        mlp.add(Dense(1, activation='relu', kernel_regularizer=l2_reg))
    return mlp

def build_models(n_users, n_items, user_dim=32, item_dim=64,
                 n_hidden=1, hidden_size=64, dropout=0, l2_reg=0):
    """Build models to train a deep triplet network"""
    user_input = Input((1,), name='user_input')
    positive_item_id_input = Input((1,), name='positive_item_id_input')
    positive_item_variables_input = Input((5,), name='positive_item_variables_input')
    negative_item_id_input = Input((1,), name='negative_item_id_input')
    negative_item_variables_input = Input((5,), name='negative_item_variables_input')

    positive_item_reward_input = Input((1,), name='positive_item_reward_input')
    negative_item_reward_input = Input((1,), name='negative_item_reward_input')

    l2_reg = None if l2_reg == 0 else l2(l2_reg)
    user_layer = Embedding(n_users, user_dim, input_length=1,
                           name='user_embedding', embeddings_regularizer=l2_reg)

    # The following embedding parameters will be shared to encode both
    # the positive and negative items.
    item_layer = Embedding(n_items, item_dim, input_length=1,
                           name="item_embedding", embeddings_regularizer=l2_reg)

    user_embedding = Flatten()(user_layer(user_input))
    positive_item_embedding = Flatten()(item_layer(positive_item_id_input))
    negative_item_embedding = Flatten()(item_layer(negative_item_id_input))


    # Similarity computation between embeddings using a MLP similarity
    positive_embeddings_tuple = Concatenate(name="positive_embeddings_tuple")(
        [user_embedding, positive_item_embedding, positive_item_variables_input])
    positive_embeddings_tuple = Dropout(dropout)(positive_embeddings_tuple)
    negative_embeddings_tuple = Concatenate(name="negative_embeddings_tuple")(
        [user_embedding, negative_item_embedding, negative_item_variables_input])
    negative_embeddings_tuple = Dropout(dropout)(negative_embeddings_tuple)

    # Instanciate the shared similarity architecture
    interaction_layers = make_interaction_mlp(
        user_dim + item_dim + 5, n_hidden=n_hidden, hidden_size=hidden_size,
        dropout=dropout, l2_reg=l2_reg)

    positive_similarity = interaction_layers(positive_embeddings_tuple)
    negative_similarity = interaction_layers(negative_embeddings_tuple)
    reward_difference = positive_item_reward_input - negative_item_reward_input

    # The triplet network model, only used for training
    triplet_loss = Lambda(margin_comparator_loss, output_shape=(1,),
                          name='comparator_loss')(
        [positive_similarity,
         negative_similarity,
         positive_item_reward_input,
         negative_item_reward_input])

    deep_triplet_model = KerasModel(inputs=[user_input,
                                       positive_item_id_input,
                                       positive_item_variables_input,
                                       positive_item_reward_input,
                                       negative_item_id_input,
                                       negative_item_variables_input,
                                       negative_item_reward_input],
                               outputs=[triplet_loss])

    # The match-score model, only used at inference
    deep_match_model = KerasModel(inputs=[user_input,
                                     positive_item_id_input,
                                     positive_item_variables_input],
                             outputs=[positive_similarity])

    return deep_match_model, deep_triplet_model

def build_user_data(history):
    state_history = history["state_history"]
    action_history = history["action_history"]
    rewards_history = history["rewards_history"]
    pos_data = defaultdict(list)
    neg_data = defaultdict(list)
    neutral_data = defaultdict(list)

    for s, a, r in zip(state_history, action_history, rewards_history):
        user = s[0][0]
        if r > 0:
            pos_data[user].append(s[a])
        else:
            neg_data[user].append(s[a])
        for i in range(len(s)):
            if i != a:
                neutral_data[user].append(s[i])

    for user in pos_data:
        pos_data[user] = sorted(pos_data[user], key=lambda x: x[2])

    return pos_data, neg_data, neutral_data


def sample_triplets(pos_data, neg_data, neutral_data):
    user_ids = []
    positive_item_ids = []
    positive_item_variables = []
    positive_item_rewards = []
    negative_item_ids = []
    negative_item_variables = []
    negative_item_rewards = []
    for user_id, pos_list in pos_data.items():
        if len(neg_data[user_id]) != 0:
            for pos in pos_list:
                user_ids.append(user_id)
                positive_item_ids.append(pos[1])
                positive_item_variables.append(pos[3:])
                positive_item_rewards.append(pos[2])
                neg_i = np.random.randint(0, len(neg_data[user_id]))
                neg = neg_data[user_id][neg_i]
                negative_item_ids.append(neg[1])
                negative_item_variables.append(neg[3:])
                negative_item_rewards.append(0)
        if len(neutral_data[user_id]) != 0:
            for pos in pos_list:
                user_ids.append(user_id)
                positive_item_ids.append(pos[1])
                positive_item_variables.append(pos[3:])
                positive_item_rewards.append(pos[2])
                neg_i = np.random.randint(0, len(neutral_data[user_id]))
                neg = neutral_data[user_id][neg_i]
                negative_item_ids.append(neg[1])
                negative_item_variables.append(neg[3:])
                negative_item_rewards.append(0)
        if len(pos_list) >= 2:
            for i, pos in enumerate(pos_list[1:]):
                user_ids.append(user_id)
                positive_item_ids.append(pos[1])
                positive_item_variables.append(pos[3:])
                positive_item_rewards.append(pos[2])
                neg_i = np.random.randint(0, i+1)
                neg = pos_list[neg_i]
                negative_item_ids.append(neg[1])
                negative_item_variables.append(neg[3:])
                negative_item_rewards.append(neg[2])

    return [user_ids, positive_item_ids, positive_item_variables, positive_item_rewards,
            negative_item_ids, negative_item_variables, negative_item_rewards]

class TripletLossReward(Model):
    def __init__(self, alpha = 0, user_dim=10, item_dim=10, n_hidden=1,
                 hidden_size=30, dropout=0, l2_reg=0):
        self.alpha = alpha
        self.parameters = {
            "user_dim": user_dim,
            "item_dim": item_dim,
            "n_hidden": n_hidden,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "l2_reg": l2_reg
        }

    def train(self, history):
        nb_users = history["nb_users"]
        nb_items = history["nb_items"]
        self.deep_match_model, self.deep_triplet_model = \
            build_models(nb_users, nb_items, **self.parameters)
        self.deep_triplet_model.compile(loss=identity_loss, optimizer="adam")
        pos_data, neg_data, neutral_data = build_user_data(history)
        for i in range(500):
            # Sample new negatives to build different triplets at each epoch
            triplet_inputs = sample_triplets(pos_data, neg_data, neutral_data)

            fake_y = np.ones_like(triplet_inputs[0])

            # Fit the model incrementally by doing a single pass over the
            # sampled triplets.
            self.deep_triplet_model.fit(triplet_inputs, fake_y, shuffle=True,
                                        batch_size=16, epochs=1, verbose=0)

    def predict(self, input_data):
        next_state = input_data["state"]
        #self.last_state = next_state
        scores = np.zeros(len(next_state))
        prices = np.zeros(len(next_state))
        for j, item_spec in enumerate(next_state):
            inputs = [np.array(item_spec[0]).reshape((1, 1)),
                      np.array(item_spec[1]).reshape((1, 1)),
                      np.array(item_spec[3:]).reshape((1, 5))]
            scores[j] = self.deep_match_model.predict(inputs)
            prices[j] = item_spec[2]
        action = np.argmax(scores + self.alpha * prices)
        return action


    def store_reward(self, reward):
        self.last_r     = reward
