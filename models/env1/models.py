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



class Constant:
    def __init__(self, constant):
        self.constant = constant

    def train(self, history):
        pass

    def predict(self, input_data):
        return self.constant

    def store_reward(self, reward):
        pass


class CorrelationAgent:
    def __init__(self):
        pass

    def train(self, history):

        self.data = pd.DataFrame(np.array([history['item_history'],
                                           history['user_history'],
                                           history['rating_history']]).T,
                    columns=['item', 'user', 'rating'])

        self.ratings = self.data.pivot_table(columns=['item'],
                                        index=['user'], values=['rating']).fillna(0).values

        self.sim = self.similarity(self.ratings)
        self.predicted_table = self.predict_ratings(self.ratings, self.sim, self.phi)

        self.user_history = []
        self.item_history = []
        self.predicted_scores = []
        self.true_scores = []


    def predict(self, input_data):
        self.user_history.append(input_data['next_user'])
        self.item_history.append(input_data['next_item'])

        self.last_pred = self.predicted_table[self.user_history[-1], self.item_history[-1]]

        return self.last_pred

    def store_reward(self, true_score):
        self.predicted_scores.append(self.last_pred)
        self.true_scores.append(true_score)

    def similarity(self, ratings):
        # vecteur contenant pour chaque utilisateur le nombre de notes données
        r_user = (ratings>0).sum(axis=1)

        # vecteur contenant pour chaque utilisateur la moyenne des notes données
        m_user = np.divide(ratings.sum(axis=1) , r_user, where=r_user!=0)

        # Notes recentrées par la moyenne par utilisateur : chaque ligne i contient le vecteur \bar r_i
        ratings_ctr = ratings.T - ((ratings.T!=0) * m_user)
        ratings_ctr = ratings_ctr.T

        # Matrice de Gram, contenant les produits scalaires
        sim = ratings_ctr.dot(ratings_ctr.T)

        # Renormalisation
        norms = np.array([np.sqrt(np.diagonal(sim))])
        #assert np.all(norms != 0)

        sim = np.divide(sim , norms, where=norms!=0)
        sim = np.divide(sim, norms.T, where=norms.T != 0)
        # (En numpy, diviser une matrice par un vecteur ligne (resp. colonne)
        # revient à diviser chaque ligne (resp. colonne) terme à terme par les éléments du vecteur)

        return sim

    def phi(self, x):
        return x * (x > 0)

    def predict_ratings(self, ratings,sim,phi=(lambda x:x)):
        wsum_sim = np.abs(phi(sim)).dot(ratings>0)
#         assert np.all(wsum_sim != 0)
        return np.divide(phi(sim).dot(ratings) , wsum_sim, where= wsum_sim!=0)


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
        self.user_history.append(input_data['next_user'])
        self.item_history.append(input_data['next_user'])

        self.last_pred = self.algo.predict(uid = self.user_history[-1], iid = self.item_history[-1]).est
        return self.last_pred


    def store_reward(self, true_score):
        self.predicted_scores.append(self.last_pred)
        self.true_scores.append(true_score)



from keras.layers import Input, Embedding, Flatten, Dot
from keras.models import Model

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
                    batch_size=100, epochs=40, validation_split=0.1,
                    shuffle=True)

    def _predict(self):
#         return np.clip(self.model.predict([[self.user_history[-1]],
#                                            [self.item_history[-1]]]), a_min=1, a_max=5)
          return float(self.model.predict([[self.user_history[-1]],
                                            [self.item_history[-1]]]))

    def predict(self, input_data):
        self.user_history.append(input_data['next_user'])
        self.item_history.append(input_data['next_item'])

        self.step_counter += 1

        self.last_pred = self._predict()

        return self.last_pred

    def store_reward(self, true_score):

        self.predicted_scores.append(self.last_pred)
        self.true_scores.append(true_score)

        if self.retrain:
            if self.step_counter % self.retrain_after == 0:

                new_memory = np.array(list(zip(self.user_history[:-1],
                                                               self.item_history[:-1],
                                                               self.true_scores)))

                self.model.fit([new_memory[:, 1], new_memory[:, 0]], new_memory[:, 2],
                        batch_size=self.retrain_after//4, epochs=10, validation_split=0.1,
                        shuffle=True)    
