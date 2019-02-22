from DeepMFModel import DeepMFCovariatesAgent

class RecommenderModel():
    def __init__(self, key, history):
        kwargs = {
            'embedding_size' : 30,
            'loss_name' : 'mse',
        }
        self.model = DeepMFCovariatesAgent(key, history, **kwargs)

    def predict(self, input_data):
        prediction = self.model.step(input_data)
        return prediction
