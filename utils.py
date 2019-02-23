
import requests
import matplotlib.pylab as plt
import numpy as np

class ENV3:
    def __init__(self, url, key):
        self.url = url
        self.key = key

    def reset(self):
        r = requests.get(url=self.url+'/reset',
                     params = {
                         'user_id':self.key,
                     })
        try:
            return r.json()
        except:
            print(r.content)
            raise

    def predict(self, predicted_score):
        r = requests.get(url=self.url + '/predict',
                     params = {
                         'user_id':self.key,
                         'recommended_item':predicted_score
                     })
        try:
            return r.json()
        except:
            print(r.content)
            raise


import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_model_trajectories(model_res_dict):
    #f, (ax1) = plt.subplots(1, 1, sharey=False, figsize=(16, 8))
    plt.figure(figsize=(12, 10))
    cmap = get_cmap(len(model_res_dict) + 1)

    sorted_model_name_scores = []

    for i, (model_name, model_res) in enumerate(model_res_dict.items()):

        cumsum_rewards = []
        conversion_rates = []
        for rewards in model_res:
            cumsum_reward = np.cumsum(rewards)
            cumsum_rewards.append(cumsum_reward)
            conversion_rates.append(np.mean(np.array(rewards)>0))


        mean = np.mean(cumsum_rewards, axis=0)
        ub = np.min(cumsum_rewards, axis=0)
        lb = np.max(cumsum_rewards, axis=0)


        plt.fill_between(range(mean.shape[0]), ub, lb,
                         color=cmap(i), alpha=.3)
        # plot the mean on top
        score = mean[-1]
        mean_conversion_rate = np.mean(np.array(conversion_rates))

        plt.plot(mean, color=cmap(i),
                 label='%s-conversion=%.2f-score=%d' % (model_name, mean_conversion_rate, score))

        sorted_model_name_scores.append((model_name, int(score), mean_conversion_rate))

    plt.legend(loc=2, prop={'size': 15});

    return pd.DataFrame(sorted_model_name_scores,
                        columns=['Model', 'Score', 'Conversion Rate']).sort_values('Score', ascending=False)



def plot_model_moving_mse_mae(model_res_dict):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(16, 10))
    #plt.figure(figsize=(12, 10))
    cmap = get_cmap(len(model_res_dict) + 1)

    sorted_model_name_scores = []

    for i, (model_name, model_res) in enumerate(model_res_dict.items()):

        mses = []
        maes = []

        moving_mses = []
        moving_maes = []

        for j in range(len(model_res['true'])):

            pred = np.array(model_res['predicted'][j])
            true = np.array(model_res['true'][j])

            mse = np.sum((pred- true)**2)/ len(pred)
            mae = np.mean(np.abs(pred - true))

            mses.append(mse)
            maes.append(mae)

            moving_mse = [s/(n+1) for n, s in enumerate(np.cumsum((pred - true)**2)) ]
            moving_mae = [s/(n+1) for n, s in enumerate(np.cumsum(np.abs(pred - true))) ]

            moving_mses.append(moving_mse)
            moving_maes.append(moving_mae)

        mean_mse = np.mean(mses)
        mean_mae = np.mean(maes)


        # plot mse on ax1
        mean = np.mean(moving_mses, axis=0)
        ub = np.min(moving_mses, axis=0)
        lb = np.max(moving_mses, axis=0)

        ax1.fill_between(range(mean.shape[0]), ub, lb,
                         color=cmap(i), alpha=.3)
        # plot the mean on top

        ax1.plot(mean, color=cmap(i),
                 label='%s-mse=%.2f' % (model_name, mean_mse))

        # plot mae on ax2
        mean = np.mean(moving_maes, axis=0)
        ub = np.min(moving_maes, axis=0)
        lb = np.max(moving_maes, axis=0)

        ax2.fill_between(range(mean.shape[0]), ub, lb,
                         color=cmap(i), alpha=.3)
        # plot the mean on top

        ax2.plot(mean, color=cmap(i),
                 label='%s-mae=%.2f' % (model_name, mean_mae))

        sorted_model_name_scores.append((model_name, mean_mae, mean_mse))

    ax1.legend(loc=1, prop={'size': 15});
    ax2.legend(loc=1, prop={'size': 15});

    return pd.DataFrame(sorted_model_name_scores,
                        columns=['Model', 'mae', 'mse']).sort_values('mse', ascending=True)
