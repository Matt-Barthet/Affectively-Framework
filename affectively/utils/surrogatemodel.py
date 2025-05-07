from typing import Literal

import importlib_resources
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class KNNSurrogateModel:
    def __init__(self,
                 k: int,
                 game: Literal["Heist", "Pirates", "Solid"],
                 cluster=0):
        """
		Generate a KNN surrogate model.

		Args:
			k: The number of neighbors.
			game: The game name.
		"""
        self.x_train = None
        self.y_train = None
        self.k = k
        self.scaler = MinMaxScaler()
        self.game = game
        self.cluster = cluster
        self.target_behavior, self.target_arousal = [], []
        self.surrogate_length = 0
        self.columns = []
        self.load_data()

    def __call__(self, state):
        distances = np.array(np.sqrt(np.sum((state - self.x_train) ** 2, axis=1)))
        k_indices = np.array(np.argsort(distances)[:self.k])
        k_labels = np.array(self.y_train)[k_indices]
        if self.k == 1:
            return self.y_train[k_indices][0]
        else:
            weights = 1 / (distances[k_indices] + 1e-5)
            weighted_sum = np.sum(weights * k_labels)
            total_weights = np.sum(weights)
            predicted_class = weighted_sum / total_weights
        # print(k_labels, predicted_class)
        return predicted_class, k_indices

    def load_and_clean(self, filename: str, preference: bool):
        data = pd.read_csv(filename)
        print(f"Before cleaning {len(data)}", end=" ")

        if self.cluster > 0:
            cluster_members = pd.read_csv(f"./affectively/datasets/{self.game}_cluster_book.csv")
            cluster_members = cluster_members[cluster_members['Cluster'] == self.cluster]
            data = data[data['[control]player_id'].isin(cluster_members['[control]player_id'])]

        data.drop(columns=['[control]player_id'], inplace=True)

        if preference:
            arousals = data['[output]ranking'].values
            data = data.drop(columns=['[output]ranking'])
        else:
            arousals = data['[output]arousal'].values
            data = data.drop(columns=['[output]arousal'])
            self.surrogate_length = len(data.columns) 

        self.columns = list(data.columns)
        print(f"-- After cleaning {len(data)}")
        return data, arousals

    def load_data(self):
        fname = f'./affectively/datasets/{self.game}_3000ms.csv'
        fname_train = f'./affectively/datasets/{self.game}_3000ms_downsampled_pairs.csv'
        unscaled_data, _ = self.load_and_clean(fname, False)
        self.x_train, self.y_train = self.load_and_clean(fname_train, True)
        self.scaler.fit(unscaled_data.values)
