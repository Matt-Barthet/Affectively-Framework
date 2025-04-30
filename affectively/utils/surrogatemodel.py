from typing import Literal

import importlib_resources
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def trim_pad(array, length, keep_max=True):
    array = np.asarray(array)
    adjusted_array = array
    if array.shape[0] < length:
        pad_width = [(0, length - array.shape[0])] + [(0, 0)] * (array.ndim - 1)
        adjusted_array = np.pad(array, pad_width=pad_width, mode='edge')

    if keep_max:
        for value in range(1, len(adjusted_array)):
            if adjusted_array[value] < adjusted_array[value - 1]:
                adjusted_array[value] = adjusted_array[value - 1]
    return adjusted_array


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
        data = data.loc[:, data.apply(pd.Series.nunique) != 1]

        if preference:
            data = data[data['Ranking'] != "stable"]
            arousals = data['Ranking'].values
            label_mapping = {"decrease": 0.0, "increase": 1.0}
            arousals = [label_mapping[label] for label in arousals]
            data = data.drop(columns=['Player', 'Ranking'])

            if self.cluster != 0:
                data = data[data['Cluster_L'] == self.cluster]
        else:
            if self.cluster != 0:
                data = data[data['Cluster'] == self.cluster]

            arousals = data['[output]arousal'].values
            participant_list = data['[control]player_id'].unique()

            human_arousal = []
            human_scores = []
            for participant in participant_list:
                sub_df = data[data['[control]player_id'] == participant]
                arousal_trace = sub_df['[output]arousal'].values
                score_trace = sub_df['playerScore'].values
                human_scores.append(trim_pad(score_trace[:40], 40))
                human_arousal.append(trim_pad(arousal_trace[:40], 40))

            self.target_behavior = np.mean(np.stack(human_scores, axis=-1), axis=1)
            self.target_arousal = np.mean(np.stack(human_arousal, axis=-1), axis=1)
            data = data.drop(columns=['[control]player_id', '[output]arousal'])

        # data = data[data.columns[~data.columns.str.contains("Score")]]
        data = data[data.columns[~data.columns.str.contains("botRespawn")]]

        if self.game == "Solid":
            data = data[data.columns[~data.columns.str.contains("DeltaDistance")]]
            data = data[data.columns[~data.columns.str.contains("DeltaRotation")]]

        data = data[data.columns[~data.columns.str.startswith("Cluster")]]
        data = data[data.columns[~data.columns.str.startswith("Time_Index")]]
        data = data[data.columns[~data.columns.str.contains("arousal")]]

        print(data)
        return data, arousals

    def load_data(self):
        fname = f'./affectively/datasets/{self.game}_nonorm_with_clusters.csv'
        fname_train = f'./affectively/datasets/{self.game}_pairs_classification.csv'
        unscaled_data, _ = self.load_and_clean(fname, False)
        self.x_train, self.y_train = self.load_and_clean(fname_train, True)
        self.scaler.fit(unscaled_data.values)
