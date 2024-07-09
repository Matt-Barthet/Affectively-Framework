import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class KNNSurrogateModel:
    def __init__(self, k, game):
        self.x_train = None
        self.y_train = None
        self.k = k
        self.scaler = MinMaxScaler()
        self.game = game
        self.max_score = 0
        if game == "Heist":
            self.max_score = 500
        elif game == "pirates":
            self.max_score = 460
        else:
            self.max_score = 24
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
        return predicted_class, k_indices

    def load_and_clean(self, filename, preference):
        data = pd.read_csv(filename)
        data = data.loc[:, data.apply(pd.Series.nunique) != 1]
        if preference:
            data = data[data['Ranking'] != "stable"]
            arousals = data['Ranking'].values
            label_mapping = {"decrease": 0.0, "increase": 1.0}
            arousals = [label_mapping[label] for label in arousals]
            data = data.drop(columns=['Player', 'Ranking'])
        else:
            arousals = data['[output]arousal'].values
            participant_list = data['[control]player_id'].unique()
            human_arousal = []
            for participant in participant_list:
                sub_df = data[data['[control]player_id'] == participant]
                max_score = np.max(sub_df['playerScore'])
                human_arousal.append(max_score / self.max_score)  # Keep normalized score
            data = data.drop(columns=['[control]player_id', '[output]arousal'])
        if self.game == "Solid":
            data = data[data.columns[~data.columns.str.contains("botRespawn")]]
        data = data[data.columns[~data.columns.str.startswith("Cluster")]]
        data = data[data.columns[~data.columns.str.startswith("Time_Index")]]
        data = data[data.columns[~data.columns.str.contains("arousal")]]
        if self.game != "Heist":
            data = data[data.columns[~data.columns.str.contains("Score")]]
        return data, arousals

    def load_data(self):
        unscaled_data, _ = self.load_and_clean(f'./Datasets/{self.game}_3000ms_nonorm_with_clusters.csv', False)
        self.x_train, self.y_train = self.load_and_clean(f'./Datasets/{self.game}_3000ms_pairs_classification_downsampled.csv', True)
        self.scaler.fit(unscaled_data.values)
