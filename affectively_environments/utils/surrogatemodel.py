from typing import Literal

import importlib_resources
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class KNNSurrogateModel:
	def __init__(self,
	             k: int,
	             game: Literal["Heist", "Pirates", "Solid"]):
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
		self.max_score = 0
		if game == "Heist":
			self.max_score = 500
		elif game == "Pirates":
			self.max_score = 460
		elif game == "Solid":
			self.max_score = 24
		else:
			raise ValueError(f"Game {game} not supported.")
		self.load_data()
	
	def __call__(self,
	             state):  # TODO: Type hinting, better docstring
		"""
		Compute a prediction using the surrogate model.
		
		Args:
			state: The game state.

		Returns: A prediction and the neighbours indices.

		"""
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
	
	def load_and_clean(self,
	                   filename: str,
	                   preference: bool):  # TODO: Type hinting, better docstring
		"""
		Load affect data and clean the data.
		
		Args:
			filename: The data file name.
			preference: ???

		Returns: Game data and arousal values.

		"""
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
		"""
		Load the arousal data for the selected game.
		"""
		fname = importlib_resources.files(
			'affectively_environments') / f'datasets/{self.game}_3000ms_nonorm_with_clusters.csv'
		fname_train = importlib_resources.files(
			'affectively_environments') / f'datasets/{self.game}_3000ms_pairs_classification_downsampled.csv'
		unscaled_data, _ = self.load_and_clean(fname, False)
		self.x_train, self.y_train = self.load_and_clean(fname_train, True)
		self.scaler.fit(unscaled_data.values)
