import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import copy
from abc import ABC, abstractmethod


class AbstractSurrogateModel(ABC):

    def __init__(self, game, cluster, classifier, preference):
        self.target_behavior, self.target_arousal, self.columns, self.players, self.arousals = [], [], [], [], []
        self.game = game
        self.cluster = cluster
        self.surrogate_length = 0
        self.classifier = classifier
        self.preference = preference
        self.test_result = None
        self.models, self.scalers = [], []
        self.best_params = {}
        self.cluster_score, self.cluster_arousal = [], []
        self._setup_paths()
        
        self.data, self.x_train, self.y_train = None, None, None
        self.output_size = 2 if self.preference and self.classifier else 3
        self.load_data()

        self.load_model()
        
        if len(self.models) == 0:
            self.train_model()
        else:
            self.evaluate_ensemble()
    
    @abstractmethod
    def _setup_paths(self):
        pass
    
    @abstractmethod
    def _create_model(self, **kwargs):
        pass
    
    @abstractmethod
    def _train_single_model(self, x_train, y_train, x_val, y_val, **hyperparams):
        pass
    
    @abstractmethod
    def _predict_single(self, model, scaler, state):
        pass
    
    @abstractmethod
    def _save_single_model(self, model, scaler, index):
        pass
    
    @abstractmethod
    def _load_single_model(self, path, scaler_path):
        pass
    
    @abstractmethod
    def get_hyperparameter_space(self):
        pass
    

    def __call__(self, state, leave_out=-1):
        if len(self.models) == 0:
            return 0
        
        state = np.array(state).reshape(1, -1)
        predictions = []
        
        for id, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            if id != leave_out != -1:
                continue

            state_scaled = scaler.transform(state)
            # if np.min(state_scaled) < 0 or np.max(state_scaled) > 1:
            #     print(f"Values outside of range: Max={np.max(state_scaled):.3f}@{self.columns[np.argmax(state_scaled)]}(other={np.where(state_scaled > 1)[0]})", end=", ")
            #     print(f"Min={np.min(state_scaled):.3f}@{self.columns[np.argmin(state_scaled)]}(other={np.where(state_scaled < 0)[0]})")
            
            state_scaled = np.clip(state_scaled, 0, 1)
            predictions.append(self._predict_single(model, scaler, state_scaled))
        
        avg_prediction = np.mean(predictions, axis=0)
        if self.classifier:
            return np.argmax(avg_prediction, axis=1)[0]
        else:
            return avg_prediction[0] if len(avg_prediction.shape) == 1 else avg_prediction[0][0]
    

    def load_data(self):
        pref_suff = '_downsampled_pairs' if self.preference else ''
        fname = f'./affectively/datasets/{self.game.lower()}_3000ms{pref_suff}.csv'
        
        self.data = pd.read_csv(fname)
        if self.cluster > 0:
            cluster_members = pd.read_csv(f"./affectively/datasets/{self.game.lower()}_cluster_book.csv")
            cluster_members = cluster_members[cluster_members['Cluster'] == self.cluster]
            self.data = self.data[self.data['[control]player_id'].isin(cluster_members['[control]player_id'])]
        
        self.players = self.data['[control]player_id']
        
        scores, arousals = [], []

        if not self.preference:
            for player in self.players.unique():
                try:
                    player_mask = self.data['[control]player_id'] == player
                    player_scores = np.pad(self.data.loc[player_mask, 'playerScore'].values[:40], (0, max(0, 40 - len(self.data.loc[player_mask, 'playerScore'].values))), 'edge')
                    player_arousals = np.pad(self.data.loc[player_mask, '[output]arousal'].values[:40], (0, max(0, 40 - len(self.data.loc[player_mask, '[output]arousal'].values))), 'edge')
                    scores.append(player_scores)
                    scaler = MinMaxScaler()
                    scaled_arousals = scaler.fit_transform(player_arousals.reshape(-1, 1)).flatten()
                    arousals.append(scaled_arousals)
                except:
                    print("Skipping header error")

            scores_stacked = np.stack(scores, axis=1)
            self.cluster_score = np.mean(scores_stacked, axis=1)

            arousals_stacked = np.stack(arousals, axis=1)
            self.cluster_arousal = np.mean(arousals_stacked, axis=1)
            self.cluster_score = np.round(self.cluster_score)
            print(self.cluster)

            cluster_step = 3.0   # seconds per cluster sample
            fine_step = 0.2    # seconds per fine sample
            repeat_factor = int(cluster_step / fine_step)  # 3 / 0.25 = 12

            # Repeat each value 12 times
            self.cluster_score = np.repeat(self.cluster_score, repeat_factor)
            self.cluster_arousal = np.repeat(self.cluster_arousal, repeat_factor)

            from matplotlib import pyplot as plt
            plt.errorbar(np.arange(len(self.cluster_arousal)), self.cluster_arousal, label='Cluster Arousal', alpha=0.7)
            plt.errorbar(np.arange(len(self.cluster_score)), self.cluster_score / 24, label='Cluster Score', alpha=0.7, color='orange')
            plt.show()

        if self.preference and self.classifier:
            arousals = self.data['[output]ranking'].values
            self.data = self.data.drop(columns=['[output]ranking', '[output]delta'])


        elif self.preference and not self.classifier:
            arousals = self.data['[output]delta'].values

            data_copy = self.data.copy()

            for player in self.players.unique():
                player_mask = data_copy['[control]player_id'] == player
                player_arousals = data_copy.loc[player_mask, '[output]delta'].values
                scaler = MinMaxScaler()
                scaled_arousals = scaler.fit_transform(player_arousals.reshape(-1, 1)).flatten()
                data_copy.loc[player_mask, '[output]delta'] = scaled_arousals

            arousals = data_copy['[output]delta'].values
            self.data = self.data.drop(columns=['[output]ranking', '[output]delta'])

        elif not self.preference:
            data_copy = self.data.copy()
            arousal_labels = []
            valid_indices = []
            threshold_scale = 0.1  

            for player in self.players.unique():
                player_mask = data_copy['[control]player_id'] == player
                player_arousals = data_copy.loc[player_mask, '[output]arousal'].values
                scaler = MinMaxScaler()
                scaled_arousals = scaler.fit_transform(player_arousals.reshape(-1, 1)).flatten()
                data_copy.loc[player_mask, '[output]arousal'] = scaled_arousals

                mean = np.mean(scaled_arousals)

                upper = mean + threshold_scale
                lower = mean - threshold_scale

                for i, val in zip(data_copy[player_mask].index, scaled_arousals):
                    if val > upper:
                        arousal_labels.append(1)
                        valid_indices.append(i)
                    elif val < lower:
                        arousal_labels.append(0)
                        valid_indices.append(i)

            if self.classifier:
                self.data = data_copy.loc[valid_indices].reset_index(drop=True)
                arousals = np.array(arousal_labels)
            else:
                arousals = data_copy['[output]arousal'].values

            self.data = self.data.drop(columns=['[output]arousal'])

        self.players = self.data['[control]player_id'] # Update players
        self.data.drop(columns=['[control]player_id'], inplace=True)

        self.surrogate_length = len(self.data.columns)
        self.columns = self.data.columns.tolist()
        if self.preference:
            self.surrogate_length = int(self.surrogate_length / 2)

        self.arousals = arousals

    def load_model(self):
        counter = 0
        
        while True:
            model_path_counter = self.model_path.replace(self.model_extension, f'_{counter}{self.model_extension}')
            if not os.path.exists(model_path_counter):
                break
            
            scaler_path = self.model_path.replace(self.model_extension, f'_scaler_{counter}.pkl')
            
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                break
            
            model, params = self._load_single_model(model_path_counter, scaler_path)
            if params:
                self.best_params = params
            
            self.models.append(copy.deepcopy(model))
            self.scalers.append(copy.deepcopy(scaler))
            
            counter += 1
    

    def save_models(self):
        os.makedirs(f"affectively/models/{self.game.lower()}/", exist_ok=True)
        for i in range(len(self.models)):
            self._save_single_model(self.models[i], self.scalers[i], i)
    

    def evaluate_ensemble(self):

        group_kfold = GroupKFold(n_splits=5)
        accuracies, cccs = [], []
        baseline_scores = []
        
        for id, (train_idx, val_idx) in enumerate(group_kfold.split(self.data, self.arousals, groups=self.players.values)):
            x_val = self.data.values[val_idx]
            y_val = self.arousals[val_idx]
            y_train = self.arousals[train_idx]
            
            fold_predictions = []
            
            for entry in x_val:
                prediction = self(entry, id)
                fold_predictions.append(prediction)
            
            fold_predictions = np.array(fold_predictions)

            if self.classifier:
                fold_predictions = np.asarray(fold_predictions, dtype=int)
                accuracy = (fold_predictions == y_val).sum() / len(y_val)
                accuracies.append(accuracy)
                y_train = np.array(y_train, dtype=int)
                y_val = np.array(y_val, dtype=int)
                majority_vote = np.bincount(y_train).argmax()
                baseline_acc = np.sum(y_val == majority_vote) / len(y_val)
                baseline_scores.append(baseline_acc)
            else:
                ccc = self.CCC(y_val, fold_predictions)
                cccs.append(ccc)
                baseline_ccc = self.CCC(y_val, np.full_like(y_val, np.mean(y_train)))
                baseline_scores.append(baseline_ccc)
        
        if self.classifier:
            self.test_result = {
                'score': np.mean(accuracies),
                'baseline': np.mean(baseline_scores),
                'metric': 'accuracy'
            }
            print(f"Ensemble accuracy: {np.mean(accuracies):.4f} vs baseline: {np.mean(baseline_scores):.4f}")
        else:
            self.test_result = {
                'score': np.mean(cccs),
                'baseline': np.mean(baseline_scores),
                'metric': 'ccc'
            }
            print(f"Ensemble CCC: {np.mean(cccs):.4f} vs baseline CCC: {np.mean(baseline_scores):.4f}")
    

    def CCC(self, y_true, y_pred):
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        cov = np.cov(y_true, y_pred)[0][1]
        
        ccc = 2 * cov / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        return ccc
    
    
    def train_model(self):
        hyperparams = self.get_hyperparameter_space()
        
        param_combinations = list(itertools.product(*hyperparams.values()))
        param_names = list(hyperparams.keys())
        
        print(f"Testing {len(param_combinations)} hyperparameter combinations...")
        
        best_score = -float('inf')
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_names, params))
            print(f"  Testing combination {i+1}/{len(param_combinations)}...")
            
            cv_scores = []
            group_kfold = GroupKFold(n_splits=5)
            models, scalers = [], []
            
            for fold, (train_idx, val_idx) in enumerate(group_kfold.split(self.data, self.arousals, groups=self.players.values)):
                x_train, x_val = self.data.values[train_idx], self.data.values[val_idx]
                y_train, y_val = self.arousals[train_idx], self.arousals[val_idx]
                
                scaler = MinMaxScaler()
                x_train = scaler.fit_transform(x_train)
                x_val = scaler.transform(x_val)
                
                score, model = self._train_single_model(x_train, y_train, x_val, y_val, **param_dict)
                if score is None:
                    break
                
                cv_scores.append(score)
                models.append(copy.deepcopy(model))
                scalers.append(copy.deepcopy(scaler))
            
            if len(cv_scores) == 0:
                continue
            
            avg_score = np.mean(cv_scores)
            
            if avg_score > best_score:
                best_score = avg_score
                self.best_params = param_dict
                self.models = models.copy()
                self.scalers = scalers.copy()
                self.save_models()
                self.evaluate_ensemble()