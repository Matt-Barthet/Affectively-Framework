import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import os
import pickle
import matplotlib.pyplot as plt
import copy
from scipy.stats import pearsonr


class KNNSurrogateModel:

    def __init__(self, game, cluster, classifier, preference):
        self.target_behavior, self.target_arousal, self.columns, self.players, self.arousals = [], [], [], [], []
        self.game = game
        self.cluster = cluster
        self.surrogate_length = 0
        self.classifier = classifier
        self.preference = preference
        self.test_result = None

        classifier_suff = 'classifier' if self.classifier else 'regressor'
        pref_suff = 'preferences' if self.preference else ''
        self.model_path = f'./affectively/models/{game}/best_knn_model_{game}_cluster_{self.cluster}_{classifier_suff}_{pref_suff}.pkl'

        # Data loading
        self.data, self.x_train, self.y_train = None, None, None
        self.output_size = 2 if self.preference and self.classifier else 3
        self.load_data()

        # Model loading
        self.models, self.scalers = [], []
        self.load_model()

        # Model training
        if len(self.models) == 0:
            self.best_params = {}
            self.train_model()
        else:
            self.evaluate_ensemble()

    def __call__(self, state, leave_out=-1):
        if len(self.models) == 0:
            return 0
        
        state = np.array(state).reshape(1, -1)
        predictions = []
        
        for id, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            if id != leave_out != -1:
                continue
            state_scaled = scaler.transform(state)
            
            if self.classifier:
                # Get probability predictions for classification
                proba = model.predict_proba(state_scaled)
                predictions.append(proba)
            else:
                # Get regression prediction
                pred = model.predict(state_scaled)
                predictions.append(pred)
        
        avg_prediction = np.mean(predictions, axis=0)
        
        if self.classifier:
            return np.argmax(avg_prediction, axis=1)[0]
        else:
            return avg_prediction[0]
    
    def load_data(self):
        pref_suff = '_downsampled_pairs' if self.preference else ''
        fname = f'./affectively/datasets/{self.game}_3000ms{pref_suff}.csv'
    
        self.data = pd.read_csv(fname)
        if self.cluster > 0:
            cluster_members = pd.read_csv(f"./affectively/datasets/{self.game}_cluster_book.csv")
            cluster_members = cluster_members[cluster_members['Cluster'] == self.cluster]
            self.data = self.data[self.data['[control]player_id'].isin(cluster_members['[control]player_id'])]

        self.players = self.data['[control]player_id']

        if self.preference and self.classifier:
            arousals = self.data['[output]ranking'].values
            self.data = self.data.drop(columns=['[output]ranking', '[output]delta'])
        elif self.preference and not self.classifier:
            arousals = self.data['[output]delta'].values
            self.data = self.data.drop(columns=['[output]ranking', '[output]delta'])
        elif not self.preference:
            arousals = self.data['[output]arousal'].values
            data_copy = self.data.copy()
            for player in self.players.unique():
                player_mask = data_copy['[control]player_id'] == player
                player_arousal = data_copy.loc[player_mask, '[output]arousal']
                arousal_scaler = MinMaxScaler()
                scaled_values = arousal_scaler.fit_transform(player_arousal.values.reshape(-1, 1))
                data_copy.loc[player_mask, '[output]arousal'] = scaled_values.flatten()
            arousals = data_copy['[output]arousal'].values

            if self.classifier:
                arousal_labels = []
                for arousal in arousals:
                    if arousal < 0.33:
                        arousal_labels.append(0)
                    elif arousal <= 0.66:
                        arousal_labels.append(1)
                    else:
                        arousal_labels.append(2)
                arousals = np.asarray(arousal_labels)
                
            self.data = self.data.drop(columns=['[output]arousal'])

        self.data.drop(columns=['[control]player_id'], inplace=True)
        self.surrogate_length = len(self.data.columns) 
        self.arousals = arousals

    def load_model(self):
        counter = 0

        while True:
            model_path_counter = self.model_path.replace('.pkl', f'_{counter}.pkl')
            if not os.path.exists(model_path_counter):
                break

            scaler_path = self.model_path.replace('.pkl', f'_scaler_{counter}.pkl')

            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                break
            
            with open(model_path_counter, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            self.best_params = model_data.get('hyperparams', {})
            
            self.models.append(copy.deepcopy(model))
            self.scalers.append(copy.deepcopy(scaler))

            print(f"Model loaded from {model_path_counter}")
            counter += 1

    def save_models(self):
        os.makedirs(f"affectively/models/{self.game}", exist_ok=True)
        task_type = 'preference' if self.preference else 'regression'
        
        for i in range(len(self.models)):
            model = self.models[i]
            scaler = self.scalers[i]

            model_path = self.model_path.replace('.pkl', f'_{i}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'hyperparams': self.best_params,
                    'task_type': task_type
                }, f)
            print(f"Model {i} saved to {model_path}")
        
            scaler_path = self.model_path.replace('.pkl', f'_scaler_{i}.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        print(f"Models saved successfully")

    def evaluate_knn(self, k, distance_metric, weights, x_train, y_train, x_val, y_val):
        """Evaluate a single KNN model configuration"""
        if self.classifier:
            model = KNeighborsClassifier(
                n_neighbors=k,
                metric=distance_metric,
                weights=weights
            )
        else:
            model = KNeighborsRegressor(
                n_neighbors=k,
                metric=distance_metric,
                weights=weights
            )
        
        model.fit(x_train, y_train)
        
        if self.classifier:
            train_score = model.score(x_train, y_train)
            val_score = model.score(x_val, y_val)
            return val_score, train_score, model
        else:
            # For regression, calculate CCC instead of R²
            train_pred = model.predict(x_train)
            val_pred = model.predict(x_val)
            train_ccc = self.CCC(y_train, train_pred)
            val_ccc = self.CCC(y_val, val_pred)
            return val_ccc, train_ccc, model

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
                accuracy = (fold_predictions == y_val).sum() / len(y_val)
                accuracies.append(accuracy)
                
                y_train_int = np.array(y_train, dtype=int)
                y_val_int = np.array(y_val, dtype=int)
                majority_vote = np.bincount(y_train_int).argmax()
                baseline_acc = np.sum(y_val_int == majority_vote) / len(y_val_int)
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
        """Calculate the Concordance Correlation Coefficient (CCC)"""
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        cov = np.cov(y_true, y_pred)[0][1]
        
        ccc = 2 * cov / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        return ccc

    def train_model(self):
        # KNN hyperparameters
        hyperparams = {
            'k': [3, 5, 7, 9, 11, 15, 21, 31],
            'distance_metric': ['euclidean', 'manhattan', 'minkowski'],
            'weights': ['uniform', 'distance']
        }
        
        param_combinations = list(itertools.product(
            hyperparams['k'],
            hyperparams['distance_metric'],
            hyperparams['weights']
        ))
        
        print(f"Testing {len(param_combinations)} hyperparameter combinations...")
        
        best_score = -float('inf')
        
        for i, (k, distance_metric, weights) in enumerate(param_combinations):
            print(f"  Testing combination {i+1}/{len(param_combinations)}: k={k}, metric={distance_metric}, weights={weights}")
            
            cv_scores = []
            group_kfold = GroupKFold(n_splits=5)
            models, scalers = [], []
            
            for fold, (train_idx, val_idx) in enumerate(group_kfold.split(self.data, self.arousals, groups=self.players.values)):
                # print(f"    Fold {fold+1}/5")
                x_train, x_val = self.data.values[train_idx], self.data.values[val_idx]
                y_train, y_val = self.arousals[train_idx], self.arousals[val_idx]
                
                # Scale features
                scaler = MinMaxScaler()
                x_train = scaler.fit_transform(x_train)
                x_val = scaler.transform(x_val)
                
                # Skip if k is greater than number of training samples
                if k > len(x_train):
                    # print(f"    Skipping k={k} as it's larger than training samples ({len(x_train)})")
                    break
                
                val_score, train_score, model = self.evaluate_knn(
                    k, distance_metric, weights, x_train, y_train, x_val, y_val
                )
                
                cv_scores.append(val_score)
                models.append(copy.deepcopy(model))
                scalers.append(copy.deepcopy(scaler))
            
            if len(cv_scores) == 0:
                continue
                
            avg_score = np.mean(cv_scores)
            
            if avg_score > best_score:
                best_score = avg_score
                self.best_params = {
                    'k': k,
                    'distance_metric': distance_metric,
                    'weights': weights
                }
                self.models = models.copy()
                self.scalers = scalers.copy()
                
                # print(f"    New best score: {best_score:.4f}")
                # self.save_models()
                self.evaluate_ensemble()


if __name__ == "__main__":
    results = []
    
    for game in ['solid', 'fps', 'platform']:
        for pref in [True, False]:
            for classifier in [True, False]:
                for cluster in [0, 1, 2, 3, 4]:
                    print(f"\nTraining KNN for game={game}, cluster={cluster}, classifier={classifier}, preference={pref}")
                    model = KNNSurrogateModel(game=game, cluster=cluster, classifier=classifier, preference=pref)
                    
                    if model.test_result is not None:
                        results.append({
                            'game': game,
                            'cluster': cluster,
                            'classifier': classifier,
                            'preference': pref,
                            'score': model.test_result['score'],
                            'baseline': model.test_result['baseline'],
                            'metric': model.test_result['metric']
                        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('knn_training_results.csv', index=False)
    print(f"\nResults saved to knn_training_results.csv")