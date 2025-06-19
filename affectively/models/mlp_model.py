import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch
import os
import pickle
from torch.optim import Adam
import matplotlib.pyplot as plt
import copy 
from scipy.stats import pearsonr
from affectively.models import Classifier, Regressor

class MLPSurrogateModel:

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
        self.model_path = f'./affectively/models/{game}/best_model_{game}_cluster_{self.cluster}_{classifier_suff}_{pref_suff}.pth'

        # Data loading
        self.data, self.x_train, self.y_train = None, None, None
        self.output_size = 2 if self.preference and self.classifier else 3
        self.load_data()

        # Model loading
        self.models, self.scalers = [], []
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
            model.eval()
            with torch.no_grad():
                if id != leave_out != -1:
                    continue
                state_scaled = scaler.transform(state)
                state_tensor = torch.tensor(state_scaled, dtype=torch.float32).to(self.device)
                output = model(state_tensor)
                
                if self.classifier:
                    predictions.append(torch.softmax(output, dim=1).cpu().numpy())
                else:
                    predictions.append(output.cpu().numpy())
        
        avg_prediction = np.mean(predictions, axis=0)
        if self.classifier:
            return np.argmax(avg_prediction, axis=1)[0]
        else:
            return avg_prediction[0][0]
    
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

            model_path_counter = self.model_path.replace('.pth', f'_{counter}.pth')
            if not os.path.exists(model_path_counter):
                break

            scaler_path = self.model_path.replace('.pth', f'_scaler_{counter}.pkl')

            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                break
            
            checkpoint = torch.load(model_path_counter, map_location=self.device)
            
            if self.classifier:
                model = Classifier(
                    checkpoint['input_size'], 
                    checkpoint['hidden_size'], 
                    checkpoint['output_size'],
                    checkpoint['dropout_rate']
                )
            else:
                model = Regressor(
                    checkpoint['input_size'], 
                    checkpoint['hidden_size'],
                    checkpoint['dropout_rate']
                )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

            self.models.append(copy.deepcopy(model))
            self.scalers.append(copy.deepcopy(scaler))

            # print(f"Model loaded from {model_path_counter}")

            counter+=1



    def save_models(self):

        os.makedirs(f"affectively/models/{self.game}", exist_ok=True)
        task_type = 'preference' if self.preference else 'regression'
        
        for i in range(len(self.models)):
            model = self.models[i]
            scaler = self.scalers[i]
            input_size = model.fc1.in_features
            hidden_size = model.fc1.out_features
            output_size = model.fc2.out_features if self.classifier else 1

            model_path = self.model_path.replace('.pth', f'_{i}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'hidden_size': hidden_size,
                'output_size': output_size,
                'dropout_rate': model.dropout.p,
                'hyperparams': self.best_params,
                'task_type': task_type
            }, model_path)
            print(f"Model {i} saved to {model_path}")
        
            scaler_path = self.model_path.replace('.pth', f'_scaler_{i}.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        print(f"Model saved to {self.model_path}")
        print(f"Scaler saved to {scaler_path}")
    

    def training_run(self, criterion, x_train, y_train, x_val, y_val, model, lr):
        optimizer = Adam(model.parameters(), lr=lr)
        best_loss = float('inf')
        patience_counter = 0
        patience = 30
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []


        while True:
            model.train()
            outputs = model(x_train)
            loss = criterion(outputs, y_train) if self.classifier else criterion(outputs.squeeze(), y_train)
            train_losses.append(loss.item())

            if self.classifier:
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_train).sum().item() / y_train.size(0)
                train_accuracies.append(accuracy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val) if self.classifier else criterion(val_outputs.squeeze(), y_val)
                val_losses.append(val_loss.item())

                if self.classifier:
                    _, predicted = torch.max(val_outputs, 1)
                    score = (predicted == y_val).sum().item() / y_val.size(0)
                    val_accuracies.append(score)
                    
                if val_loss < best_loss: 
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    break
        

        return best_loss, model, (train_losses, val_losses, train_accuracies, val_accuracies)


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
        """
        Calculate the Concordance Correlation Coefficient (CCC) between true and predicted values.
        """
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        cov = np.cov(y_true, y_pred)[0][1]
        
        ccc = 2 * cov / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        return ccc

    def train_model(self):

        hyperparams = {'hidden_size': [8, 16, 32, 64, 128],
            'learning_rate': [0.0001, 0.001],
            'dropout_rate': [0, 0.1, 0.2, 0.3, 0.4]}
        
        param_combinations = list(itertools.product(
            hyperparams['hidden_size'], 
            hyperparams['learning_rate'], 
            hyperparams['dropout_rate']
        ))
        
        print(f"Testing {len(param_combinations)} hyperparameter combinations...")
        
        best_loss = float('inf')
        
        for i, (hidden_size, lr, dropout_rate) in enumerate(param_combinations):
            print(f"  Testing combination {i+1}/{len(param_combinations)}...")
            
            cv_scores = []
            group_kfold = GroupKFold(n_splits=5)
            models, scalers = [],[]
            for fold, (train_idx, val_idx) in enumerate(group_kfold.split(self.data, self.arousals, groups=self.players.values)):
                # print(f"    Fold {fold+1}/5")
                x_train, x_val = self.data.values[train_idx], self.data.values[val_idx]
                y_train, y_val = self.arousals[train_idx], self.arousals[val_idx]
                
                self.scaler = MinMaxScaler()
                x_train = self.scaler.fit_transform(x_train)
                x_val = self.scaler.transform(x_val)

                x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
                y_train = torch.tensor(y_train, dtype=torch.long if self.classifier else torch.float32).to(self.device)
                x_val = torch.tensor(x_val, dtype=torch.float32).to(self.device)
                y_val = torch.tensor(y_val, dtype=torch.long if self.classifier else torch.float32).to(self.device)

                input_size = x_train.shape[1]
                
                if self.classifier:
                    model = Classifier(input_size, hidden_size, self.output_size, dropout_rate).to(self.device)
                    criterion = nn.CrossEntropyLoss()
                else:
                    model = Regressor(input_size, hidden_size, dropout_rate).to(self.device)
                    criterion = nn.MSELoss()
                
                loss, model, (train_losses, val_losses, train_accuracies, val_accuracies) = self.training_run(criterion, x_train, y_train, x_val, y_val, model, lr)
                cv_scores.append(loss)
                models.append(copy.deepcopy(model))
                scalers.append(copy.deepcopy(self.scaler))

            loss = np.mean(cv_scores)

            if loss < best_loss:
                best_loss = loss
                self.best_params = {
                    'hidden_size': hidden_size, 
                    'learning_rate': lr, 
                    'dropout_rate': dropout_rate,
                }
                # plot_training_curves(
                #     train_losses, val_losses, train_acc=train_accuracies,
                #     val_acc=val_accuracies, baseline=baseline, game=self.game, cluster=self.cluster)
                self.models = models.copy()
                self.scalers = scalers.copy()
                self.save_models()
                self.evaluate_ensemble()

if __name__== "__main__":
    
    results = []
    
    for classifier in [False, True]:
        for pref in [True, False]:
            for game in ['solid', 'fps', 'platform']:
                for cluster in [0, 1, 2, 3, 4]:
                    print(f"\nTraining MLP for game={game}, cluster={cluster}, classifier={classifier}, preference={pref}")
                    model = MLPSurrogateModel(game=game, cluster=cluster, classifier=classifier, preference=pref)
                    
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
            pref_suff = 'preferences' if pref else ''
            classifier_suff = 'classifier' if classifier else 'regressor'
            results_df.to_csv(f'./affectively/models/mlp_training_results_{pref_suff}_{classifier_suff}.csv', index=False)
            print(f"\nResults saved to mlp_training_results.csv")