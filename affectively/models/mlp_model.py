from affectively.models.base_model import AbstractSurrogateModel
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import pickle
import copy
from affectively.models.models import Classifier, Regressor


class MLPSurrogateModel(AbstractSurrogateModel):
    def __init__(self, game, cluster, classifier, preference):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        super().__init__(game, cluster, classifier, preference)
    
    def _setup_paths(self):
        classifier_suff = 'classifier' if self.classifier else 'regressor'
        pref_suff = 'preferences' if self.preference else ''
        self.model_path = f'./affectively/models/{self.game}/{classifier_suff}/MLP/best_model_{self.game}_cluster_{self.cluster}_{classifier_suff}_{pref_suff}.pth'
        self.model_extension = '.pth'
    
    def _create_model(self, input_size, hidden_size, dropout_rate):
        if self.classifier:
            return Classifier(input_size, hidden_size, self.output_size, dropout_rate).to(self.device)
        else:
            return Regressor(input_size, hidden_size, dropout_rate).to(self.device)
    
    def _train_single_model(self, x_train, y_train, x_val, y_val, **hyperparams):
        x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.long if self.classifier else torch.float32).to(self.device)
        x_val = torch.tensor(x_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.long if self.classifier else torch.float32).to(self.device)
        
        input_size = x_train.shape[1]
        model = self._create_model(input_size, hyperparams['hidden_size'], hyperparams['dropout_rate'])
        
        criterion = nn.CrossEntropyLoss() if self.classifier else nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=hyperparams['learning_rate'])
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 30
        
        while True:
            model.train()
            outputs = model(x_train)
            loss = criterion(outputs, y_train) if self.classifier else criterion(outputs.squeeze(), y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val) if self.classifier else criterion(val_outputs.squeeze(), y_val)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
        
        return -best_loss.cpu().numpy(), model
    
    def _predict_single(self, model, scaler, state):
        model.eval()
        with torch.no_grad():
            state_scaled = scaler.transform(state)
            state_tensor = torch.tensor(state_scaled, dtype=torch.float32).to(self.device)
            output = model(state_tensor)
            
            if self.classifier:
                return torch.softmax(output, dim=1).cpu().numpy()
            else:
                return output.cpu().numpy()
    
    def _save_single_model(self, model, scaler, index):
        input_size = model.fc1.in_features
        hidden_size = model.fc1.out_features
        output_size = model.fc2.out_features if self.classifier else 1
        
        model_path = self.model_path.replace('.pth', f'_{index}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'dropout_rate': model.dropout.p,
            'hyperparams': self.best_params,
            'task_type': 'preference' if self.preference else 'regression'
        }, model_path)
        print(f"Model {index} saved to {model_path}")
        
        scaler_path = self.model_path.replace('.pth', f'_scaler_{index}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    def _load_single_model(self, path, scaler_path):
        checkpoint = torch.load(path, map_location=self.device)
        
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
        
        return model, checkpoint.get('hyperparams', {})
    
    def get_hyperparameter_space(self):
        return {
            'hidden_size': [8, 16, 32, 64, 128],
            'learning_rate': [0.0001, 0.001],
            'dropout_rate': [0, 0.1, 0.2, 0.3, 0.4]
        }


if __name__ == "__main__":
    import pandas as pd
    
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
    results_df.to_csv(f'./affectively/models/results/mlp_training_results.csv', index=False)
