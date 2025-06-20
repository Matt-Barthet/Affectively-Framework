from base_model import AbstractSurrogateModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pickle
import copy


class RFSurrogateModel(AbstractSurrogateModel):
    def _setup_paths(self):
        classifier_suff = 'classifier' if self.classifier else 'regressor'
        pref_suff = 'preferences' if self.preference else ''
        self.model_path = f'./affectively/models/{self.game}/{classifier_suff}/RF/Cluster_{self.cluster}_{classifier_suff}_{pref_suff}_rf.pkl'
        self.model_extension = '.pkl'
    
    def _create_model(self, **kwargs):
        if self.classifier:
            return RandomForestClassifier(**kwargs, random_state=42, n_jobs=-1)
        else:
            return RandomForestRegressor(**kwargs, random_state=42, n_jobs=-1)
    
    def _train_single_model(self, x_train, y_train, x_val, y_val, **hyperparams):
        model = self._create_model(**hyperparams)
        model.fit(x_train, y_train)
        
        if self.classifier:
            val_pred = model.predict(x_val)
            val_score = np.mean(val_pred == y_val)
        else:
            val_pred = model.predict(x_val)
            val_score = -np.mean((val_pred - y_val) ** 2)
        
        return val_score, model
    
    def _predict_single(self, model, scaler, state):
        state_scaled = scaler.transform(state)
        
        if self.classifier:
            return model.predict_proba(state_scaled)
        else:
            pred = model.predict(state_scaled)
            return pred.reshape(1, -1)
    
    def _save_single_model(self, model, scaler, index):
        model_path = self.model_path.replace('.pkl', f'_{index}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model {index} saved to {model_path}")
        
        scaler_path = self.model_path.replace('.pkl', f'_scaler_{index}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    def _load_single_model(self, path, scaler_path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model, None
    
    def get_hyperparameter_space(self):
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }


if __name__ == "__main__":
    import pandas as pd
    
    results = []
    
    for classifier in [False, True]:
        for pref in [True, False]:
            for game in ['solid', 'fps', 'platform']:
                for cluster in [0, 1, 2, 3, 4]:
                    print(f"\nTraining RF for game={game}, cluster={cluster}, classifier={classifier}, preference={pref}")
                    model = RFSurrogateModel(game=game, cluster=cluster, classifier=classifier, preference=pref)
                    
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
    results_df.to_csv(f'./affectively/models/results/rf_training_results.csv', index=False)
