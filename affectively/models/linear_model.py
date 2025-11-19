from affectively.models.base_model import AbstractSurrogateModel
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import pickle
import copy
import os


class LinearSurrogateModel(AbstractSurrogateModel):
    def _setup_paths(self):
        classifier_suff = 'classifier' if self.classifier else 'regressor'
        pref_suff = 'preferences' if self.preference else ''
        self.model_path = f'./affectively/models/{self.game}/{classifier_suff}/Linear/Cluster_{self.cluster}_{classifier_suff}_{pref_suff}_linear.pkl'
        self.model_extension = '.pkl'
    
    def _create_model(self, **kwargs):
        if self.classifier:
            return LogisticRegression(**kwargs, random_state=42, max_iter=1000)
        else:
            return LinearRegression(**kwargs)
    
    def _train_single_model(self, x_train, y_train, x_val, y_val, **hyperparams):
        model = self._create_model(**hyperparams)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        if self.classifier:
            val_pred = np.round(val_pred)
            val_score = np.mean(val_pred == y_val)
        else:
            val_score = -np.mean((val_pred - y_val) ** 2)
        
        return val_score, model
    
    def _predict_single(self, model, scaler, state):
        
        if self.classifier:
            return model.predict_proba(state)
        else:
            pred = model.predict(state)
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
        if self.classifier:
            return {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
        else:
            return {}  # LinearRegression has no hyperparameters that typically need tuning


if __name__ == "__main__":
    import pandas as pd
    
    results = []
    
    for classifier in [False, True]:
        for pref in [True, False]:
            for game in ['solid', 'fps', 'platform']:
                pref_suff = 'preferences' if pref else ''
                classifier_suff = 'classifier' if classifier else 'regressor'
                os.makedirs(f'./affectively/models/{game}/{classifier_suff}/Linear/', exist_ok=True)

                for cluster in [0, 1, 2, 3, 4]:
                    print(f"\nTraining LM for game={game}, cluster={cluster}, classifier={classifier}, preference={pref}")
                    model = LinearSurrogateModel(game=game, cluster=cluster, classifier=classifier, preference=pref)
                    
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
    results_df.to_csv(f'./affectively/models/results/linear_training_results.csv', index=False)
