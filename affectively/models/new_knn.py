from base_model import AbstractSurrogateModel
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import pickle


class KNNSurrogateModel(AbstractSurrogateModel):
    def _setup_paths(self):
        classifier_suff = 'classifier' if self.classifier else 'regressor'
        pref_suff = 'preferences' if self.preference else ''
        self.model_path = f'./affectively/models/{self.game}/{classifier_suff}/KNN/Cluster_{self.cluster}_{classifier_suff}_{pref_suff}.pkl'
        self.model_extension = '.pkl'
    
    def _create_model(self, **kwargs):
        if self.classifier:
            return KNeighborsClassifier(**kwargs)
        else:
            return KNeighborsRegressor(**kwargs)
    
    def _train_single_model(self, x_train, y_train, x_val, y_val, **hyperparams):
        k = hyperparams.get('k', 5)
        
        if k > len(x_train):
            return None, None
        
        model_params = {k: v for k, v in hyperparams.items() if k != 'k'}
        model_params['n_neighbors'] = k
        
        model = self._create_model(**model_params)
        model.fit(x_train, y_train)
        
        if self.classifier:
            val_score = model.score(x_val, y_val)
        else:
            val_pred = model.predict(x_val)
            val_score = self.CCC(y_val, val_pred)
        
        return val_score, model
    
    def _predict_single(self, model, scaler, state):
        state_scaled = scaler.transform(state)
        
        if self.classifier:
            return model.predict_proba(state_scaled)
        else:
            return model.predict(state_scaled)
    
    def _save_single_model(self, model, scaler, index):
        model_path = self.model_path.replace('.pkl', f'_{index}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'hyperparams': self.best_params,
                'task_type': 'preference' if self.preference else 'regression'
            }, f)
        print(f"Model {index} saved to {model_path}")
        
        scaler_path = self.model_path.replace('.pkl', f'_scaler_{index}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    def _load_single_model(self, path, scaler_path):
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        params = model_data.get('hyperparams', {})
        
        return model, params
    
    def get_hyperparameter_space(self):
        return {
            'k': [3, 5, 7, 9, 11, 15, 21, 31],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'weights': ['uniform', 'distance']
        }


import os

if __name__ == "__main__":
    import pandas as pd
    
    results = []
    
    for game in ['solid', 'fps', 'platform']:
        for pref in [True, False]:
            for classifier in [True, False]:
                for cluster in [0, 1, 2, 3, 4]:

                    pref_suff = 'preferences' if pref else ''
                    classifier_suff = 'classifier' if classifier else 'regressor'
                    os.makedirs(f'./affectively/models/{game}/{classifier_suff}/KNN/', exist_ok=True)

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
    results_df.to_csv('./affectively/models/results/knn_training_results.csv', index=False)
    print(f"\nResults saved to knn_training_results.csv")