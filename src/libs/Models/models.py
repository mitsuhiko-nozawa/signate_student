from .base import BaseModel
from catboost import Pool, CatBoostRegressor, CatBoost
import lightgbm as lgb
import pickle
import numpy as np
from sklearn.metrics import f1_score

TH = 0.39 # 学習のmetricに使うと過学習する？
def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.where(y_hat < TH, 0, 1)  
    return 'f1', f1_score(y_true, y_hat), True

class LGBM_Model(BaseModel):
    def get_model(self):
        return None

    def fit(self, train_X, train_y, valid_X, valid_y):
        train = lgb.Dataset(train_X.values, train_y)
        valid = lgb.Dataset(valid_X.values, valid_y)
        self.model = lgb.train(
            self.model_param,
            train, 
            valid_sets=valid, 
            verbose_eval=100,
            #feval=lgb_f1_score
        )

    def predict(self, X):
        return self.model.predict(X.values)
    
    def save_weight(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def read_weight(self, fname):
        self.model = pickle.load(open(fname, 'rb'))



class CatBoost_Model(BaseModel):
    def get_model(self):
        if self.model_param is None:
            return None
        else:
            return CatBoostRegressor(**self.model_param)

    def fit(self, train_X, train_y, valid_X, valid_y):
        categorical_features_indices = np.where(train_X.dtypes == "object")[0]
        train_data = Pool(train_X, train_y, cat_features=categorical_features_indices)
        valid_data = Pool(valid_X, valid_y, cat_features=categorical_features_indices)
        self.model.fit(
            train_data,
            eval_set=valid_data, 
            early_stopping_rounds=100, 
            use_best_model=True,
            verbose=90,
        )

    def predict(self, X):
        preds = self.model.predict(X)
        return np.where(preds < 0, 0, preds)
    
    def save_weight(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def read_weight(self, fname):
        self.model = pickle.load(open(fname, 'rb'))




def threshold_optimization(y_true, y_pred, metrics=None):
    def f1_opt(x):
        if metrics is not None:
            score = -metrics(y_true, y_pred >= x)
        else:
            raise NotImplementedError
        return score
    result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
    best_threshold = result['x'].item()
    return best_threshold

# 後で定義するモデルは確率で結果を出力するので、そこから最適なf1をgetする関数を定義
def optimized_f1(y_true, y_pred):
    bt = threshold_optimization(y_true, y_pred, metrics=f1_score)
    score = f1_score(y_true, y_pred >= bt)
    return score