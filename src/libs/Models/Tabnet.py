from .base import BaseModel
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

import pickle
import numpy as np

class Tabnet_Classifier_Model(BaseModel):
    def get_model(self):
        return None

    def fit(self, train_X, train_y, valid_X, valid_y):
        cat_feats = self.model_param["cat_feats"]
        categorical_features_indices = [train_X.columns.to_list().index(feat) for feat in cat_feats]
        continuous_features_indices = [i for i in range(train_X.shape[1]) if i not in categorical_features_indices]
        #self.model_param["params"]["cat_idxs"] = categorical_features_indices#カテゴリ変数
        #self.model_param["params"]["cat_dims"] = list(train_X.iloc[:,categorical_features_indices].nunique() + 1000)
        #self.model_param["params"]["cat_emb_dim"] = 20
        self.model_param["params"]["optimizer_fn"] = Adam
        self.model_param["params"]["optimizer_params"] = dict(lr = 2e-2)
        self.model_param["params"]["scheduler_fn"] = ReduceLROnPlateau
        self.model_param["params"]["scheduler_params"] = dict(mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9)
        
        train_X.fillna(-999, inplace=True)
        valid_X.fillna(-999, inplace=True)
        train_X = train_X.astype(float)
        valid_X = valid_X.astype(float)
        device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")
        
        self.pretrainer = TabNetPretrainer(**self.model_param["params"])
        self.pretrainer.device = device
        self.pretrainer.fit(
            X_train=train_X.values,
            eval_set=[train_X.values],
            max_epochs=10,
            patience=20, 
            batch_size=256, 
            virtual_batch_size=64,
            num_workers=1, 
            drop_last=True
            )        
        self.model_param["params"]["cat_idxs"] = categorical_features_indices#カテゴリ変数
        self.model_param["params"]["cat_dims"] = list(train_X.iloc[:,categorical_features_indices].nunique() + 1000)
        self.model_param["params"]["cat_emb_dim"] = 20
        self.model_param["params"]["optimizer_params"] = dict(lr = 2e-2, weight_decay = 1e-5)
        self.model_param["params"]["scheduler_fn"] = OneCycleLR
        self.model_param["params"]["scheduler_params"] = dict(max_lr=5e-2, steps_per_epoch=int(train_X.shape[0] / 256), epochs=30, is_batch_level=True)

        #self.model = TabNetRegressor(**self.model_param["params"])
        self.model = TabNetClassifier(**self.model_param["params"])
        self.model.device = device
        self.model.fit(
            X_train=train_X.values,
            y_train=train_y.values.reshape(-1, ),
            eval_set=[(valid_X.values, valid_y.values.reshape(-1, ))],
            eval_name = ["val"],
            eval_metric = ["logloss"],
            max_epochs=30,
            patience=20, 
            batch_size=256, 
            virtual_batch_size=64,
            num_workers=1, 
            drop_last=False,
            loss_fn=CrossEntropyLoss(),
            from_unsupervised=self.pretrainer,
        )

    def predict(self, X):
        X = X.fillna(-999).astype(float)
        preds = self.model.predict_proba(X.values)[:,1]
        return np.where(preds < 0, 0, preds)
    
    def save_weight(self, path):
        path = path.replace(".pkl", "")
        self.model.save_model(path)

    def read_weight(self, fname):
        self.model = TabNetRegressor()
        fname = fname.replace(".pkl", ".zip")
        self.model.load_model(fname)