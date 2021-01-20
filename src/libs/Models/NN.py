from .base import BaseModel
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.nn import CrossEntropyLoss
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.nn.utils import weight_norm

import pickle
import numpy as np

from utils_nn import *


class NN_Model(BaseModel):
    def get_model(self):
        self.parse_param(self.model_param)
        return None

    def fit(self, train_X, train_y, valid_X, valid_y):
        self.n_numfeats = train_X.shape[1]-self.n_embfeats
        self.model = eval(self.model_param["model_type"])(self.embedding_dim, self.max_seq_len, n_embfeats=self.n_embfeats, n_numfeats=self.n_numfeats)

        self.optimizer = eval(self.optimizer)(self.model.parameters(), **self.optimizer_param)
        self.scheduler = eval(self.scheduler)(optimizer=self.optimizer, **self.scheduler_param)
        self.model.to(self.DEVICE)
        trainloader = get_dataloader(train_X, train_y, self.emb_cols, self.batch_size, mode="train", model_type=self.model_type)
        validloader = get_dataloader(valid_X, valid_y, self.emb_cols, self.batch_size, mode="valid", model_type=self.model_type)
        
        self.model = run_training(self.model, trainloader, validloader, self.MAX_EPOCH, self.optimizer, self.scheduler, self.loss_fn, self.early_stopping, verbose=self.verbose, device=self.DEVICE, fold=self.fold, seed=self.seed, path=self.WORK_DIR)


    def predict(self, X):
        if self.model is None:
            self.n_numfeats = X.shape[1]-self.n_embfeats
            self.model = eval(self.model_param["model_type"])(self.embedding_dim, self.max_seq_len, n_embfeats=self.n_embfeats, n_numfeats=self.n_numfeats)

        self.model.load_state_dict(torch.load(osp.join( self.WORK_DIR, "weight" f"{self.seed}_{self.fold}.pt")), self.DEVICE)
        testloader = get_dataloader(X, None, self.emb_cols, self.batch_size, mode="test", model_type=self.model_type)
        preds = inference_fn(self.model, testloader, self.DEVICE)
        return np.where(preds < 0, 0, preds)
    
    def save_weight(self, path):
        pass
        #path = path.replace(".pkl", "")
        #self.model.save_model(path)

    def read_weight(self, fname):
        pass

    def parse_param(self, params):
        self.DEVICE = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
        self.WORK_DIR = params["WORK_DIR"]
        self.model_type = params["model_type"]
        self.emb_cols = params["emb_cols"]
        self.n_embfeats = len(self.emb_cols)
        self.early_stopping = params["early_stopping"]
        self.MAX_EPOCH = params["MAX_EPOCH"]
        self.embedding_dim = params["embedding_dim"]
        self.max_seq_len = params["max_seq_len"]
        self.verbose = params["verbose_step"]
        self.batch_size = params["batch_size"]

        self.optimizer_param = params["optimizer_param"]
        self.scheduler_param = params["scheduler_param"]
        self.optimizer = params["optimizer"]
        self.scheduler = params["scheduler"]
        self.loss_fn = eval(params["loss_fn"])()

        self.seed = self.model_param["random_state"] 
        self.fold = self.model_param["fold"]


    





class Base_NN(nn.Module):
    def __init__(self, embedding_dim, max_seq_len, n_embfeats, n_numfeats):
        super(Base_NN, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_embfeats = n_embfeats
        self.n_numfeats = n_numfeats
        self.embs = [Embedding_Module(self.embedding_dim) for i in range(self.n_embfeats)]
        self.bn1 = nn.BatchNorm1d(self.n_embfeats)
        self.dense1 = nn.Linear(self.n_embfeats, self.n_embfeats)
        self.relu1 = nn.ReLU()

        self.bn2 = nn.BatchNorm1d(self.n_numfeats)
        self.dropout2 = nn.Dropout(0.4)
        self.dense2 = nn.Linear(self.n_numfeats, self.n_numfeats//2)
        self.relu2 = nn.ReLU()

        self.h3 = self.n_embfeats + self.n_numfeats//2
        self.h4 = 40
        self.bn3 = nn.BatchNorm1d(self.h3)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.Linear(self.h3, self.h4)
        self.relu3 = nn.ReLU()

        self.bn4 = nn.BatchNorm1d(self.h4)
        self.dense4 = nn.Linear(self.h4, 1)


    
    def forward(self, x1, x2):
        emb_li = []
        for i in range(self.n_embfeats):
            emb_li.append(self.embs[i](x1[:,i]))
        x1 = torch.cat(emb_li, axis=1)
        
        x1 = self.bn1(x1)
        x1 = self.dense1(x1)
        x1 = self.relu1(x1)

        x2 = self.bn2(x2)
        x2 = self.dropout2(x2)
        x2 = self.dense2(x2) ### 
        x2 = self.relu2(x2)
        x = torch.cat([x1, x2], axis=1)
        
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        x = self.relu3(x)

        x = self.bn4(x)
        x = self.dense4(x)
        
        return x


class Embedding_Module(nn.Module):
    def __init__(self, embedding_dim):
        super(Embedding_Module, self).__init__()
        self.module = nn.Sequential(
            nn.Embedding(num_embeddings=800, embedding_dim=embedding_dim),
            #nn.Dropout(0.2),
            #nn.BatchNorm1d(embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=1),
            nn.ReLU(),
        )        
    def forward(self, x1):
        return self.module(x1)

