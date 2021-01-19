import re
from abc import ABCMeta, abstractmethod

import sys, os
sys.path.append("../")
import os.path as osp
from utils import trace, timer

class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_param=None):
        self.model_param = model_param
        self.model = self.get_model()

    
    @abstractmethod
    def fit(self, train_X, train_y, valid_X, valid_y):
        raise NotImplementedError
    
    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abstractmethod   
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def save_weight(self, path):
        raise NotImplementedError

    @abstractmethod
    def read_weight(self, path):
        raise NotImplementedError    

