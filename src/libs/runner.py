from preprocessing import Preprocessing
from learning import Learning
from predicting import Predicting
from _logging import Logging


class Runner():
    """
    全体の工程
    ・特徴量作成
    ・データの読み出し
    ・学習、weightと特徴量の名前の保存
    ・ログ(mlflow, feature_importances, )
    """
    def __init__(self, param):
        self.exp_param = param["exp_param"]
        self.prepro_param = param["prepro_param"]
        self.train_param = param["train_param"]
        self.pred_param = param["train_param"]
        self.log_param = param
        self.prepro_param.update(self.exp_param)
        self.train_param.update(self.exp_param)

    def __call__(self):
        Preprocessor = Preprocessing(self.prepro_param)
        Learner = Learning(self.train_param)
        Predictor = Predicting(self.pred_param)
        Logger = Logging(self.log_param)
        Preprocessor()
        Learner()
        Predictor()
        Logger()
    