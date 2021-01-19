import os.path as osp
import numpy as np
import pandas as pd
from .base import Feature
import datetime
from sklearn.decomposition import LatentDirichletAllocation
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class default_features(Feature):
    def create_features(self):
        return self.create_default_features()

class countVectorizer_svd(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        for df in [train_df, test_df]:
            df["html_content"] = df["html_content"].map(lambda x : BeautifulSoup(x).get_text())
            df["html_content"] = df["html_content"].map(lambda x : re.sub("[^a-zA-Z1-9]"," ", x)) 
            df["html_content"] = df["html_content"].map(lambda x : re.sub("  ","", x)) 
            df["html_content"] = df["html_content"].map(lambda x : x.lower()) 
        # stop words 未処理
        n_comp = 10
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
        svd = TruncatedSVD(n_components=n_comp)
        feats = [f"countVec_{i}" for i in range(n_comp)]

        #vectorizer.fit(train_df.append(test_df)["html_content"])
        vals = vectorizer.fit_transform(train_df.append(test_df)["html_content"])
        svd.fit(vals)
        tr_svd_feats = pd.DataFrame(svd.transform(vectorizer.transform(train_df["html_content"])), columns=feats)
        te_svd_feats = pd.DataFrame(svd.transform(vectorizer.transform(test_df["html_content"])), columns=feats)
        return tr_svd_feats, te_svd_feats

class tfidfVectorizer_svd(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        for df in [train_df, test_df]:
            df["html_content"] = df["html_content"].map(lambda x : BeautifulSoup(x).get_text())
            df["html_content"] = df["html_content"].map(lambda x : re.sub("[^a-zA-Z1-9]"," ", x)) 
            df["html_content"] = df["html_content"].map(lambda x : re.sub("  ","", x)) 
            df["html_content"] = df["html_content"].map(lambda x : x.lower()) 
        # stop words 未処理
        n_comp = 80
        vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
        svd = TruncatedSVD(n_components=n_comp)
        feats = [f"tfidfVec_{i}" for i in range(n_comp)]

        #vectorizer.fit(train_df.append(test_df)["html_content"])
        vals = vectorizer.fit_transform(train_df.append(test_df)["html_content"])
        svd.fit(vals)
        tr_svd_feats = pd.DataFrame(svd.transform(vectorizer.transform(train_df["html_content"])), columns=feats)
        te_svd_feats = pd.DataFrame(svd.transform(vectorizer.transform(test_df["html_content"])), columns=feats)
        return tr_svd_feats, te_svd_feats

class targetMeanEncode(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        use_cols = []
        col = "country_tgtMean"
        use_cols.append(col)
        mean_country_df = train_df.groupby("country", as_index=False)["state"].mean().rename(columns={"state" : col})
        train_df = pd.merge(train_df, mean_country_df, on=["country"], how="left")
        test_df = pd.merge(test_df, mean_country_df, on=["country"], how="left")
        return train_df[use_cols], test_df[use_cols]
