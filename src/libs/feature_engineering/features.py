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
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.cluster import KMeans
from xfeat import TargetEncoder
import nltk


class default_features(Feature):
    def create_features(self):
        return self.create_default_features()

class countVectorizer_svd(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        TR_SIZE = train_df.shape[0]
        for df in [train_df, test_df]:
            df["html_content"] = df["html_content"].map(lambda x : BeautifulSoup(x).get_text())
            df["html_content"] = df["html_content"].map(lambda x : re.sub("[^a-zA-Z1-9]"," ", x)) 
            df["html_content"] = df["html_content"].map(lambda x : re.sub("  ","", x)) 
            df["html_content"] = df["html_content"].map(lambda x : x.lower()) 
        # stop words 未処理
        n_comp = 10
        stopwords = nltk.corpus.stopwords.words('english')
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=stopwords, max_features=12000)
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        feats = [f"countVec_{i}" for i in range(n_comp)]

        #vectorizer.fit(train_df.append(test_df)["html_content"])
        vals = vectorizer.fit_transform(train_df.append(test_df)["html_content"])
        vals = svd.fit_transform(vals)
        tr_svd_feats = pd.DataFrame(vals[:TR_SIZE], columns=feats)
        te_svd_feats = pd.DataFrame(vals[TR_SIZE:], columns=feats)
        return tr_svd_feats, te_svd_feats

class tfidfVectorizer_svd(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        TR_SIZE = train_df.shape[0]
        for df in [train_df, test_df]:
            df["html_content"] = df["html_content"].map(lambda x : BeautifulSoup(x).get_text())
            df["html_content"] = df["html_content"].map(lambda x : re.sub("[^a-zA-Z1-9]"," ", x)) 
            df["html_content"] = df["html_content"].map(lambda x : re.sub("  ","", x)) 
            df["html_content"] = df["html_content"].map(lambda x : x.lower()) 
        # stop words 未処理
        n_comp = 80
        stopwords = nltk.corpus.stopwords.words('english')
        vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=stopwords, max_features=12000)
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        feats = [f"tfidfVec_{i}" for i in range(n_comp)]

        #vectorizer.fit(train_df.append(test_df)["html_content"])
        vals = vectorizer.fit_transform(train_df.append(test_df)["html_content"])
        vals = svd.fit_transform(vals)
        tr_svd_feats = pd.DataFrame(vals[:TR_SIZE], columns=feats)
        te_svd_feats = pd.DataFrame(vals[TR_SIZE:], columns=feats)
        return tr_svd_feats, te_svd_feats

class targetMeanEncode(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        use_cols = []
        for col in ["country", "goal", "duration", "category1"]: #, "category2"
            feat_name = f"{col}_tgtMean"
            use_cols.append(feat_name)
            mean_df = train_df.groupby(col, as_index=False)["state"].mean().rename(columns={"state" : feat_name})
            train_df = pd.merge(train_df, mean_df, on=[col], how="left")
            test_df = pd.merge(test_df, mean_df, on=[col], how="left")

        return train_df[use_cols], test_df[use_cols]

class oof_targetMeanEncode(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        use_cols = []
        enc_cols = ["country", "goal", "duration", "category1", "category2"]
        colss = [
            ["goal", "duration"],
            ["goal", "country"],
            ["duration", "country"],
            ["country", "category1"],
            #["duration", "category2"],

        ]

        for cols in colss:
            feat_name = "_".join(cols)
            enc_cols.append(feat_name)
            for df in [train_df, test_df]: 
                df[feat_name] = ""
                df[feat_name] = df[feat_name].str.cat(df[col].astype(str) for col in cols)

        for col in enc_cols:
            feat_name = f"{col}_oof-tgtMean"
            use_cols.append(feat_name)
            kf = KFold(n_splits=5, shuffle=False, random_state=0)
            encoder = TargetEncoder(
                input_cols=[col], 
                target_col="state",
                fold=kf,
                output_suffix="_oof-tgtMean"
            )

            train_df = encoder.fit_transform(train_df)
            test_df = encoder.transform(test_df)

        return train_df[use_cols], test_df[use_cols]


class coutEncode(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        use_cols = []
        enc_cols = ["country", "goal", "duration", "category1", "category2"]
        colss = [
            ["goal", "duration"],
            ["goal", "country"],
            ["duration", "country"],
            ["country", "category1"],
        ]

        for cols in colss:
            feat_name = "_".join(cols)
            enc_cols.append(feat_name)
            for df in [train_df, test_df]: 
                df[feat_name] = ""
                df[feat_name] = df[feat_name].str.cat(df[col].astype(str) for col in cols)

        for col in enc_cols:
            feat_name = f"{col}_counts"
            use_cols.append(feat_name)
            temp_df = train_df.append(test_df)
            vc = temp_df[col].value_counts()

            train_df[feat_name] = train_df[col].map(vc)
            test_df[feat_name] = test_df[col].map(vc)

        return train_df[use_cols], test_df[use_cols]

class prod_features(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        use_cols = []
        colss = [
            ["goal", "duration"],
            ["goal", "country"],
            ["country", "category1"],
            ["duration", "country"],
        ]

        for cols in colss:
            feat_name = "_".join(cols)
            use_cols.append(feat_name)
            for df in [train_df, test_df]: 
                df[feat_name] = ""
                df[feat_name] = df[feat_name].str.cat(df[col].astype(str) for col in cols)
        return train_df[use_cols], test_df[use_cols]

class tfidfVectorizer_tsne(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        TR_SIZE = train_df.shape[0]
        for df in [train_df, test_df]:
            df["html_content"] = df["html_content"].map(lambda x : BeautifulSoup(x).get_text())
            df["html_content"] = df["html_content"].map(lambda x : re.sub("[^a-zA-Z1-9]"," ", x)) 
            df["html_content"] = df["html_content"].map(lambda x : re.sub("  ","", x)) 
            df["html_content"] = df["html_content"].map(lambda x : x.lower()) 
        # stop words 未処理
        n_comp = 3
        vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
        tsne = TSNE(n_components=n_comp, random_state=0, n_iter=1000)
        feats = [f"tfidfT-SNE_{i}" for i in range(n_comp)]

        #vectorizer.fit(train_df.append(test_df)["html_content"])
        vals = vectorizer.fit_transform(train_df.append(test_df)["html_content"])
        vals = tsne.fit_transform(vals)
        tr_tsne_feats = pd.DataFrame(vals[:TR_SIZE], columns=feats)
        te_tsne_feats = pd.DataFrame(vals[TR_SIZE:], columns=feats)
        return tr_tsne_feats, te_tsne_feats

class TSNE_Kmeans(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        TR_SIZE = train_df.shape[0]
        n_comp=3
        train_df = pd.concat([train_df, pd.read_feather(osp.join(self.ROOT, "features", "train", "tfidfVectorizer_tsne.feather"))], axis=1)
        test_df = pd.concat([test_df, pd.read_feather(osp.join(self.ROOT, "features", "test", "tfidfVectorizer_tsne.feather"))], axis=1)
        feats = [f"tfidfT-SNE_{i}" for i in range(n_comp)]
        km = KMeans(random_state=0)
        temp_df = train_df.append(test_df)
        vals = km.fit(temp_df[feats]).labels_
        train_df["KMeans"] = vals[:TR_SIZE]
        test_df["KMeans"] = vals[TR_SIZE:]
        return pd.get_dummies(train_df["KMeans"],prefix="KMeans_"), pd.get_dummies(test_df["KMeans"],prefix="KMeans_")

class word_Count(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        for df in [train_df, test_df]:
            df["html_content"] = df["html_content"].map(lambda x : BeautifulSoup(x).get_text())
            df["html_content"] = df["html_content"].map(lambda x : re.sub("[^a-zA-Z1-9]"," ", x)) 
            df["html_content"] = df["html_content"].map(lambda x : re.sub("  ","", x)) 
            df["html_content"] = df["html_content"].map(lambda x : x.lower()) 
            df["word_Count"] = df["html_content"].map(lambda x : len(x.split()))
        # stop words 未処理
        return train_df[["word_Count"]], test_df[["word_Count"]]

class html_tagCount(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        use_cols = ["a_tagCount", "img_tagCount", "p_tagCount"]
        for df in [train_df, test_df]:
            df["soupobj"] = df["html_content"].map(lambda x : BeautifulSoup(x))
            df["a_tagCount"] = df["soupobj"].map(lambda x : len(x.find_all('a')))
            df["img_tagCount"] = df["soupobj"].map(lambda x : len(x.find_all('img')))
            df["p_tagCount"] = df["soupobj"].map(lambda x : len(x.find_all('p')))

        return train_df[use_cols], test_df[use_cols]

class boolean_feats(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        use_cols = []
        cat2_g = train_df.groupby("category2")["state"].mean()
        bool_vals = cat2_g[(cat2_g >= 0.9) | (cat2_g <= 0.1)].index.to_list()
        for val in bool_vals:
            feat_name = f"is_{val}"
            use_cols.append(feat_name)
            for df in [train_df, test_df]:
                df[feat_name] = (df["category2"] == val).astype(int).values
            

        return train_df[use_cols], test_df[use_cols]
