"""
M1 Model for subcategory selection.
This model depends on the following files:
- vectorizer: datasets/m1_data/sk_vectorizer.joblib
- lsi_text_transformed: datasets/m1_data/lsi_text_transformed.joblib
- model: models/lsi.joblib
- lsi_index: index/sk_lsi_index.joblib
- index_category: index/M2_index_category.pickle
"""
import os
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

dir_path = os.path.dirname(os.path.realpath(__file__))

class M1Lsi:
    def __init__(self):
        self.vectorizer = joblib.load(f"{dir_path}/../datasets/m1_data/sk_vectorizer.joblib")
        self.lsi_text_transformed = joblib.load(f"{dir_path}/../datasets/m1_data/lsi_text_transformed.joblib", mmap_mode=None)
        self.model = joblib.load(f"{dir_path}/../models/lsi.joblib", mmap_mode=None)
        self.lsi_index = joblib.load(f"{dir_path}/../index/sk_lsi_index.joblib", mmap_mode=None)

        with open(f"{dir_path}/../index/M2_index_category.pickle", "rb") as filehandle:
            self.index_category = pickle.load(filehandle)

    def query_category(self, test_query, threshold=0.86, group_count=3):
        Xquery = self.vectorizer.transform([test_query])
        q_topic_lsi = self.model.transform(Xquery)
        lsi_simularity = cosine_similarity(q_topic_lsi, self.lsi_text_transformed)
        ndf = pd.DataFrame(self.lsi_index)
        ndf['lsi_sim'] = lsi_simularity[0]
        ndf = ndf.groupby(0).mean().sort_values(by = 'lsi_sim', ascending=False).reset_index()
        ndf = ndf[(ndf[1] > group_count) & (ndf['lsi_sim'] > threshold)]

        result_obj = []
        if len(ndf) > 0:
            ndf = ndf[0:5].reset_index(drop = True)
            for x in range(len(ndf)):
                curr_idx = int(ndf.loc[x,0])
                item_obj = {
                    "label": self.index_category[curr_idx],
                    "index": curr_idx,
                }
                result_obj.append(item_obj)

        return result_obj


if __name__=="__main__":
    m1_obj = M1Lsi()

    res = m1_obj.query_category("ps4", threshold=0.86)
    print(res)
