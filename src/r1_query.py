"""
R1 Models
Dependencies:
- "datasets/R1_Rank_mtx"
- "index/R1_asin_labels.data"
- "index/M2R1_big_dico.pickle"
"""
import pandas as pd
import random
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
## tf_idf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import joblib
import gensim
import gzip
import json
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
import pickle
# from keras.preprocessing.text import Tokenizer
import collections
from collections import Counter
import re as regex
from scipy.sparse import csr_matrix
import os
import warnings
# warnings.filterwarnings('ignore')
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from gensim import utils
from bs4 import BeautifulSoup as BSHTML
from gensim.parsing.preprocessing import preprocess_string, STOPWORDS #, remove_stopword_tokens

from db.query_product import (
    query_products_from_r1_index
)

dir_path = os.path.dirname(os.path.realpath(__file__))

class R1Baseline:
    def __init__(self, extra_feature):
        # load asin/title
        with open(f"{dir_path}/../index/M2R1_big_dico.pickle", "rb") as filehandle:
            self.big_index = pickle.load(filehandle)
        
        with open(f"{dir_path}/../index/M2_index_category.pickle", "rb") as filehandle:
            self.index_category = pickle.load(filehandle)
        
        # load index/category   
        # with open(f"{dir_path}/../index/R1_asin_labels.data", "rb") as filehandle:
            # self.asin_labels = np.array(pickle.load(filehandle))
            
        self.mtx_load=self.load_sparse_csr(f"{dir_path}/../datasets/r1_data/R1_Rank_mtx")
        self.cols = ["top_features","top_value","top_sellers","top_ratings"]
        self.extra_feature = extra_feature
        self.stemmer = PorterStemmer() 
        
        if self.extra_feature:
            self.tfidf_mtx = self.load_sparse_csr(f"{dir_path}/../datasets/r1_data/R1_tfidf_mtx")
            self.model = joblib.load(f"{dir_path}/../models/R1_tfidf.joblib")
            
    def load_sparse_csr(self, filename):
        # here we need to add .npz extension manually
        loader = np.load(filename + '.npz')
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
    
    def extra_feat(self, q, arr1):
        stop_words= set(stopwords.words('english'))
        liste=[self.stemmer.stem(w) for w in word_tokenize(q) if not w.lower() in stop_words]
        string=' '.join(element for element in liste)       
        quer =self.model.transform([string])
        
        simi = cosine_similarity(arr1,quer) 
        
        return simi
                    
    async def lambda_R1(self, q, cat_idx, filters):
        # retrieving data
        info = self.big_index[cat_idx]
        cat = info[0]
        start = info[1]
        finish = info[2]
        col_num = self.cols.index(filters)

        arr = self.mtx_load
        filtered_arr = arr[start:finish,col_num].toarray()
        # print("filtered", filtered_arr.shape)
        
        if self.extra_feature:
            arr1 = self.tfidf_mtx[start:finish,]
            simi = self.extra_feat(q, arr1)
            # weight can be adjusted
            # print("simi", simi.shape)
            
            filtered_arr = (filtered_arr * .3) + (simi *.7)
            # print("filtered_arr", filtered_arr.shape)
        
        try:
            res_index = np.argsort(-filtered_arr.reshape(1,-1))[0][:7] + start
        except:
            res_index = np.argsort(-filtered_arr.reshape(1,-1))[0] + start
        print(" ")
        print(" Query: '{}'".format(q))
        print(" Category: {}\n Filter: '{}'".format(cat, filters))
        print(" ")

        # result = self.asin_labels[res_index]

        result = await query_products_from_r1_index(res_index.tolist())
        return result

def test_run():
    smart = True
    obj = R1Baseline(extra_feature = smart) 
    dico = {'a tablet Apple 64GB': 604}

    liste_filters = ["top_features","top_value","top_sellers","top_ratings"]
    for query, category in dico.items():
        # randomly selects a filter value
        cat_idx = category
        filters = random.choice(tuple(liste_filters))
        filters = "top_sellers"
        result = obj.lambda_R1(query, cat_idx, filters)

        for i, element in enumerate(result):
            print(element)

if __name__=="__main__":
    test_run()
