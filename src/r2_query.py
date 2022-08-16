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

''' Changes versus R1 are:
1- It supports the following features ["top_quality","top_value","top_sellers","top_ratings","top_reviews", "top_matches"]
2- "top_matches" should be the default instead of one of the others "top" as is it the purest from a similarity standpoint
3- IS smart, as it brings back the initial query and factors in similarity
4- IS even smarter for "top_value" and "top_quality" and "top_reviews" as we are extractinv the sentiment from the reviews.
'''

class R2Smart:
    def __init__(self, extra_feature):
        # load asin/title
        with open(f"{dir_path}/../index/M2R1_big_dico.pickle", "rb") as filehandle:
            self.big_index = pickle.load(filehandle)
        
        with open(f"{dir_path}/../index/M2_index_category.pickle", "rb") as filehandle:
            self.index_category = pickle.load(filehandle)
        
        # load index/category   
        # with open(f"{dir_path}/../index/R1_asin_labels.data", 'rb') as filehandle:
        #     self.asin_labels = np.array(pickle.load(filehandle))
        
        ################# CHANGE TO R2 RANK #######################
        self.mtx_load=self.load_sparse_csr(f"{dir_path}/../datasets/r2_data/R2_Rank_mtx")
        self.cols = ["top_quality","top_value","top_sellers","top_ratings","top_reviews"]
        ##########################################################
        
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
                    
    async def lambda_R2(self, q, cat_idx, filters):
        # retrieving asin ranking data based on the big index of the category/asin
        info = self.big_index[cat_idx]
        cat = info[0]
        start = info[1]
        finish = info[2]
        try:
            col_num = self.cols.index(filters)
        except:
            col_num = self.cols.index("top_ratings")

        arr = self.mtx_load
        filtered_arr = arr[start:finish,col_num].toarray()
        
        if self.extra_feature:
            arr1 = self.tfidf_mtx[start:finish,]
            simi = self.extra_feat(q, arr1)
            
            # weight can be adjusted
            weight = 0.2 # is the weight of the ranking in the returned results
            
            # if the feature "top_matches" rank is selected, then weight = 0 to other factors
            # and rank is returned based on similarity only
            top_matches = filters != "top_matches"
            
            #############################################################
            filtered_arr = (filtered_arr * weight * top_matches) + (simi)
            #############################################################
        
        try:
            res_index = np.argsort(-filtered_arr.reshape(1,-1))[0][:7] + start
        except:
            res_index = np.argsort(-filtered_arr.reshape(1,-1))[0] + start

        # result = self.asin_labels[res_index]
        result = await query_products_from_r1_index(res_index.tolist())

        return result

dico ={
    'black backpack for laptop': 628,
    'European AC DC adapter': 91,
    'headphone': 803,
    'I want a Kingston sd card': 536,
    'dell inspiron laptop': 699,
    'canon camera': 153,
    'screen for laptop': 709,
    'garmin sports watch': 984,
    'a tablet Apple': 604,
    'a television': 972,
    'microsoft office': 525,
    'Apple laptop case': 650,
    '64GB kingston SD card': 536,
    'books': 1010,
    'The holy bible': 740,
    'a garmin sports watch forerunner': 798,
    'a waterproof watch garmin': 984
}

if __name__=="__main__":
    smart = True
    obj = R2Smart(extra_feature = smart) 
    liste_filters = obj.cols

    for query, category in dico.items():
        # randomly selects a filter value
        cat_idx = category
        # filters = random.choice(tuple(liste_filters))
        filters = "top_matches"
        print(" ")
        print(" Query: '{}'".format(query))
        print(" Category: {}, Filter: {}".format(obj.index_category[cat_idx], filters))
        result = obj.lambda_R2(query, cat_idx, filters)

        for i, element in enumerate(result):  
            try:
                print(i+1, element[0], element[1][:108])
            except:
                print(i+1, element[0], element[1])
