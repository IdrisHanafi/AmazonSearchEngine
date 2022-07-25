"""
M2 Models
Dependencies:
- 'datasets/m2_data/M2_tfidf_mtx'
- 'models/M2_tfidf.joblib'
- 'index/M2_category_labels.data'
- 'datasets/m2_data/M2_rank.data'
- 'index/M2_category_index.pickle'

Lambda function: takes a query and returns the top 5 category keys of recommendations
INPUT files: "M2_tfidf_mtx.csv" , "M2.joblib" ,  "M2_categories.json" + query
OUTPUT: top 5 tuples of matching categoris
1- Creates a class tfidf and download dependencies in the first instance, tries to save in cache
2- Takes each query and vectorize
3- Calculate simirality tuples
4- Ranks recommendation based on similarity and np.log(count of asin in each category
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

dir_path = os.path.dirname(os.path.realpath(__file__))

class tfidf:
    def __init__(self):
        # load cleaned group_by data sets and models from disk
        self.mtx=self.load_sparse_csr(f"{dir_path}/../datasets/m2_data/M2_tfidf_mtx")        
        self.M2 = joblib.load(f"{dir_path}/../models/M2_tfidf.joblib")
        self.stemmer = PorterStemmer() 
        
        with open(f"{dir_path}/../index/M2_category_labels.data", "rb") as filehandle:
            self.category = pickle.load(filehandle)
            
        with open(f"{dir_path}/../datasets/m2_data/M2_rank.data", "rb") as filehandle:
            self.rank = np.array(pickle.load(filehandle))
            
        with open(f"{dir_path}/../index/M2_category_index.pickle", "rb") as filehandle:
            self.category_index = pickle.load(filehandle)

            
    def load_sparse_csr(self, filename):
        # here we need to add .npz extension manually
        loader = np.load(filename + '.npz')
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape']) 
        
    ## retrieve match tfidf (LAMBDA FUNCTION)
    def M2_query_to_category_function(self,q):
        stop_words= set(stopwords.words('english'))
        liste=[self.stemmer.stem(w) for w in word_tokenize(q) if not w.lower() in stop_words]
        string=' '.join(element for element in liste)       
        quer = self.M2.transform([string])
        simi = cosine_similarity(self.mtx,quer) 
        
        simi_mtx = simi * self.rank # rank is a coeff that you can tweak np.log(gb["count"] + 1) * 0.35
        results = list(self.category[np.argsort(-simi_mtx,axis=0)[:5].reshape(5,)])    
        # print(results)
        # return results, np.round(np.max(simi),2)

        result_obj = {}
        for res_item in results:
            result_obj[self.category_index[res_item]] = res_item
        
        return result_obj, results, np.round(np.max(simi),2)


def make_calls():
    ob=tfidf()    

    k=["AC adapter", "I want a screen","DESKTOP HP","headphone", "I want an sd card",
          "computer","camera","screen","car GPS","sports watch",
          "a tablet Apple","a television","microsoft office","laptop case",
           "data science","hewlet packard"]

    dico={}
    for q in k:
        print(q)
        result_obj, result, maxi = ob.M2_query_to_category_function(q)
        
        if maxi> 0.2:
            print("Query: {}, simi_max {}".format(q,maxi))
            for i,category in enumerate(result):
                if i==0:
                    dico[q]=[category,ob.category_index[category]]
                print(i+1," ",category)
            print(" ")
        else:
            print("There is no sufficient match for '{}', please enter new terms".format(q))
            print(" ")


if __name__=="__main__":
    make_calls()
